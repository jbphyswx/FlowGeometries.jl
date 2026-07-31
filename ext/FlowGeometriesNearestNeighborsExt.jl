module FlowGeometriesNearestNeighborsExt

using NearestNeighbors: NearestNeighbors
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# k-d-tree neighbor construction for `UnstructuredGrid` (overrides the throwing fallback).
# Cartesian: tree on (x, y). Spherical: tree on the unit-sphere embedding, where nearest-by-chord is
# exactly nearest-by-great-circle — which also makes longitude wrap for free, since λ and λ+2π embed
# to the same point.

"""
    _shift_set(periodic, period)

Offsets to replicate a point set by: `(0,)` in a non-wrapping direction, `(0, -L, +L)` in a wrapping
one. Zero comes first so the originals occupy the first block of the replicated set.
"""
@inline _shift_set(p::Bool, L::T) where {T} = p ? (zero(T), -L, L) : (zero(T),)

"""
    _ghost_points(pts, periodic, period) -> (all_pts, nghost)

Replicate the `D × N` point matrix once per combination of periodic image offsets, originals in the
first `N` columns. A wrapping domain is searched by placing the images and running an ordinary
Euclidean query over them: a node near one face then finds the nodes across the opposite face at
their true wrapped separation.
"""
function _ghost_points(
    pts::AbstractMatrix{T}, periodic::NTuple{D,Bool}, period::NTuple{D,T},
) where {D,T}
    shifts = ntuple(d -> _shift_set(periodic[d], period[d]), Val(D))
    N = size(pts, 2)
    ng = prod(map(length, shifts))
    out = similar(pts, D, N * ng)
    g = 0
    # Column-major over the per-direction shift sets, so the all-zero combination comes first and the
    # originals occupy columns `1:N`.
    for ci in CartesianIndices(map(eachindex, shifts))
        cols = (g * N + 1):((g + 1) * N)
        for d in 1:D
            δ = shifts[d][ci[d]]
            @views out[d, cols] .= pts[d, :] .+ δ
        end
        g += 1
    end
    return out, ng
end

"""
    _accept_candidates!(nbrs, base, cands, i, N, kmax) -> Int

Append the candidates for node `i` to `nbrs` (which already holds `base` entries for this node),
mapping periodic images back to ORIGINAL indices and dropping self-references and repeats — one node
can be reached through several images. Nearest-first order is preserved. Returns the new count for
this node, stopping at `kmax`.
"""
@inline function _accept_candidates!(nbrs::Vector{Int}, lo::Int, m::Int, cands, i::Int, N::Int, kmax::Int)
    @inbounds for t in cands
        m == kmax && break
        j = mod1(t, N)          # periodic image → original node
        j == i && continue
        seen = false
        for q in 1:m
            if nbrs[lo + q] == j
                seen = true
                break
            end
        end
        seen && continue
        m += 1
        nbrs[lo + m] = j
    end
    return m
end

# Queries go one point at a time through `knn!`/`inrange!` into reused buffers; the batch forms
# return a `Vector{Vector{…}}`, i.e. two heap vectors per query point.
#
# The loops sit behind a FUNCTION BARRIER: `KDTree(pts)` infers only to
# `KDTree{V,Euclidean,Float64,V1} where {V<:SVector{_,Float64}, V1<:AbstractVector}`, so calling
# `knn!` on it in the same function is a dynamic dispatch costing two allocations per query.
# Passing the tree as an argument makes it concrete.

function _knn_loop!(nbrs::Vector{Int}, ptr::Vector{Int}, tree, pts::AbstractMatrix, N::Int, kq::Int, ask::Int)
    ibuf = Vector{Int}(undef, ask)
    dbuf = Vector{float(eltype(pts))}(undef, ask)   # knn! requires exactly the tree's distance type
    @inbounds ptr[1] = 1
    @inbounds for i in 1:N
        NearestNeighbors.knn!(ibuf, dbuf, tree, view(pts, :, i), ask, true)
        ptr[i + 1] = ptr[i] + _accept_candidates!(nbrs, (i - 1) * kq, 0, ibuf, i, N, kq)
    end
    # Compact the fixed-stride blocks down onto the exact CSR offsets.
    @inbounds for i in 1:N
        src = (i - 1) * kq
        dst = ptr[i] - 1
        if dst != src
            for q in 1:(ptr[i + 1] - ptr[i])
                nbrs[dst + q] = nbrs[src + q]
            end
        end
    end
    resize!(nbrs, ptr[end] - 1)
    return nbrs, ptr
end

function _csr_from_knn(pts::AbstractMatrix, k::Integer, ng::Int = 1)
    N = size(pts, 2) ÷ ng
    if N < 2
        return Int[], ones(Int, N + 1)
    end
    kq = min(Int(k), N - 1)
    # Ask for enough candidates that `kq` DISTINCT originals survive even when several images of the
    # same node sit closer than the next real neighbor.
    ask = min(kq * ng + 1, size(pts, 2))
    return _knn_loop!(
        Vector{Int}(undef, N * kq), Vector{Int}(undef, N + 1),
        NearestNeighbors.KDTree(pts), pts, N, kq, ask,
    )
end

function _radius_loop!(ptr::Vector{Int}, tree, pts::AbstractMatrix, N::Int, r::Real)
    # A radius query has no fixed degree bound, so CSR is grown directly rather than sized from a
    # `maximum(length, lists)` pass over materialized lists.
    #
    # This does not reach O(1) allocations the way the knn path does: `inrange!` costs 3 per query
    # inside `_inrange`'s own setup, not in the candidate buffer — `inrangecount` over the same
    # views and tree allocates nothing.
    nbrs = Int[]
    sizehint!(nbrs, 8N)
    cands = Int[]                     # `inrange!` pushes into this; `empty!` keeps the capacity
    @inbounds ptr[1] = 1
    for i in 1:N
        empty!(cands)
        NearestNeighbors.inrange!(cands, tree, view(pts, :, i), r, true)
        lo = length(nbrs)
        resize!(nbrs, lo + length(cands))
        m = _accept_candidates!(nbrs, lo, 0, cands, i, N, length(cands))
        resize!(nbrs, lo + m)
        @inbounds ptr[i + 1] = ptr[i] + m
    end
    return nbrs, ptr
end

function _csr_from_radius(pts::AbstractMatrix, r::Real, ng::Int = 1)
    N = size(pts, 2) ÷ ng
    if N < 2
        return Int[], ones(Int, N + 1)
    end
    return _radius_loop!(Vector{Int}(undef, N + 1), NearestNeighbors.KDTree(pts), pts, N, r)
end

# Points go to the tree as a contiguous `D × N` matrix: `KDTree` takes that form directly, and it
# avoids one heap allocation per node.
function Grids._build_kdtree_neighbors(
    ::Geometry.AbstractCartesianGeometry{T}, coords::NTuple{D,AbstractVector{T}};
    k::Integer = 6, radius::Union{Nothing,Real} = nothing,
    periodic::NTuple{D,Bool} = ntuple(_ -> false, Val(D)),
    period = ntuple(_ -> zero(T), Val(D)),
) where {D, T<:AbstractFloat}
    n = length(coords[1])
    pts = similar(coords[1], T, D, n)
    for d in 1:D
        @views pts[d, :] .= coords[d]
    end
    all_pts, ng = any(periodic) ? _ghost_points(pts, periodic, NTuple{D,T}(period)) : (pts, 1)
    return radius === nothing ? _csr_from_knn(all_pts, k, ng) :
                                _csr_from_radius(all_pts, T(radius), ng)
end

# `(λ, φ)` embeds on the unit sphere and `(λ, φ, r)` at its own radius: nearest-by-chord is then
# nearest-by-great-circle, and longitude wraps for free because λ and λ+2π embed to the same point —
# so no ghost images are needed in either case.
function Grids._build_kdtree_neighbors(
    geo::Geometry.AbstractSphericalGeometry{T}, coords::NTuple{D,AbstractVector{T}};
    k::Integer = 6, radius::Union{Nothing,Real} = nothing,
    periodic::NTuple{D,Bool} = ntuple(d -> d == 1, Val(D)),
    period = ntuple(d -> d == 1 ? T(2π) : zero(T), Val(D)),
) where {D, T<:AbstractFloat}
    D == 2 || D == 3 || throw(ArgumentError(
        "a spherical node set is `(λ, φ)` or `(λ, φ, r)`; got $D coordinate vectors",
    ))
    λ, φ = coords[1], coords[2]
    n = length(λ)
    pts = similar(λ, T, 3, n)
    if D == 2
        @views pts[1, :] .= cos.(φ) .* cos.(λ)
        @views pts[2, :] .= cos.(φ) .* sin.(λ)
        @views pts[3, :] .= sin.(φ)
    else
        r = coords[3]
        @views pts[1, :] .= r .* cos.(φ) .* cos.(λ)
        @views pts[2, :] .= r .* cos.(φ) .* sin.(λ)
        @views pts[3, :] .= r .* sin.(φ)
    end
    radius === nothing && return _csr_from_knn(pts, k)
    # In 2-D the embedding is the UNIT sphere, so a physical arc has to become a unit chord; in 3-D the
    # embedding carries true lengths and the radius is already a chord.
    #
    # `2sin(σ/2)` is only monotone in `σ` up to `σ = π`, and turns back down after: past an antipodal
    # radius it would shrink, reaching zero at a full circumference and excluding everything. An arc of
    # `πR` or more already spans the sphere, so the chord saturates at the diameter.
    rq = if D == 2
        σ = T(radius) / Geometry.radius(geo)
        σ ≥ T(π) ? T(2) : T(2) * sin(σ / T(2))
    else
        T(radius)
    end
    return _csr_from_radius(pts, rq)
end

# ---------------------------------------------------------------------------
# Reusable spatial index for ball queries
# ---------------------------------------------------------------------------
#
# The construction path above builds a tree and discards it, having no more use for it. This one is kept
# and queried repeatedly, and it goes through the same embedding, so the two cannot diverge.
#
# `index_within!` is only required to return a SUPERSET of the ball — the caller re-tests every candidate
# with the geometry's own `distance`. That is what lets the embedding be a lower bound rather than exact,
# which matters for the ellipsoid: the ECEF chord is ≤ the geodesic, so a chord query over-returns and the
# caller's gate trims it.

struct BallIndex{TR,T}
    tree::TR
    pts::Matrix{T}   # the embedded points, reused as query vectors
    n::Int           # originals; a ghosted tree holds `n * ng` columns
    embed::Symbol    # :cartesian, or :arc / :chord for what the embedded distance means
    R::T             # sphere radius, for the arc→chord transform; zero where there is none
end

# One embedding routine for every architecture: cell centres come out of `Grids._raw_coords`, which every
# grid type provides, so nothing here needs to know how coordinates are stored.
function _grid_points(grid::Grids.AbstractGrid{G,T}) where {G,T}
    msk = Grids.mask(grid)
    n = length(msk)
    lin = LinearIndices(size(msk))
    first_pt = Grids._raw_coords(grid, Tuple(first(CartesianIndices(size(msk))))...)
    D = length(first_pt)
    raw = Matrix{T}(undef, D, n)
    @inbounds for ci in CartesianIndices(size(msk))
        p = Grids._raw_coords(grid, Tuple(ci)...)
        for d in 1:D
            raw[d, lin[ci]] = p[d]
        end
    end
    return raw, D
end

function _grid_points(grid::Grids.UnstructuredGrid{T,G,N}) where {T,G,N}
    n = length(Grids.mask(grid))
    raw = Matrix{T}(undef, N, n)
    @inbounds for k in 1:n
        p = Grids._raw_coords(grid, k)
        for d in 1:N
            raw[d, k] = p[d]
        end
    end
    return raw, N
end

function _embed(geo::Geometry.AbstractCartesianGeometry{T}, raw::Matrix{T}, grid) where {T}
    D = size(raw, 1)
    per = ntuple(d -> Grids.isperiodic(grid, d), D)
    prd = ntuple(d -> T(Grids.period(grid, d)), D)
    pts, ng = any(per) ? _ghost_points(raw, per, prd) : (raw, 1)
    return pts, ng, :cartesian, zero(T)
end

# `spherical_to_cartesian` rather than the same formula written again here: at `(λ, φ, r)` the metric IS
# the Euclidean chord of this embedding, so sharing the transform is what makes the bound below exact
# rather than merely close. Longitude needs no ghost images either way — `λ` and `λ+2π` embed to the same
# point.
function _embed(geo::Geometry.AbstractSphericalGeometry{T}, raw::Matrix{T}, _grid) where {T}
    D = size(raw, 1)
    n = size(raw, 2)
    pts = Matrix{T}(undef, 3, n)
    @inbounds for k in 1:n
        c = Geometry.spherical_to_cartesian(geo, ntuple(d -> raw[d, k], D))
        pts[1, k] = c.x; pts[2, k] = c.y; pts[3, k] = c.z
    end
    # At two coordinates the embedding sits on the reference sphere and the metric is the ARC, so a
    # radius has to be converted; at three it is the chord already, and passes through.
    return pts, 1, D ≥ 3 ? :chord : :arc, T(Geometry.radius(geo))
end

# The ECEF chord is a lower bound on the geodesic, so a chord query over-returns — which is exactly what
# an index is allowed to do, the caller's Vincenty gate deciding membership.
function _embed(geo::Geometry.AbstractEllipsoidalGeometry{T}, raw::Matrix{T}, _grid) where {T}
    n = size(raw, 2)
    D = size(raw, 1)
    pts = Matrix{T}(undef, 3, n)
    @inbounds for k in 1:n
        c = Geometry.geodetic_to_cartesian(geo, ntuple(d -> raw[d, k], D))
        pts[1, k] = c.x; pts[2, k] = c.y; pts[3, k] = c.z
    end
    return pts, 1, :chord, zero(T)
end

const _IndexableGrid = Union{Grids.StructuredGrid,Grids.CurvilinearGrid,Grids.UnstructuredGrid}

Grids.has_spatial_index(::_IndexableGrid) = true

# Dispatched on the concrete architectures, so this ADDS a method rather than overwriting the
# `AbstractGrid` fallback in `Grids` — overwriting is an error during precompilation.
function Grids.spatial_index(
    grid::_IndexableGrid,
)
    raw, _D = _grid_points(grid)
    geo = Grids.grid_geometry(grid)
    pts, ng, embed, R = _embed(geo, raw, grid)
    n = size(pts, 2) ÷ ng
    return BallIndex(NearestNeighbors.KDTree(pts), pts, n, embed, R)
end

# A physical radius as a radius in the embedding. `2sin(σ/2)` is monotone only to `σ = π`, so an arc of an
# antipodal distance or more saturates at the diameter instead of turning back down.
@inline function _query_radius(ix::BallIndex{TR,T}, r::Real) where {TR,T}
    ix.embed === :arc || return T(r)
    σ = T(r) / ix.R
    return σ ≥ T(π) ? T(2) * ix.R : T(2) * ix.R * sin(σ / T(2))
end

function Grids.index_within!(buf::AbstractVector{<:Integer}, ix::BallIndex, grid, I, r)
    lin = I isa Integer ? Int(I) : LinearIndices(size(Grids.mask(grid)))[CartesianIndex(I)]
    empty!(buf)
    # `inrange!` pushes into a caller-owned buffer; the batch `inrange` allocates a fresh vector per
    # query, which dominates the cost of a small ball.
    NearestNeighbors.inrange!(buf, ix.tree, view(ix.pts, :, lin), _query_radius(ix, r), false)
    # Ghost images collapse back onto their originals. The caller's distance gate and its own
    # minimum-image reduction decide membership, so a duplicate is harmless — but it would make the
    # fold visit one cell twice, which a counting or summing `f` would get wrong.
    if ix.n != size(ix.pts, 2)
        @inbounds for t in eachindex(buf)
            buf[t] = mod1(buf[t], ix.n)
        end
    end
    # Ascending, so an indexed query returns exactly what the unindexed scan returns — same list, same
    # fold order, same CSR bytes — and loading this extension changes speed and nothing else. It also
    # makes the dedup above a linear pass instead of `unique!`'s hash set. `QuickSort` because the
    # default stable algorithm allocates a scratch buffer, which is the cost this path exists to avoid.
    sort!(buf; alg = Base.Sort.QuickSort)
    return _dedup_sorted!(buf)
end

@inline function _dedup_sorted!(v::AbstractVector{<:Integer})
    length(v) < 2 && return v
    w = 1
    @inbounds for t in 2:length(v)
        v[t] == v[w] && continue
        w += 1
        v[w] = v[t]
    end
    return resize!(v, w)
end

end # module
