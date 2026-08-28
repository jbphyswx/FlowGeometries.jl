module FlowGeometriesNearestNeighborsExt

using NearestNeighbors: NearestNeighbors
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# k-d-tree neighbor construction for `UnstructuredGrid` (overrides the throwing fallback).
# Cartesian: tree on (x, y). Spherical: tree on the unit-sphere embedding, where nearest-by-chord is
# exactly nearest-by-great-circle — which also makes longitude wrap for free, since λ and λ+2π embed
# to the same point.

# Point replication and the per-geometry embedding live in `Grids`, so this extension and the in-package
# index search the same space by construction rather than by two implementations agreeing.
using FlowGeometries.Grids: _ghost_points

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
#
# Every query is made in the tree's OWN point type. NearestNeighbors converts anything else per call,
# and that conversion is the whole per-query allocation: measured against this tree, a column view
# costs 224 B and a plain `Vector` 176 B, where the static point costs nothing.
@inline _qpoint(::NearestNeighbors.NNTree{V}, q::NTuple) where {V} = V(q)

@inline _qcol(::NearestNeighbors.NNTree{V}, pts::AbstractMatrix, j::Integer) where {V} =
    V(ntuple(d -> @inbounds(pts[d, j]), Val(length(V))))

function _knn_loop!(nbrs::Vector{Int}, ptr::Vector{Int}, tree, pts::AbstractMatrix, N::Int, kq::Int, ask::Int)
    ibuf = Vector{Int}(undef, ask)
    dbuf = Vector{float(eltype(pts))}(undef, ask)   # knn! requires exactly the tree's distance type
    @inbounds ptr[1] = 1
    @inbounds for i in 1:N
        NearestNeighbors.knn!(ibuf, dbuf, tree, _qcol(tree, pts, i), ask, true)
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
    nbrs = Int[]
    sizehint!(nbrs, 8N)
    cands = Int[]                     # `inrange!` pushes into this; `empty!` keeps the capacity
    @inbounds ptr[1] = 1
    for i in 1:N
        empty!(cands)
        NearestNeighbors.inrange!(cands, tree, _qcol(tree, pts, i), r, true)
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
    # At the reference radius, not the unit sphere, so this is the same space `Grids.embedded_points`
    # produces and the one radius conversion below serves both. Scaling leaves the knn ordering alone.
    R = T(Geometry.radius(geo))
    emb = if D == 2
        @views pts[1, :] .= R .* cos.(φ) .* cos.(λ)
        @views pts[2, :] .= R .* cos.(φ) .* sin.(λ)
        @views pts[3, :] .= R .* sin.(φ)
        Grids.ArcEmbedding(R)
    else
        r = coords[3]
        @views pts[1, :] .= r .* cos.(φ) .* cos.(λ)
        @views pts[2, :] .= r .* cos.(φ) .* sin.(λ)
        @views pts[3, :] .= r .* sin.(φ)
        Grids.ChordEmbedding()
    end
    radius === nothing && return _csr_from_knn(pts, k)
    return _csr_from_radius(pts, Grids.embedded_radius(emb, T(radius)))
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

# The embedding is a type parameter, so the radius conversion below resolves at compile time instead of
# branching on a stored tag once per query.
struct BallIndex{TR,T,MT<:AbstractMatrix{T},E<:Grids.AbstractEmbedding}
    tree::TR
    pts::MT          # the embedded points, reused as query vectors
    n::Int           # originals; a ghosted tree holds `n * ng` columns
    embedding::E
    # Shortest nonzero period among the replicated directions, `Inf` where nothing is replicated. Two
    # images of one cell are a whole lattice vector apart, so both can be within `r` of a query only if
    # `2r` reaches this — below it a query cannot return a duplicate and needs no dedup at all.
    min_period::T
end

const _IndexableGrid = Union{Grids.StructuredGrid,Grids.CurvilinearGrid,Grids.UnstructuredGrid}

Grids.has_spatial_index(::_IndexableGrid) = true

# Dispatched on the concrete architectures, so this ADDS a method rather than overwriting the
# `AbstractGrid` fallback in `Grids` — overwriting is an error during precompilation.
function Grids.spatial_index(grid::_IndexableGrid)
    pts, ng, embedding = Grids.embedded_points(grid)
    T = eltype(pts)
    D = Grids.ncoordinates(grid)
    minper = ng == 1 ? T(Inf) :
        minimum(T(Grids.isperiodic(grid, d) ? Grids.period(grid, d) : Inf) for d in 1:D)
    return BallIndex(NearestNeighbors.KDTree(pts), pts, size(pts, 2) ÷ ng, embedding, minper)
end

@inline _query_radius(ix::BallIndex, r::Real) = Grids.embedded_radius(ix.embedding, r)

# The point form of the tree query. A tree searches replicated points, so one cell can come back through
# several images and the caller would visit it twice; folding the images back and dropping the repeats
# keeps the contract that a candidate is offered once. Below half the shortest period no two images can
# both be within `r`, so there is nothing to drop and that pass is skipped.
function Grids.fold_candidates_at(f::F, acc, ix::BallIndex{TR,T}, q, r, scratch) where {F,TR,T}
    cands = scratch === nothing ? Int[] : empty!(scratch)
    # `inrange!` into a caller-owned buffer, for the same reason the cell form uses it: the batch
    # `inrange` allocates a fresh vector per query, which dominates the cost of a small ball.
    NearestNeighbors.inrange!(cands, ix.tree, _qpoint(ix.tree, map(T, q)), _query_radius(ix, r), false)
    if ix.n == size(ix.pts, 2) || 2 * r < ix.min_period
        @inbounds for t in cands
            acc = f(acc, mod1(t, ix.n))
        end
        return acc
    end
    @inbounds for t in eachindex(cands)
        cands[t] = mod1(cands[t], ix.n)
    end
    sort!(cands; alg = Base.Sort.QuickSort)
    @inbounds for t in eachindex(_dedup_sorted!(cands))
        acc = f(acc, cands[t])
    end
    return acc
end

function Grids.index_within!(buf::AbstractVector{<:Integer}, ix::BallIndex, grid, I, r)
    lin = I isa Integer ? Int(I) : LinearIndices(size(Grids.mask(grid)))[CartesianIndex(I)]
    empty!(buf)
    # `inrange!` pushes into a caller-owned buffer; the batch `inrange` allocates a fresh vector per
    # query, which dominates the cost of a small ball.
    NearestNeighbors.inrange!(buf, ix.tree, _qcol(ix.tree, ix.pts, lin), _query_radius(ix, r), false)
    ix.n == size(ix.pts, 2) && return buf     # nothing replicated: no image can repeat a cell
    # Images collapse back onto their originals.
    @inbounds for t in eachindex(buf)
        buf[t] = mod1(buf[t], ix.n)
    end
    # Two images of one cell are a whole lattice vector apart, so a query narrower than half the shortest
    # period cannot have found both, and there is nothing to remove. Skipping the sort matters: it is
    # `O(m log m)` on an `O(m)` query, and the wide radius that needs it is the rare case.
    2 * r < ix.min_period && return buf
    # Sorted only so the dedup below is a linear pass rather than `unique!`'s hash set. `QuickSort`
    # because the default algorithm allocates a scratch buffer, which this path exists to avoid.
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
