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
    rq = D == 2 ? T(2) * sin(T(radius) / Geometry.radius(geo) / T(2)) : T(radius)
    return _csr_from_radius(pts, rq)
end

end # module
