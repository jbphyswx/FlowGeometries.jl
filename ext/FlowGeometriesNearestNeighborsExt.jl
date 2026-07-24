module FlowGeometriesNearestNeighborsExt

using NearestNeighbors: NearestNeighbors
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# k-d-tree neighbor construction for `UnstructuredGrid` (overrides the throwing fallback).
# Cartesian: tree on (x, y). Spherical: tree on unit-sphere embedding (chord ≅ great-circle order).

function _csr_from_knn(pts::AbstractVector, k::Integer)
    N = length(pts)
    if N < 2
        return Int[], ones(Int, N + 1)
    end
    kq = min(k, N - 1)
    tree = NearestNeighbors.KDTree(pts)
    idxs, _ = NearestNeighbors.knn(tree, pts, kq + 1, true)
    rowlen = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        rowlen[i] = min(count(!=(i), idxs[i]), kq)
    end
    ptr = Vector{Int}(undef, N + 1)
    ptr[1] = 1
    @inbounds for i in 1:N
        ptr[i + 1] = ptr[i] + rowlen[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    @inbounds for i in 1:N
        cursor = ptr[i]
        stop = ptr[i + 1] - 1
        for j in idxs[i]
            j == i && continue
            cursor > stop && break
            nbrs[cursor] = j
            cursor += 1
        end
    end
    return nbrs, ptr
end

function _csr_from_radius(pts::AbstractVector, r::Real)
    N = length(pts)
    if N < 2
        return Int[], ones(Int, N + 1)
    end
    tree = NearestNeighbors.KDTree(pts)
    lists = NearestNeighbors.inrange(tree, pts, r, false)
    ptr = Vector{Int}(undef, N + 1)
    ptr[1] = 1
    @inbounds for i in 1:N
        ptr[i + 1] = ptr[i] + (length(lists[i]) - 1)
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    @inbounds for i in 1:N
        cursor = ptr[i]
        for j in lists[i]
            j == i && continue
            nbrs[cursor] = j
            cursor += 1
        end
    end
    return nbrs, ptr
end

function Grids._build_kdtree_neighbors(
    ::Geometry.AbstractCartesianGeometry{T}, x::AbstractVector{T}, y::AbstractVector{T};
    k::Integer = 6, radius::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    pts = [NTuple{2,T}(x[i], y[i]) for i in eachindex(x)]
    return radius === nothing ? _csr_from_knn(pts, k) : _csr_from_radius(pts, T(radius))
end

function Grids._build_kdtree_neighbors(
    geo::Geometry.AbstractSphericalGeometry{T}, x::AbstractVector{T}, y::AbstractVector{T};
    k::Integer = 6, radius::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    pts = [
        NTuple{3,T}(cos(y[i]) * cos(x[i]), cos(y[i]) * sin(x[i]), sin(y[i]))
        for i in eachindex(x)
    ]
    if radius === nothing
        return _csr_from_knn(pts, k)
    else
        arc = T(radius) / geo.R
        chord_radius = T(2) * sin(arc / T(2))
        return _csr_from_radius(pts, chord_radius)
    end
end

end # module
