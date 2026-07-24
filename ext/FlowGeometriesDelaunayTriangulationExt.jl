module FlowGeometriesDelaunayTriangulationExt

using DelaunayTriangulation: DelaunayTriangulation as DT
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# Exact per-node Cartesian Voronoi-cell areas (overrides throwing fallback).

function Grids._voronoi_areas(
    ::Geometry.AbstractCartesianGeometry{T}, x::AbstractVector{T}, y::AbstractVector{T},
) where {T<:AbstractFloat}
    N = length(x)
    pts = [(Float64(x[i]), Float64(y[i])) for i in 1:N]
    tri = DT.triangulate(pts)
    vorn = DT.voronoi(tri; clip = true)
    areas = Vector{T}(undef, N)
    for i in DT.each_polygon_index(vorn)
        areas[i] = T(DT.get_area(vorn, i))
    end
    return areas
end

end # module
