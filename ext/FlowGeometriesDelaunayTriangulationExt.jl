module FlowGeometriesDelaunayTriangulationExt

using DelaunayTriangulation: DelaunayTriangulation as DT
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# Exact per-node planar Voronoi-cell areas, clipped to the convex hull (overrides the throwing
# fallback in `Grids._voronoi_areas`).

"""
    _assert_not_collinear(x, y, N)

Throw if the point set spans no area. Uses the farthest point from the first as the baseline
direction — the numerically best-conditioned choice available in one pass — and measures every
point's perpendicular distance from that line against a tolerance scaled by the set's own extent.
"""
function _assert_not_collinear(x::AbstractVector{T}, y::AbstractVector{T}, N::Int) where {T}
    @inbounds x1, y1 = x[1], y[1]
    d2max = zero(T)
    k = 1
    @inbounds for i in 2:N
        d2 = (x[i] - x1)^2 + (y[i] - y1)^2
        if d2 > d2max
            d2max = d2
            k = i
        end
    end
    d2max > 0 || throw(ArgumentError(
        "all $N points are identical, so they span no area; supply `areas` explicitly",
    ))
    @inbounds ux, uy = x[k] - x1, y[k] - y1
    scale = sqrt(d2max)
    tol = scale * sqrt(eps(T)) * 8
    @inbounds for i in 2:N
        # |cross| / |u| is the perpendicular distance from the baseline.
        if abs((x[i] - x1) * uy - (y[i] - y1) * ux) / scale > tol
            return nothing
        end
    end
    throw(ArgumentError(
        "all $N points are collinear to within $(tol), so the Voronoi cells are unbounded and have " *
        "no finite area; supply `areas` explicitly for such a grid",
    ))
end

function Grids._voronoi_tessellation(
    ::Geometry.AbstractCartesianGeometry{T}, x::AbstractVector{T}, y::AbstractVector{T},
) where {T<:AbstractFloat}
    N = length(x)
    length(y) == N || throw(DimensionMismatch("x/y length mismatch"))
    N ≥ 3 || throw(ArgumentError(
        "a planar Voronoi tessellation needs at least 3 non-collinear points (got $N)",
    ))

    # A degenerate (collinear) point set has no triangulation, and feeding one in surfaces as an
    # opaque internal error. The precondition is `O(N)`, so it is checked up front; a `try`/`catch`
    # around the triangulation swallows genuine bugs and interrupts alongside it.
    _assert_not_collinear(x, y, N)

    # The tessellation runs in Float64 whatever `T` is: the orientation and incircle predicates are
    # exact-arithmetic and defined there. Only the resulting areas are converted back to `T`.
    pts = [(Float64(x[i]), Float64(y[i])) for i in 1:N]
    tri = DT.triangulate(pts)
    vorn = DT.voronoi(tri; clip = true)

    # `zeros`, never `undef`: the triangulation skips duplicate input points, which are then assigned
    # no polygon. A skipped point owns no region, so zero is its area, and an unwritten `undef` slot
    # would escape as a plausible-looking one.
    areas = zeros(T, N)
    @inbounds for i in DT.each_polygon_index(vorn)
        (1 ≤ i ≤ N) || continue
        areas[i] = T(DT.get_area(vorn, i))
    end

    # The triangulation the areas came from, returned alongside them. Solid triangles only: a ghost
    # triangle names the boundary, and has no node to be a cell of.
    # The narrowest integer that indexes this node set, as the neighbour arrays use.
    I = N ≤ typemax(Int32) ? Int32 : Int
    cell_nodes = I[]
    sizehint!(cell_nodes, 3 * DT.num_solid_triangles(tri))
    for τ in DT.each_solid_triangle(tri)
        i, j, k = DT.triangle_vertices(τ)
        push!(cell_nodes, I(i), I(j), I(k))
    end
    nc = length(cell_nodes) ÷ 3
    cell_ptr = Vector{I}(undef, nc + 1)
    @inbounds for c in 1:(nc + 1)
        cell_ptr[c] = I(3 * (c - 1) + 1)
    end
    return areas, Grids.CellMesh(cell_ptr, cell_nodes, N)
end

end # module
