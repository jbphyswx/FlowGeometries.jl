module FlowGeometriesDelaunayTriangulationExt

using DelaunayTriangulation: DelaunayTriangulation as DT
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# `Grids._voronoi_tessellation` for a planar node set: each node's exact Voronoi-cell area and the
# Delaunay triangles as its cells, clipped to the convex hull along a bounded direction and repeating
# along a periodic one.

"""
    _assert_not_collinear(pts)

Throw if the points span no area: every point lies on the line through the first and another distinct
one, by DelaunayTriangulation's exact orientation predicate on the coordinates the triangulation receives.
"""
function _assert_not_collinear(pts::Vector{NTuple{2,Float64}})
    N = length(pts)
    p1 = pts[1]
    k = findfirst(!=(p1), pts)
    k === nothing && throw(ArgumentError(
        "all $N points are identical, so they span no area; supply `areas` explicitly",
    ))
    pk = pts[k]
    any(p -> DT.orient_predicate(p1, pk, p) != 0, pts) && return nothing
    throw(ArgumentError(
        "all $N points are collinear, so the Voronoi cells are unbounded and have no finite area; " *
        "supply `areas` explicitly for such a grid",
    ))
end

function Grids._voronoi_tessellation(
    ::Type{I}, ::Geometry.AbstractCartesianGeometry{T}, x::AbstractVector{T}, y::AbstractVector{T},
    period::NTuple{2,Real},
) where {I<:Integer, T<:AbstractFloat}
    N = length(x)
    length(y) == N || throw(DimensionMismatch("x/y length mismatch"))
    N ≥ 3 || throw(ArgumentError(
        "a planar Voronoi tessellation needs at least 3 non-collinear points (got $N)",
    ))

    # The tessellation runs in Float64 whatever `T` is: the orientation and incircle predicates are
    # exact-arithmetic and defined there. Only the resulting areas are converted back to `T`.
    P = (Float64(period[1]), Float64(period[2]))
    pts = [(Float64(x[i]), Float64(y[i])) for i in 1:N]

    # A collinear point set has no triangulation; the check is `O(N)` and runs before one is attempted.
    _assert_not_collinear(pts)
    any(>(0), P) || return _cells(DT.triangulate(pts), collect(1:N), N, T, I)
    return _periodic_cells(pts, P, N, T, I)
end

# Along a direction of period `P[d] > 0` each node is folded into the box `[lo_d, lo_d + P_d)` and the
# images within a strip `w` past either face join it. A node's cell, and every triangle at it, is the
# periodic one once each such triangle's circumdisk lies inside the strip, where the images are complete;
# the strip doubles until it does, the whole neighbouring copies being the last resort. A bounded
# direction is clipped to the hull of what is triangulated, which over the box is the hull of the
# repeating set once the neighbouring copies are whole, so with one the strip starts there.
function _periodic_cells(
    pts::Vector{NTuple{2,Float64}}, P::NTuple{2,Float64}, N::Int, ::Type{T}, ::Type{I},
) where {T,I}
    lo = ntuple(d -> P[d] > 0 ? minimum(p -> p[d], pts) : 0.0, Val(2))
    fold(v, d) = P[d] > 0 ? lo[d] + mod(v - lo[d], P[d]) : v
    base = [(fold(p[1], 1), fold(p[2], 2)) for p in pts]
    band = ntuple(d -> extrema(p -> p[d], base), Val(2))
    w = all(>(0), P) ? 4 * sqrt(P[1] * P[2] / N) : maximum(P)
    shifts = [(a, b) for a in (P[1] > 0 ? (-1:1) : (0:0)), b in (P[2] > 0 ? (-1:1) : (0:0)) if (a, b) != (0, 0)]
    while true
        whole = (P[1] == 0 || w >= P[1]) && (P[2] == 0 || w >= P[2])
        ext, src = copy(base), collect(1:N)
        for (a, b) in shifts, i in 1:N
            q = (base[i][1] + a * P[1], base[i][2] + b * P[2])
            _in_strip(q, lo, P, w) || continue
            push!(ext, q)
            push!(src, i)
        end
        tri = DT.triangulate(ext)
        _strip_holds(tri, ext, N, lo, P, w, band) && return _cells(tri, src, N, T, I)
        whole && throw(ArgumentError(
            "the node set is too sparse for its period: a Delaunay circle at a node reaches past the " *
            "neighbouring copies of the domain; supply `areas` explicitly"))
        w *= 2
    end
end

@inline _in_strip(q, lo, P, w) =
    all(d -> P[d] == 0 || (lo[d] - w <= q[d] < lo[d] + P[d] + w), 1:2)

# Whether every solid triangle at an original node has its circumdisk inside the strip along each
# periodic direction. Along a bounded direction the nodes span `band`, so only the part of a disk over
# that span can hold one.
function _strip_holds(tri, ext, N::Int, lo, P, w, band)
    for τ in DT.each_solid_triangle(tri)
        i, j, k = DT.triangle_vertices(τ)
        (i <= N || j <= N || k <= N) || continue
        a, b, c = ext[i], ext[j], ext[k]
        # The circumcentre about vertex `a`, where the differences are formed exactly.
        bx, by = b[1] - a[1], b[2] - a[2]
        cx, cy = c[1] - a[1], c[2] - a[2]
        d = 2 * (bx * cy - by * cx)
        iszero(d) && return false
        b2, c2 = bx^2 + by^2, cx^2 + cy^2
        ux, uy = (cy * b2 - by * c2) / d, (bx * c2 - cx * b2) / d
        R = hypot(ux, uy)
        o = (a[1] + ux, a[2] + uy)
        for e in 1:2
            P[e] > 0 || continue
            f = 3 - e
            h = R
            if P[f] == 0
                gap = max(band[f][1] - o[f], o[f] - band[f][2], 0.0)
                gap >= R && continue
                h = sqrt(R^2 - gap^2)
            end
            (o[e] - h >= lo[e] - w && o[e] + h <= lo[e] + P[e] + w) || return false
        end
    end
    return true
end

# The areas of the first `N` points' cells, and the triangles as cells over the nodes `src` maps each
# point to: one copy of each, the one whose lowest-numbered node is that node's own point. The node and
# cell ids are `I`, the offsets `Int`.
function _cells(tri, src::Vector{Int}, N::Int, ::Type{T}, ::Type{I}) where {T,I<:Integer}
    vorn = DT.voronoi(tri; clip = true)

    # `zeros`, never `undef`: the triangulation skips duplicate input points, which are then assigned
    # no polygon. A skipped point owns no region, so zero is its area, and every slot the loop below
    # leaves unwritten already holds it.
    areas = zeros(T, N)
    @inbounds for i in DT.each_polygon_index(vorn)
        (1 ≤ i ≤ N) || continue
        areas[i] = T(DT.get_area(vorn, i))
    end

    # Solid triangles only: a ghost triangle names the boundary, and has no node to be a cell of.
    Grids._check_ids(I, max(N, DT.num_solid_triangles(tri)))
    cell_nodes = I[]
    sizehint!(cell_nodes, 3 * DT.num_solid_triangles(tri))
    for τ in DT.each_solid_triangle(tri)
        i, j, k = DT.triangle_vertices(τ)
        m = argmin(v -> src[v], (i, j, k))
        m <= N || continue
        push!(cell_nodes, I(src[i]), I(src[j]), I(src[k]))
    end
    nc = length(cell_nodes) ÷ 3
    cell_ptr = Vector{Int}(undef, nc + 1)
    @inbounds for c in 1:(nc + 1)
        cell_ptr[c] = 3 * (c - 1) + 1
    end
    return areas, Grids.CellMesh(cell_ptr, cell_nodes, N)
end

end # module
