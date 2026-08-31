# ---------------------------------------------------------------------------
# Icosahedral geodesic
# ---------------------------------------------------------------------------

"""
    IcosahedralGrid(geometry, frequency; mask = nothing)

The frequency-`ν` icosahedral geodesic as a grid: `10ν² + 2` vertices, twelve of them pentagonal and
the rest hexagonal.

It stores `ν`, the geometry and the mask. Vertex ids are assigned by topology — the twelve corners, then
each macro-edge's interior, then each face's — so a vertex's position, its neighbours and its dual-cell
area are each arithmetic in `(ν, id)`: reading the numbering backwards gives the lattice positions the
vertex occupies, and everything follows from those.

A vertex sits on one face if a face owns it, two if a macro-edge does, and five if it is a corner. That
count is its valence, so the twelve corners are the grid's twelve pentagons.

The measure is the spherical Voronoi dual, computed from the vertex's own incident triangles — each
triangle split among its three vertices by the arcs from its circumcenter to its edge midpoints. It
costs a fan of spherical excesses per cell, so use [`measure_array`](@ref) when sweeping every cell.

Coordinates are `(λ, φ)`: [`coords`](@ref)`(grid, id)` reads one vertex, and [`materialize`](@ref) gives
the whole cloud as dense vectors.
"""
struct IcosahedralGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    B<:AbstractVector{Bool},
} <: AbstractGrid{G,T}
    geometry::G
    frequency::Int
    mask::B
end

function IcosahedralGrid(
    geometry::Geometry.AbstractSphericalGeometry{T}, frequency::Integer; mask = nothing,
) where {T<:AbstractFloat}
    ν = Int(frequency)
    ν ≥ 1 || throw(ArgumentError("icosahedral frequency must be ≥ 1, got $ν"))
    n = SphericalSampling.icosahedral_nvertices(ν)
    m = mask === nothing ? AllActive((n,)) : mask
    length(m) == n || throw(DimensionMismatch(
        "mask holds $(length(m)) entries for a grid of $n vertices",
    ))
    return IcosahedralGrid{T,typeof(geometry),typeof(m)}(geometry, ν, m)
end

IcosahedralGrid(frequency::Integer; kwargs...) =
    IcosahedralGrid(Geometry.SphericalGeometry(), frequency; kwargs...)

@inline _from_fields(
    ::Type{<:IcosahedralGrid},
    geometry::G, frequency::Int, mask::B,
) where {T,G<:Geometry.AbstractSphericalGeometry{T},B} =
    IcosahedralGrid{T,G,B}(geometry, frequency, mask)

"""
    frequency(grid::IcosahedralGrid) -> Int

The subdivision frequency `ν`: the grid has `10ν² + 2` vertices and `20ν²` triangles.
"""
@inline frequency(grid::IcosahedralGrid) = getfield(grid, :frequency)

# ---- the three traits -------------------------------------------------------

@inline cell_address(::IcosahedralGrid) = FlatCells()
@inline adjacency_source(::IcosahedralGrid) = FormulaNeighbors()

# A lattice step and its negation are both in the six directions.
@inline has_symmetric_adjacency(::IcosahedralGrid) = true
@inline candidate_source(::IcosahedralGrid) = IndexedCandidates()

# ---- coordinates ------------------------------------------------------------

@inline ncoordinates(::IcosahedralGrid) = 2

@inline function _raw_coords(grid::IcosahedralGrid{T}, id::Integer) where {T}
    v = SphericalSampling._ico_vertex_dir(T, Int(id), frequency(grid))
    θ = acos(clamp(v[3], -one(T), one(T)))
    ϕ = atan(v[2], v[1])
    ϕ < 0 && (ϕ += T(2π))
    return (ϕ, SphericalSampling.geographic_latitude(θ))
end

coordinates(grid::IcosahedralGrid) = throw(ArgumentError(
    "an IcosahedralGrid stores no coordinate arrays — a vertex's position is arithmetic in " *
    "(frequency, id). Use `coords(grid, id)` for one vertex, or `Grids.materialize(grid)` for the " *
    "whole cloud.",
))

# ---- topology and span ------------------------------------------------------

@inline topology(::IcosahedralGrid) = (Periodic(), Bounded())
@inline period(grid::IcosahedralGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? T(2π) : zero(T)

# The geodesic covers the sphere, with two vertices at the poles of the base icosahedron's own frame.
@inline bounds(grid::IcosahedralGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? (zero(T), T(2π)) : (-T(π) / 2, T(π) / 2)

@inline origin(grid::IcosahedralGrid, d::Integer) = bounds(grid, d)[1]

# ---- measure ----------------------------------------------------------------

"""
    _ico_dual_share(V, A, B) -> T

Vertex `V`'s share of the spherical triangle `(V, A, B)`: the quadrilateral `(V, M_VA, O, M_VB)` with
`O` the circumcenter and `M` the edge midpoints, as two spherical excesses.

Those arcs are the triangle's perpendicular bisectors — `O` is equidistant from all three vertices and
each midpoint from its two — so the share is exactly `V`'s Voronoi cell restricted to that triangle, and
the three shares tile the triangle.
"""
@inline function _ico_dual_share(V::NTuple{3,T}, A::NTuple{3,T}, B::NTuple{3,T}) where {T}
    ux = A[1] - V[1]; uy = A[2] - V[2]; uz = A[3] - V[3]
    vx = B[1] - V[1]; vy = B[2] - V[2]; vz = B[3] - V[3]
    O = SphericalSampling._ico_norm3((uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx))
    (O[1] * V[1] + O[2] * V[2] + O[3] * V[3]) < 0 && (O = (-O[1], -O[2], -O[3]))
    Mva = SphericalSampling._ico_norm3((V[1] + A[1], V[2] + A[2], V[3] + A[3]))
    Mvb = SphericalSampling._ico_norm3((V[1] + B[1], V[2] + B[2], V[3] + B[3]))
    return Geometry.spherical_excess(V, Mva, O) + Geometry.spherical_excess(V, O, Mvb)
end

"""
    _ico_vertex_measure(grid, k, ν) -> T

The dual area of vertex `k`, from its own incident triangles: the fan of
[`_ico_dual_share`](@ref)s over every face the vertex sits on.
"""
function _ico_vertex_measure(grid::IcosahedralGrid{T}, k::Int, ν::Int) where {T}
    V = SphericalSampling._ico_vertex_dir(T, k, ν)
    occ, nocc = SphericalSampling._ico_occurrences(k, ν)
    a = zero(T)
    @inbounds for t in 1:nocc
        fc, i, j = occ[t]
        a = SphericalSampling._ico_fold_incident_triangles(a, fc, i, j, ν) do acc, o1, o2
            return acc + _ico_dual_share(V, SphericalSampling._ico_vertex_dir(T, o1, ν),
                                         SphericalSampling._ico_vertex_dir(T, o2, ν))
        end
    end
    return T(Geometry.radius(grid_geometry(grid)))^2 * a
end

# Descending, by three compares.
@inline function _sorted3(a::Int, b::Int, c::Int)
    a < b && ((a, b) = (b, a))
    b < c && ((b, c) = (c, b))
    a < b && ((a, b) = (b, a))
    return (a, b, c)
end

"""
    _ico_class(k, ν) -> (b, c)

Vertex `k`'s symmetry class: the two smallest of its barycentric parts `(i, j, ν−i−j)`, the largest
being `ν` minus their sum.

All twenty faces are the same spherical triangle and the geodesic is built the same way on each, so a
vertex's dual area depends only on its barycentric position within a face; the face being equilateral,
only on that triple up to order. A vertex on a macro-edge or at a corner sits on several faces, and an
isometry of the icosahedron carrying it to another vertex of its class carries those faces with it, so
the areas agree there too.

The classes are the partitions of `ν` into three parts, `O(ν²/12)` of them against `10ν² + 2` vertices.
"""
@inline function _ico_class(k::Int, ν::Int)
    occ, _ = SphericalSampling._ico_occurrences(k, ν)
    _fc, i, j = @inbounds occ[1]
    _a, b, c = _sorted3(i, j, ν - i - j)
    return (b, c)
end

# Face 1's lattice node `(b, c)`, the member of the class that stands for all of them. Congruent cells
# then hold one number, so a sum over them does not depend on which member each came from.
@inline _ico_class_vertex(b::Int, c::Int, ν::Int) = SphericalSampling._ico_lattice_id(1, b, c, ν)

function measure(grid::IcosahedralGrid{T}, id::Integer) where {T}
    ν = frequency(grid)
    b, c = _ico_class(Int(id), ν)
    return _ico_vertex_measure(grid, _ico_class_vertex(b, c, ν), ν)
end

@inline measure(grid::IcosahedralGrid) = GridMeasure(grid)

# The dual cells tile the sphere, the three shares of every triangle summing to it.
@inline _total_measure(grid::IcosahedralGrid{T}) where {T} =
    T(4π) * T(Geometry.radius(grid_geometry(grid)))^2

"""
    measure_array(grid::IcosahedralGrid) -> Vector

The dual-cell areas of every vertex, the fan of spherical excesses evaluated once per symmetry class —
see [`_ico_class`](@ref). The two smallest barycentric parts key a `(ν+1)²` table.
"""
function measure_array(grid::IcosahedralGrid{T}) where {T}
    ν = frequency(grid)
    n = length(grid)
    out = Vector{T}(undef, n)
    tbl = fill(T(NaN), ν + 1, ν + 1)
    @inbounds for k in 1:n
        b, c = _ico_class(k, ν)
        v = tbl[b + 1, c + 1]
        if isnan(v)
            v = _ico_vertex_measure(grid, _ico_class_vertex(b, c, ν), ν)
            tbl[b + 1, c + 1] = v
        end
        out[k] = v
    end
    return out
end

# ---- materialization --------------------------------------------------------

# The vertex writer, walking the owning entities in id order so no id is decoded.
function materialize(grid::IcosahedralGrid{T}) where {T}
    n = length(grid)
    p = SphericalSampling.icosahedral_vertices!(
        Vector{T}(undef, n), Vector{T}(undef, n), frequency(grid),
    )
    return (p.λ, p.φ)
end

# ---- display ----------------------------------------------------------------

Base.show(io::IO, grid::IcosahedralGrid{T}) where {T} =
    print(io, "IcosahedralGrid{", T, "}(ν=", frequency(grid), ", ", length(grid), " vertices)")

function Base.show(io::IO, ::MIME"text/plain", grid::IcosahedralGrid{T}) where {T}
    ν = frequency(grid)
    println(io, "IcosahedralGrid{", T, "} ν=", ν, " (", count(mask(grid)), "/", length(grid),
            " active)")
    println(io, "  geometry:  ", grid_geometry(grid))
    println(io, "  cells:     12 pentagons, ", length(grid) - 12, " hexagons; ", 20 * ν^2,
            " triangles")
    print(io, "  measure:   ", sum(measure(grid)), " total")
end
