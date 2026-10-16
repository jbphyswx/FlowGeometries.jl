# ---------------------------------------------------------------------------
# Icosahedral geodesic
# ---------------------------------------------------------------------------

"""
    IcosahedralGrid(geometry, frequency; mask = nothing)

The frequency-`ν` icosahedral geodesic as a grid: `10ν² + 2` vertices, twelve of them pentagonal and
the rest hexagonal.

It stores `ν`, the geometry, the mask and one dual-cell area per symmetry class of vertex. Vertex ids are
assigned by topology — the twelve corners, then each macro-edge's interior, then each face's — so a
vertex's position and its neighbours are arithmetic in `(ν, id)`: reading the numbering backwards gives
the lattice positions the vertex occupies, and everything follows from those.

A vertex sits on one face if a face owns it, two if a macro-edge does, and five if it is a corner. That
count is its valence, so the twelve corners are the grid's twelve pentagons.

The measure is the spherical Voronoi dual, from the vertex's own incident triangles — each triangle split
among its three vertices by the arcs from its circumcenter to its edge midpoints. It depends on the
vertex's class alone (see [`_ico_class`](@ref)), so the `O(ν²/12)` class areas are evaluated at
construction and a vertex reads its class's.

Coordinates are `(λ, φ)`: [`coords`](@ref)`(grid, id)` reads one vertex, and [`materialize`](@ref) gives
the whole cloud as dense vectors.
"""
struct IcosahedralGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    B<:AbstractVector{Bool},
    A<:AbstractVector{T},
    O<:AbstractVector{Int},
} <: AbstractGrid{G,T}
    geometry::G
    frequency::Int
    mask::B
    class_area::A        # the dual area of each class `(b, c)`, at `class_offset[c + 1] + b − c + 1`
    class_offset::O      # classes with smallest part below `c`, for `c = 0 … ⌊ν/3⌋`
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
    offs = _ico_class_offsets(ν)
    R² = T(Geometry.radius(geometry))^2
    area = Vector{T}(undef, last(offs))
    @inbounds for c in 0:(ν ÷ 3), b in c:((ν - c) ÷ 2)
        area[offs[c + 1] + b - c + 1] = R² * _ico_vertex_share(T, _ico_class_vertex(b, c, ν), ν)
    end
    return IcosahedralGrid{T,typeof(geometry),typeof(m),typeof(area),typeof(offs)}(geometry, ν, m, area,
                                                                                 offs)
end

IcosahedralGrid(frequency::Integer; kwargs...) =
    IcosahedralGrid(Geometry.SphericalGeometry(), frequency; kwargs...)

@inline _from_fields(
    ::Type{<:IcosahedralGrid},
    geometry::G, frequency::Int, mask::B, class_area::A, class_offset::O,
) where {T,G<:Geometry.AbstractSphericalGeometry{T},B,A<:AbstractVector{T},O} =
    IcosahedralGrid{T,G,B,A,O}(geometry, frequency, mask, class_area, class_offset)

# The classes `(b, c)`, `c ≤ b ≤ ν − b − c`, counted by smallest part: `⌊(ν − c)/2⌋ − c + 1` of them for
# each `c`, the last entry their total.
function _ico_class_offsets(ν::Int)
    offs = Vector{Int}(undef, ν ÷ 3 + 2)
    offs[1] = 0
    @inbounds for c in 0:(ν ÷ 3)
        offs[c + 2] = offs[c + 1] + (ν - c) ÷ 2 - c + 1
    end
    return offs
end

"""
    frequency(grid::IcosahedralGrid) -> Int

The subdivision frequency `ν`: the grid has `10ν² + 2` vertices and `20ν²` triangles.
"""
@inline frequency(grid::IcosahedralGrid) = getfield(grid, :frequency)

@inline sampling(grid::IcosahedralGrid) = SphericalSampling.IcosahedralSampling(frequency(grid))

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
    return SphericalSampling._unit_lonlat(v[1], v[2], v[3])
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
    _ico_vertex_share(T, k, ν) -> T

The dual area of vertex `k` on the unit sphere, from its own incident triangles: on every face the vertex
sits on, its Voronoi share of each triangle around it.

The edges come from the lattice. With `p` a node's barycentric point before normalization and
`d = p_A − p_V`, a small-integer combination of the face's corner differences over `ν`,

    A − V = d/|p_A| − V (2 p_V·d + |d|²) / (|p_A| (|p_V| + |p_A|)),

which keeps the relative precision of `T` however short the edge.
"""
function _ico_vertex_share(::Type{T}, k::Int, ν::Int) where {T}
    occ, nocc = SphericalSampling._ico_occurrences(k, ν)
    a = zero(T)
    @inbounds for t in 1:nocc
        fc, i, j = occ[t]
        a += _ico_face_share(T, fc, i, j, ν)
    end
    return a
end

# Lattice node `(i, j)`'s dual share of face `fc`, over the face's triangles around it.
@inline function _ico_face_share(::Type{T}, fc::Int, i::Int, j::Int, ν::Int) where {T}
    base = SphericalSampling._icosahedron_base(T)
    ia, ib, ic = @inbounds SphericalSampling._ICOSAHEDRON_FACES[fc]
    A0 = @inbounds base[ia]
    eB = map(-, (@inbounds base[ib]), A0)
    eC = map(-, (@inbounds base[ic]), A0)
    lattice(a, b) = map((x, y) -> (T(a) * x + T(b) * y) / T(ν), eB, eC)
    pV = map(+, A0, lattice(i, j))
    rV = sqrt(Geometry._dot3(pV, pV))
    V = map(x -> x / rV, pV)
    function edge((a, b))
        d = lattice(a - i, b - j)
        p = map(+, pV, d)
        r = sqrt(Geometry._dot3(p, p))
        return Geometry._axpy3(-(2 * Geometry._dot3(pV, d) + Geometry._dot3(d, d)) / (r * (rV + r)), V,
                               map(x -> x / r, d))
    end
    return SphericalSampling._ico_fold_incident_triangles(zero(T), i, j, ν) do acc, n1, n2
        return acc + Geometry._dual_share(V, edge(n1), edge(n2))
    end
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

@inline function measure(grid::IcosahedralGrid, id::Integer)
    b, c = _ico_class(Int(id), frequency(grid))
    return @inbounds getfield(grid, :class_area)[getfield(grid, :class_offset)[c + 1] + b - c + 1]
end

@inline measure(grid::IcosahedralGrid) = GridMeasure(grid)

# The dual cells tile the sphere, the three shares of every triangle summing to it.
@inline _total_measure(grid::IcosahedralGrid{T}) where {T} =
    T(4π) * T(Geometry.radius(grid_geometry(grid)))^2

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
