# ---------------------------------------------------------------------------
# A rectilinear mesh read in another frame
# ---------------------------------------------------------------------------

"""
    RotatedGrid(base, rotation, map)

A rectilinear grid whose cells are reported in a rotated frame: the mesh is `base`'s, and a cell's
position is `map(rotation, λ, φ)` evaluated where it is asked for.

[`rotate`](@ref) and [`unrotate`](@ref) build one. `map` is which direction the frame goes —
[`Geometry.rotate`](@ref) for a mesh given in geographic coordinates and reported in the rotated frame,
[`Geometry.unrotate`](@ref) for the reverse — and it is a singleton function type, so the choice
resolves at compile time and the grid stays `isbits` apart from what `base` already holds.

Nothing is materialized. A rotation is an isometry of the sphere, so the cell measure carries over
exactly and this shares `base`'s array. The mesh, the mask, the index topology and the wrap lengths are
`base`'s too, all being properties of the cell lattice. What differs is where each cell sits, which
costs two `sincos` pairs per coordinate access.
"""
struct RotatedGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    GB<:AbstractStructuredGrid,
    F,
} <: AbstractGrid{G,T}
    base::GB
    rotation::Geometry.PoleRotation{T}
    map::F
end

function RotatedGrid(base::AbstractStructuredGrid{G,T}, rot::Geometry.PoleRotation, map::F) where {G,T,F}
    ncoordinates(base) == 2 || throw(ArgumentError(
        "a pole rotation acts on a `(λ, φ)` surface, so it needs a 2-coordinate grid; got " *
        "$(ncoordinates(base))",
    ))
    G <: Geometry.AbstractSphericalGeometry ||
        throw(ArgumentError("a pole rotation needs a spherical geometry; got $G"))
    r = Geometry.similar_rotation(T, rot)
    return RotatedGrid{T,G,typeof(base),F}(base, r, map)
end

@inline _from_fields(::Type{<:RotatedGrid}, base::GB, rotation::Geometry.PoleRotation{T},
                     map::F) where {T,GB,F} =
    RotatedGrid{T,typeof(grid_geometry(base)),GB,F}(base, rotation, map)

"""
    base_grid(grid::RotatedGrid) -> AbstractStructuredGrid

The rectilinear mesh underneath. Its axes are the coordinates a derivative differences along; the
rotation changes where cells are reported, leaving the lattice as it is.
"""
@inline base_grid(grid::RotatedGrid) = getfield(grid, :base)

"""
    rotation(grid::RotatedGrid) -> PoleRotation

The frame the mesh is read in.
"""
@inline rotation(grid::RotatedGrid) = getfield(grid, :rotation)

@inline grid_geometry(grid::RotatedGrid) = grid_geometry(base_grid(grid))
@inline mask(grid::RotatedGrid) = mask(base_grid(grid))
@inline size_tuple(grid::RotatedGrid) = size_tuple(base_grid(grid))
@inline topology(grid::RotatedGrid) = topology(base_grid(grid))
@inline period(grid::RotatedGrid, d::Integer) = period(base_grid(grid), d)
@inline coordinate_names(grid::RotatedGrid) = coordinate_names(base_grid(grid))

# ---- lattice spacing --------------------------------------------------------

# How the cells are spaced along an index direction is a property of the lattice, and the lattice is
# `base`'s, so each of these is `base`'s answer. `bounds` and `origin` below are the rotated frame's,
# a rotation moving the patch off the base's per-axis span.
@inline isuniform(grid::RotatedGrid, d::Integer) = isuniform(base_grid(grid), d)
@inline isuniform(grid::RotatedGrid) = isuniform(base_grid(grid))
@inline spacing_trait(grid::RotatedGrid, v::Val) = spacing_trait(base_grid(grid), v)
@inline spacing(grid::RotatedGrid, d::Integer) = spacing(base_grid(grid), d)
@inline minimum_spacing(grid::RotatedGrid, d::Integer) = minimum_spacing(base_grid(grid), d)
@inline maximum_spacing(grid::RotatedGrid, d::Integer) = maximum_spacing(base_grid(grid), d)
@inline local_spacing(grid::RotatedGrid, d::Integer, i::Integer) =
    local_spacing(base_grid(grid), d, i)
@inline cell_width(grid::RotatedGrid, d::Integer, i::Integer) = cell_width(base_grid(grid), d, i)
@inline cell_widths(grid::RotatedGrid, d::Integer) = cell_widths(base_grid(grid), d)

# ---- the three traits -------------------------------------------------------

@inline cell_address(::RotatedGrid) = CartesianCells()

# The cell lattice is the base's, so neighbours are still index offsets; the frame moves where a cell
# sits, leaving its adjacency alone.
@inline adjacency_source(::RotatedGrid) = IndexStencilNeighbors()

# …but a rotated `(λ, φ)` is not axis-aligned, so a per-axis window cannot bound the candidates for a
# ball query, and an index has to.
@inline candidate_source(::RotatedGrid) = IndexedCandidates()

# ---- coordinates ------------------------------------------------------------

@inline ncoordinates(::RotatedGrid) = 2

@inline function _raw_coords(grid::RotatedGrid{T}, I::Vararg{Integer,2}) where {T}
    λ, φ = _raw_coords(base_grid(grid), I...)
    return grid.map(rotation(grid), λ, φ)
end

coordinates(grid::RotatedGrid) = throw(ArgumentError(
    "a RotatedGrid stores no coordinate arrays: a rotated `(λ, φ)` depends on both mesh coordinates, " *
    "so it has no per-axis form. Use `coords(grid, i, j)` for one cell, `Grids.materialize(grid)` for " *
    "the whole cloud, or `Grids.base_grid(grid)` for the mesh's own axes.",
))

# ---- measure ----------------------------------------------------------------

# A rotation is an isometry, so every cell keeps the area it had, and the base's measure array is
# shared.
@inline measure(grid::RotatedGrid) = measure(base_grid(grid))
@inline measure(grid::RotatedGrid, I::Vararg{Integer,2}) = measure(base_grid(grid), I...)

# ---- span -------------------------------------------------------------------

# A rotation moves the patch off the base's per-axis bounds, so these are the sphere's own.
@inline bounds(grid::RotatedGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? (-T(π), T(π)) : (-T(π) / 2, T(π) / 2)

@inline origin(grid::RotatedGrid, d::Integer) = bounds(grid, d)[1]

# ---- display ----------------------------------------------------------------

Base.show(io::IO, grid::RotatedGrid{T}) where {T} = print(
    io, "RotatedGrid{", T, "}(", join(size_tuple(grid), "×"), " cells, pole (",
    rotation(grid).λp, ", ", rotation(grid).φp, "))",
)

function Base.show(io::IO, ::MIME"text/plain", grid::RotatedGrid{T}) where {T}
    println(io, "RotatedGrid{", T, "} ", join(size_tuple(grid), "×"), " (", count(mask(grid)), "/",
            length(grid), " active)")
    println(io, "  geometry:  ", grid_geometry(grid))
    println(io, "  pole:      (", rotation(grid).λp, ", ", rotation(grid).φp, ")")
    print(io, "  mesh:      ", nameof(typeof(base_grid(grid))))
end