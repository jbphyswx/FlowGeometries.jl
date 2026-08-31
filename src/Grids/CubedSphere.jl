# ---------------------------------------------------------------------------
# Cubed sphere
# ---------------------------------------------------------------------------

"""
    CubedSphereGrid(geometry, n; mask = nothing)

The gnomonic cubed sphere as a grid: six `n × n` panels of equiangular cells, `6n²` in all.

It stores `n`, the geometry and the mask. A cell's coordinates are its panel-local angles through the
gnomonic map, its neighbours are the panel-interior offsets and the exact seam fold, and its area is the
closed-form solid angle of its gnomonic rectangle — so all three are arithmetic in `(n, cell)`, and
`Base.summarysize` is the same at every `n`.

Cells are numbered panel by panel with `i` fastest, matching
[`SphericalSampling.cubed_sphere_points`](@ref). [`panel_cell`](@ref) and [`cell_id`](@ref) convert
between a cell id and its `(panel, i, j)`.

Coordinates are `(λ, φ)`: [`coords`](@ref)`(grid, k)` reads one cell, and [`materialize`](@ref) gives the
whole cloud as dense vectors.
"""
struct CubedSphereGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    B<:AbstractVector{Bool},
} <: AbstractGrid{G,T}
    geometry::G
    n::Int
    mask::B
end

function CubedSphereGrid(
    geometry::Geometry.AbstractSphericalGeometry{T}, n::Integer; mask = nothing,
) where {T<:AbstractFloat}
    nn = Int(n)
    nn ≥ 1 || throw(ArgumentError("cubed-sphere n must be ≥ 1, got $nn"))
    ncell = 6 * nn * nn
    m = mask === nothing ? AllActive((ncell,)) : mask
    length(m) == ncell || throw(DimensionMismatch(
        "mask holds $(length(m)) entries for a grid of $ncell cells",
    ))
    return CubedSphereGrid{T,typeof(geometry),typeof(m)}(geometry, nn, m)
end

CubedSphereGrid(n::Integer; kwargs...) =
    CubedSphereGrid(Geometry.SphericalGeometry(), n; kwargs...)

@inline _from_fields(
    ::Type{<:CubedSphereGrid},
    geometry::G, n::Int, mask::B,
) where {T,G<:Geometry.AbstractSphericalGeometry{T},B} =
    CubedSphereGrid{T,G,B}(geometry, n, mask)

"""
    panel_size(grid) -> Int

Cells across one panel of a panel layout: `n` for the cubed sphere, whose panels are `n × n`.
"""
@inline panel_size(grid::CubedSphereGrid) = getfield(grid, :n)

"""
    npanels(grid) -> Int

How many panels the layout has — six for the cubed sphere, two for Yin–Yang.
"""
@inline npanels(::CubedSphereGrid) = 6

"""
    panel_cell(grid, k) -> (panel, i, j)

Which panel cell `k` is on and where on it, the inverse of [`cell_id`](@ref).
"""
@inline panel_cell(grid::CubedSphereGrid, k::Integer) =
    SphericalSampling._cubed_unlin(Int(k), panel_size(grid))

"""
    cell_id(grid, panel, i, j) -> Int

The cell id of `(panel, i, j)`, the inverse of [`panel_cell`](@ref).
"""
@inline cell_id(grid::CubedSphereGrid, panel::Integer, i::Integer, j::Integer) =
    SphericalSampling._cubed_lin(Int(panel), Int(i), Int(j), panel_size(grid))

# ---- the three traits -------------------------------------------------------

@inline cell_address(::CubedSphereGrid) = FlatCells()
@inline adjacency_source(::CubedSphereGrid) = FormulaNeighbors()

# The seam fold maps a panel exit to the entry that maps back.
@inline has_symmetric_adjacency(::CubedSphereGrid) = true
@inline candidate_source(::CubedSphereGrid) = IndexedCandidates()

# ---- coordinates ------------------------------------------------------------

@inline ncoordinates(::CubedSphereGrid) = 2

@inline function _raw_coords(grid::CubedSphereGrid{T}, k::Integer) where {T}
    n = panel_size(grid)
    f, i, j = SphericalSampling._cubed_unlin(Int(k), n)
    return SphericalSampling._cubed_cell_lonlat(T, n, f, i, j)
end

coordinates(grid::CubedSphereGrid) = throw(ArgumentError(
    "a CubedSphereGrid stores no coordinate arrays — a cell's position is the gnomonic map of its " *
    "panel-local angles. Use `coords(grid, k)` for one cell, or `Grids.materialize(grid)` for the " *
    "whole cloud.",
))

# ---- topology and span ------------------------------------------------------

@inline topology(::CubedSphereGrid) = (Periodic(), Bounded())
@inline period(grid::CubedSphereGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? T(2π) : zero(T)

# Six panels tile the sphere, so both spans are known without reading a cell.
@inline bounds(grid::CubedSphereGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? (zero(T), T(2π)) : (-T(π) / 2, T(π) / 2)

@inline origin(grid::CubedSphereGrid, d::Integer) = bounds(grid, d)[1]

# ---- measure ----------------------------------------------------------------

"""
    _gnomonic_solid_angle(X1, X2, Y1, Y2) -> T

The solid angle subtended by the gnomonic rectangle `[X1, X2] × [Y1, Y2]` on the unit sphere, exactly.

`atan(XY / √(1 + X² + Y²))` at a corner is the solid angle of the rectangle from the panel centre to
that corner, so the four corners combine by inclusion–exclusion. Over a whole panel it gives `4π/6`.
"""
@inline _gnomonic_corner(X::T, Y::T) where {T} = atan(X * Y / sqrt(one(T) + X * X + Y * Y))

@inline _gnomonic_solid_angle(X1::T, X2::T, Y1::T, Y2::T) where {T} =
    _gnomonic_corner(X2, Y2) - _gnomonic_corner(X1, Y2) -
    _gnomonic_corner(X2, Y1) + _gnomonic_corner(X1, Y1)

# A cell's area depends on `(i, j)` alone: the six panels are rigid images of one another, and the
# solid angle is invariant under the orthogonal map between them.
@inline function measure(grid::CubedSphereGrid{T}, k::Integer) where {T}
    n = panel_size(grid)
    _, i, j = SphericalSampling._cubed_unlin(Int(k), n)
    X1, X2, Y1, Y2 = SphericalSampling._cubed_cell_edges(T, n, i, j)
    R² = T(Geometry.radius(grid_geometry(grid)))^2
    return R² * _gnomonic_solid_angle(X1, X2, Y1, Y2)
end

@inline measure(grid::CubedSphereGrid) = GridMeasure(grid)

@inline _total_measure(grid::CubedSphereGrid{T}) where {T} =
    T(4π) * T(Geometry.radius(grid_geometry(grid)))^2

# ---- display ----------------------------------------------------------------

Base.show(io::IO, grid::CubedSphereGrid{T}) where {T} =
    print(io, "CubedSphereGrid{", T, "}(n=", panel_size(grid), ", ", length(grid), " cells)")

function Base.show(io::IO, ::MIME"text/plain", grid::CubedSphereGrid{T}) where {T}
    n = panel_size(grid)
    println(io, "CubedSphereGrid{", T, "} 6 × ", n, " × ", n, " (", count(mask(grid)), "/",
            length(grid), " active)")
    println(io, "  geometry:  ", grid_geometry(grid))
    print(io, "  measure:   ", sum(measure(grid)), " total")
end
