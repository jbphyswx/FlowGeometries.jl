# ---------------------------------------------------------------------------
# Yin–Yang
# ---------------------------------------------------------------------------

"""
    YinYangGrid(geometry, nlon, nlat; mask = nothing)

The Kageyama–Sato Yin–Yang grid: two `nlon × nlat` lat–lon panels covering
`[-3π/4, 3π/4] × [-π/4, π/4]` in their own frames, `2·nlon·nlat` cells in all.

It stores `nlon`, `nlat`, the geometry and the mask. In its own frame each panel is a separable lat–lon
patch, and yang is yin rigidly rotated, so a cell's coordinates are the panel formula composed with a
rotation — [`SphericalSampling._yin_yang_rotate`](@ref) — rather than a second coordinate array. A
cell's area depends on its latitude row alone.

Cells are numbered yin first with `i` fastest, then yang the same way, matching
[`SphericalSampling.spherical_points`](@ref)`(YinYangSampling(), nlon, nlat)`. [`panel_cell`](@ref) and
[`cell_id`](@ref) convert between a cell id and its `(panel, i, j)`.

!!! note "The panels overlap"
    The two panels overlap by construction, so the cell areas sum to `3√2πR²` — 6.07% more than the
    sphere, at every resolution. That excess is the grid's real geometry, and integrating over both
    panels needs a partition-of-unity weight for the shared region, which is a modelling choice made on
    top of these areas. The panels are also not cross-linked: a cell's neighbours are its own panel's,
    which is the standard Yin–Yang discrete topology — the panels couple through interpolation.
"""
struct YinYangGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    B<:AbstractVector{Bool},
} <: AbstractGrid{G,T}
    geometry::G
    nlon::Int
    nlat::Int
    mask::B
end

function YinYangGrid(
    geometry::Geometry.AbstractSphericalGeometry{T}, nlon::Integer, nlat::Integer; mask = nothing,
) where {T<:AbstractFloat}
    nl = Int(nlon)
    nt = Int(nlat)
    (nl ≥ 1 && nt ≥ 1) || throw(ArgumentError("Yin–Yang nlon/nlat must be ≥ 1, got $nl and $nt"))
    ncell = 2 * nl * nt
    m = mask === nothing ? AllActive((ncell,)) : mask
    length(m) == ncell || throw(DimensionMismatch(
        "mask holds $(length(m)) entries for a grid of $ncell cells",
    ))
    return YinYangGrid{T,typeof(geometry),typeof(m)}(geometry, nl, nt, m)
end

YinYangGrid(nlon::Integer, nlat::Integer; kwargs...) =
    YinYangGrid(Geometry.SphericalGeometry(), nlon, nlat; kwargs...)

@inline _from_fields(
    ::Type{<:YinYangGrid},
    geometry::G, nlon::Int, nlat::Int, mask::B,
) where {T,G<:Geometry.AbstractSphericalGeometry{T},B} =
    YinYangGrid{T,G,B}(geometry, nlon, nlat, mask)

@inline npanels(::YinYangGrid) = 2

"""
    panel_shape(grid::YinYangGrid) -> (nlon, nlat)

The cell counts across one panel.
"""
@inline panel_shape(grid::YinYangGrid) = (getfield(grid, :nlon), getfield(grid, :nlat))

@inline _panel_cells(grid::YinYangGrid) = getfield(grid, :nlon) * getfield(grid, :nlat)

@inline function panel_cell(grid::YinYangGrid, k::Integer)
    nlon, _ = panel_shape(grid)
    np = _panel_cells(grid)
    q, r = divrem(Int(k) - 1, np)
    j, i = divrem(r, nlon)
    return (q + 1, i + 1, j + 1)
end

@inline function cell_id(grid::YinYangGrid, panel::Integer, i::Integer, j::Integer)
    nlon, _ = panel_shape(grid)
    return (Int(panel) - 1) * _panel_cells(grid) + (Int(j) - 1) * nlon + Int(i)
end

# ---- the three traits -------------------------------------------------------

@inline cell_address(::YinYangGrid) = FlatCells()
@inline adjacency_source(::YinYangGrid) = FormulaNeighbors()
@inline candidate_source(::YinYangGrid) = IndexedCandidates()

# ---- coordinates ------------------------------------------------------------

@inline ncoordinates(::YinYangGrid) = 2

@inline function _raw_coords(grid::YinYangGrid{T}, k::Integer) where {T}
    nlon, nlat = panel_shape(grid)
    p, i, j = panel_cell(grid, k)
    λ, φ = SphericalSampling._yin_yang_panel_coords(T, nlon, nlat, i, j)
    return p == 1 ? (λ, φ) : SphericalSampling._yin_yang_rotate(λ, φ)
end

coordinates(grid::YinYangGrid) = throw(ArgumentError(
    "a YinYangGrid stores no coordinate arrays — a cell's position is its panel-local lat–lon, " *
    "rotated onto the sphere for yang. Use `coords(grid, k)` for one cell, or " *
    "`Grids.materialize(grid)` for the whole cloud.",
))

# ---- topology and span ------------------------------------------------------

# Yin's own frame spans three quarters of the circle, so no direction of the CELL numbering wraps: the
# two panels are separate patches and neither closes on itself.
@inline topology(::YinYangGrid) = (Bounded(), Bounded())
@inline period(grid::YinYangGrid{T}, d::Integer) where {T} =
    (_checked_direction((1, 2), d); zero(T))

# The union of the two panels covers the sphere, and yang reaches both poles.
@inline bounds(grid::YinYangGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? (zero(T), T(2π)) : (-T(π) / 2, T(π) / 2)

@inline origin(grid::YinYangGrid, d::Integer) = bounds(grid, d)[1]

# ---- measure ----------------------------------------------------------------

# A panel cell is a lat–lon patch in its own frame, so its area integrates to
# `R²·Δλ·(sin(φ+Δφ/2) − sin(φ−Δφ/2)) = R²·Δλ·2sin(Δφ/2)·cos φ` — independent of `λ`, and the same on
# both panels because yang is a rigid rotation of yin.
@inline function measure(grid::YinYangGrid{T}, k::Integer) where {T}
    nlon, nlat = panel_shape(grid)
    _, _, j = panel_cell(grid, k)
    Δλ = (T(3π) / 2) / T(nlon)
    Δφ = (T(π) / 2) / T(nlat)
    _, φ = SphericalSampling._yin_yang_panel_coords(T, nlon, nlat, 1, j)
    return T(Geometry.radius(grid_geometry(grid)))^2 * Δλ * 2 * sin(Δφ / 2) * cos(φ)
end

@inline measure(grid::YinYangGrid) = GridMeasure(grid)

# `Σ_j cos φ_j` over the `nlat` rows, twice over, so this is `O(nlat)` rather than `O(nlon·nlat)`.
function _total_measure(grid::YinYangGrid{T}) where {T}
    nlon, nlat = panel_shape(grid)
    Δλ = (T(3π) / 2) / T(nlon)
    Δφ = (T(π) / 2) / T(nlat)
    c = T(Geometry.radius(grid_geometry(grid)))^2 * Δλ * 2 * sin(Δφ / 2)
    s = zero(T)
    @inbounds for j in 1:nlat
        _, φ = SphericalSampling._yin_yang_panel_coords(T, nlon, nlat, 1, j)
        s += cos(φ)
    end
    return 2 * T(nlon) * c * s
end

# ---- display ----------------------------------------------------------------

Base.show(io::IO, grid::YinYangGrid{T}) where {T} =
    print(io, "YinYangGrid{", T, "}(", panel_shape(grid)[1], "×", panel_shape(grid)[2],
          " per panel, ", length(grid), " cells)")

function Base.show(io::IO, ::MIME"text/plain", grid::YinYangGrid{T}) where {T}
    nlon, nlat = panel_shape(grid)
    println(io, "YinYangGrid{", T, "} 2 × ", nlon, " × ", nlat, " (", count(mask(grid)), "/",
            length(grid), " active)")
    println(io, "  geometry:  ", grid_geometry(grid))
    print(io, "  measure:   ", sum(measure(grid)), " total (panels overlap)")
end
