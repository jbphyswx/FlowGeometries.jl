module Grids

using ..Geometry: Geometry

# Public API via `FlowGeometries.Grids.*` or `FlowGeometries.coords` (parent rebind). No exports.

"""
    AbstractGrid{G<:AbstractGeometry, T<:AbstractFloat}

Supertype for all grid architectures.
"""
abstract type AbstractGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} end

"""
    AbstractStructuredGrid{G,T} <: AbstractGrid{G,T}

Rectilinear grids.  Default: [`StructuredGrid`](@ref).
"""
abstract type AbstractStructuredGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} <: AbstractGrid{G,T} end

"""
    AbstractCurvilinearGrid{G,T} <: AbstractGrid{G,T}

Curvilinear grids.  Default: [`CurvilinearGrid`](@ref).
"""
abstract type AbstractCurvilinearGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} <: AbstractGrid{G,T} end

"""
    AbstractUnstructuredGrid{G,T} <: AbstractGrid{G,T}

Unstructured / node grids.  Default: [`UnstructuredGrid`](@ref).
"""
abstract type AbstractUnstructuredGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} <: AbstractGrid{G,T} end

# Common interface queries
grid_geometry(grid::AbstractGrid) = grid.geometry

"""
    isperiodic(grid, dim) -> Bool

Whether axis `dim` (1 = x, 2 = y — longitude/latitude for a spherical grid) is periodic, i.e. the filter footprint should wrap
across that boundary. Non-periodic by default; `StructuredGrid` carries explicit per-axis flags.
"""
isperiodic(::AbstractGrid, ::Integer) = false

# ---------------------------------------------------------------------------
# Structured Grid
# ---------------------------------------------------------------------------

"""
    StructuredGrid{G, T, N, Ax, AT, BT}

Structured (rectilinear) `N`-dimensional grid: one coordinate vector per axis (`axes`), an N-D cell
`measure` (length in 1D, area in 2D, volume in 3D), an N-D active `mask`, and per-axis `periodic`
flags. `N = 1, 2, 3`.

`Ax` is a heterogeneous `NTuple{N, AbstractVector{T}}` — each axis independently keeps whatever
concrete `AbstractVector{T}` type it was constructed with (a `Range`, a plain `Vector`, or any other
subtype); there is deliberately no shared single vector type forcing e.g. `x` and `y` to match.
This matters beyond storage: a `Range`'s type is a compile-time proof of constant spacing that
`Filtering.build_footprint` dispatches on to select an exact fast path with no runtime check, and
that proof would be destroyed if both axes were forced into a common (possibly abstract, possibly
`Vector`) type.

The first two axes carry the convenience aliases `grid.x` / `grid.y` (= `axes[1]` / `axes[2]`),
and `grid.areas` aliases `grid.measure`, so 2D code reads naturally.
"""
struct StructuredGrid{
    G<:Geometry.AbstractGeometry,
    T<:AbstractFloat,
    N,
    Ax<:NTuple{N,AbstractVector{T}},
    AT<:AbstractArray{T,N},
    BT<:AbstractArray{Bool,N},
} <: AbstractStructuredGrid{G, T}
    geometry::G
    axes::Ax                # coordinate vector per axis (axes[1]=x, axes[2]=y, axes[3]=z)
    measure::AT             # N-D cell measure (area in 2D, volume in 3D)
    mask::BT                # N-D active mask (true=active/included, false=excluded)
    periodic::NTuple{N,Bool} # per-axis periodicity for footprint wrapping
end

# Convenience field aliases: x/y for the first two axes, areas ≡ measure (keeps 2D code/users
# working). `getfield` is used internally to avoid recursion.
@inline function Base.getproperty(grid::StructuredGrid, name::Symbol)
    if name === :x
        return @inbounds getfield(grid, :axes)[1]
    elseif name === :y
        return @inbounds getfield(grid, :axes)[2]
    elseif name === :areas
        return getfield(grid, :measure)
    else
        return getfield(grid, name)
    end
end

size_tuple(grid::StructuredGrid) = size(grid.mask)

@inline isperiodic(grid::StructuredGrid, dim::Integer) = grid.periodic[dim]

# ---------------------------------------------------------------------------
# Point accessors: NamedTuple default; coords! / coords(S, ...) for other storage
# ---------------------------------------------------------------------------

"""
    _raw_coords(grid, I...) -> NTuple

Positional coordinate values at indices `I` (axis order). Internal; prefer [`coords`](@ref).
"""
@inline function _raw_coords(grid::StructuredGrid{G,T,N}, I::Vararg{Integer,N}) where {G,T,N}
    return ntuple(d -> @inbounds(grid.axes[d][I[d]]), N)
end

# CurvilinearGrid / UnstructuredGrid _raw_coords are defined after those types exist.

"""
    _construct_point(::Type{S}, vals::NTuple{N,T})

Build a point of type `S` from positional values.
"""
@inline _construct_point(::Type{NTuple{N,T}}, vals::NTuple{N,T}) where {N,T} = vals
@inline _construct_point(::Type{Tuple}, vals::NTuple) = vals
@inline function _construct_point(::Type{Vector{T}}, vals::NTuple{N,T}) where {N,T}
    return T[vals...]
end
@inline function _construct_point(::Type{NamedTuple}, geo::Geometry.AbstractGeometry, vals::NTuple{N,T}) where {N,T}
    return Geometry.named_point(geo, vals)
end
# Generic fallback: `S(vals...)` (covers many user types)
@inline _construct_point(::Type{S}, vals::NTuple) where {S} = S(vals...)

"""
    coords(grid, I...) -> NamedTuple

Point at indices `I`, as a geometry-named `NamedTuple`:
- Cartesian: `(x=, y=)` or `(x=, y=, z=)`
- Spherical: `(λ=, φ=)` or `(λ=, φ=, r=)`

See also [`coords!`](@ref), [`coords(::Type, grid, I...)`](@ref).
"""
@inline function coords(grid::AbstractGrid, I::Vararg{Integer})
    return Geometry.named_point(grid_geometry(grid), _raw_coords(grid, I...))
end

"""
    coords(::Type{S}, grid, I...)

Construct the point as type `S` (e.g. `NTuple{2,Float64}`, `Vector{Float64}`,
or `SVector{2,Float64}` when StaticArrays is loaded).
"""
@inline function coords(::Type{S}, grid::AbstractGrid, I::Vararg{Integer}) where {S}
    return _construct_point(S, _raw_coords(grid, I...))
end

"""
    coords!(out, grid, I...) -> out

Write positional coordinates into a preallocated `AbstractVector` (`length(out) ≥ N`).
"""
@inline function coords!(out::AbstractVector, grid::AbstractGrid, I::Vararg{Integer})
    vals = _raw_coords(grid, I...)
    N = length(vals)
    length(out) ≥ N || throw(DimensionMismatch("out length $(length(out)) < point dimension $N"))
    @inbounds for d in 1:N
        out[d] = vals[d]
    end
    return out
end

@inline area(grid::StructuredGrid{G,T,N}, I::Vararg{Integer,N}) where {G,T,N} = grid.measure[I...]
@inline isactive(grid::StructuredGrid{G,T,N}, I::Vararg{Integer,N}) where {G,T,N} = grid.mask[I...]

# Auto-detect axis-1 periodicity: on a spherical grid axis 1 is longitude, and a longitude axis
# that closes the full 2π circle (to within one cell) is periodic; a regional span is NOT.
# Cartesian axes are opt-in only (there's no analogous auto-detectable physical closure).
_auto_periodic_x(::Geometry.AbstractCartesianGeometry, x::AbstractVector) = false
function _auto_periodic_x(::Geometry.AbstractSphericalGeometry, x::AbstractVector{T}) where {T<:AbstractFloat}
    length(x) > 2 || return false
    dλ = x[2] - x[1]
    iszero(dλ) && return false
    return isapprox(x[end] - x[1] + dλ, T(2π); atol = abs(dλ))
end

_periodic_tuple(p::NTuple{2,Bool}) = p
_periodic_tuple(p::Bool) = (p, false)
_periodic_tuple(p::Tuple{Bool}) = (p[1], false)

"""
    _local_spacing(x, i, period=nothing) -> (h_m, h_p)

Zero-allocation one-sided coordinate gaps around index `i` of a 1D axis `x`: `h_m = x[i]-x[i-1]`
and `h_p = x[i+1]-x[i]`. This is the single primitive both the per-cell area computation below and
the nonuniform derivative stencils (`Derivatives.ddx!`/`ddy!`) are built on — always a scalar
subtraction of two already-stored array elements, never a heap allocation, so it's safe to call
per grid point in a hot loop.

`period`, if given (e.g. `2π` for a periodic longitude axis), makes the boundary gaps *wrap*
instead of vanishing: at `i==1`, `h_m` is the gap to the unwrapped previous point `x[n]-period`;
at `i==n`, `h_p` is the gap to the unwrapped next point `x[1]+period`. Pass `nothing` (default) for
a non-periodic axis, where boundary gaps are simply zero (the caller then falls back to a one-sided
stencil).
"""
@inline function _local_spacing(
    x::AbstractVector{T}, i::Integer, period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    if period === nothing
        h_m = i > 1 ? @inbounds(x[i] - x[i-1]) : zero(T)
        h_p = i < n ? @inbounds(x[i+1] - x[i]) : zero(T)
    else
        p = T(period)
        h_m = i > 1 ? @inbounds(x[i] - x[i-1]) : @inbounds(x[1] - (x[n] - p))
        h_p = i < n ? @inbounds(x[i+1] - x[i]) : @inbounds((x[1] + p) - x[n])
    end
    return h_m, h_p
end

"""
    _to_axis(T, x) -> AbstractVector{T}

Adapt axis input `x` to element type `T` while preserving its concrete type whenever possible —
never force a `Range` (or any other `AbstractVector` subtype) into a plain `Vector` just to get the
eltype to match. This matters beyond just avoiding an unnecessary allocation: a `Range`'s type is a
*proof* of constant spacing that later code (`Filtering.build_footprint`) dispatches on to select an
exact, zero-runtime-check fast path — collapsing it into a `Vector` would silently destroy that
guarantee. Four non-overlapping, unambiguous methods:
- `AbstractRange{T}` (already the right eltype): passthrough, zero cost.
- `AbstractRange` (wrong eltype): reconstruct as a `Range` of eltype `T` (still provably uniform).
- `AbstractVector{T}` (already the right eltype, not a `Range`): passthrough, zero cost.
- `AbstractVector` (wrong eltype, not a `Range`): the one genuine case that must copy.
"""
_to_axis(::Type{T}, x::AbstractRange{T}) where {T<:AbstractFloat} = x
_to_axis(::Type{T}, x::AbstractRange) where {T<:AbstractFloat} =
    range(T(first(x)); step = T(step(x)), length = length(x))
_to_axis(::Type{T}, x::AbstractVector{T}) where {T<:AbstractFloat} = x
_to_axis(::Type{T}, x::AbstractVector) where {T<:AbstractFloat} = convert(Vector{T}, x)

"""
    _min_gap(x) -> minimum consecutive |gap|, or Inf if length(x) < 2

Smallest spacing found anywhere on axis `x`. Used to build a conservative (safe, never
under-covering) search-radius bound for a genuinely nonuniform axis: since real distance checks
still gate what's actually included, using the smallest gap anywhere can only widen the search
window, never cause a missed in-range cell.
"""
function _min_gap(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    n < 2 && return T(Inf)
    @inbounds m = abs(x[2] - x[1])
    @inbounds for i in 3:n
        g = abs(x[i] - x[i-1])
        g < m && (m = g)
    end
    return m
end

"""
    _cell_width(x, i, period=nothing) -> width

Per-cell coordinate width at index `i` of a 1D axis of cell-centered samples `x`: the centered
width `(|h_m|+|h_p|)/2` at interior cells (and, when `period` is given, at the wrapped boundary too —
see [`_local_spacing`](@ref)); for a genuinely non-periodic boundary, the one-sided gap to the
single neighbour; zero for a length-1 axis. For a *uniform* axis every width equals the constant
step, so this is a strict generalization of the old single-Δ convention (bit-for-bit identical on
uniform grids) that is also correct for genuinely nonuniform axes. Zero-allocation (built on
[`_local_spacing`](@ref); no array is materialized).

A "width" is a physical measure and so must be non-negative regardless of whether the axis happens
to be stored increasing or decreasing (e.g. `lat = π/2 .- θ`, the natural FastSphericalHarmonics
recipe, or any dataset storing latitude/depth/pressure levels in descending order) — `_local_spacing`
itself returns SIGNED gaps (needed as-is by the nonuniform derivative stencils in `Derivatives.jl`,
where the sign correctly encodes index-vs-coordinate direction), so the `abs` is applied here, at the
one place that turns a spacing into an area/volume/length contribution.
"""
@inline function _cell_width(
    x::AbstractVector{T}, i::Integer, period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    # A singleton axis contributes a multiplicative IDENTITY (one), not zero, to a measure that's a
    # plain product of per-axis widths (e.g. Cartesian area = Δx·Δy) — this correctly degenerates
    # area -> length when one axis has no real extent, instead of forcing the whole product to zero.
    # (This convention is wrong for the spherical R²cosφ·Δλ·Δφ area formula, which has its own
    # explicit singleton-axis handling in the spherical `StructuredGrid` constructor below, using the
    # correct lower-dimensional arc-length formula rather than substituting a placeholder here.)
    n == 1 && return one(T)
    h_m, h_p = _local_spacing(x, i, period)
    if period === nothing
        i == 1 && return abs(h_p)
        i == n && return abs(h_m)
    end
    return (abs(h_m) + abs(h_p)) / T(2)
end

"""
    StructuredGrid(geometry, x, y, mask; periodic = nothing)

Build a structured (rectilinear) grid, pre-computing cell areas from the geometry. `x`/`y` are
longitude/latitude (radians) for a `SphericalGeometry`, or physical distance for a `CartesianGeometry`.

`periodic` controls footprint wrapping per axis (`(x, y)`); pass a `Bool` (applied to x) or
an `NTuple{2,Bool}`. When omitted, axis-1 periodicity is auto-detected — on a spherical grid, an
`x` (longitude) axis spanning the full circle is treated as periodic, a regional span is not — and
axis 2 (`y`) is non-periodic.
"""
function StructuredGrid(
    geometry::G,
    x::AbstractVector,
    y::AbstractVector,
    mask::AbstractMatrix{Bool};
    periodic = nothing,
) where {
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T}
}
    # Adapt x/y to the geometry float type T, preserving concrete type (a `Range` stays a
    # `Range` — see `_to_axis`).
    x_T = _to_axis(T, x)
    y_T = _to_axis(T, y)

    Nx = length(x_T)
    Ny = length(y_T)

    per = periodic === nothing ? (_auto_periodic_x(geometry, x_T), false) : _periodic_tuple(periodic)
    # A periodic SPHERICAL longitude axis spans exactly one full turn (2π radians) — but a periodic
    # CARTESIAN axis is measured in physical distance (meters), where 2π has no meaning at all. Using
    # the spherical constant unconditionally here was a real bug: it silently corrupted the boundary
    # cell width (and hence its filter weight) for any periodic Cartesian grid, since `_cell_width`
    # would wrap using a period of ~6.28 units instead of the actual domain length — e.g. producing a
    # wildly wrong (even negative) cell width, and a correspondingly nonsensical filter weight, at the
    # seam. For Cartesian, the period is the real extent plus the (assumed-uniform) first-cell
    # spacing, matching this file's existing "extent + one spacing" periodicity convention.
    x_period = if per[1]
        G <: Geometry.AbstractSphericalGeometry{T} ? T(2π) : (x_T[end] - x_T[1] + (x_T[2] - x_T[1]))
    else
        nothing
    end

    # Pre-allocate areas matrix (this single O(Nx*Ny) allocation happens once, at grid
    # construction — not a hot path — so computing a genuine per-cell area below costs nothing
    # extra beyond what the old single-Δ version already allocated).
    areas = Matrix{T}(undef, Nx, Ny)

    # Populate cell areas from the REAL per-cell axis spacing (via `_cell_width`, zero-allocation
    # scalar lookups), not a single global Δ read from just the first two samples — this is what
    # makes area computation correct for genuinely nonuniform axes while being bit-for-bit
    # identical to the old behavior on a uniform axis (where every per-cell width equals the same
    # constant step).
    if G <: Geometry.AbstractCartesianGeometry{T}
        @inbounds for j in 1:Ny
            Δφ = _cell_width(y_T, j)
            for i in 1:Nx
                areas[i, j] = _cell_width(x_T, i, x_period) * Δφ
            end
        end
    elseif Nx == 1 && Ny == 1
        fill!(areas, one(T))   # single point: no meaningful measure, dimensionless placeholder
    elseif Ny == 1
        # Zonal transect (singleton latitude): the measure genuinely degenerates from area
        # (R²cosφ·Δλ·Δφ) to arc length along that circle of latitude (R·cosφ·Δλ) — NOT
        # `area_element` with a placeholder Δφ, which would leave a spurious extra factor of R
        # (`area_element`'s R² assumes TWO angular differentials; only one survives here).
        R = geometry.R
        φ = y_T[1]
        cosφ = cos(φ)
        @inbounds for i in 1:Nx
            areas[i, 1] = R * cosφ * _cell_width(x_T, i, x_period)
        end
    elseif Nx == 1
        # Meridional transect (singleton longitude): arc length along the meridian, R·Δφ — a
        # meridian's arc-length element has no cosφ factor (unlike a parallel's).
        R = geometry.R
        @inbounds for j in 1:Ny
            areas[1, j] = R * _cell_width(y_T, j)
        end
    else
        @inbounds for j in 1:Ny
            φ = y_T[j]
            Δφ = _cell_width(y_T, j)
            for i in 1:Nx
                Δλ = _cell_width(x_T, i, x_period)
                areas[i, j] = Geometry.area_element(geometry, φ, Δλ, Δφ)
            end
        end
    end

    axes = (x_T, y_T)
    return StructuredGrid{G, T, 2, typeof(axes), typeof(areas), typeof(mask)}(
        geometry, axes, areas, mask, per,
    )
end

"""
    StructuredGrid(geometry, x, mask; periodic = false)

Build a 1D Cartesian grid (cell length `dx`). `periodic` is a `Bool` (default `false`).
"""
function StructuredGrid(
    geometry::G,
    x::AbstractVector,
    mask::AbstractVector{Bool};
    periodic::Bool = false,
) where {T<:AbstractFloat, G<:Geometry.AbstractCartesianGeometry{T}}
    x_T = _to_axis(T, x)
    n = length(x_T)
    # Genuine per-cell measure (not a `fill(geometry.dx, ...)` broadcast of the geometry's nominal
    # scalar spacing) — identical to the old value on a uniform axis, and correct for a nonuniform
    # one. One-time O(n) cost at construction, not a hot path.
    measure = Vector{T}(undef, n)
    @inbounds for i in 1:n
        measure[i] = _cell_width(x_T, i)
    end
    axes = (x_T,)
    return StructuredGrid{G, T, 1, typeof(axes), typeof(measure), typeof(mask)}(
        geometry, axes, measure, mask, (periodic,),
    )
end

"""
    StructuredGrid(geometry, x, y, z, mask; periodic = nothing)

Build a 3D grid. For `CartesianGeometry`, cell volume is `dx·dy·dz` (the geometry must carry a
non-zero `dz`). For `SphericalGeometry`, `z` is the RADIUS axis — the absolute physical distance from
the planet center at each level, not a depth/height offset from a reference radius (which would force
picking an ocean-vs-atmosphere sign convention with no natural default) — and needs at least 2 levels
(a single level is the 2D/2.5D case; use the 2-argument `(x, y)` constructor instead). Cell volume
is the genuine spherical-shell element `r²·cosφ·Δλ·Δφ·Δr` at each level's own local radius (see
[`Geometry.volume_element`](@ref)).

`periodic` is a `Bool` (applied to axis 1, `x`) or an `NTuple{3,Bool}`; when omitted, axis-1
periodicity is auto-detected the same way the 2D constructor does (on a spherical grid, an `x`
(longitude) axis spanning the full circle is periodic, a regional span is not; Cartesian is never
auto-periodic).
"""
function StructuredGrid(
    geometry::G,
    x::AbstractVector,
    y::AbstractVector,
    z::AbstractVector,
    mask::AbstractArray{Bool,3};
    periodic = nothing,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    x_T = _to_axis(T, x)
    y_T = _to_axis(T, y)
    z_T = _to_axis(T, z)
    Nx, Ny, Nz = length(x_T), length(y_T), length(z_T)

    per = if periodic === nothing
        (_auto_periodic_x(geometry, x_T), false, false)
    elseif periodic isa Bool
        (periodic, false, false)
    else
        NTuple{3,Bool}(periodic)
    end

    # Genuine per-cell measure (not a `fill(dx*dy*dz, ...)` broadcast of the geometry's nominal
    # scalar spacing) — identical to the old value on a uniform axis, correct for a nonuniform one.
    measure = Array{T,3}(undef, Nx, Ny, Nz)
    if G <: Geometry.AbstractCartesianGeometry{T}
        @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
            measure[i, j, k] = _cell_width(x_T, i) * _cell_width(y_T, j) * _cell_width(z_T, k)
        end
    else
        Nz > 1 || throw(ArgumentError(
            "a true 3D spherical StructuredGrid needs at least 2 radius levels (got $Nz); a single " *
            "level is the 2D/2.5D case — use the (geometry, x, y, mask) constructor instead.",
        ))
        x_period = per[1] ? T(2π) : nothing
        @inbounds for k in 1:Nz
            Δr = _cell_width(z_T, k)
            rk = z_T[k]
            for j in 1:Ny
                φ = y_T[j]
                Δφ = _cell_width(y_T, j)
                for i in 1:Nx
                    Δλ = _cell_width(x_T, i, x_period)
                    measure[i, j, k] = Geometry.volume_element(geometry, rk, φ, Δλ, Δφ, Δr)
                end
            end
        end
    end

    axes = (x_T, y_T, z_T)
    return StructuredGrid{G, T, 3, typeof(axes), typeof(measure), typeof(mask)}(
        geometry, axes, measure, mask, per,
    )
end

# ---------------------------------------------------------------------------
# Curvilinear Grid
# ---------------------------------------------------------------------------

"""
    CurvilinearGrid{T, G, ML, MA, B}

Curvilinear grid where the cell-center coordinates are 2D arrays (e.g. an orthogonal curvilinear
mesh from a structured-grid ocean/atmosphere model). `x`/`y` are the `Nx × Ny` cell-center
coordinates; `x_corner`/`y_corner` are the `(Nx+1) × (Ny+1)` cell-vertex coordinates from
which the exact quadrilateral cell `areas` are computed directly, rather than a cell-center spacing
approximation.

# Type parameters
- `T`: coordinate float type. `G<:AbstractGeometry{T}` is tied to it (a mismatched-eltype geometry is
  a type error, not a silent promotion) — hence `T` precedes `G` (Julia forbids the forward
  reference `G<:AbstractGeometry{T}, T` needed to keep the `{G,T}` order).
- `ML`: matrix type shared by the four coordinate arrays (`x`/`y`/`x_corner`/`y_corner` — a
  mesh's own coordinate arrays are legitimately almost always the same concrete type).
- `MA`: matrix type of the derived `areas` field — independent of `ML`, since it is a computed field
  with no reason to match the coordinate arrays' storage type.
- `B`: matrix type of the active `mask`.
"""
struct CurvilinearGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T},
    ML<:AbstractMatrix{T},
    MA<:AbstractMatrix{T},
    B<:AbstractMatrix{Bool},
} <: AbstractCurvilinearGrid{G, T}
    geometry::G
    x::ML         # X/λ cell-center coordinate array (Nx × Ny)
    y::ML         # Y/φ cell-center coordinate array (Nx × Ny)
    x_corner::ML  # X/λ cell-vertex coordinate array ((Nx+1) × (Ny+1))
    y_corner::ML  # Y/φ cell-vertex coordinate array ((Nx+1) × (Ny+1))
    areas::MA       # pre-computed exact quadrilateral cell areas (Nx × Ny)
    mask::B         # active mask (true=active/included, false=excluded)
end

size_tuple(grid::CurvilinearGrid) = size(grid.mask)

@inline function _raw_coords(grid::CurvilinearGrid, i::Integer, j::Integer)
    return (grid.x[i, j], grid.y[i, j])
end

@inline area(grid::CurvilinearGrid, i::Integer, j::Integer) = grid.areas[i, j]
@inline isactive(grid::CurvilinearGrid, i::Integer, j::Integer) = grid.mask[i, j]

# ---------------------------------------------------------------------------
# Curvilinear grid construction: corner-based exact quadrilateral cell areas
# ---------------------------------------------------------------------------

# Adapt a coordinate matrix to element type `T`, preserving type when the eltype already matches
# (no needless copy); otherwise materialize a `Matrix{T}`.
_to_mat(::Type{T}, M::AbstractMatrix{T}) where {T<:AbstractFloat} = M
_to_mat(::Type{T}, M::AbstractMatrix) where {T<:AbstractFloat} = convert(Matrix{T}, M)

"""
    _centers_to_corners(C) -> K

Reconstruct an `(n+1) × (m+1)` cell-vertex array from an `n × m` cell-center array `C` by averaging
the (up to four) surrounding centers, with a linearly-extrapolated one-cell ghost ring so the true
domain-boundary vertices are placed a half-cell outside the outermost centers. Used only when the
caller does not supply explicit corner arrays; requires `n, m ≥ 2`.
"""
function _centers_to_corners(C::AbstractMatrix{T}) where {T<:AbstractFloat}
    n, m = size(C)
    (n >= 2 && m >= 2) || throw(ArgumentError(
        "auto-deriving curvilinear cell corners needs a grid of at least 2×2 centers; " *
        "supply `x_corner`/`y_corner` explicitly for a smaller grid",
    ))
    # Padded centers with a linearly-extrapolated one-cell ghost ring.
    P = Matrix{T}(undef, n + 2, m + 2)
    @inbounds for j in 1:m, i in 1:n
        P[i+1, j+1] = C[i, j]
    end
    @inbounds for j in 1:m
        P[1,   j+1] = T(2) * C[1, j] - C[2, j]
        P[n+2, j+1] = T(2) * C[n, j] - C[n-1, j]
    end
    @inbounds for i in 1:n
        P[i+1, 1]   = T(2) * C[i, 1] - C[i, 2]
        P[i+1, m+2] = T(2) * C[i, m] - C[i, m-1]
    end
    # Ghost corners via bilinear extrapolation from the adjacent ghost edges.
    @inbounds begin
        P[1, 1]     = P[1, 2] + P[2, 1] - P[2, 2]
        P[n+2, 1]   = P[n+2, 2] + P[n+1, 1] - P[n+1, 2]
        P[1, m+2]   = P[1, m+1] + P[2, m+2] - P[2, m+1]
        P[n+2, m+2] = P[n+2, m+1] + P[n+1, m+2] - P[n+1, m+1]
    end
    # Each vertex is the average of the 2×2 padded centers surrounding it.
    K = Matrix{T}(undef, n + 1, m + 1)
    @inbounds for j in 1:(m+1), i in 1:(n+1)
        K[i, j] = (P[i, j] + P[i+1, j] + P[i, j+1] + P[i+1, j+1]) / T(4)
    end
    return K
end

# Exact planar quadrilateral area (shoelace) for a Cartesian cell with the four vertices in order.
@inline function _quad_area(
    ::Geometry.AbstractCartesianGeometry{T},
    x1::T, y1::T, x2::T, y2::T, x3::T, y3::T, x4::T, y4::T,
) where {T}
    return T(0.5) * abs(x1 * (y2 - y4) + x2 * (y3 - y1) + x3 * (y4 - y2) + x4 * (y1 - y3))
end

# Exact spherical-triangle area (L'Huilier) from three (λ, φ) vertices; great-circle side lengths.
@inline function _sph_triangle_area(
    geo::Geometry.AbstractSphericalGeometry{T}, p1::NTuple{2,T}, p2::NTuple{2,T}, p3::NTuple{2,T},
) where {T}
    R = geo.R
    a = Geometry.distance(geo, p2, p3) / R   # arc angle opposite p1
    b = Geometry.distance(geo, p1, p3) / R
    c = Geometry.distance(geo, p1, p2) / R
    s = (a + b + c) / T(2)
    t = tan(s / T(2)) * tan((s - a) / T(2)) * tan((s - b) / T(2)) * tan((s - c) / T(2))
    E = T(4) * atan(sqrt(max(zero(T), t)))    # spherical excess
    return R^2 * E
end

# Exact spherical quadrilateral area = sum of the two triangles (p1,p2,p3) and (p1,p3,p4).
@inline function _quad_area(
    geo::Geometry.AbstractSphericalGeometry{T},
    λ1::T, φ1::T, λ2::T, φ2::T, λ3::T, φ3::T, λ4::T, φ4::T,
) where {T}
    p1 = (λ1, φ1); p2 = (λ2, φ2)
    p3 = (λ3, φ3); p4 = (λ4, φ4)
    return _sph_triangle_area(geo, p1, p2, p3) + _sph_triangle_area(geo, p1, p3, p4)
end

# Compute the Nx×Ny exact quadrilateral cell areas from the (Nx+1)×(Ny+1) corner arrays.
function _corner_areas(
    geometry::Geometry.AbstractGeometry{T}, xc::AbstractMatrix{T}, yc::AbstractMatrix{T},
    Nx::Integer, Ny::Integer,
) where {T<:AbstractFloat}
    areas = Matrix{T}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        # Cell (i,j) has vertices (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1) (counter-clockwise in index space).
        areas[i, j] = _quad_area(
            geometry,
            xc[i, j],     yc[i, j],
            xc[i+1, j],   yc[i+1, j],
            xc[i+1, j+1], yc[i+1, j+1],
            xc[i, j+1],   yc[i, j+1],
        )
    end
    return areas
end

"""
    CurvilinearGrid(geometry, x, y, mask; x_corner=nothing, y_corner=nothing)

Build a curvilinear grid from `Nx × Ny` cell-center coordinate arrays `x`/`y`, computing
exact quadrilateral cell areas from the `(Nx+1) × (Ny+1)` cell-vertex arrays. Supply
`x_corner`/`y_corner` for exact corners (e.g. from the source model's own cell-vertex grid);
otherwise they are reconstructed
from the centers (see [`_centers_to_corners`](@ref)), which requires at least a 2×2 grid.

Spherical cell areas use the exact spherical-quadrilateral (L'Huilier excess) area; Cartesian cells
use the exact planar shoelace area. The grid is treated as non-periodic.
"""
function CurvilinearGrid(
    geometry::G,
    x::AbstractMatrix,
    y::AbstractMatrix,
    mask::AbstractMatrix{Bool};
    x_corner::Union{Nothing,AbstractMatrix} = nothing,
    y_corner::Union{Nothing,AbstractMatrix} = nothing,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    size(x) == size(y) || throw(ArgumentError("x and y must have the same size"))
    size(x) == size(mask) || throw(ArgumentError("x/y and mask must have the same size"))
    Nx, Ny = size(x)

    x_T = _to_mat(T, x)
    y_T = _to_mat(T, y)
    xc = x_corner === nothing ? _centers_to_corners(x_T) : _to_mat(T, x_corner)
    yc = y_corner === nothing ? _centers_to_corners(y_T) : _to_mat(T, y_corner)
    size(xc) == (Nx + 1, Ny + 1) || throw(ArgumentError(
        "x_corner/y_corner must be (Nx+1)×(Ny+1) = $((Nx+1, Ny+1)); got $(size(xc))",
    ))
    size(yc) == size(xc) || throw(ArgumentError("x_corner and y_corner must have the same size"))

    areas = _corner_areas(geometry, xc, yc, Nx, Ny)
    return CurvilinearGrid{T, G, typeof(x_T), typeof(areas), typeof(mask)}(
        geometry, x_T, y_T, xc, yc, areas, mask,
    )
end

"""
    CurvilinearGrid(geometry, x, y, areas, mask; x_corner=nothing, y_corner=nothing)

Build a curvilinear grid from cell-center coordinates with caller-supplied cell `areas` (common when
a dataset ships its own cell areas). Corner arrays are still stored (reconstructed from the centers
if not supplied) but the supplied `areas` are used verbatim rather than recomputed.
"""
function CurvilinearGrid(
    geometry::G,
    x::AbstractMatrix,
    y::AbstractMatrix,
    areas::AbstractMatrix{<:Real},
    mask::AbstractMatrix{Bool};
    x_corner::Union{Nothing,AbstractMatrix} = nothing,
    y_corner::Union{Nothing,AbstractMatrix} = nothing,
) where {T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    size(x) == size(y) || throw(ArgumentError("x and y must have the same size"))
    size(x) == size(mask) || throw(ArgumentError("x/y and mask must have the same size"))
    size(x) == size(areas) || throw(ArgumentError("x/y and areas must have the same size"))

    x_T = _to_mat(T, x)
    y_T = _to_mat(T, y)
    areas_T = _to_mat(T, areas)
    xc = x_corner === nothing ? _centers_to_corners(x_T) : _to_mat(T, x_corner)
    yc = y_corner === nothing ? _centers_to_corners(y_T) : _to_mat(T, y_corner)

    return CurvilinearGrid{T, G, typeof(x_T), typeof(areas_T), typeof(mask)}(
        geometry, x_T, y_T, xc, yc, areas_T, mask,
    )
end

# ---------------------------------------------------------------------------
# Unstructured Grid
# ---------------------------------------------------------------------------

"""
    UnstructuredGrid{T, G, V, VA, B, VI}

Unstructured mesh (e.g. radial data, finite volume, or triangular mesh) where coords are 1D vectors.

# Type parameters
- `T`: coordinate float type. `G<:AbstractGeometry{T}` is tied to it (a mismatched-eltype geometry is
  a type error, not a silent promotion) — hence `T` precedes `G` (Julia forbids the forward reference
  `G<:AbstractGeometry{T}, T` needed to keep the `{G,T}` order), matching the same convention
  [`CurvilinearGrid`](@ref) uses.
- `V`: vector type shared by `x`/`y` (a node's own coordinate vectors are legitimately almost
  always the same concrete type).
- `VA`: vector type of the derived `areas` field — independent of `V`, since it is frequently a
  computed field (Voronoi tessellation) with no reason to match the coordinate vectors' storage type.
- `B`/`VI`: mask and CSR index-array storage types.

Neighbor adjacency is stored CSR-style (flat `neighbor_nbrs` + `neighbor_ptr` offsets, node `t` owns
`neighbor_ptr[t]:neighbor_ptr[t+1]-1`) rather than as a `Vector{Vector{Int}}` — the data is immutable
after construction, so there's no reason to pay for `Nnodes` separately-heap-allocated per-node
`Vector`s (cache-unfriendly pointer-chasing, one allocation per node) when one contiguous block (two
allocations total) holds the same information, matching the CSR footprint / WLSQ plan convention used by coarse-graining consumers.
"""
struct UnstructuredGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T},
    V<:AbstractVector{T},
    VA<:AbstractVector{T},
    B<:AbstractVector{Bool},
    VI<:AbstractVector{Int},
} <: AbstractUnstructuredGrid{G, T}
    geometry::G
    x::V       # X/λ vector for each node (Nnodes)
    y::V       # Y/φ vector for each node (Nnodes)
    areas::VA    # Area of each grid cell / control volume (Nnodes)
    mask::B      # active mask (true=active/included, false=excluded) (Nnodes)
    neighbor_nbrs::VI  # flat neighbor-index array (CSR)
    neighbor_ptr::VI   # CSR offsets, length Nnodes+1
end

"""
    UnstructuredGrid(geometry, x, y, areas, mask)

Convenience constructor with no neighbor adjacency (every node reports zero neighbors) — for
scattered-point spectral filtering (FINUFFT/NUFSHT), which doesn't need one. Real-space filtering /
derivatives on an `UnstructuredGrid` need actual adjacency; build it explicitly (e.g. via the k-d-tree
constructor below) and pass it via the full 7-argument constructor.
"""
function UnstructuredGrid(
    geometry::Geometry.AbstractGeometry{T}, x::AbstractVector, y::AbstractVector,
    areas::AbstractVector, mask::AbstractVector{Bool},
) where {T<:AbstractFloat}
    N = length(x)
    return UnstructuredGrid(geometry, x, y, areas, mask, Int[], ones(Int, N + 1))
end

# ---------------------------------------------------------------------------
# Unstructured grid construction: k-d-tree adjacency + (optional) Voronoi areas
# ---------------------------------------------------------------------------
#
# Two extension hook points (fallbacks that throw until a consumer package loads the relevant
# weakdep and overrides these methods): a k-d-tree neighbor query (`NearestNeighbors.jl`) and a
# per-node Voronoi-cell area (`DelaunayTriangulation.jl` for Cartesian, `Quickhull.jl` for spherical,
# dispatched on the geometry type since each needs a different tessellation library).

"""
    _build_kdtree_neighbors(geometry, x, y; k=6, radius=nothing) -> (nbrs::Vector{Int}, ptr::Vector{Int})

Extension hook: build CSR neighbor adjacency via a k-d tree. Overridden by
a consumer NearestNeighbors extension (load `using NearestNeighbors`). `radius`, if given,
switches to an all-neighbors-within-`radius` query (mutually exclusive with `k`); `radius` is in the
grid's physical distance units (`Geometry.distance` — meters for `SphericalGeometry`, geometry's own
units for `CartesianGeometry`), NOT a raw chord/angle.
"""
function _build_kdtree_neighbors(
    geometry::Geometry.AbstractGeometry, x::AbstractVector, y::AbstractVector;
    k::Integer = 6, radius::Union{Nothing,Real} = nothing,
)
    throw(ArgumentError(
        "k-d-tree neighbor construction requires NearestNeighbors.jl — run `using NearestNeighbors` " *
        "(or build adjacency explicitly and pass it via the full 7-argument `UnstructuredGrid` constructor).",
    ))
end

"""
    _voronoi_areas(geometry, x, y) -> Vector{T}

Extension hook: exact per-node Voronoi-cell area from a Delaunay/convex-hull tessellation of the node
coordinates. Dispatched on the geometry type (each needs a different tessellation library): overridden
for `CartesianGeometry` by a consumer DelaunayTriangulation extension (load
`using DelaunayTriangulation`, planar Voronoi clipped to the point set's convex hull) and for
`SphericalGeometry` by a consumer Quickhull extension (load `using Quickhull`, spherical
Voronoi from the dual of the 3D convex hull of the unit-sphere embedding).
"""
function _voronoi_areas(geometry::Geometry.AbstractCartesianGeometry, x::AbstractVector, y::AbstractVector)
    throw(ArgumentError(
        "Cartesian Voronoi-cell areas require DelaunayTriangulation.jl — run `using DelaunayTriangulation` " *
        "(or supply `areas` explicitly to the `UnstructuredGrid` constructor).",
    ))
end
function _voronoi_areas(geometry::Geometry.AbstractSphericalGeometry, x::AbstractVector, y::AbstractVector)
    throw(ArgumentError(
        "Spherical Voronoi-cell areas require Quickhull.jl — run `using Quickhull` " *
        "(or supply `areas` explicitly to the `UnstructuredGrid` constructor).",
    ))
end

"""
    UnstructuredGrid(geometry, x, y, mask; k=6, radius=nothing, areas=nothing)

Build an `UnstructuredGrid` with REAL neighbor adjacency, via a k-d-tree nearest-neighbor query
(`NearestNeighbors.jl`; brute-force O(N²) doesn't scale) — either the `k` nearest neighbors per node
(default `k=6`), or every neighbor within a physical `radius` (pass `radius` to switch; mutually
exclusive with `k`). For `SphericalGeometry` the tree is built on the 3D Cartesian embedding of the
nodes (nearest-by-chord-distance is exactly nearest-by-great-circle-distance — exact, not an
approximation).

`areas`: supply per-node cell areas explicitly (common for a real dataset that ships its own), or
leave `nothing` to auto-compute exact Voronoi-cell areas from a Delaunay/convex-hull tessellation
(`DelaunayTriangulation.jl` for Cartesian, `Quickhull.jl` for spherical — see [`_voronoi_areas`](@ref)).
"""
function UnstructuredGrid(
    geometry::Geometry.AbstractGeometry{T}, x::AbstractVector, y::AbstractVector, mask::AbstractVector{Bool};
    k::Integer = 6, radius::Union{Nothing,Real} = nothing, areas::Union{Nothing,AbstractVector} = nothing,
) where {T<:AbstractFloat}
    x_T = _to_axis(T, x)
    y_T = _to_axis(T, y)
    nbrs, ptr = _build_kdtree_neighbors(geometry, x_T, y_T; k = k, radius = radius)
    areas_T = areas === nothing ? _voronoi_areas(geometry, x_T, y_T) : _to_axis(T, areas)
    return UnstructuredGrid(geometry, x_T, y_T, areas_T, mask, nbrs, ptr)
end

"""
    neighbors(grid::UnstructuredGrid, idx::Integer) -> AbstractVector{Int}

Neighbor node indices of node `idx`, as a zero-copy view into the CSR-flattened adjacency storage.
"""
@inline function neighbors(grid::UnstructuredGrid, idx::Integer)
    lo = grid.neighbor_ptr[idx]
    hi = grid.neighbor_ptr[idx+1] - 1
    return view(grid.neighbor_nbrs, lo:hi)
end

size_tuple(grid::UnstructuredGrid) = (length(grid.mask),)

@inline function _raw_coords(grid::UnstructuredGrid, idx::Integer)
    return (grid.x[idx], grid.y[idx])
end

@inline area(grid::UnstructuredGrid, idx::Integer) = grid.areas[idx]
@inline isactive(grid::UnstructuredGrid, idx::Integer) = grid.mask[idx]

end # module
