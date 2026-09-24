# ---------------------------------------------------------------------------
# Structured Grid
# ---------------------------------------------------------------------------

"""
    AxisSummary{T}

One direction's reductions: the values [`origin`](@ref), [`bounds`](@ref), [`extent`](@ref),
[`minimum_spacing`](@ref) and [`maximum_spacing`](@ref) report.

A stretched axis answers each of these by scanning, and a distance query reads them through
[`Connectivity.MetricTopology`](@ref FlowGeometries.Connectivity.MetricTopology), which is the default
`topology` on every per-cell entry point. Holding them keeps that construction `O(1)` and keeps a query
off the coordinates.

`isbits`, five numbers per direction, filled once by [`_axis_summary`](@ref): one pass per axis, the
`O(∑ Nᵈ)` the measure factors already cost. `first` is kept apart from `lo` because a descending axis
starts at its largest value.
"""
struct AxisSummary{T<:AbstractFloat}
    first::T
    lo::T
    hi::T
    min_gap::T
    max_gap::T
end

"""
    StructuredGrid{T, G, N, S, TP, C, AT, BT}

Rectilinear `N`-dimensional grid, for any `N`: one coordinate vector per direction (`coordinates`),
an N-D cell `measure` (length in 1-D, area in 2-D, volume in 3-D, the N-D measure in general), an N-D
active `mask`, per-direction `topology`, and the wrap `period` of each periodic direction.

# Type parameters
- `T`: coordinate float type. `G<:AbstractGeometry{T}` is tied to it, so a mismatched-eltype geometry
  raises a type error. `T` therefore precedes `G`: Julia binds type parameters left to right, and
  `G<:AbstractGeometry{T}` requires `T` already bound. [`CurvilinearGrid`](@ref) and
  [`UnstructuredGrid`](@ref) carry the same convention.
- `N`: number of coordinate directions.
- `S`: the `SphericalSampling` recipe the axes came from, or `Nothing` for axes given directly.
  A zero-size singleton, so it costs nothing to carry and lets quadrature exactness, the matching
  latitude weights and a rebuild at another resolution dispatch on which node set this is — none of
  which the coordinates alone determine. See [`sampling`](@ref).
- `TP`: the per-direction [`AbstractTopology`](@ref) — singletons, so no storage, readable from the type.
- `C`: a heterogeneous `NTuple{N,AbstractVector{T}}`. Each axis independently keeps whatever concrete
  `AbstractVector{T}` type it was constructed with — an [`Axes.UniformAxis`](@ref), a plain `Vector`, a
  device array — so one axis can be uniform while another is stretched. A `UniformAxis`'s type is a
  compile-time proof of constant spacing that [`spacing_trait`](@ref) and [`spacing`](@ref) read
  without touching a coordinate, and a common vector type across the axes erases it.
- `AT`, `BT`: array types of the derived `measure` and of the active `mask`.
"""
struct StructuredGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T},
    N,
    S,
    TP<:NTuple{N,AbstractTopology},
    C<:NTuple{N,AbstractVector{T}},
    AT<:AbstractArray{T,N},
    BT<:AbstractArray{Bool,N},
} <: AbstractStructuredGrid{G, T}
    geometry::G
    coordinates::C            # one coordinate vector per direction
    measure::AT               # N-D cell measure (length/area/volume by dimension)
    mask::BT                  # N-D active mask (true = active/included)
    topology::TP              # per-direction closure (singletons: no storage)
    period::NTuple{N,T}       # wrap length per direction; meaningless where Bounded
    sampling::S               # the node-set recipe, or `nothing`; zero-size where it is one
    stats::NTuple{N,AxisSummary{T}}   # per-direction reductions; see `AxisSummary`
end

"""
    sampling(grid) -> AbstractSphericalSampling | Nothing

The node-set recipe `grid`'s axes were built from, or `nothing` where they were given directly.

Coordinates do not determine it: Gauss–Legendre and an arbitrary lat–lon grid are the same numbers to
within round-off, and only the recipe says which quadrature is exact on them. Keeping it means a grid
can be asked for its matching weights, or rebuilt at another resolution, after construction.
"""
@inline sampling(grid::AbstractGrid) = nothing
@inline sampling(grid::StructuredGrid) = getfield(grid, :sampling)

"""
    rebuild(grid, fields::NamedTuple) -> grid

`grid` with the named fields replaced, its type parameters re-derived from what the new fields are.

The hook a storage change goes through — moving a grid's arrays to a device, rewrapping them — so one
generic method serves every layout and no reconstruction elsewhere spells out a parameter list of its
own.

`fields` need only name what changes; everything else is carried over.
"""
@inline function rebuild(grid::G, fields::NamedTuple) where {G<:AbstractGrid}
    unknown = Base.setdiff(keys(fields), fieldnames(G))
    isempty(unknown) || throw(ArgumentError(
        "$(nameof(G)) has no field $(join(unknown, ", ")); it has $(join(fieldnames(G), ", "))",
    ))
    return _from_fields(G, map(n -> get(fields, n, getfield(grid, n)), fieldnames(G))...)
end

# Every type parameter is determined by the field types, so these re-derive the whole list from the
# values a rebuild is given; a storage change is what alters them. One line per layout, beside the
# struct it mirrors
#
# The leading argument is the grid's own type, and names the layout: two layouts can hold the same field
# types, a geometry and a resolution parameter and a mask.
# `stats` travels as a value. `rebuild` serves a storage change — moving arrays to a device,
# rewrapping them — which leaves every coordinate unchanged
@inline _from_fields(
    ::Type{<:StructuredGrid},
    geometry::G, coordinates::C, measure::AT, mask::BT, topology::TP, period::NTuple{N,T},
    sampling::S, stats::NTuple{N,AxisSummary{T}},
) where {T,G<:Geometry.AbstractGeometry{T},N,S,TP,C,AT,BT} =
    StructuredGrid{T,G,N,S,TP,C,AT,BT}(geometry, coordinates, measure, mask, topology, period,
                                       sampling, stats)

@inline topology(grid::StructuredGrid) = getfield(grid, :topology)
@inline period(grid::StructuredGrid, d::Integer) =
    @inbounds getfield(grid, :period)[_checked_direction(getfield(grid, :period), d)]

# The reductions each direction was summarised with at construction. The tuple is homogeneous, so a
# runtime `d` indexes it type-stably, and each accessor below is one read of a number.
@inline _axis_stat(grid::StructuredGrid, d::Integer) =
    @inbounds getfield(grid, :stats)[_checked_direction(getfield(grid, :stats), d)]

@inline origin(grid::StructuredGrid, d::Integer) = _axis_stat(grid, d).first
@inline bounds(grid::StructuredGrid, d::Integer) = (s = _axis_stat(grid, d); (s.lo, s.hi))
@inline minimum_spacing(grid::StructuredGrid, d::Integer) = _axis_stat(grid, d).min_gap
@inline maximum_spacing(grid::StructuredGrid, d::Integer) = _axis_stat(grid, d).max_gap

# ---------------------------------------------------------------------------
# Point accessors: NamedTuple default; coords! / coords(S, ...) for other storage
# ---------------------------------------------------------------------------

# Tail-split, not `for d in 1:N`: the coordinate tuple is heterogeneous when the directions have
# different axis types, and indexing it with a loop variable is a dynamic lookup.
@inline _checkaxes(::Tuple{}, ::Tuple{}) = nothing
@inline function _checkaxes(c::Tuple, I::Tuple)
    checkbounds(first(c), first(I))
    return _checkaxes(Base.tail(c), Base.tail(I))
end

"""
    _raw_coords(grid, I...) -> NTuple

Positional coordinate values at indices `I`. Internal; prefer [`coords`](@ref).
"""
@inline function _raw_coords(grid::AbstractStructuredGrid{G,T}, I::Vararg{Integer,N}) where {G,T,N}
    c = coordinates(grid)
    @boundscheck _checkaxes(c, I)
    return ntuple(d -> @inbounds(c[d][I[d]]), Val(N))
end

# CurvilinearGrid / UnstructuredGrid _raw_coords are defined after those types exist.

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
    coords(::Type{S}, grid, I...) -> S

Construct the point as type `S` — `Tuple`, `NTuple{N,T}`, `NamedTuple`, `Vector{T}`, or
`SVector{N,T}`/`MVector{N,T}` when StaticArrays is loaded. See
[`Geometry.build_point`](@ref) for how `S` is assembled.
"""
@inline function coords(::Type{S}, grid::AbstractGrid, I::Vararg{Integer}) where {S}
    vals = _raw_coords(grid, I...)
    names = Geometry.point_names(grid_geometry(grid), Val(length(vals)))
    return Geometry.build_point(S, names, vals)
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

# Auto-detect axis-1 periodicity: on a spherical grid axis 1 is longitude, and a longitude axis
# that closes the full 2π circle (to within one cell) is periodic; a regional span is bounded.
# Cartesian axes are opt-in only, having no analogous auto-detectable physical closure.
#
# Compares magnitudes, so the answer is the same in either storage order.
_auto_periodic_x(::Geometry.AbstractCartesianGeometry, x::AbstractVector) = false
_auto_periodic_x(
    ::Union{Geometry.AbstractSphericalGeometry,Geometry.AbstractEllipsoidalGeometry},
    x::AbstractVector{T},
) where {T<:AbstractFloat} = _closes_circle(x)

function _closes_circle(x::AbstractVector{T}) where {T<:AbstractFloat}
    length(x) > 2 || return false
    dλ = abs(x[2] - x[1])
    iszero(dλ) && return false
    return isapprox(abs(x[end] - x[1]) + dλ, T(2π); atol = dλ)
end

"""
    _wrap_sign(x) -> ±1

`+1` for an ascending axis and `-1` for a descending one: the sign that turns a period magnitude into
the wrapped neighbour's offset in index order. Defined in [`Axes`](@ref) — it is a property of the axis
alone, and the discretization layer needs the same answer.
"""
@inline _wrap_sign(x::AbstractVector) = Axes.wrap_sign(x)

"""
    _to_axis(T, x) -> AbstractVector{T}

Adapt axis input `x` to element type `T`, keeping whatever is known about its spacing and never
collapsing a provably uniform axis into a plain `Vector`.

An axis already of element type `T` is **kept exactly as it is**, whatever type it is. A caller's own
range subtype, a `StepRangeLen` whose `TwicePrecision` internals they want, a `BigFloat`-backed range —
all pass through untouched. Nothing needs converting to get the uniform fast paths: those dispatch on
[`Axes.spacing_trait`](@ref), which is `UniformSpacing()` for every `AbstractRange`, so the caller's own
range takes them.

Conversion happens only where the element type must change, and there an arbitrary range subtype cannot
generically be rebuilt at a new eltype. That case becomes an [`Axes.UniformAxis`](@ref)`{T}`, which is
also how a `Float32` axis stops carrying `Float64` internals. Call [`Axes.uniform_axis`](@ref) to
convert at the call site.

Four methods, ordered so nothing is ambiguous (a `StepRangeLen{T}` is both an `AbstractRange` and an
`AbstractVector{T}`, so the parameterized range form is needed to break that tie):
- `AbstractRange{T}` / `AbstractVector{T}`: passthrough, zero cost, type preserved.
- `AbstractRange` (wrong eltype): rebuilt as `UniformAxis{T}`, still uniform and now `isbits`.
- `AbstractVector` (wrong eltype): copied with `similar`, so a device-resident array stays in its own
  storage.
"""
_to_axis(::Type{T}, x::AbstractRange{T}) where {T<:AbstractFloat} = x
_to_axis(::Type{T}, x::AbstractRange) where {T<:AbstractFloat} = Axes.uniform_axis(T, x)
_to_axis(::Type{T}, x::AbstractVector{T}) where {T<:AbstractFloat} = x
_to_axis(::Type{T}, x::AbstractVector) where {T<:AbstractFloat} = copyto!(similar(x, T), x)

"""
    _min_gap(x) -> minimum consecutive |gap|, or Inf if length(x) < 2

Smallest spacing found anywhere on axis `x`. It bounds the search radius for a genuinely nonuniform
axis: a distance check still gates what is included, so the smallest gap anywhere can only widen the
search window, and no in-range cell is missed.
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
    _max_gap(x) -> maximum consecutive |gap|, or 0 if length(x) < 2

Largest spacing anywhere on axis `x`, the counterpart of [`_min_gap`](@ref). Together they bound how
far an index window must reach to cover a given physical distance.
"""
function _max_gap(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    n < 2 && return zero(T)
    @inbounds m = abs(x[2] - x[1])
    @inbounds for i in 3:n
        g = abs(x[i] - x[i-1])
        g > m && (m = g)
    end
    return m
end

# A uniform axis has one gap, known from its type — no scan.
_min_gap(x::AbstractRange{T}) where {T<:AbstractFloat} =
    length(x) < 2 ? T(Inf) : abs(T(step(x)))
_max_gap(x::AbstractRange{T}) where {T<:AbstractFloat} =
    length(x) < 2 ? zero(T) : abs(T(step(x)))

"""
    _axis_summary(x) -> AxisSummary

Reduce axis `x` in a single pass. A uniform axis answers from `first`, `step` and `length`, reading no
element.
"""
function _axis_summary(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && return AxisSummary{T}(T(NaN), T(Inf), T(-Inf), T(Inf), zero(T))
    @inbounds f = x[1]
    n == 1 && return AxisSummary{T}(f, f, f, T(Inf), zero(T))
    @inbounds l = x[n]
    lo, hi = f ≤ l ? (f, l) : (l, f)
    @inbounds mn = mx = abs(x[2] - x[1])
    @inbounds for i in 3:n
        g = abs(x[i] - x[i - 1])
        g < mn && (mn = g)
        g > mx && (mx = g)
    end
    return AxisSummary{T}(f, lo, hi, mn, mx)
end

function _axis_summary(x::AbstractRange{T}) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && return AxisSummary{T}(T(NaN), T(Inf), T(-Inf), T(Inf), zero(T))
    f, l = first(x), last(x)
    lo, hi = f ≤ l ? (f, l) : (l, f)
    g = n < 2 ? (T(Inf), zero(T)) : (abs(T(step(x))), abs(T(step(x))))
    return AxisSummary{T}(f, lo, hi, g[1], g[2])
end

# A period is a length, so this is a magnitude and does not change sign with the axis's storage
# order. On a uniform axis it is exactly `n·|Δ|`, the axis's own closure.
@inline _cartesian_period(x::AbstractVector{T}) where {T} =
    length(x) < 2 ? one(T) : @inbounds(abs(x[end] - x[1]) + abs(x[2] - x[1]))

@inline _cartesian_period(x::AbstractRange{T}) where {T<:AbstractFloat} =
    length(x) < 2 ? one(T) : T(length(x)) * abs(T(step(x)))

"""
    _curvilinear_topology(geometry, x, topo) -> NTuple{2,AbstractTopology}

Per-direction closure. Absent a caller's choice, direction 1 is auto-detected from the first row of
centres read as a longitude-like axis, as [`StructuredGrid`](@ref) does.
"""
function _curvilinear_topology(
    geometry::Geometry.AbstractGeometry, x::AbstractArray{<:Any,N}, topo,
) where {N}
    topo === nothing || return _as_topology(Val(N), topo)
    # Auto-detection reads direction 1 along its own line through the first cell of every other.
    line = @view x[:, ntuple(_ -> 1, Val(N - 1))...]
    wraps = size(x, 1) ≥ 3 && _auto_periodic_x(geometry, line)
    return ntuple(d -> d == 1 && wraps ? Periodic() : Bounded(), Val(N))
end

"""
    _curvilinear_periods(geometry, centers, topology, period) -> NTuple{N,T}

Wrap length per periodic direction, zero where bounded. Spherical longitude closes at `2π`; a
Cartesian direction's comes from that direction's own line of centres.
"""
function _curvilinear_periods(
    geometry::Geometry.AbstractGeometry{T}, centers::NTuple{N,AbstractArray{T,N}},
    tp::NTuple{N,AbstractTopology}, period,
) where {N, T<:AbstractFloat}
    given = _period_tuple(Val(N), T, period)
    return ntuple(Val(N)) do d
        if !_is_periodic(tp[d])
            zero(T)
        elseif given[d] !== nothing
            T(given[d])
        elseif geometry isa Geometry.AbstractSphericalGeometry ||
               geometry isa Geometry.AbstractEllipsoidalGeometry
            d == 1 ? T(2π) : throw(ArgumentError(
                "direction $d of a spherical curvilinear grid has no intrinsic period; pass `period`",
            ))
        else
            # Direction `d`'s own line: vary index `d`, hold every other at 1.
            _cartesian_period(@view centers[d][ntuple(k -> k == d ? Colon() : 1, Val(N))...])
        end
    end
end

"""
    _half_gaps(x, period) -> (below, above)

Half the distance from each centre of an axis of `N ≥ 2` centres to its two faces, `below` toward
smaller coordinates and `above` toward larger. Faces sit midway between neighbours; at a bounded end the
outer face is as far out as the inner one, and at a periodic end it is half the seam gap out, so
`below[i] + above[i]` is `Discretization.cell_width(x, i, period)`. Each entry is half a difference of two
stored coordinates, so no face position is rounded.
"""
function _half_gaps(x::AbstractVector{T}, period::Union{Nothing,Real}) where {T<:AbstractFloat}
    n = length(x)
    d = abs.(view(x, 2:n) .- view(x, 1:(n - 1))) ./ T(2)
    prev = similar(x, T, n)      # toward index i - 1
    next = similar(x, T, n)      # toward index i + 1
    @views prev[2:n] .= d
    @views next[1:(n - 1)] .= d
    if period === nothing
        @views prev[1:1] .= d[1:1]
        @views next[n:n] .= d[(n - 1):(n - 1)]
    else
        g = abs(T(period) - abs(@inbounds(x[n]) - @inbounds(x[1]))) / T(2)
        @views prev[1:1] .= g
        @views next[n:n] .= g
    end
    return Axes.wrap_sign(x) > 0 ? (prev, next) : (next, prev)
end

# `π/2 - T(π)/2`, the rounding error of `T(π)/2`, so `(T(π)/2 - |φ|) + _half_pi_lo(T)` is a colatitude
# carrying no error from `π/2` itself.
@inline _half_pi_lo(::Type{Float64}) = 6.123233995736766e-17
@inline _half_pi_lo(::Type{Float32}) = -4.371139f-8
@inline _half_pi_lo(::Type{T}) where {T<:AbstractFloat} = T(big(π) / 2 - big(T(π)) / 2)

"""
    _polar_cell(φ, below, above) -> (lo, width, north)

The latitude cell centred at `φ`, in colatitude from the pole nearer `φ` (`north` says which): the span
`[lo, lo + width]`, clamped to `[0, π]`. The centre's colatitude carries `π/2` as `T(π)/2` plus its
rounding error, so near a pole it is as accurate as `φ`, and the span is built from the half-gaps
without forming a face.
"""
@inline function _polar_cell(φ::T, below::T, above::T) where {T<:AbstractFloat}
    north = !signbit(φ)
    c = (T(π) / 2 - abs(φ)) + _half_pi_lo(T)
    toward, away = north ? (above, below) : (below, above)
    return (max(c - toward, zero(T)), min(toward, c) + min(away, T(π) - c), north)
end

# `|sin φ₊ - sin φ₋|` for the latitude cell, the area of its band on the unit sphere per radian of
# longitude, as `2 sin(t̄) sin(w/2)` about the mid-colatitude `t̄` and width `w`: a narrow band and a
# polar one lose no digits to cancellation.
@inline function _band_sine(φ::T, below::T, above::T) where {T<:AbstractFloat}
    lo, w, _ = _polar_cell(φ, below, above)
    return 2 * sin(lo + w / 2) * sin(w / 2)
end

@inline _band_arc(φ::T, below::T, above::T) where {T<:AbstractFloat} = _polar_cell(φ, below, above)[2]

# `(r₊³ - r₋³)/3`, the radial integral of `r²` over the cell, the lower face clamped at the origin.
@inline function _shell_cube(r::T, below::T, above::T) where {T<:AbstractFloat}
    lo = max(r - below, zero(T))
    w = r ≥ below ? below + above : max(r + above, zero(T))
    hi = lo + w
    return w * (lo * lo + lo * hi + hi * hi) / T(3)
end

# `f(centre, below, above)` over every cell of an axis of `n ≥ 2` centres.
@inline function _per_cell(f::F, x::AbstractVector, period) where {F}
    below, above = _half_gaps(x, period)
    return f.(x, below, above)
end

"""
    _gl8(f, φ, below, above, nodes) -> T

`∫ f(φ′, cos φ′) dφ′` over the latitude cell of [`_polar_cell`](@ref), by 8-point Gauss–Legendre in
colatitude, `cos φ′` passed as the sine of the colatitude. The spheroid's latitude integrands are
analytic within a distance of about 3 of the real axis, so the rule is exact to round-off for any cell
on the sphere.
"""
@inline function _gl8(
    f::F, φ::T, below::T, above::T, nodes::Tuple{NTuple{8,T},NTuple{8,T}},
) where {F,T<:AbstractFloat}
    lo, w, north = _polar_cell(φ, below, above)
    x, wt = nodes
    h = w / 2
    s = zero(T)
    @inbounds for k in 1:8
        t = (lo + h) + h * x[k]
        lat = (T(π) / 2 - t) + _half_pi_lo(T)
        s += wt[k] * f(north ? lat : -lat, sin(t))
    end
    return h * s
end

function _gl8_nodes(::Type{T}) where {T<:AbstractFloat}
    r = SphericalSampling._gauss_legendre_μ(T, 8)
    return (ntuple(k -> @inbounds(r.μ[k]), Val(8)), ntuple(k -> @inbounds(r.w[k]), Val(8)))
end

"""
    _measure_factors(geometry, axes, periods) -> NTuple{N,AbstractVector}

Per-axis factors whose outer product is the cell measure: `measure[I...] == prod(w[d][I[d]])`.

A cell spans the faces of [`_half_gaps`](@ref), latitude faces clamped to the poles, and its measure
is the exact integral of the metric over that span: Cartesian `Δx·Δy·Δz`; spherical
`Δλ · R²(sin φ₊ - sin φ₋)`, or `Δλ · (sin φ₊ - sin φ₋) · (r₊³ - r₋³)/3` with a radius direction; on a
spheroid `Δλ · ∫ M(φ)N(φ)cos φ dφ`. Building the measure as an outer product of these factors keeps the
result in whatever array type the axes use, and exposes the separability to a caller that can exploit
it.

A degenerate (length-1) angular axis drops its differential, so a zonal transect measures arc length
`R·cosφ·Δλ` along its circle of latitude and a meridional one measures `R·Δφ`.
"""
function _measure_factors(
    ::G, axes::NTuple{N,AbstractVector{T}}, periods::NTuple{N,Union{Nothing,Real}},
) where {N, T<:AbstractFloat, G<:Geometry.AbstractCartesianGeometry{T}}
    return ntuple(d -> Discretization.cell_widths(axes[d], periods[d]), Val(N))
end

# A lone longitude axis measures arc length; with no latitude direction there is no `cosφ` factor.
function _measure_factors(
    geometry::G, axes::NTuple{1,AbstractVector{T}}, periods::NTuple{1,Union{Nothing,Real}},
) where {T<:AbstractFloat, G<:Geometry.AbstractSphericalGeometry{T}}
    return (Geometry.radius(geometry) .* Discretization.cell_widths(axes[1], periods[1]),)
end

function _measure_factors(
    geometry::G, axes::NTuple{2,AbstractVector{T}}, periods::NTuple{2,Union{Nothing,Real}},
) where {T<:AbstractFloat, G<:Geometry.AbstractSphericalGeometry{T}}
    λ, φ = axes
    R = Geometry.radius(geometry)
    Nλ, Nφ = length(λ), length(φ)
    if Nλ == 1 && Nφ == 1
        return (Axes.ConstantVector(one(T), 1), Axes.ConstantVector(one(T), 1))  # a point has no extent
    elseif Nφ == 1
        return (Discretization.cell_widths(λ, periods[1]),
                Axes.ConstantVector(R * cos(@inbounds φ[1]), 1))
    elseif Nλ == 1
        return (Axes.ConstantVector(one(T), 1), R .* _per_cell(_band_arc, φ, periods[2]))
    end
    return (Discretization.cell_widths(λ, periods[1]), (R^2) .* _per_cell(_band_sine, φ, periods[2]))
end

# 3-D and above: `(Δλ)·(sin φ₊ - sin φ₋)·((r₊³ - r₋³)/3)`, further directions entering as plain widths.
function _measure_factors(
    geometry::G, axes::NTuple{N,AbstractVector{T}}, periods::NTuple{N,Union{Nothing,Real}},
) where {N, T<:AbstractFloat, G<:Geometry.AbstractSphericalGeometry{T}}
    λ, φ, r = axes[1], axes[2], axes[3]
    return (
        Discretization.cell_widths(λ, periods[1]),
        length(φ) == 1 ? cos.(φ) : _per_cell(_band_sine, φ, periods[2]),
        _per_cell(_shell_cube, r, periods[3]),
        ntuple(d -> Discretization.cell_widths(axes[d + 3], periods[d + 3]), Val(N - 3))...,
    )
end

# An ellipsoid's surface element is `M(φ)·N(φ)cosφ·Δλ·Δφ`, separable exactly as the sphere's is — only
# the latitude factor differs. With no latitude direction the parallel is the equator, radius `a`.
function _measure_factors(
    geometry::G, axes::NTuple{1,AbstractVector{T}}, periods::NTuple{1,Union{Nothing,Real}},
) where {T<:AbstractFloat, G<:Geometry.AbstractEllipsoidalGeometry{T}}
    return (Geometry.semimajor_axis(geometry) .* Discretization.cell_widths(axes[1], periods[1]),)
end

function _measure_factors(
    geometry::G, axes::NTuple{2,AbstractVector{T}}, periods::NTuple{2,Union{Nothing,Real}},
) where {T<:AbstractFloat, G<:Geometry.AbstractEllipsoidalGeometry{T}}
    λ, φ = axes
    Nλ, Nφ = length(λ), length(φ)
    if Nλ == 1 && Nφ == 1
        return (Axes.ConstantVector(one(T), 1), Axes.ConstantVector(one(T), 1))
    elseif Nφ == 1
        φ1 = @inbounds φ[1]
        return (Discretization.cell_widths(λ, periods[1]),
                Axes.ConstantVector(Geometry.prime_vertical_radius(geometry, φ1) * cos(φ1), 1))
    end
    nodes = _gl8_nodes(T)
    # Closures, so each broadcast runs over the axis alone — a geometry is not a broadcastable scalar.
    arc(p, b, a) = _gl8((t, _) -> Geometry.meridional_radius(geometry, t), p, b, a, nodes)
    band(p, b, a) = _gl8(p, b, a, nodes) do t, c
        Geometry.meridional_radius(geometry, t) * Geometry.prime_vertical_radius(geometry, t) * c
    end
    Nλ == 1 && return (Axes.ConstantVector(one(T), 1), _per_cell(arc, φ, periods[2]))
    return (Discretization.cell_widths(λ, periods[1]), _per_cell(band, φ, periods[2]))
end

"""
    _quadrature_weights(geometry, sampling, axes) -> AbstractVector | Nothing

The sampling's latitude quadrature weights when they are the measure's latitude factor: a 2-D spherical
grid on a sampling that has weights, both directions resolved. `nothing` otherwise, and the measure is
the geometric one.
"""
_quadrature_weights(_, _, _) = nothing

const _WeightedSampling = Union{
    SphericalSampling.AbstractGaussLegendreSampling, SphericalSampling.AbstractDriscollHealySampling,
    SphericalSampling.AbstractClenshawCurtisSampling,
}

function _quadrature_weights(
    ::Geometry.AbstractSphericalGeometry{T}, s::_WeightedSampling, ax::NTuple{2,AbstractVector{T}},
) where {T<:AbstractFloat}
    (length(ax[1]) ≥ 2 && length(ax[2]) ≥ 2) || return nothing
    return SphericalSampling.latitude_weights(T, s, length(ax[2]))
end

"""
    _cell_measure(geometry, axes, periods, weights)

The grid's stored measure: a [`SeparableMeasure`](@ref) wherever the metric factors per axis, and a
[`SlabMeasure`](@ref) where one pair of directions is coupled and the rest are not. Given latitude
quadrature `weights`, the latitude factor is `R²·wⱼ`, so `Σ measure·f` is the sampling's quadrature.
"""
_cell_measure(geometry, ax, per, ::Nothing) = SeparableMeasure(_measure_factors(geometry, ax, per))

function _cell_measure(
    geometry::Geometry.AbstractSphericalGeometry{T}, ax::NTuple{2,AbstractVector{T}},
    per::NTuple{2,Union{Nothing,Real}}, w::AbstractVector{T},
) where {T<:AbstractFloat}
    λ, φ = ax
    length(w) == length(φ) || throw(DimensionMismatch(
        "$(length(w)) latitude weights for a latitude axis of $(length(φ))",
    ))
    wφ = (Geometry.radius(geometry)^2) .* copyto!(similar(φ, T, length(φ)), w)
    return SeparableMeasure((Discretization.cell_widths(λ, per[1]), wφ))
end

# Geodetic `(λ, φ, h)`: the volume element `(N(φ)+h)cosφ·(M(φ)+h)` offsets each curvature radius by the
# height, so φ and h are coupled and no product of per-axis factors reproduces it. Longitude enters
# none of it, so φ and h alone are stored together — see [`SlabMeasure`](@ref). Further directions
# still enter as plain widths. The height integral is exact and the latitude one [`_gl8`](@ref).
function _cell_measure(
    geometry::G, ax::NTuple{N,AbstractVector{T}}, per::NTuple{N,Union{Nothing,Real}}, ::Nothing,
) where {N, T<:AbstractFloat, G<:Geometry.AbstractEllipsoidalGeometry{T}}
    N ≥ 3 || return SeparableMeasure(_measure_factors(geometry, ax, per))
    λ, φ, h = ax[1], ax[2], ax[3]
    wλ = Discretization.cell_widths(λ, per[1])
    rest = ntuple(d -> Discretization.cell_widths(ax[d + 3], per[d + 3]), Val(N - 3))
    nodes = _gl8_nodes(T)
    bφ, aφ = _half_gaps(φ, per[2])
    bh, ah = _half_gaps(h, per[3])
    Nh = length(h)
    # ∫(N+h)(M+h) dh = NM·Δh + (N+M)·Δ(h²)/2 + Δ(h³)/3, each difference in a form that does not cancel.
    function cell(p, pb, pa, hk, hb, ha)
        dh = hb + ha
        lo, hi = hk - hb, hk + ha
        h2 = dh * (lo + hi) / T(2)
        h3 = dh * (lo * lo + lo * hi + hi * hi) / T(3)
        return _gl8(p, pb, pa, nodes) do t, c
            Nt = Geometry.prime_vertical_radius(geometry, t)
            Mt = Geometry.meridional_radius(geometry, t)
            c * (Nt * Mt * dh + (Nt + Mt) * h2 + h3)
        end
    end
    slab = cell.(φ, bφ, aφ, reshape(h, 1, Nh), reshape(bh, 1, Nh), reshape(ah, 1, Nh))
    return SlabMeasure(wλ, slab, rest)
end

"""
    StructuredGrid(geometry, axes...; mask, topology, period, periodic, sampling)
    StructuredGrid(geometry, axes..., mask; topology, period, periodic, sampling)

Build a rectilinear grid in **any** number of dimensions from one coordinate vector per direction,
pre-computing the separable cell measure from the geometry.

Each axis may independently be uniform or stretched, and keeps whichever it is in its own type — see
[`isuniform`](@ref). Axes are adapted to the geometry's float type `T` by [`_to_axis`](@ref), which
preserves uniformity and keeps a device-resident axis on its device.

For a `SphericalGeometry` the directions are `(λ, φ, r, …)`: longitude, geographic latitude, and — in
3-D and above — the absolute radius from the origin. (A
[`Geometry.SpheroidGeometry`](@ref FlowGeometries.Geometry.SpheroidGeometry)'s third direction is a
height above the surface instead.) A cell's measure is the exact integral of the metric between its
faces, latitude faces clamped to the poles: `R·Δλ`, `R²·Δλ·(sin φ₊ - sin φ₋)` and
`Δλ·(sin φ₊ - sin φ₋)·(r₊³ - r₋³)/3`, `Δλ·∫M(φ)N(φ)cos φ dφ` on a spheroid, with further directions
entering as plain widths. A `CartesianGeometry` measure is the product of the per-direction widths.

# Keywords
- `mask`: an `N`-D `Bool` array of active cells. Omit it (or pass `nothing`) for an all-active grid,
  which stores only its size — see [`AllActive`](@ref). It may also be given positionally, after the
  axes.
- `topology`: per-direction closure. Accepts [`Periodic`](@ref)/[`Bounded`](@ref) instances, a tuple
  of them, a `Bool`, or a tuple of `Bool`s; a single value or a short tuple applies to the leading
  directions and the rest are `Bounded`. When omitted, direction 1 is auto-detected — on a spherical
  grid a longitude axis spanning the full circle is `Periodic` and a regional span is not, in either
  storage order — and every other direction is `Bounded`.
- `period`: the wrap length of each periodic direction. Omit it and the axis's own closure is used:
  `2π` for spherical longitude, and `extent + one spacing` for a Cartesian direction, which is exact
  for a uniform axis (`n·|Δ|`). A **nonuniform** periodic Cartesian direction has no closure to infer,
  its seam gap being undetermined by its samples, so `period` is required there.
- `periodic`: the `topology` keyword under another name, for a `Bool` or a tuple of them.
- `sampling`: the [`SphericalSampling.AbstractSphericalSampling`](@ref) the axes came from, kept on
  the grid (see [`sampling`](@ref)). On a 2-D spherical grid a sampling with latitude quadrature
  weights sets the latitude factor of the measure to `R²·wⱼ`, so `sum(f .* measure(grid))` is its
  quadrature.
"""
function StructuredGrid(
    geometry::Geometry.AbstractGeometry{T},
    args...;
    mask = nothing,
    topology = nothing,
    period = nothing,
    periodic = nothing,
    sampling = nothing,
) where {T<:AbstractFloat}
    axes_in, mask_pos = _split_axes_mask(args)
    mask === nothing || mask_pos === nothing ||
        throw(ArgumentError("mask given both positionally and as a keyword"))
    return _structured_grid(
        geometry, axes_in, mask_pos === nothing ? mask : mask_pos,
        topology === nothing ? periodic : topology, period, sampling,
    )
end

# A trailing `Bool` array is unambiguously the mask; an axis is never a `Bool` vector.
_split_axes_mask(args::Tuple{}) = ((), nothing)
function _split_axes_mask(args::Tuple)
    last_arg = args[end]
    if last_arg isa AbstractArray{Bool}
        return Base.front(args), last_arg
    end
    return args, nothing
end

# `weights` are the sampling's latitude quadrature weights when the caller already holds them from the
# solve that placed the axes.
function _structured_grid(
    geometry::G, axes_in::NTuple{N,AbstractVector}, mask, topo, period, sampling,
    weights = nothing,
) where {N, T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    N ≥ 1 || throw(ArgumentError("a StructuredGrid needs at least one axis"))
    ax = ntuple(d -> _to_axis(T, axes_in[d]), Val(N))
    dims = ntuple(d -> length(ax[d]), Val(N))

    tp = topo === nothing ?
        ntuple(d -> (d == 1 && _auto_periodic_x(geometry, ax[1])) ? Periodic() : Bounded(), Val(N)) :
        _as_topology(Val(N), topo)

    per = _wrap_periods(geometry, ax, tp, period)
    # A bounded direction passes `nothing`, selecting the width kernel's non-wrapping branch.
    period_args = ntuple(d -> _is_periodic(tp[d]) ? per[d] : nothing, Val(N))

    if G <: Geometry.AbstractSphericalGeometry{T} && N ≥ 3
        dims[3] > 1 || throw(ArgumentError(
            "a spherical StructuredGrid with a radius direction needs at least 2 radius levels " *
            "(got $(dims[3])); a single level is the 2-D surface case — pass just (λ, φ).",
        ))
    end

    m = mask === nothing ? AllActive(dims) : mask
    size(m) == dims || throw(DimensionMismatch(
        "mask size $(size(m)) does not match the axis lengths $dims",
    ))

    w = weights === nothing ? _quadrature_weights(geometry, sampling, ax) : weights
    measure = _cell_measure(geometry, ax, period_args, w)
    # One pass per axis, alongside the pass the measure factors already make.
    stats = ntuple(d -> _axis_summary(ax[d]), Val(N))
    return StructuredGrid{
        T, G, N, typeof(sampling), typeof(tp), typeof(ax), typeof(measure), typeof(m),
    }(geometry, ax, measure, m, tp, per, sampling, stats)
end

# Wrap length per direction; zero where bounded, so it never reads as usable.
function _wrap_periods(
    geometry::Geometry.AbstractGeometry{T}, ax::NTuple{N,AbstractVector{T}},
    tp::NTuple{N,AbstractTopology}, period,
) where {N, T<:AbstractFloat}
    given = _period_tuple(Val(N), T, period)
    return ntuple(Val(N)) do d
        if !_is_periodic(tp[d])
            zero(T)
        elseif given[d] !== nothing
            T(given[d])
        else
            _inferred_period(geometry, ax[d], d)
        end
    end
end

_period_tuple(::Val{N}, ::Type{T}, ::Nothing) where {N,T} = ntuple(_ -> nothing, Val(N))
_period_tuple(::Val{N}, ::Type{T}, p::Real) where {N,T} =
    ntuple(d -> d == 1 ? p : nothing, Val(N))
_period_tuple(::Val{N}, ::Type{T}, p::Tuple) where {N,T} =
    ntuple(d -> d ≤ length(p) ? p[d] : nothing, Val(N))
_period_tuple(::Val{N}, ::Type{T}, p::AbstractVector) where {N,T} =
    ntuple(d -> d ≤ length(p) ? p[d] : nothing, Val(N))

# Longitude closes after exactly one turn whatever its samples look like.
_inferred_period(
    ::Union{Geometry.AbstractSphericalGeometry{T},Geometry.AbstractEllipsoidalGeometry{T}},
    ::AbstractVector, d::Int,
) where {T} =
    d == 1 ? T(2π) : throw(ArgumentError(
        "direction $d of a spherical grid has no intrinsic period; pass `period` explicitly",
    ))

# "extent + one spacing" is exact only when there IS one spacing; on a stretched axis the samples do
# not determine the seam gap, so it must be given.
function _inferred_period(
    ::Geometry.AbstractCartesianGeometry{T}, x::AbstractVector, d::Int,
) where {T}
    Axes.isuniform(x) || length(x) < 2 || throw(ArgumentError(
        "a periodic Cartesian direction with NONUNIFORM spacing has no period to infer (its seam " *
        "gap is not determined by its samples); pass `period` explicitly for direction $d",
    ))
    return _cartesian_period(x)
end
