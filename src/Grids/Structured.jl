# ---------------------------------------------------------------------------
# Structured Grid
# ---------------------------------------------------------------------------

"""
    AxisStats

Everything about one axis that does not depend on the query, reduced once when the grid is built:
gaps, span, and whether the axis type proves constant spacing. Each of these is otherwise an `O(N_d)`
scan — or a dynamic lookup, since the coordinate tuple is heterogeneous — for an answer that cannot
change over the grid's life.
"""
struct AxisStats{T<:AbstractFloat}
    min_gap::T
    max_gap::T
    step::T          # signed constant spacing where `uniform`, `NaN` otherwise
    first_value::T
    min_value::T
    max_value::T
    uniform::Bool    # from the axis TYPE, captured at construction
end

"""
    _axis_stats(x) -> AxisStats
"""
@inline function _axis_stats(x::AbstractVector{T}) where {T<:AbstractFloat}
    uni = Axes.isuniform(x)
    stp = uni ? T(Axes.spacing(x)) : T(NaN)
    lo, hi = _min_gap(x), _max_gap(x)
    isempty(x) && return AxisStats{T}(T(lo), T(hi), stp, T(NaN), T(Inf), T(-Inf), uni)
    mn, mx = extrema(x)
    return AxisStats{T}(T(lo), T(hi), stp, T(@inbounds first(x)), T(mn), T(mx), uni)
end

# A coordinate FIELD has no axis, so no gap and no spacing — but its span is still an invariant, and one
# that a search radius has to be bounded against. Reduced once, like the rest.
function _axis_stats(x::AbstractArray{T}) where {T<:AbstractFloat}
    isempty(x) && return AxisStats{T}(T(Inf), zero(T), T(NaN), T(NaN), T(Inf), T(-Inf), false)
    mn, mx = extrema(x)
    return AxisStats{T}(T(Inf), zero(T), T(NaN), T(@inbounds first(x)), T(mn), T(mx), false)
end

"""
    StructuredGrid{T, G, N, S, TP, C, AT, BT}

Rectilinear `N`-dimensional grid, for any `N`: one coordinate vector per direction (`coordinates`),
an N-D cell `measure` (length in 1-D, area in 2-D, volume in 3-D, the N-D measure in general), an N-D
active `mask`, per-direction `topology`, and the wrap `period` of each periodic direction.

# Type parameters
- `T`: coordinate float type. `G<:AbstractGeometry{T}` is tied to it, so a mismatched-eltype geometry
  is a type error rather than a silent promotion — hence `T` precedes `G`, Julia forbidding the
  forward reference a `{G,T}` order would need. [`CurvilinearGrid`](@ref) and
  [`UnstructuredGrid`](@ref) carry the same convention.
- `N`: number of coordinate directions.
- `S`: the `SphericalSampling` recipe the axes came from, or `Nothing` for axes given directly.
  A zero-size singleton, so it costs nothing to carry and lets quadrature exactness, the matching
  latitude weights and a rebuild at another resolution dispatch on which node set this is — none of
  which the coordinates alone determine. See [`sampling`](@ref).
- `TP`: the per-direction [`AbstractTopology`](@ref) — singletons, so no storage, readable from the type.
- `C`: a heterogeneous `NTuple{N,AbstractVector{T}}`. Each axis independently keeps whatever concrete
  `AbstractVector{T}` type it was constructed with (an [`Axes.UniformAxis`](@ref), a plain `Vector`, a
  device array, or any other subtype); there is deliberately no shared vector type forcing the axes to
  match. This matters beyond storage: a `UniformAxis`'s type is a compile-time proof of constant
  spacing that [`spacing_trait`](@ref) and [`spacing`](@ref) read without touching a coordinate, and
  forcing the axes into a common type would destroy it. One axis can be uniform while another is
  stretched.
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
    stats::NTuple{N,AxisStats{T}}   # reduced once; see AxisStats
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

The hook a storage change goes through — moving a grid's arrays to a device, rewrapping them — so that
one generic method serves every layout and adding a field to one does not silently leave a
reconstruction elsewhere spelling out a parameter list that no longer matches.

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
# values rather than carrying over the ones the old grid happened to have — which is the point, since a
# storage change is exactly what alters them. One line per layout, beside the struct it mirrors, so a
# field added to one cannot leave a stale parameter list somewhere that loads only with a weak
# dependency.
#
# The leading argument is the grid's own type, and it serves only to name the layout — two layouts can
# hold the same field types, a geometry and a resolution parameter and a mask. The parameter list below
# is built from the types of the field VALUES, since a field whose type changed is the reason to rebuild
# in the first place.
@inline _from_fields(
    ::Type{<:StructuredGrid},
    geometry::G, coordinates::C, measure::AT, mask::BT, topology::TP, period::NTuple{N,T},
    sampling::S, stats::NTuple{N,AxisStats{T}},
) where {T,G<:Geometry.AbstractGeometry{T},N,S,TP,C,AT,BT} =
    StructuredGrid{T,G,N,S,TP,C,AT,BT}(geometry, coordinates, measure, mask, topology, period,
                                       sampling, stats)

"""
    axis_stats(grid) -> NTuple{N,AxisStats}
    axis_stats(grid, d) -> AxisStats

The cached per-axis reductions. Homogeneous whatever the axis types are, so reading one with a runtime
direction index stays type-stable.
"""
@inline axis_stats(grid::StructuredGrid) = getfield(grid, :stats)
@inline axis_stats(grid::StructuredGrid, d::Integer) = @inbounds axis_stats(grid)[d]

# Reading the cache rather than the axis. Two things follow: the answer is `O(1)` where it was a scan,
# and it is type-stable for a runtime `d`, which indexing the heterogeneous coordinate tuple is not.
# The `AbstractGrid` fallbacks stay for subtypes and architectures without the field.
@inline minimum_spacing(grid::StructuredGrid, d::Integer) = axis_stats(grid, d).min_gap
@inline maximum_spacing(grid::StructuredGrid, d::Integer) = axis_stats(grid, d).max_gap
@inline isuniform(grid::StructuredGrid, d::Integer) = axis_stats(grid, d).uniform

@inline function spacing(grid::StructuredGrid, d::Integer)
    st = axis_stats(grid, d)
    st.uniform || throw(ArgumentError(
        "direction $d is not uniform; use `minimum_spacing`/`maximum_spacing`, `local_spacing` at " *
        "one index, or `cell_width`/`cell_widths`",
    ))
    return st.step
end

@inline topology(grid::StructuredGrid) = getfield(grid, :topology)
@inline period(grid::StructuredGrid, d::Integer) =
    @inbounds getfield(grid, :period)[_checked_direction(getfield(grid, :period), d)]

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
@inline function _raw_coords(grid::StructuredGrid{T, G,N}, I::Vararg{Integer,N}) where {G,T,N}
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
# that closes the full 2π circle (to within one cell) is periodic; a regional span is NOT.
# Cartesian axes are opt-in only (there's no analogous auto-detectable physical closure).
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
also how a `Float32` axis stops carrying `Float64` internals. To convert deliberately rather than by
side effect, call [`Axes.uniform_axis`](@ref).

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

# A period is a LENGTH, so this is a magnitude and does not change sign with the axis's storage
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
    _measure_factors(geometry, axes, periods) -> NTuple{N,AbstractVector}

Per-axis factors whose outer product is the cell measure: `measure[I...] == prod(w[d][I[d]])`.

Every rectilinear cell measure this package supports is separable in exactly this way — Cartesian
`Δx·Δy·Δz`, and spherical `R²cosφ·Δλ·Δφ` = `(Δλ) · (R²cosφ·Δφ)` or `r²cosφ·Δλ·Δφ·Δr` =
`(Δλ) · (cosφ·Δφ) · (r²·Δr)`. Building the measure as an outer product of these factors rather than
by a nested scalar loop keeps the result in whatever array type the axes use, and makes the
separability available to callers that can exploit it.

Degenerate (length-1) angular axes are handled by dropping the differential that no longer exists,
so a zonal transect measures arc length `R·cosφ·Δλ` along its circle of latitude and a meridional
one measures `R·Δφ` — not an area formula with a placeholder substituted in.
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
        return (Axes.ConstantVector(one(T), 1), R .* Discretization.cell_widths(φ, periods[2]))
    end
    return (Discretization.cell_widths(λ, periods[1]),
            (R^2) .* cos.(φ) .* Discretization.cell_widths(φ, periods[2]))
end

# 3-D and above: `(Δλ)·(cosφ·Δφ)·(r²·Δr)`, further directions entering as plain widths.
function _measure_factors(
    geometry::G, axes::NTuple{N,AbstractVector{T}}, periods::NTuple{N,Union{Nothing,Real}},
) where {N, T<:AbstractFloat, G<:Geometry.AbstractSphericalGeometry{T}}
    λ, φ, r = axes[1], axes[2], axes[3]
    return (
        Discretization.cell_widths(λ, periods[1]),
        cos.(φ) .* Discretization.cell_widths(φ, periods[2]),
        (r .^ 2) .* Discretization.cell_widths(r, periods[3]),
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
    # Closures, so each broadcast runs over `φ` alone — a geometry is not a broadcastable scalar.
    _area_factor(φj) = Geometry.meridional_radius(geometry, φj) *
                       Geometry.prime_vertical_radius(geometry, φj) * cos(φj)
    _mer_factor(φj) = Geometry.meridional_radius(geometry, φj)
    if Nλ == 1 && Nφ == 1
        return (Axes.ConstantVector(one(T), 1), Axes.ConstantVector(one(T), 1))
    elseif Nφ == 1
        φ1 = @inbounds φ[1]
        return (Discretization.cell_widths(λ, periods[1]),
                Axes.ConstantVector(Geometry.prime_vertical_radius(geometry, φ1) * cos(φ1), 1))
    elseif Nλ == 1
        return (Axes.ConstantVector(one(T), 1),
                _mer_factor.(φ) .* Discretization.cell_widths(φ, periods[2]))
    end
    return (Discretization.cell_widths(λ, periods[1]),
            _area_factor.(φ) .* Discretization.cell_widths(φ, periods[2]))
end

"""
    _cell_measure(geometry, axes, periods)

The grid's stored measure: a [`SeparableMeasure`](@ref) wherever the metric factors per axis, and a
[`SlabMeasure`](@ref) where one pair of directions is coupled and the rest are not.
"""
_cell_measure(geometry, ax, per) = SeparableMeasure(_measure_factors(geometry, ax, per))

# Geodetic `(λ, φ, h)`: the volume element `(N(φ)+h)cosφ·(M(φ)+h)` offsets BOTH curvature radii by the
# height, so φ and h are coupled and no product of per-axis factors reproduces it. Longitude enters
# none of it, so only that PAIR is stored together — see [`SlabMeasure`](@ref). Further directions
# still enter as plain widths.
function _cell_measure(
    geometry::G, ax::NTuple{N,AbstractVector{T}}, per::NTuple{N,Union{Nothing,Real}},
) where {N, T<:AbstractFloat, G<:Geometry.AbstractEllipsoidalGeometry{T}}
    N ≥ 3 || return SeparableMeasure(_measure_factors(geometry, ax, per))
    λ, φ, h = ax[1], ax[2], ax[3]
    wλ = Discretization.cell_widths(λ, per[1])
    wφ = Discretization.cell_widths(φ, per[2])
    wh = Discretization.cell_widths(h, per[3])
    rest = ntuple(d -> Discretization.cell_widths(ax[d + 3], per[d + 3]), Val(N - 3))
    slab = similar(wλ, T, (length(φ), length(h)))
    @inbounds for k in eachindex(h), j in eachindex(φ)
        φj, hk = φ[j], h[k]
        slab[j, k] = (Geometry.prime_vertical_radius(geometry, φj) + hk) * cos(φj) *
                     (Geometry.meridional_radius(geometry, φj) + hk) * wφ[j] * wh[k]
    end
    return SlabMeasure(wλ, slab, rest)
end

"""
    StructuredGrid(geometry, axes...; mask = nothing, topology = nothing, period = nothing)
    StructuredGrid(geometry, axes..., mask; topology = nothing, period = nothing)

Build a rectilinear grid in **any** number of dimensions from one coordinate vector per direction,
pre-computing the separable cell measure from the geometry.

Each axis may independently be uniform or stretched, and keeps whichever it is in its own type — see
[`isuniform`](@ref). Axes are adapted to the geometry's float type `T` by [`_to_axis`](@ref), which
preserves uniformity and keeps a device-resident axis on its device.

For a `SphericalGeometry` the directions are `(λ, φ, r, …)`: longitude, geographic latitude, and — in
3-D and above — the absolute radius from the origin, not an offset from a reference radius. Measures
are the metric elements `R·Δλ`, `R²cosφ·Δλ·Δφ` and `r²cosφ·Δλ·Δφ·Δr`, with further directions entering
as plain widths. A `CartesianGeometry` measure is the product of the per-direction widths.

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
  for a uniform axis (`n·|Δ|`). A **nonuniform** periodic Cartesian direction has no well-defined
  closure to infer — its seam gap is not determined by its samples — so `period` is required there
  determine.
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

function _structured_grid(
    geometry::G, axes_in::NTuple{N,AbstractVector}, mask, topo, period, sampling,
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

    measure = _cell_measure(geometry, ax, period_args)
    stats = ntuple(d -> _axis_stats(ax[d]), Val(N))
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
