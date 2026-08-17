module Grids

using ..Execution: Execution
using ..Axes: Axes
using ..Geometry: Geometry
using ..Discretization: Discretization

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

# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------
# Coordinate STORAGE is positional: every grid holds an `NTuple{N,<:AbstractArray{T}}` of coordinate
# arrays, one per coordinate direction, reached by index through `coordinates(grid, d)`. Coordinate
# NAMES come from the geometry (`Geometry.point_names`), so `grid.x` exists on a Cartesian grid and
# `grid.λ` on a spherical one — never both, and never one standing in for the other.

# These reach fields with `getfield`, never `grid.field`: `getproperty` below is overloaded to
# resolve coordinate names, so property access from inside the interface would recurse.
@inline grid_geometry(grid::AbstractGrid) = getfield(grid, :geometry)

"""
    coordinates(grid) -> NTuple{N,AbstractArray}
    coordinates(grid, d::Integer) -> AbstractArray

The grid's coordinate arrays, or just direction `d`'s. Shape depends on the grid architecture: a
`StructuredGrid` stores one 1-D axis vector per direction, a `CurvilinearGrid` one `N`-D array
per direction, an `UnstructuredGrid` one value per node. Direction order matches
[`Geometry.point_names`](@ref) — `(x, y, z)` for Cartesian, `(λ, φ, r)` for spherical.

See also [`axis`](@ref) (the rectilinear spelling), [`coords`](@ref) (a single point).
"""
@inline coordinates(grid::AbstractGrid) = getfield(grid, :coordinates)
@inline coordinates(grid::AbstractGrid, d::Integer) = @inbounds coordinates(grid)[d]

"""
    axis(grid::AbstractStructuredGrid, d::Integer) -> AbstractVector

Direction `d`'s coordinate axis. Only rectilinear grids have axes; this is
[`coordinates`](@ref) under the name that is exact for them.
"""
@inline axis(grid::AbstractStructuredGrid, d::Integer) = coordinates(grid, d)

# ---------------------------------------------------------------------------
# Spacing and span
# ---------------------------------------------------------------------------

"""
    isuniform(grid, d) -> Bool
    isuniform(grid) -> Bool

Whether coordinate direction `d` has constant spacing known from its TYPE (all directions, for the
no-`d` form). This is the compile-time answer, so it can select a fast path by dispatch rather than
by a runtime scan — see [`Axes.spacing_trait`](@ref) for the trait it reads and
No code path inspects coordinate VALUES to decide this; the answer comes from the type alone.

A curvilinear or unstructured grid is never uniform: its coordinates are per-cell fields, not axes.
"""
@inline isuniform(grid::AbstractGrid, d::Integer) = Axes.isuniform(coordinates(grid, d))
@inline isuniform(grid::AbstractGrid) = all(Axes.isuniform, coordinates(grid))

"""
    spacing(grid, d) -> T

The constant spacing of coordinate direction `d`, available without reading any coordinate. Signed,
so a descending axis reports a negative spacing. Raises for a direction that is not
[`isuniform`](@ref) — for a nonuniform one use [`minimum_spacing`](@ref) / [`maximum_spacing`](@ref)
for its range of gaps, [`local_spacing`](@ref) for the gaps at one index, or [`cell_width`](@ref) /
[`cell_widths`](@ref) for the width of a cell.
"""
@inline spacing(grid::AbstractGrid, d::Integer) = Axes.spacing(coordinates(grid, d))

"""
    origin(grid, d) -> T

The first coordinate along direction `d`.
"""
@inline origin(grid::AbstractGrid, d::Integer) = first(coordinates(grid, d))

"""
    bounds(grid, d) -> (lo, hi)

Smallest and largest coordinate along direction `d`, ordered `lo ≤ hi` regardless of whether the
direction is stored ascending or descending. These are the extreme SAMPLE positions (cell centres),
not the outer cell boundaries.
"""
@inline bounds(grid::AbstractGrid, d::Integer) = extrema(coordinates(grid, d))

"""
    extent(grid, d) -> T

`hi - lo` from [`bounds`](@ref): the span covered by direction `d`'s samples. Zero for a singleton
direction.
"""
@inline function extent(grid::AbstractGrid, d::Integer)
    lo, hi = bounds(grid, d)
    return hi - lo
end

"""
    minimum_spacing(grid, d) -> T

Smallest gap between consecutive samples along direction `d`, as a non-negative magnitude. `O(1)` when
the direction is [`isuniform`](@ref) and `O(N_d)` otherwise. With [`maximum_spacing`](@ref) it bounds
how far an index window must reach to cover a given physical distance, which is what a
neighbourhood-by-distance query needs on a stretched axis. They are also the exact test of whether a
stretched axis happens to be equally spaced: its gaps are identical when the two are equal.

A direction of fewer than two samples has no gap, and reports `Inf` — the identity for `min`.
"""
@inline minimum_spacing(grid::AbstractStructuredGrid, d::Integer) = _min_gap(coordinates(grid, d))

"""
    maximum_spacing(grid, d) -> T

Largest gap between consecutive samples along direction `d`, the counterpart of
[`minimum_spacing`](@ref). A direction of fewer than two samples reports `0`, the identity for `max`.
"""
@inline maximum_spacing(grid::AbstractStructuredGrid, d::Integer) = _max_gap(coordinates(grid, d))

# Selecting ONE axis by a runtime direction out of the coordinate tuple, whose entries may be different
# types. `ntuple(…, Val(N))` unrolls where every direction is wanted; only one is here, so the tuple is
# walked by a recursive tail-split instead — the same shape as `_checkaxes`. Each branch sees a
# concretely typed axis and every branch returns the same concrete type, so the result infers and
# nothing allocates, where `coordinates(grid, d)` alone would be a dynamic lookup.
@inline _at_axis(::F, ::Tuple{}, d::Integer) where {F} =
    throw(ArgumentError("direction $d is outside this grid's directions"))
@inline _at_axis(f::F, c::Tuple, d::Integer) where {F} =
    d == 1 ? f(first(c)) : _at_axis(f, Base.tail(c), d - 1)

"""
    local_spacing(grid, d, i) -> (h_m, h_p)

The one-sided coordinate gaps around index `i` along direction `d`:
[`Discretization.local_spacing`](@ref) on that direction's axis, with the wrap period taken from the
grid, so a periodic seam is right without the caller supplying it.

Signed, so a descending axis reports negative gaps — see the axis-level form for why, and
[`cell_width`](@ref) for the non-negative width built from them. Allocation-free, so this is the
per-point form to call inside a loop assembling a finite-difference operator.
"""
@inline function local_spacing(grid::AbstractStructuredGrid, d::Integer, i::Integer)
    # Branching here rather than passing a `Union{Nothing,T}` period: both arms return the same
    # concrete tuple type, where the union would have to be split inside the callee on every call.
    return isperiodic(grid, d) ?
        _at_axis(x -> Discretization.local_spacing(x, i, period(grid, d)), coordinates(grid), d) :
        _at_axis(x -> Discretization.local_spacing(x, i, nothing), coordinates(grid), d)
end

"""
    cell_width(grid, d, i) -> T

The coordinate width of cell `i` along direction `d`: [`Discretization.cell_width`](@ref) on that
direction's axis, with the grid's own wrap period. Non-negative whichever way the axis is stored.

This is the coordinate width, not the cell measure — on a sphere [`measure`](@ref) is `R²cosφ·Δλ·Δφ`
and this is the `Δλ` or `Δφ` in it, which [`measure_factors`](@ref) does not expose separately
because it folds the metric into the factor it multiplies.
"""
@inline function cell_width(grid::AbstractStructuredGrid, d::Integer, i::Integer)
    return isperiodic(grid, d) ?
        _at_axis(x -> Discretization.cell_width(x, i, period(grid, d)), coordinates(grid), d) :
        _at_axis(x -> Discretization.cell_width(x, i, nothing), coordinates(grid), d)
end

"""
    cell_widths(grid, d) -> AbstractVector

[`cell_width`](@ref) along the whole of direction `d`, as
[`Discretization.cell_widths`](@ref) on its axis with the grid's own wrap period. A uniform direction
gets an [`Axes.ConstantVector`](@ref), so nothing is materialized.
"""
@inline function cell_widths(grid::AbstractStructuredGrid, d::Integer)
    return isperiodic(grid, d) ?
        Discretization.cell_widths(coordinates(grid, d), period(grid, d)) :
        Discretization.cell_widths(coordinates(grid, d), nothing)
end

"""
    coordinate_names(grid) -> NTuple{N,Symbol}

The grid's coordinate names, from its geometry: `(:x, :y[, :z])` or `(:λ, :φ[, :r])`.
"""
@inline coordinate_names(grid::AbstractGrid) =
    Geometry.point_names(grid_geometry(grid), Val(length(coordinates(grid))))

# Geometry-correct named access (`grid.x` / `grid.λ`), resolved from `coordinate_names`. A name that
# does not belong to this geometry is a `FieldError`, not a silent alias for the wrong quantity.
@inline function Base.getproperty(grid::AbstractGrid, name::Symbol)
    d = findfirst(==(name), coordinate_names(grid))
    d === nothing && return getfield(grid, name)
    return @inbounds coordinates(grid)[d]
end

@inline Base.propertynames(grid::AbstractGrid) = (coordinate_names(grid)..., fieldnames(typeof(grid))...)

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

"""
    AbstractTopology

How a coordinate direction closes: [`Periodic`](@ref) or [`Bounded`](@ref).

Carried in the grid's type, so [`isperiodic`](@ref) is a compile-time answer and callers can dispatch
on it. The cell measure depends on it — a wrapped boundary cell has a width a bounded one does not —
so it is part of what the grid is, not a flag attached to it.
"""
abstract type AbstractTopology end

"""
    Periodic()

A direction that wraps: the last cell's neighbour is the first, one [`period`](@ref) away.
"""
struct Periodic <: AbstractTopology end

"""
    Bounded()

A direction that ends: its first and last cells each have one neighbour rather than two.
"""
struct Bounded <: AbstractTopology end

"""
    topology(grid) -> NTuple{N,AbstractTopology}
    topology(grid, d) -> AbstractTopology

The grid's per-direction topology. Singletons, so this occupies no storage.
"""
topology(grid::AbstractGrid) = map(_ -> Bounded(), coordinates(grid))
@inline topology(grid::AbstractGrid, d::Integer) = topology(grid)[d]

"""
    periodic_flags(grid) -> NTuple{N,Bool}

[`topology`](@ref) as one `Bool` per direction. Const-folds, and unlike indexing the heterogeneous
topology tuple it stays type-stable under a runtime direction.
"""
@inline periodic_flags(grid::AbstractGrid) = map(_is_periodic, topology(grid))

@inline _is_periodic(::Periodic) = true
@inline _is_periodic(::Bounded) = false

"""
    isperiodic(grid, d) -> Bool

Whether coordinate direction `d` wraps. See [`topology`](@ref) for the type-level form and
[`period`](@ref) for the wrap length.
"""
@inline isperiodic(grid::AbstractGrid, d::Integer) =
    @inbounds periodic_flags(grid)[_checked_direction(periodic_flags(grid), d)]

# A direction is a dimension selector, not an array index, so an out-of-range one is an
# `ArgumentError` — the same one `_at_axis` raises — rather than whatever the underlying tuple read
# happens to do. Without this the read is `@inbounds` on a runtime index: under `--check-bounds=yes`
# it raises `BoundsError` from an internal, and under the default it is simply out of bounds.
@inline function _checked_direction(t::Tuple, d::Integer)
    1 ≤ d ≤ length(t) ||
        throw(ArgumentError("direction $d is outside this grid's directions"))
    return d
end

# Normalize a caller's `topology` to `NTuple{N,AbstractTopology}`. `Bool`s are accepted, and a scalar
# or short tuple applies to the leading directions with the rest `Bounded`.
_as_topology(::Val{N}, t::AbstractTopology) where {N} = ntuple(d -> d == 1 ? t : Bounded(), Val(N))
_as_topology(::Val{N}, b::Bool) where {N} = _as_topology(Val(N), b ? Periodic() : Bounded())
_as_topology(::Val{N}, t::NTuple{N,AbstractTopology}) where {N} = t
_as_topology(::Val{N}, t::Union{Tuple,AbstractVector}) where {N} =
    ntuple(d -> d ≤ length(t) ? _topology_of(t[d]) : Bounded(), Val(N))

@inline _topology_of(t::AbstractTopology) = t
@inline _topology_of(b::Bool) = b ? Periodic() : Bounded()

"""
    SeparableMeasure(factors)

The cell measure of a rectilinear grid, stored as its per-axis factors rather than materialized.

Every measure this package supports on such a grid is a product of one factor per axis (see
[`_measure_factors`](@ref)), so the `∏ Nᵈ` entries carry only `∑ Nᵈ` numbers. It is a genuine
`AbstractArray`: indexing, broadcasting and `collect` behave as for the dense equivalent.

`sum` is specialized to `∏ᵈ ∑ᵢ wᵈᵢ`, which is `O(∑ Nᵈ)` rather than `O(∏ Nᵈ)`.
"""
struct SeparableMeasure{T,N,F<:NTuple{N,AbstractVector{T}}} <: AbstractArray{T,N}
    factors::F
end

@inline Base.size(m::SeparableMeasure) = map(length, m.factors)
Base.IndexStyle(::Type{<:SeparableMeasure}) = IndexCartesian()

@inline function Base.getindex(m::SeparableMeasure{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(m, I...)
    return prod(ntuple(d -> @inbounds(m.factors[d][I[d]]), Val(N)))
end

# ---------------------------------------------------------------------------
# Separable reductions
# ---------------------------------------------------------------------------
#
# A reduction over the outer product factors into a reduction over the axes when the pair
# `(f, op)` permits it, turning `O(∏ Nᵈ)` into `O(∑ Nᵈ)`. Dispatch is on that PAIR, not on the
# reduction's name: `mapreduce` is where `sum`, `prod`, `maximum`, `minimum`, `sum(f, ·)`, `count` and
# the `dims` forms all arrive, so one method each covers them, and any pair not listed here falls
# through to the generic dense path and stays correct.
#
# `f` must be MULTIPLICATIVE — `f(xy) = f(x)f(y)` — for a sum or product to factor. `identity`, `abs`,
# `abs2`, `sqrt` and `inv` are; `exp` and `log` are not, and must not take this path.
const _MultiplicativeF = Union{
    typeof(identity), typeof(abs), typeof(abs2), typeof(sqrt), typeof(inv),
}

# ∑_{i,j} f(wxᵢ·wyⱼ) = (∑ᵢ f(wxᵢ))(∑ⱼ f(wyⱼ))
Base.mapreduce(f::_MultiplicativeF, ::typeof(Base.add_sum), m::SeparableMeasure) =
    prod(fac -> sum(f, fac), m.factors)

# ∏_{i,j} f(wxᵢ·wyⱼ) = ∏ᵈ (∏ᵢ f(wᵈᵢ))^(∏_{e≠d} Nᵉ)
function Base.mapreduce(f::_MultiplicativeF, ::typeof(Base.mul_prod), m::SeparableMeasure{T,N}) where {T,N}
    n = size(m)
    total = prod(n)
    return prod(ntuple(d -> prod(f, m.factors[d])^(total ÷ n[d]), Val(N)))
end

Base.mapreduce(f::_MultiplicativeF, ::typeof(max), m::SeparableMeasure) = _sep_extrema(f, m)[2]
Base.mapreduce(f::_MultiplicativeF, ::typeof(min), m::SeparableMeasure) = _sep_extrema(f, m)[1]
Base.extrema(m::SeparableMeasure) = _sep_extrema(identity, m)
Base.extrema(f::_MultiplicativeF, m::SeparableMeasure) = _sep_extrema(f, m)

"""
    _sep_extrema(f, m) -> (lo, hi)

Smallest and largest `f(cell)` over a [`SeparableMeasure`](@ref), from the per-axis extremes.

A product's extremes are attained with every factor at one of its own endpoints, so all `2^N` endpoint
combinations are formed and the best taken. That is exact for factors of ANY sign — `∏ maximum` alone
would be wrong the moment a factor could go negative — and it costs `O(∑ Nᵈ + 2^N)` against the dense
`O(∏ Nᵈ)`.
"""
function _sep_extrema(f, m::SeparableMeasure{T,N}) where {T,N}
    isempty(m) && throw(ArgumentError("extrema of an empty SeparableMeasure is undefined"))
    ends = ntuple(d -> extrema(m.factors[d]), Val(N))
    lo = hi = f(prod(ntuple(d -> ends[d][1], Val(N))))
    for corner in 0:(2^N - 1)
        v = f(prod(ntuple(d -> ends[d][1 + ((corner >> (d - 1)) & 1)], Val(N))))
        v < lo && (lo = v)
        v > hi && (hi = v)
    end
    return (lo, hi)
end

# Reducing direction `d` away replaces its factor by the single number that direction contributes, so
# the result is still separable and still costs O(∑ Nᵈ).
function Base.sum(m::SeparableMeasure{T,N}; dims = :) where {T,N}
    dims === (:) && return mapreduce(identity, Base.add_sum, m)
    reduced = ntuple(Val(N)) do d
        d in dims ? Axes.ConstantVector(sum(m.factors[d]), 1) : m.factors[d]
    end
    return SeparableMeasure(reduced)
end

# The largest cell sits where every factor is largest, so the index is the per-axis argmax — no scan of
# the product. Only valid while the factors are non-negative, which a measure's are by construction;
# a factor that is not falls back to the generic search.
function Base.findmax(m::SeparableMeasure{T,N}) where {T,N}
    all(fac -> minimum(fac) ≥ 0, m.factors) || return @invoke findmax(m::AbstractArray)
    I = ntuple(d -> argmax(m.factors[d]), Val(N))
    return (m[I...], CartesianIndex(I))
end

function Base.findmin(m::SeparableMeasure{T,N}) where {T,N}
    all(fac -> minimum(fac) ≥ 0, m.factors) || return @invoke findmin(m::AbstractArray)
    I = ntuple(d -> argmin(m.factors[d]), Val(N))
    return (m[I...], CartesianIndex(I))
end

Base.argmax(m::SeparableMeasure) = findmax(m)[2]
Base.argmin(m::SeparableMeasure) = findmin(m)[2]

# ---------------------------------------------------------------------------
# Separability-preserving broadcast
# ---------------------------------------------------------------------------
#
# Converting a 2000×2000 grid's measure to other units is `c .* m`, which through the generic array
# path turns 32 bytes into 32 MB. The operations that keep the measure a PRODUCT of per-axis factors
# stay a measure; everything else falls through to Base and materializes, which is correct, just dense.
#
# Non-negative factors are an invariant of this type — widths are `abs`-ed at construction — and
# `findmax`/`findmin` above rely on it, since "largest cell at the per-axis argmax" is a statement
# about non-negative factors only. A scale that would break it therefore does NOT stay lazy.

@inline _scaled_factor(w::Axes.ConstantVector, c) = Axes.ConstantVector(w.value * c, length(w))
@inline _scaled_factor(w::AbstractVector, c) = w .* c

# Scaling the product means scaling exactly one factor.
_scale_separable(m::SeparableMeasure{T,N}, c::T) where {T,N} = SeparableMeasure(
    ntuple(d -> d == 1 ? _scaled_factor(m.factors[d], c) : m.factors[d], Val(N)),
)

function _broadcast_scale(m::SeparableMeasure{T,N}, c::Real) where {T,N}
    cT = convert(T, c)
    cT ≥ zero(T) && return _scale_separable(m, cT)
    return collect(m) .* cT
end

Base.broadcasted(::typeof(*), c::Real, m::SeparableMeasure) = _broadcast_scale(m, c)
Base.broadcasted(::typeof(*), m::SeparableMeasure, c::Real) = _broadcast_scale(m, c)
Base.broadcasted(::typeof(/), m::SeparableMeasure, c::Real) = _broadcast_scale(m, inv(c))

# `f(∏ wᵈ) = ∏ f(wᵈ)` exactly when `f` is multiplicative — the same condition the reductions above
# dispatch on, and the same `f`s.
@inline _mapped_factor(f, w::Axes.ConstantVector) = Axes.ConstantVector(f(w.value), length(w))
@inline _mapped_factor(f, w::AbstractVector) = f.(w)

Base.broadcasted(f::_MultiplicativeF, m::SeparableMeasure{<:Any,N}) where {N} =
    SeparableMeasure(ntuple(d -> _mapped_factor(f, m.factors[d]), Val(N)))

@inline _combine_factors(op, u::Axes.ConstantVector, v::Axes.ConstantVector) =
    Axes.ConstantVector(op(u.value, v.value), length(u))
@inline _combine_factors(op, u::AbstractVector, v::AbstractVector) = op.(u, v)

# An elementwise product or quotient of two separable arrays is separable factor by factor.
for op in (:*, :/)
    @eval function Base.broadcasted(
        ::typeof($op), a::SeparableMeasure{T,N}, b::SeparableMeasure{T,N},
    ) where {T,N}
        size(a) == size(b) || throw(DimensionMismatch(
            "separable measures of size $(size(a)) and $(size(b)) do not match",
        ))
        return SeparableMeasure(ntuple(d -> _combine_factors($op, a.factors[d], b.factors[d]), Val(N)))
    end
end

"""
    measure_factors(grid) -> NTuple{N,AbstractVector} or nothing

The grid's per-axis measure factors when it has them, else `nothing`. Callers that can exploit
separability (a zonal mean weights by one factor only, a global integral is a product of sums) can
avoid touching `∏ Nᵈ` values at all.
"""
@inline measure_factors(grid::AbstractGrid) = _measure_factors_of(measure(grid))
# Also on a measure directly: a separability-preserving broadcast returns one of these, so the factors
# of the result have to be reachable without going back through a grid.
@inline measure_factors(m::AbstractArray) = _measure_factors_of(m)
@inline _measure_factors_of(m::SeparableMeasure) = m.factors
@inline _measure_factors_of(::AbstractArray) = nothing

"""
    measure_array(grid) -> Array

The cell measure materialized densely. This is `∏ Nᵈ` values — only ask for it when a dense array is
genuinely required; [`measure`](@ref) already indexes and broadcasts.
"""
measure_array(grid::AbstractGrid) = collect(measure(grid))

"""
    AllActive(size)

The mask of a grid where every cell participates, stored as its size alone. `getindex` is a
constant the compiler can fold, and `count` is `length` without a scan.
"""
struct AllActive{N} <: AbstractArray{Bool,N}
    size::NTuple{N,Int}
end

@inline Base.size(m::AllActive) = m.size
Base.IndexStyle(::Type{<:AllActive}) = IndexCartesian()

@inline function Base.getindex(m::AllActive{N}, I::Vararg{Int,N}) where {N}
    @boundscheck checkbounds(m, I...)
    return true
end

Base.count(m::AllActive) = length(m)
Base.all(m::AllActive) = true
Base.any(m::AllActive) = length(m) > 0
Base.sum(m::AllActive) = length(m)
Base.prod(m::AllActive) = true
Base.minimum(m::AllActive) = _allactive_reduce(m, "minimum")
Base.maximum(m::AllActive) = _allactive_reduce(m, "maximum")
Base.extrema(m::AllActive) = (maximum(m), maximum(m))
# `f::Function` rather than `f`: Base's own `all(f::Function, ::AbstractArray)` would otherwise be
# equally specific, and the pair ambiguous.
Base.count(f::Function, m::AllActive) = f(true) ? length(m) : 0
Base.all(f::Function, m::AllActive) = isempty(m) ? true : f(true)
Base.any(f::Function, m::AllActive) = isempty(m) ? false : f(true)
Base.findfirst(m::AllActive{N}) where {N} =
    isempty(m) ? nothing : CartesianIndex(ntuple(_ -> 1, Val(N)))
Base.findall(m::AllActive) = collect(CartesianIndices(size(m)))

_allactive_reduce(m::AllActive, name) =
    isempty(m) ? throw(ArgumentError("$name of an empty AllActive is undefined")) : true

"""
    measure(grid, I...) -> T

Cell measure at index `I`: length in 1-D, area in 2-D, volume in 3-D, or the node's control-volume
size on an unstructured grid. [`area`](@ref) is the 2-D spelling of the same quantity.
"""
@inline measure(grid::AbstractGrid) = getfield(grid, :measure)

# `@boundscheck` + `@inbounds` body: elided at an `@inbounds` call site, so a hot loop pays nothing,
# while an ordinary call errors instead of returning a value read past the end of the array.
@inline function measure(grid::AbstractGrid, I::Vararg{Integer})
    m = measure(grid)
    @boundscheck checkbounds(m, I...)
    return @inbounds m[I...]
end

"""
    area(grid, I...) -> T

[`measure`](@ref) under its 2-D name.
"""
@inline area(grid::AbstractGrid, I::Vararg{Integer}) = measure(grid, I...)

"""
    isactive(grid, I...) -> Bool

Whether cell/node `I` participates (`false` = masked out).
"""
@inline mask(grid::AbstractGrid) = getfield(grid, :mask)

@inline function isactive(grid::AbstractGrid, I::Vararg{Integer})
    m = mask(grid)
    @boundscheck checkbounds(m, I...)
    return @inbounds m[I...]
end

Base.size(grid::AbstractGrid) = size(mask(grid))
Base.size(grid::AbstractGrid, d::Integer) = size(mask(grid), d)
Base.length(grid::AbstractGrid) = length(mask(grid))
Base.ndims(grid::AbstractGrid) = ndims(mask(grid))
Base.eltype(::AbstractGrid{G,T}) where {G,T} = T
Base.axes(grid::AbstractGrid) = axes(mask(grid))

"""
    size_tuple(grid) -> NTuple{N,Int}

`size(grid)`, kept as a named function for call sites that read better spelled out.
"""
size_tuple(grid::AbstractGrid) = size(grid)

function Base.show(io::IO, ::MIME"text/plain", grid::AbstractGrid)
    println(io, nameof(typeof(grid)), "{", eltype(grid), "} ", join(size(grid), "×"),
            " (", count(mask(grid)), "/", length(grid), " active)")
    println(io, "  geometry:  ", grid_geometry(grid))
    for (d, name) in enumerate(coordinate_names(grid))
        c = coordinates(grid, d)
        rng = isempty(c) ? "empty" : string(minimum(c), " … ", maximum(c))
        per = isperiodic(grid, d) ? ", periodic" : ""
        println(io, "  ", rpad(String(name), 10), rng, "  (", join(size(c), "×"), per, ")")
    end
    print(io, "  measure:   ", sum(measure(grid)), " total")
end

Base.show(io::IO, grid::AbstractGrid) =
    print(io, nameof(typeof(grid)), "{", eltype(grid), "}(", join(size(grid), "×"), ")")

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
    StructuredGrid{G, T, N, TP, C, AT, BT}

Rectilinear `N`-dimensional grid, for any `N`: one coordinate vector per direction (`coordinates`),
an N-D cell `measure` (length in 1-D, area in 2-D, volume in 3-D, the N-D measure in general), an N-D
active `mask`, per-direction `topology`, and the wrap `period` of each periodic direction.

`TP` is the per-direction [`AbstractTopology`](@ref): singletons, so no storage, and readable from
the type.

`C` is a heterogeneous `NTuple{N,AbstractVector{T}}` — each axis independently keeps whatever
concrete `AbstractVector{T}` type it was constructed with (an [`Axes.UniformAxis`](@ref), a plain
`Vector`, a device array, or any other subtype); there is deliberately no shared vector type forcing
the axes to match. This matters beyond storage: a `UniformAxis`'s type is a compile-time proof of
constant spacing that [`isuniform`](@ref) and [`spacing`](@ref) read without touching a coordinate,
and forcing the axes into a common type would destroy it. One axis can be uniform while another is
stretched.
"""
struct StructuredGrid{
    G<:Geometry.AbstractGeometry,
    T<:AbstractFloat,
    N,
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
    stats::NTuple{N,AxisStats{T}}   # reduced once; see AxisStats
end

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
@inline function _raw_coords(grid::StructuredGrid{G,T,N}, I::Vararg{Integer,N}) where {G,T,N}
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

The grid's stored measure: a [`SeparableMeasure`](@ref) wherever the metric factors, and a dense array
where it genuinely does not.
"""
_cell_measure(geometry, ax, per) = SeparableMeasure(_measure_factors(geometry, ax, per))

# Geodetic `(λ, φ, h)`: the volume element `(N(φ)+h)cosφ·(M(φ)+h)` offsets BOTH curvature radii by the
# height, so φ and h are coupled and no product of per-axis factors reproduces it. Stored dense; further
# directions still enter as plain widths.
function _cell_measure(
    geometry::G, ax::NTuple{N,AbstractVector{T}}, per::NTuple{N,Union{Nothing,Real}},
) where {N, T<:AbstractFloat, G<:Geometry.AbstractEllipsoidalGeometry{T}}
    N ≥ 3 || return SeparableMeasure(_measure_factors(geometry, ax, per))
    λ, φ, h = ax[1], ax[2], ax[3]
    wλ = Discretization.cell_widths(λ, per[1])
    wφ = Discretization.cell_widths(φ, per[2])
    wh = Discretization.cell_widths(h, per[3])
    rest = ntuple(d -> Discretization.cell_widths(ax[d + 3], per[d + 3]), Val(N - 3))
    dims = ntuple(d -> length(ax[d]), Val(N))
    out = similar(wλ, T, dims)
    @inbounds for I in CartesianIndices(dims)
        i, j, k = I[1], I[2], I[3]
        φj, hk = φ[j], h[k]
        v = (Geometry.prime_vertical_radius(geometry, φj) + hk) * cos(φj) *
            (Geometry.meridional_radius(geometry, φj) + hk) * wλ[i] * wφ[j] * wh[k]
        for d in 1:(N - 3)
            v *= rest[d][I[d + 3]]
        end
        out[I] = v
    end
    return out
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
) where {T<:AbstractFloat}
    axes_in, mask_pos = _split_axes_mask(args)
    mask === nothing || mask_pos === nothing ||
        throw(ArgumentError("mask given both positionally and as a keyword"))
    return _structured_grid(
        geometry, axes_in, mask_pos === nothing ? mask : mask_pos,
        topology === nothing ? periodic : topology, period,
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
    geometry::G, axes_in::NTuple{N,AbstractVector}, mask, topo, period,
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
    return StructuredGrid{G, T, N, typeof(tp), typeof(ax), typeof(measure), typeof(m)}(
        geometry, ax, measure, m, tp, per, stats,
    )
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

"""
    apply_stencil!(out, field, grid, dim; order=1, nodes=order+1, active_only=true, masked=zero) -> out

[`Discretization.apply_stencil!`](@ref) with the axis, wrap period and mask taken from `grid`, so a
periodic direction wraps and an inactive cell is honoured without restating any of it.

Only a rectilinear direction has a 1-D axis to difference along, so this is a `StructuredGrid` method.
"""
function Discretization.apply_stencil!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::StructuredGrid{G,T,N},
    dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
    active_only::Bool = true, masked = zero(S), backend = nothing,
    policy::Discretization.AbstractMaskPolicy = Discretization.BlankMasked(), scratch = nothing,
) where {S,G,T,N,NA}
    _check_batched(out, field, grid, Val(N), Int(dim))
    msk = active_only && !(mask(grid) isa AllActive) ? mask(grid) : nothing
    return Discretization.apply_stencil!(
        out, field, coordinates(grid, dim), dim;
        order = order, nodes = nodes,
        period = isperiodic(grid, dim) ? period(grid, dim) : nothing,
        mask = msk, masked = masked, backend = backend, policy = policy, scratch = scratch,
    )
end

# The two samples of direction `d` that bracket `v`, with their weights. A periodic direction wraps:
# past the last sample the pair is `(n, 1)` across the seam, where `interpolation_weights` alone would
# clamp and return the endpoint value.
@inline function _interp_pair(grid::StructuredGrid{G,T,N}, d::Int, v::T) where {G,T,N}
    x = coordinates(grid, d)
    n = length(x)
    n == 1 && return (1, 1, one(T), zero(T))
    if isperiodic(grid, d)
        L = T(period(grid, d))
        if L > 0
            lo = T(axis_stats(grid, d).min_value)
            v = lo + mod(v - lo, L)
            @inbounds x1, xn = T(x[1]), T(x[n])
            asc = xn ≥ x1
            beyond = asc ? v > xn : v < xn
            if beyond
                h = asc ? (x1 + L) - xn : (x1 - L) - xn
                t = iszero(h) ? zero(T) : (v - xn) / h
                return (n, 1, one(T) - t, t)
            end
        end
    end
    i, w = Discretization.interpolation_weights(x, v)
    return (Int(i), Int(i) + 1, w[1], w[2])
end

function Discretization.interpolate(
    field::AbstractArray{S,N}, grid::StructuredGrid{G,T,N}, p::NTuple{N,Real};
    active_only::Bool = true, masked = S(NaN),
    policy::Discretization.AbstractMaskPolicy = Discretization.BlankMasked(),
) where {S,G,T,N}
    _check_interp(field, grid, Val(N), policy)
    return _interp_at(field, grid, p, 0, active_only, masked, policy)
end

@inline function _check_interp(
    field::AbstractArray{<:Any,NA}, grid::StructuredGrid{G,T,N}, ::Val{N}, policy,
) where {NA,G,T,N}
    policy isa Discretization.ShiftWithinRun && Discretization._interp_mask_error(policy)
    NA ≥ N || throw(DimensionMismatch("field has $NA axes but the grid has $N"))
    ntuple(d -> size(field, d), Val(N)) == size_tuple(grid) || throw(DimensionMismatch(
        "field's leading $N axes $(ntuple(d -> size(field, d), Val(N))) do not match the grid " *
        "$(size_tuple(grid))",
    ))
    return nothing
end

# One batch element, at linear offset `off` into the field. The bracketing cell and its weights are a
# property of the POINT and the grid, so a batched call solves them once and calls this per element.
function _interp_at(
    field::AbstractArray{S}, grid::StructuredGrid{G,T,N}, p::NTuple{N,Real}, off::Int,
    active_only::Bool, masked, policy,
) where {S,G,T,N}
    prs = ntuple(d -> _interp_pair(grid, d, T(p[d])), Val(N))
    msk = active_only && !(mask(grid) isa AllActive) ? mask(grid) : nothing
    sz = size_tuple(grid)
    acc = zero(S)
    wsum = zero(T)
    # The `2^N` corners of the bracketing cell, each weighted by the product of its per-axis weights.
    @inbounds for c in CartesianIndices(ntuple(_ -> 2, Val(N)))
        w = prod(ntuple(d -> c[d] == 1 ? prs[d][3] : prs[d][4], Val(N)))
        iszero(w) && continue
        I = ntuple(d -> c[d] == 1 ? prs[d][1] : prs[d][2], Val(N))
        if msk !== nothing && !msk[I...]
            policy isa Discretization.BlankMasked && return masked
            continue                                   # `ReduceInRun`: drop it and renormalize
        end
        # Linear, so one expression serves the unbatched field and a slice of a batched one.
        lin = I[N]
        for d in (N - 1):-1:1
            lin = (lin - 1) * sz[d] + I[d]
        end
        acc += S(w) * S(field[off + lin])
        wsum += w
    end
    return wsum > 0 ? acc / S(wsum) : masked
end

@inline Discretization.interpolate(
    field::AbstractArray, grid::StructuredGrid, p::Geometry.PointLike; kwargs...,
) =
    Discretization.interpolate(field, grid, Geometry.as_ntuple(p); kwargs...)

"""
    interpolate!(out, field, grid, p; active_only=true, masked=NaN, policy=BlankMasked()) -> out

[`Discretization.interpolate`](@ref) at one coordinate for a field carrying trailing BATCH axes: `out`
receives one value per batch element, in the order those axes are laid out.

The bracketing cell and its `2^N` corner weights depend on the point and the grid, not on the data, so
they are solved once here and applied to every element — less work than one `interpolate` per slice.
"""
function Discretization.interpolate!(
    out::AbstractVector{S}, field::AbstractArray{<:Any,NA}, grid::StructuredGrid{G,T,N},
    p::NTuple{N,Real}; active_only::Bool = true, masked = S(NaN),
    policy::Discretization.AbstractMaskPolicy = Discretization.BlankMasked(),
) where {S,G,T,N,NA}
    _check_interp(field, grid, Val(N), policy)
    n = prod(size_tuple(grid))
    nb = length(field) ÷ n
    length(out) == nb || throw(DimensionMismatch(
        "out holds $(length(out)) values but the field carries $nb batch elements",
    ))
    @inbounds for b in 1:nb
        out[b] = _interp_at(field, grid, p, (b - 1) * n, active_only, masked, policy)
    end
    return out
end

@inline Discretization.interpolate!(
    out::AbstractVector, field::AbstractArray, grid::StructuredGrid, p::Geometry.PointLike; kwargs...,
) = Discretization.interpolate!(out, field, grid, Geometry.as_ntuple(p); kwargs...)

# The allocating form, as everywhere else in the package: `spherical_points!`/`spherical_points`,
# `latitude_weights!`/`latitude_weights`.
function Discretization.interpolate(
    field::AbstractArray{S,NA}, grid::StructuredGrid{G,T,N}, p::NTuple{N,Real}; kwargs...,
) where {S,G,T,N,NA}
    n = prod(size_tuple(grid))
    return Discretization.interpolate!(Vector{S}(undef, length(field) ÷ n), field, grid, p; kwargs...)
end

function Discretization.derivative!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::StructuredGrid{G,T,N},
    dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
    active_only::Bool = true, masked = zero(S), backend = nothing,
    policy::Discretization.AbstractMaskPolicy = Discretization.BlankMasked(), scratch = nothing,
) where {S,G,T,N,NA}
    Discretization.apply_stencil!(out, field, grid, dim; order = order, nodes = nodes,
                                  active_only = active_only, masked = masked, backend = backend,
                                  policy = policy, scratch = scratch)
    return _scale_by_metric!(out, grid, Int(dim), masked)
end

# `out`/`field` may carry trailing BATCH axes beyond the grid's own: a `(Nx, Ny, Nb)` field against a
# 2-D grid is `Nb` fields differenced together, one pass instead of one per slice. The grid fixes how
# many leading axes are spatial, and a disagreement THERE is still a mistake and still raises — only
# extra trailing axes are new.
@inline function _check_batched(
    out::AbstractArray{<:Any,NA}, field::AbstractArray{<:Any,NA}, grid::StructuredGrid{G,T,N},
    ::Val{N}, dim::Int,
) where {NA,G,T,N}
    NA ≥ N || throw(DimensionMismatch(
        "field has $NA axes but the grid has $N",
    ))
    1 ≤ dim ≤ N || throw(ArgumentError(
        "direction $dim is outside the grid's 1:$N" *
        (dim ≤ NA ? " (it is a batch axis, which carries no stencil)" : ""),
    ))
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    ntuple(d -> size(field, d), Val(N)) == size_tuple(grid) || throw(DimensionMismatch(
        "field's leading $N axes $(ntuple(d -> size(field, d), Val(N))) do not match the grid " *
        "$(size_tuple(grid))",
    ))
    return nothing
end

# A Cartesian metric is the identity, so the derivative with respect to distance is already the one
# `apply_stencil!` wrote and there is nothing to divide by.
@inline _scale_by_metric!(
    out::AbstractArray{S,NA}, ::StructuredGrid{G,T,N}, ::Int, _masked,
) where {S,G<:Geometry.AbstractCartesianGeometry,T,N,NA} = out

function _scale_by_metric!(
    out::AbstractArray{S,NA}, grid::StructuredGrid{G,T,N}, dim::Int, masked,
) where {S,G,T,N,NA}
    geo = grid_geometry(grid)
    floor_ = Discretization.metric_floor(geo)
    sz = size_tuple(grid)
    # No scale factor depends on longitude, so it is constant along axis 1 whichever direction is
    # differenced: computed once per remaining index, then swept along the contiguous axis. Any axis-1
    # coordinate serves for the point it is evaluated at, so the first one is used.
    @inbounds x1 = first(coordinates(grid, 1))
    rest = CartesianIndices(ntuple(d -> sz[d + 1], Val(N - 1)))
    # No scale factor depends on the batch either, so `h` is solved once per SPATIAL index and reused
    # across the batch — strictly less work than the per-slice loop this replaces, which re-solved it
    # for every slice.
    #
    # Addressed linearly rather than by index tuple: `rest` walks the spatial slabs in column-major
    # order, so slab `p` starts at `(p-1)*sz[1]` and batch element `b` a whole grid further on. Writing
    # `out[i, tr..., Tuple(Ib)...]` instead costs a nested splat the compiler will not see through —
    # measured at 960 bytes per call.
    ncell = prod(sz)
    nb = length(out) ÷ ncell
    if IndexStyle(out) === IndexLinear() && !Base.has_offset_axes(out)
        @inbounds for (p, Ir) in enumerate(rest)
            pt = (x1, ntuple(d -> T(coordinates(grid, d + 1)[Ir[d]]), Val(N - 1))...)
            h = Geometry.scale_factors(geo, pt)[dim]
            sbase = (p - 1) * sz[1]
            if abs(h) ≤ floor_
                for b in 0:(nb - 1), i in 1:sz[1]
                    out[b * ncell + sbase + i] = masked
                end
            else
                inv_h = inv(h)
                for b in 0:(nb - 1), i in 1:sz[1]
                    out[b * ncell + sbase + i] *= inv_h
                end
            end
        end
        return out
    end
    # Anything that does not index linearly — an offset array, a strided view — asks the array for its
    # own indexing instead.
    batch = CartesianIndices(ntuple(d -> size(out, N + d), Val(NA - N)))
    @inbounds for Ir in rest
        pt = (x1, ntuple(d -> T(coordinates(grid, d + 1)[Ir[d]]), Val(N - 1))...)
        h = Geometry.scale_factors(geo, pt)[dim]
        tr = Tuple(Ir)
        for Ib in batch
            tb = Tuple(Ib)
            if abs(h) ≤ floor_
                for i in 1:sz[1]
                    out[i, tr..., tb...] = masked
                end
            else
                inv_h = inv(h)
                for i in 1:sz[1]
                    out[i, tr..., tb...] *= inv_h
                end
            end
        end
    end
    return out
end

"""
    axis_stencils(grid, dim; order=1, nodes=order+1) -> (indices, weights)

[`Discretization.axis_stencils`](@ref) for direction `dim` of `grid`, taking that direction's axis and
wrap period from the grid.

The table depends only on the grid, not on any field, so a caller differencing many fields along the
same direction should build it once and hand it to the `(out, field, grid, indices, weights, dim)`
form — the `(out, field, grid, dim)` form above rebuilds it on every call.
"""
function Discretization.axis_stencils(
    grid::StructuredGrid{G,T,N}, dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
) where {G,T,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    return Discretization.axis_stencils(
        coordinates(grid, dim), order, nodes;
        period = isperiodic(grid, dim) ? period(grid, dim) : nothing,
    )
end

"""
    apply_stencil!(out, field, grid, indices, weights, dim; order=1, active_only=true,
                   masked=zero, policy=BlankMasked(), backend=nothing) -> out

Apply a stencil table built by [`Discretization.axis_stencils`](@ref) — the mask, the wrap period and
the axis all come from `grid`, which is what the bare `(indices, weights)` form cannot do.

Because the axis comes too, **any mask policy works here**. The bare form has no axis to rebuild a
window from at a mask edge, so it accepts only `BlankMasked`; a caller wanting `ReduceInRun` on a
masked grid would otherwise have to give up the table and pay its rebuild on every call.

This is the form to use in a loop over fields: the table is the same for all of them, and building it
is the one part of the work that does not depend on the field.
"""
function Discretization.apply_stencil!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::StructuredGrid{G,T,N},
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix, dim::Integer;
    order::Integer = 1, active_only::Bool = true, masked = zero(S), backend = nothing,
    policy::Discretization.AbstractMaskPolicy = Discretization.BlankMasked(), scratch = nothing,
) where {S,G,T,N,NA}
    _check_batched(out, field, grid, Val(N), Int(dim))
    # Both the mask and the period are `Union{Nothing, …}` if resolved with a ternary, and a small
    # union crossing a keyword boundary boxes — 96 bytes per call on a path whose whole purpose is to
    # allocate nothing. Branching instead leaves every leaf concretely typed.
    msk = mask(grid)
    if active_only && !(msk isa AllActive)
        return _apply_tbl!(out, field, grid, indices, weights, Int(dim), Int(order), msk, masked,
                           backend, policy, scratch)
    end
    return _apply_tbl!(out, field, grid, indices, weights, Int(dim), Int(order), nothing, masked,
                       backend, policy, scratch)
end

@inline function _apply_tbl!(
    out, field, grid::StructuredGrid, indices, weights, dim::Int, order::Int, msk, masked, backend,
    policy, scratch,
)
    x = coordinates(grid, dim)
    return isperiodic(grid, dim) ?
        Discretization.apply_stencil!(out, field, x, indices, weights, dim; order = order,
                                      period = period(grid, dim), mask = msk, masked = masked,
                                      backend = backend, policy = policy, scratch = scratch) :
        Discretization.apply_stencil!(out, field, x, indices, weights, dim; order = order,
                                      period = nothing, mask = msk, masked = masked,
                                      backend = backend, policy = policy, scratch = scratch)
end

"""
    derivative!(out, field, grid, indices, weights, dim; order=1, active_only=true, masked=zero,
                policy=BlankMasked(), backend=nothing) -> out

[`Discretization.derivative!`](@ref) from a table the caller holds — the same reuse as the `apply_stencil!` form
above, for the entry point a geometry-aware caller actually uses.
"""
function Discretization.derivative!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::StructuredGrid{G,T,N},
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix, dim::Integer;
    order::Integer = 1, active_only::Bool = true, masked = zero(S), backend = nothing,
    policy::Discretization.AbstractMaskPolicy = Discretization.BlankMasked(), scratch = nothing,
) where {S,G,T,N,NA}
    Discretization.apply_stencil!(out, field, grid, indices, weights, dim; order = order,
                                  active_only = active_only, masked = masked, backend = backend,
                                  policy = policy, scratch = scratch)
    return _scale_by_metric!(out, grid, Int(dim), masked)
end

# ---------------------------------------------------------------------------
# Curvilinear Grid
# ---------------------------------------------------------------------------

"""
    CurvilinearGrid{T, G, N, TP, C, MA, B}

Curvilinear grid whose cell-center coordinates are `N`-dimensional arrays (e.g. an orthogonal
curvilinear mesh). `coordinates` holds one `N`-D cell-center array per direction and `corners` the
matching cell-vertex arrays, each one larger in every direction.

At `N = 2` the cell `measure` is computed from those corners as the exact quadrilateral area rather
than by a cell-center spacing approximation. In any other dimension the measure is the caller's to
supply: the corner-area kernel is a genuinely 2-D algorithm, not a 2-D special case of an N-D one.

# Type parameters
- `T`: coordinate float type. `G<:AbstractGeometry{T}` is tied to it (a mismatched-eltype geometry is
  a type error, not a silent promotion) — hence `T` precedes `G` (Julia forbids the forward
  reference `G<:AbstractGeometry{T}, T` needed to keep the `{G,T}` order).
- `N`: number of coordinate directions.
- `C`: tuple type shared by the center and corner coordinate arrays — a mesh's own coordinate arrays
  are legitimately almost always the same concrete type.
- `MA`: array type of the derived `measure` field — independent of `C`, since it is a computed field
  with no reason to match the coordinate arrays' storage type.
- `B`: array type of the active `mask`.
"""
struct CurvilinearGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T},
    N,
    TP<:NTuple{N,AbstractTopology},
    C<:NTuple{N,AbstractArray{T,N}},
    MA<:AbstractArray{T,N},
    B<:AbstractArray{Bool,N},
} <: AbstractCurvilinearGrid{G, T}
    geometry::G
    coordinates::C            # cell-center coordinate array per direction
    corners::C                # cell-vertex coordinate array per direction, one larger in each
    measure::MA               # cell measure
    mask::B                   # active mask (true = active/included)
    topology::TP              # per-direction closure (singletons: no storage)
    period::NTuple{N,T}       # wrap length per direction; meaningless where Bounded
    stats::NTuple{N,AxisStats{T}}   # span per direction; gaps are undefined without axes
end

@inline topology(grid::CurvilinearGrid) = getfield(grid, :topology)
@inline period(grid::CurvilinearGrid, d::Integer) =
    @inbounds getfield(grid, :period)[_checked_direction(getfield(grid, :period), d)]

@inline function _raw_coords(
    grid::CurvilinearGrid{T,G,N}, I::Vararg{Integer,N},
) where {T,G,N}
    c = coordinates(grid)
    @boundscheck checkbounds(c[1], I...)
    return ntuple(d -> @inbounds(c[d][I...]), Val(N))
end

"""
    corners(grid::CurvilinearGrid) -> NTuple{N,AbstractArray}
    corners(grid::CurvilinearGrid, d::Integer) -> AbstractArray

The cell-vertex coordinate arrays — one larger than [`coordinates`](@ref) in every direction, and in
the same direction order.
"""
@inline corners(grid::CurvilinearGrid) = getfield(grid, :corners)
@inline corners(grid::CurvilinearGrid, d::Integer) = @inbounds corners(grid)[d]

"""
    corner_coords(grid::CurvilinearGrid, I...) -> NamedTuple
    corner_coords(S, grid::CurvilinearGrid, I...) -> S

Vertex `I` of the cell-vertex array, named by the geometry exactly as [`coords`](@ref) names
cell centers.
"""
@inline function corner_coords(
    grid::CurvilinearGrid{T,G,N}, I::Vararg{Integer,N},
) where {T,G,N}
    return Geometry.named_point(grid_geometry(grid), _raw_corner_coords(grid, I...))
end

@inline function corner_coords(
    ::Type{S}, grid::CurvilinearGrid{T,G,N}, I::Vararg{Integer,N},
) where {S,T,G,N}
    return Geometry.build_point(S, coordinate_names(grid), _raw_corner_coords(grid, I...))
end

@inline function _raw_corner_coords(
    grid::CurvilinearGrid{T,G,N}, I::Vararg{Integer,N},
) where {T,G,N}
    k = corners(grid)
    @boundscheck checkbounds(k[1], I...)
    return ntuple(d -> @inbounds(k[d][I...]), Val(N))
end

# ---------------------------------------------------------------------------
# Curvilinear grid construction: corner-based exact quadrilateral cell areas
# ---------------------------------------------------------------------------

# Adapt a coordinate matrix to element type `T`, preserving the concrete array type (`similar`, not
# `Matrix{T}`) and copying only when the eltype actually differs.
_to_arr(::Type{T}, A::AbstractArray{T}) where {T<:AbstractFloat} = A
_to_arr(::Type{T}, A::AbstractArray) where {T<:AbstractFloat} = copyto!(similar(A, T), A)

"""
    _centers_to_corners(C) -> K

Reconstruct a cell-vertex array one larger in every direction from the `N`-D cell-center array `C`, by
averaging the (up to `2^N`) surrounding centers with a linearly-extrapolated one-cell ghost ring, so
the true domain-boundary vertices land a half-cell outside the outermost centers. Used only when the
caller does not supply explicit corner arrays; requires at least 2 centers across every direction.

Unlike the corner-area kernel this is dimension-generic: it is a multilinear midpoint, and the ghost
ring is a per-direction linear extension of it.
"""
function _centers_to_corners(C::AbstractArray{T,N}) where {T<:AbstractFloat, N}
    all(≥(2), size(C)) || throw(ArgumentError(
        "auto-deriving curvilinear cell corners needs at least 2 centers across every direction " *
        "(got $(size(C))); supply the `corners` arrays explicitly for a smaller grid",
    ))
    # The padded ghost ring is indexed rather than materialized: `_ghosted` returns the same value a
    # padded copy would hold, so the vertex pass reads straight from `C` and only the result is
    # allocated (one array instead of two, and one pass over the data instead of three).
    K = similar(C, T, (size(C) .+ 1)...)
    shifts = CartesianIndices(ntuple(_ -> 0:1, Val(N)))
    scale = one(T) / T(2^N)
    @inbounds for ci in CartesianIndices(K)
        I = Tuple(ci)
        acc = zero(T)
        for s in shifts
            acc += _ghosted(C, ntuple(d -> I[d] - 1 + s[d], Val(N)))
        end
        K[ci] = acc * scale
    end
    return K
end

# Value of the one-cell linearly-extrapolated ghost ring around `C` at the (possibly out-of-range)
# center index `I`. Each out-of-range direction contributes its own linear extrapolation `2·edge − inner`
# taken with every other direction clamped, and where several are out at once those contributions add
# with the shared clamped value counted once — the standard halo fill, exact for a field linear in each
# direction. Written as a flat loop rather than a recursion over directions: a recursive form cannot
# inline, and its intermediate index tuples then box (measured at 992 bytes per call).
@inline function _ghosted(C::AbstractArray{T,N}, I::NTuple{N,Int}) where {T,N}
    sz = size(C)
    cl = ntuple(d -> clamp(I[d], 1, sz[d]), Val(N))
    nout = 0
    acc = zero(T)
    @inbounds begin
        for d in 1:N
            (1 ≤ I[d] ≤ sz[d]) && continue
            nout += 1
            inner = I[d] < 1 ? 2 : sz[d] - 1
            acc += T(2) * C[cl...] - C[ntuple(k -> k == d ? inner : cl[k], Val(N))...]
        end
        nout == 0 && return C[cl...]
        return acc - T(nout - 1) * C[cl...]
    end
end



# Exact planar quadrilateral area (shoelace) over the (Nx+1)×(Ny+1) corner arrays.
function _corner_areas(
    ::G, xc::AbstractMatrix{T}, yc::AbstractMatrix{T}, Nx::Integer, Ny::Integer;
    backend = nothing,
) where {T<:AbstractFloat, G<:Geometry.AbstractCartesianGeometry{T}}
    areas = similar(xc, T, Nx, Ny)
    Execution.run_chunks(Int(Ny), backend) do rows
    @inbounds for j in rows, i in 1:Nx
        # Cell (i,j) has vertices (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1) (counter-clockwise in index space).
        x1 = xc[i, j];         y1 = yc[i, j]
        x2 = xc[i+1, j];       y2 = yc[i+1, j]
        x3 = xc[i+1, j+1];     y3 = yc[i+1, j+1]
        x4 = xc[i, j+1];       y4 = yc[i, j+1]
        areas[i, j] = T(0.5) * abs(x1 * (y2 - y4) + x2 * (y3 - y1) + x3 * (y4 - y2) + x4 * (y1 - y3))
    end
    end
    return areas
end

function _fill_dir_row!(
    buf::AbstractVector{NTuple{3,T}}, λc::AbstractMatrix{T}, φc::AbstractMatrix{T}, j::Int, nk::Int,
) where {T}
    @inbounds for i in 1:nk
        sinλ, cosλ = sincos(λc[i, j])
        sinφ, cosφ = sincos(φc[i, j])
        buf[i] = (cosφ * cosλ, cosφ * sinλ, sinφ)
    end
    return buf
end

# Exact spherical quadrilateral area, as the two triangles (p1,p2,p3) and (p1,p3,p4).
#
# Each corner's unit vector is computed ONCE — every interior vertex is shared by four cells and,
# within a cell, by both triangles, so deriving it per triangle would repeat the same trig up to
# eight times over. Only two ROWS of them are ever live at a time, though: row j is finished as soon
# as row j+1's cells are done. Holding the whole `(Nx+1)×(Ny+1)` field instead costs O(Nx·Ny) for no
# extra reuse.
function _corner_areas(
    geometry::G, λc::AbstractMatrix{T}, φc::AbstractMatrix{T}, Nx::Integer, Ny::Integer;
    backend = nothing,
) where {T<:AbstractFloat, G<:Geometry.AbstractSphericalGeometry{T}}
    nk = Nx + 1
    R2 = Geometry.radius(geometry)^2
    areas = similar(λc, T, Nx, Ny)
    # Chunked over ROWS, each chunk with its own two-row buffer: a chunk re-derives its first row
    # rather than sharing one with its neighbour, which costs one extra row of trig per chunk and
    # keeps the chunks independent.
    Execution.run_chunks(Int(Ny), backend) do rows
        lo = Vector{NTuple{3,T}}(undef, nk)
        hi = Vector{NTuple{3,T}}(undef, nk)
        _fill_dir_row!(lo, λc, φc, first(rows), nk)
        @inbounds for j in rows
            _fill_dir_row!(hi, λc, φc, j + 1, nk)
            for i in 1:Nx
                d1 = lo[i]; d2 = lo[i + 1]; d3 = hi[i + 1]; d4 = hi[i]
                areas[i, j] = R2 * (Geometry.spherical_excess(d1, d2, d3) +
                                    Geometry.spherical_excess(d1, d3, d4))
            end
            lo, hi = hi, lo      # row j+1 becomes the next cell row's lower edge
        end
    end
    return areas
end

"""
    CurvilinearGrid(geometry, coords..., mask; measure=nothing, corners=nothing, …)
    CurvilinearGrid(geometry, coords..., measure, mask; corners=nothing, …)

Build a curvilinear grid in **any** number of directions from one `N`-D cell-center coordinate array
per direction. `mask` is the trailing array; a `measure` array before it is used verbatim (common when
a dataset ships its own cell areas), and may equally be given as the `measure` keyword.

With no measure supplied, one is computed from the cell-vertex arrays — **at `N = 2` only**, as the
exact quadrilateral cell area. Spherical cells use the exact spherical-quadrilateral area, the
spherical excess of the two triangles through the cell's four corner directions (see
[`Geometry.spherical_excess`](@ref)); Cartesian cells the exact planar shoelace area. That kernel is a 2-D algorithm
rather than the 2-D case of an N-D one, so in any other dimension the measure must be given.

Pass `corners` (a tuple of arrays, each one larger than the centers in every direction) for exact
cell vertices, e.g. from the source mesh's own vertex grid; otherwise they are reconstructed from the
centers per direction (see [`_centers_to_corners`](@ref)), which requires at least 2 cells across.

`periodic` is a `Bool` (applied to direction 1) or an `NTuple{N,Bool}`. When omitted, direction-1
periodicity is auto-detected the same way as [`StructuredGrid`](@ref) (full-circle spherical
longitude), and every other direction is bounded.
"""
function CurvilinearGrid(geometry::Geometry.AbstractGeometry, args...; kwargs...)
    coords, measure, mask = _split_curvilinear_args(args)
    return _curvilinear_grid(geometry, coords, measure, mask; kwargs...)
end

# Optional trailing `mask`, optionally preceded by a `measure`; everything before them is a coordinate
# array. The mask is recognised by its element type rather than by position, so leaving it out is
# unambiguous — and leaving it out is what a fully active grid should do, since `AllActive` costs one
# size tuple where a dense all-true mask costs a load and a branch per cell.
function _split_curvilinear_args(args::Tuple)
    hasmask = !isempty(args) && last(args) isa AbstractArray{Bool}
    mask = hasmask ? last(args) : nothing
    rest = hasmask ? Base.front(args) : args
    isempty(rest) && throw(ArgumentError("a CurvilinearGrid needs at least one coordinate array"))
    # A trailing real array is the measure when there is one more array than each is dimensional:
    # `N` coordinates plus it. Otherwise it is the last coordinate.
    if length(rest) ≥ 2 && last(rest) isa AbstractArray{<:Real} &&
       !(last(rest) isa AbstractArray{Bool}) && length(rest) - 1 == ndims(last(rest))
        return (Base.front(rest), last(rest), mask)
    end
    return (rest, nothing, mask)
end

# No mask means every cell participates, which is a size rather than an array — the same default
# `StructuredGrid` has always had. Resolved in its own method rather than with a `nothing` branch
# inside the one below: that would leave `mask` a small union and cost the whole constructor its
# specialization, which measured as the suite taking three times as long.
_curvilinear_grid(
    geometry::Geometry.AbstractGeometry, coords::NTuple{N,AbstractArray}, measure_pos, ::Nothing;
    kwargs...,
) where {N} = _curvilinear_grid(geometry, coords, measure_pos, AllActive(size(first(coords)));
                                kwargs...)

function _curvilinear_grid(
    geometry::G, coords::NTuple{N,AbstractArray}, measure_pos, mask::AbstractArray{Bool,N};
    corners = nothing, measure = nothing,
    x_corner = nothing, y_corner = nothing,
    topology = nothing, period = nothing, periodic = nothing, backend = nothing,
) where {N, T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    N == ndims(mask) || throw(ArgumentError(
        "got $N coordinate arrays for a $(ndims(mask))-dimensional mask",
    ))
    all(c -> ndims(c) == N, coords) || throw(ArgumentError(
        "every coordinate array must be $N-dimensional, got $(map(ndims, coords))",
    ))
    all(c -> size(c) == size(mask), coords) || throw(ArgumentError(
        "coordinate arrays and mask must have the same size; got $(map(size, coords)) and $(size(mask))",
    ))
    centers = ntuple(d -> _to_arr(T, coords[d]), Val(N))

    given_corners = corners === nothing && N == 2 && !(x_corner === nothing && y_corner === nothing) ?
        (x_corner, y_corner) : corners
    kc = if given_corners === nothing
        ntuple(d -> _centers_to_corners(centers[d]), Val(N))
    else
        length(given_corners) == N || throw(ArgumentError(
            "expected $N corner arrays, got $(length(given_corners))",
        ))
        want = size(mask) .+ 1
        ntuple(Val(N)) do d
            k = _to_arr(T, given_corners[d])
            size(k) == want || throw(ArgumentError(
                "corner array $d must be $(want) (one larger than the centers in every direction); " *
                "got $(size(k))",
            ))
            k
        end
    end

    m = measure_pos === nothing ? measure : measure_pos
    meas = if m !== nothing
        size(m) == size(mask) || throw(ArgumentError(
            "measure size $(size(m)) does not match the coordinate arrays' $(size(mask))",
        ))
        _to_arr(T, m)
    elseif N == 2
        _corner_areas(geometry, kc[1], kc[2], size(mask, 1), size(mask, 2); backend = backend)
    else
        throw(ArgumentError(
            "a $N-dimensional CurvilinearGrid has no measure to derive: the corner-area kernel is a " *
            "2-D algorithm (exact quadrilateral area), not the 2-D case of an N-D one. Pass the cell " *
            "measure explicitly.",
        ))
    end

    tp = _curvilinear_topology(geometry, centers[1], topology === nothing ? periodic : topology)
    prd = _curvilinear_periods(geometry, centers, tp, period)
    return CurvilinearGrid{T, G, N, typeof(tp), typeof(centers), typeof(meas), typeof(mask)}(
        geometry, centers, kc, meas, mask, tp, prd,
        ntuple(d -> _axis_stats(centers[d]), Val(N)),
    )
end

"""
    rotate(grid::StructuredGrid, rot) -> CurvilinearGrid
    unrotate(grid::StructuredGrid, rot) -> CurvilinearGrid

The same mesh with its coordinates expressed in the other frame of [`Geometry.PoleRotation`](@ref)
`rot` — `unrotate` being the usual direction, taking a rotated-pole grid's rectilinear `(λ′, φ′)` axes
to the geographic coordinates of each cell.

The result is curvilinear because that is what it is: a rotated lat–lon mesh is logically rectangular
and geometrically warped, and only its own frame's axes are separable.

Two things carry over rather than being recomputed. The **cell measure** is exact, because a rotation is
an isometry of the sphere — recomputing it from the rotated corners would only add roundoff. The
**index topology** is too: it is the same mesh with the same neighbours, so a direction that wrapped
still wraps. Longitude remains an angle mod `2π` in either frame, so the wrap length is unchanged.
"""
rotate(grid::AbstractStructuredGrid, rot::Geometry.PoleRotation) =
    _reframe(Geometry.rotate, grid, rot)
unrotate(grid::AbstractStructuredGrid, rot::Geometry.PoleRotation) =
    _reframe(Geometry.unrotate, grid, rot)

function _reframe(
    pointwise::F, grid::StructuredGrid{G,T,2}, rot::Geometry.PoleRotation,
) where {F, T, G<:Geometry.AbstractSphericalGeometry{T}}
    λ, φ = coordinates(grid, 1), coordinates(grid, 2)
    _first(a, b) = T(pointwise(rot, a, b)[1])
    _second(a, b) = T(pointwise(rot, a, b)[2])
    centers = ([_first(a, b) for a in λ, b in φ], [_second(a, b) for a in λ, b in φ])
    fλ, fφ = Discretization.faces(λ), Discretization.faces(φ)
    corners = ([_first(a, b) for a in fλ, b in fφ], [_second(a, b) for a in fλ, b in fφ])
    tp = topology(grid)
    msk = mask(grid)
    return CurvilinearGrid{T, G, 2, typeof(tp), typeof(centers), Matrix{T}, typeof(msk)}(
        grid_geometry(grid), centers, corners, measure_array(grid), msk, tp,
        (period(grid, 1), period(grid, 2)),
        ntuple(d -> _axis_stats(centers[d]), Val(2)),
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
- `C`: tuple type of the per-direction node-coordinate vectors (a node set's own coordinate vectors
  are legitimately almost always the same concrete type).
- `VA`: vector type of the derived `measure` field — independent of `C`, since it is frequently a
  computed field (Voronoi tessellation) with no reason to match the coordinate vectors' storage type.
- `B`: mask storage type.
- `VN`/`VP`: CSR neighbor-list and offset storage types, independent of each other. Their element
  type is a free `Integer`, so a large mesh can carry `Int32` indices (half the memory and bandwidth
  of `Int64`, and the width GPU kernels want) without needing a separate grid type.

Neighbor adjacency is stored CSR-style (flat `neighbor_nbrs` + `neighbor_ptr` offsets, node `t` owns
`neighbor_ptr[t]:neighbor_ptr[t+1]-1`) rather than as a vector of per-node vectors — the data is
immutable after construction, so there's no reason to pay for `Nnodes` separately-heap-allocated
`Vector`s (cache-unfriendly pointer-chasing, one allocation per node) when one contiguous block (two
allocations total) holds the same information.
"""
struct UnstructuredGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T},
    N,
    C<:NTuple{N,AbstractVector{T}},
    VA<:AbstractVector{T},
    B<:AbstractVector{Bool},
    TP<:NTuple{N,AbstractTopology},
    VN<:AbstractVector{<:Integer},
    VP<:AbstractVector{<:Integer},
} <: AbstractUnstructuredGrid{G, T}
    geometry::G
    coordinates::C     # node coordinate vector per direction (Nnodes)
    measure::VA        # control-volume size of each node (Nnodes)
    mask::B            # active mask (true = active/included) (Nnodes)
    neighbor_nbrs::VN  # flat neighbor-index array (CSR)
    neighbor_ptr::VP   # CSR offsets, length Nnodes+1
    topology::TP       # per-direction closure of the enclosing domain (singletons: no storage)
    period::NTuple{N,T}       # wrap length per direction; meaningless where Bounded
    stats::NTuple{N,AxisStats{T}}   # span per direction; gaps are undefined for scattered nodes
end

@inline topology(grid::UnstructuredGrid) = getfield(grid, :topology)

"""
    UnstructuredGrid(geometry, coords::Tuple, measure, mask[, neighbor_nbrs, neighbor_ptr]; periodic, period)
    UnstructuredGrid(geometry, x, y, measure, mask[, neighbor_nbrs, neighbor_ptr]; periodic, period)

Build a node grid in **any** number of directions from one coordinate vector per direction and CSR
adjacency. Coordinates come as a tuple; the two-direction case may pass `x, y` positionally.

Omitting the CSR pair gives a grid with no adjacency — every node reports zero neighbours, which is
enough for scattered-point spectral methods that never query it. Real-space neighbourhood operations
need adjacency: build it (e.g. through the k-d-tree constructor below) and pass it in, or query by
distance with `Connectivity.neighbors_within`, which reads coordinates rather than edges.

`periodic` declares that the enclosing domain wraps in a direction, and `period` gives the wrap
length there. A scattered point set carries no axis to infer this from, so both are explicit —
except on a sphere, where longitude wraps at 2π by construction and is the default. See
[`isperiodic`](@ref) and [`period`](@ref).
"""
function UnstructuredGrid(
    geometry::G, coords::NTuple{N,AbstractVector}, measure::AbstractVector,
    mask::AbstractVector{Bool} = AllActive((length(first(coords)),)), nbrs = nothing, ptr = nothing;
    periodic = nothing, period = nothing,
) where {N, T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    c = ntuple(d -> _to_axis(T, coords[d]), N)
    n = length(c[1])
    all(v -> length(v) == n, c) || throw(ArgumentError(
        "coordinate vectors must have the same length; got $(map(length, c))",
    ))
    length(mask) == n || throw(ArgumentError("coordinates and mask must have the same length"))
    length(measure) == n || throw(ArgumentError("coordinates and measure must have the same length"))
    # No CSR pair given: every node reports zero neighbours, which `ptr` all-ones expresses.
    p = ptr === nothing ? ones(Int, n + 1) : ptr
    nb = ptr === nothing ? Int[] : nbrs
    length(p) == n + 1 || throw(ArgumentError(
        "neighbor_ptr must have length Nnodes+1 = $(n + 1); got $(length(p))",
    ))
    per, prd = _node_periodicity(geometry, Val(N), periodic, period)
    m = _to_axis(T, measure)
    return UnstructuredGrid{
        T, G, N, typeof(c), typeof(m), typeof(mask), typeof(per), typeof(nb), typeof(p),
    }(geometry, c, m, mask, nb, p, per, prd, ntuple(d -> _axis_stats(c[d]), Val(N)))
end

# Two-direction convenience forms. Node coordinates are all `AbstractVector`s, so — unlike a
# `CurvilinearGrid`, where `ndims(mask)` counts them — nothing in the types says how many of a run of
# vectors are coordinates: `(x, y, mask)` and `(x, measure, mask)` are the same call. Hence the tuple
# above for the general case, and explicit methods for the two-direction one.
UnstructuredGrid(
    geometry::Geometry.AbstractGeometry, x::AbstractVector, y::AbstractVector,
    measure::AbstractVector, mask::AbstractVector{Bool},
    nbrs::AbstractVector{<:Integer}, ptr::AbstractVector{<:Integer}; kwargs...,
) = UnstructuredGrid(geometry, (x, y), measure, mask, nbrs, ptr; kwargs...)

UnstructuredGrid(
    geometry::Geometry.AbstractGeometry, x::AbstractVector, y::AbstractVector,
    measure::AbstractVector, mask::AbstractVector{Bool}; kwargs...,
) = UnstructuredGrid(geometry, (x, y), measure, mask; kwargs...)

# Longitude on a sphere wraps at 2π whatever the point set looks like; a Cartesian box has no
# intrinsic period, so wrapping there is opt-in and the length must be given.
function _node_periodicity(
    ::Union{Geometry.AbstractSphericalGeometry{T},Geometry.AbstractEllipsoidalGeometry{T}},
    ::Val{N}, periodic, period,
) where {N, T<:AbstractFloat}
    per = _as_topology(Val(N), periodic === nothing ?
        ntuple(d -> d == 1, Val(N)) : periodic)
    prd = period === nothing ? ntuple(d -> d == 1 ? T(2π) : zero(T), Val(N)) :
        _node_period_tuple(Val(N), T, period, per)
    return per, prd
end

function _node_periodicity(
    ::Geometry.AbstractCartesianGeometry{T}, ::Val{N}, periodic, period,
) where {N, T<:AbstractFloat}
    per = _as_topology(Val(N), periodic === nothing ? ntuple(_ -> false, Val(N)) : periodic)
    any(_is_periodic, per) || return per, ntuple(_ -> zero(T), Val(N))
    period === nothing && throw(ArgumentError(
        "a periodic Cartesian node grid needs an explicit `period` (the wrap length per direction): " *
        "scattered points carry no axis to infer it from",
    ))
    return per, _node_period_tuple(Val(N), T, period, per)
end

function _node_period_tuple(::Val{N}, ::Type{T}, period, per) where {N,T}
    prd = period isa Real ? ntuple(d -> d == 1 ? T(period) : zero(T), Val(N)) : NTuple{N,T}(period)
    all(d -> !_is_periodic(per[d]) || prd[d] > 0, 1:N) || throw(ArgumentError(
        "period must be positive in every periodic direction; got $prd for topology=$per",
    ))
    return prd
end

"""
    period(grid, d) -> T

Wrap length of coordinate direction `d`, meaningful only where [`isperiodic`](@ref) holds.
"""
@inline period(grid::UnstructuredGrid, d::Integer) =
    @inbounds getfield(grid, :period)[_checked_direction(getfield(grid, :period), d)]


# ---------------------------------------------------------------------------
# Unstructured grid construction: k-d-tree adjacency + (optional) Voronoi areas
# ---------------------------------------------------------------------------
#
# Two extension hook points (fallbacks that throw until a consumer package loads the relevant
# weakdep and overrides these methods): a k-d-tree neighbor query (`NearestNeighbors.jl`) and a
# per-node Voronoi-cell area (`DelaunayTriangulation.jl` for Cartesian, `Quickhull.jl` for spherical,
# dispatched on the geometry type since each needs a different tessellation library).

"""
    _build_kdtree_neighbors(geometry, coords::Tuple; k=6, radius=nothing) -> (nbrs, ptr)

Extension hook: build CSR neighbor adjacency via a k-d tree. Overridden by
a consumer NearestNeighbors extension (load `using NearestNeighbors`). `radius`, if given,
switches to an all-neighbors-within-`radius` query (mutually exclusive with `k`); `radius` is in the
grid's physical distance units (`Geometry.distance` — meters for `SphericalGeometry`, geometry's own
units for `CartesianGeometry`), NOT a raw chord/angle.
"""
function _build_kdtree_neighbors(
    geometry::Geometry.AbstractGeometry, coords::NTuple{D,AbstractVector}; _...,
) where {D}
    throw(ArgumentError(
        "k-d-tree neighbor construction requires NearestNeighbors.jl — run `using NearestNeighbors` " *
        "(or build adjacency explicitly and pass it to `UnstructuredGrid` alongside the coordinates).",
    ))
end

# Every architecture caches its per-direction reductions, so the span accessors are reads rather than
# `extrema` over the coordinates. They sit under the search-radius bound a query evaluates, which is why
# it matters that they are `O(1)` and not merely correct.
const _StatGrid = Union{StructuredGrid,CurvilinearGrid,UnstructuredGrid}

@inline axis_stats(grid::Union{CurvilinearGrid,UnstructuredGrid}) = getfield(grid, :stats)
@inline axis_stats(grid::Union{CurvilinearGrid,UnstructuredGrid}, d::Integer) =
    @inbounds axis_stats(grid)[d]

@inline bounds(grid::_StatGrid, d::Integer) =
    (st = axis_stats(grid, d); (st.min_value, st.max_value))
@inline function extent(grid::_StatGrid, d::Integer)
    st = axis_stats(grid, d)
    return st.max_value - st.min_value
end
@inline origin(grid::_StatGrid, d::Integer) = axis_stats(grid, d).first_value

# ---------------------------------------------------------------------------
# The Euclidean space a spatial index searches in
# ---------------------------------------------------------------------------

"""
    AbstractEmbedding

How a grid's cell centres sit in the Euclidean space an index searches, and therefore what a physical
radius means there. A **type**, for the reason stencils are: the conversion is applied once per query,
so a runtime tag would leave it unresolved and put a branch — and a boxed radius — in the hot path.

[`CartesianEmbedding`](@ref), [`ChordEmbedding`](@ref), [`ArcEmbedding`](@ref).
"""
abstract type AbstractEmbedding end

"""
    CartesianEmbedding()

The coordinates themselves, replicated at the periodic images. A radius passes through unchanged.
"""
struct CartesianEmbedding <: AbstractEmbedding end

"""
    ChordEmbedding()

The embedded distance already is the metric, or a lower bound on it: `(λ, φ, r)` on a sphere, where the
metric is the 3-D chord, and geodetic coordinates on a spheroid, where the ECEF chord is at most the
geodesic. A radius passes through unchanged; under-approximating means the query over-returns, which is
what an index is allowed to do.
"""
struct ChordEmbedding <: AbstractEmbedding end

"""
    ArcEmbedding(R)

Points on the reference sphere of radius `R`, where the metric is the great-circle arc. A radius has to
be converted to the chord it subtends.
"""
struct ArcEmbedding{T<:AbstractFloat} <: AbstractEmbedding
    radius::T
end

"""
    embed_point(grid, p) -> NTuple

One coordinate tuple through the same transform [`embedded_points`](@ref) applies to the cell centres,
so a query seeded by a point searches the space the index was built in.
"""
function embed_point end

@inline embed_point(grid::AbstractGrid{G,T}, p::NTuple{D,Real}) where {G<:Geometry.AbstractCartesianGeometry,T,D} =
    ntuple(d -> T(p[d]), Val(D))

@inline function embed_point(grid::AbstractGrid{G,T}, p::NTuple{D,Real}) where {G<:Geometry.AbstractSphericalGeometry,T,D}
    c = Geometry.spherical_to_cartesian(grid_geometry(grid), p)
    return (T(c.x), T(c.y), T(c.z))
end

@inline function embed_point(grid::AbstractGrid{G,T}, p::NTuple{D,Real}) where {G<:Geometry.AbstractEllipsoidalGeometry,T,D}
    c = Geometry.geodetic_to_cartesian(grid_geometry(grid), p)
    return (T(c.x), T(c.y), T(c.z))
end

"""
    embedded_radius(embedding, r) -> T

A physical radius as a radius in the embedding.
"""
@inline embedded_radius(::CartesianEmbedding, r::T) where {T} = r
@inline embedded_radius(::ChordEmbedding, r::T) where {T} = r

# `2R·sin(σ/2)` is monotone only to `σ = π`, so an arc of an antipodal distance or more saturates at the
# diameter rather than turning back down.
@inline function embedded_radius(e::ArcEmbedding{T}, r::Real) where {T}
    σ = T(r) / e.radius
    return σ ≥ T(π) ? T(2) * e.radius : T(2) * e.radius * sin(σ / T(2))
end

"""
    _shift_set(periodic, period)

Offsets to replicate a point set by: `(0,)` in a non-wrapping direction, `(0, -L, +L)` in a wrapping
one. Zero first, so the originals occupy the first block of the replicated set.
"""
@inline _shift_set(p::Bool, L::T) where {T} = p ? (zero(T), -L, L) : (zero(T),)

"""
    _ghost_points(pts, periodic, period) -> (all_pts, nghost)

Replicate the `D × N` point matrix once per combination of periodic image offsets, originals in the
first `N` columns. A wrapping domain is then searched by an ordinary Euclidean query over the images.
"""
function _ghost_points(
    pts::AbstractMatrix{T}, periodic::NTuple{D,Bool}, period::NTuple{D,T},
) where {D,T}
    shifts = ntuple(d -> _shift_set(periodic[d], period[d]), Val(D))
    N = size(pts, 2)
    ng = prod(map(length, shifts))
    out = similar(pts, D, N * ng)
    g = 0
    for ci in CartesianIndices(map(eachindex, shifts))
        cols = (g * N + 1):((g + 1) * N)
        for d in 1:D
            δ = shifts[d][ci[d]]
            @views out[d, cols] .= pts[d, :] .+ δ
        end
        g += 1
    end
    return out, ng
end

"""
    _grid_points(grid) -> (raw, D)

Cell centres as a `D × n` matrix, in the grid's own coordinates.
"""
function _grid_points(grid::AbstractGrid{G,T}) where {G,T}
    msk = mask(grid)
    lin = LinearIndices(size(msk))
    D = length(_raw_coords(grid, Tuple(first(CartesianIndices(size(msk))))...))
    raw = Matrix{T}(undef, D, length(msk))
    @inbounds for ci in CartesianIndices(size(msk))
        p = _raw_coords(grid, Tuple(ci)...)
        for d in 1:D
            raw[d, lin[ci]] = p[d]
        end
    end
    return raw, D
end

function _grid_points(grid::UnstructuredGrid{T,G,N}) where {T,G,N}
    n = length(mask(grid))
    raw = Matrix{T}(undef, N, n)
    @inbounds for k in 1:n
        p = _raw_coords(grid, k)
        for d in 1:N
            raw[d, k] = p[d]
        end
    end
    return raw, N
end

"""
    embedded_points(grid) -> (pts, nghost, embedding)

The cell centres in the space an index searches, the number of periodic replications they carry, and
the [`AbstractEmbedding`](@ref) saying what a radius means there.

One definition, so every index searches the same space as every other and as the k-d-tree construction
path — the guarantee that an indexed query and a scan return the same cells rests on it.
"""
function embedded_points end

function embedded_points(grid::AbstractGrid{G,T}) where {G<:Geometry.AbstractCartesianGeometry,T}
    raw, D = _grid_points(grid)
    per = ntuple(d -> isperiodic(grid, d), D)
    prd = ntuple(d -> T(period(grid, d)), D)
    pts, ng = any(per) ? _ghost_points(raw, per, prd) : (raw, 1)
    return pts, ng, CartesianEmbedding()
end

# `spherical_to_cartesian` rather than the formula written again: at `(λ, φ, r)` the metric IS the
# Euclidean chord of this embedding. Longitude needs no ghost images — `λ` and `λ+2π` embed together.
function embedded_points(grid::AbstractGrid{G,T}) where {G<:Geometry.AbstractSphericalGeometry,T}
    geo = grid_geometry(grid)
    raw, D = _grid_points(grid)
    pts = Matrix{T}(undef, 3, size(raw, 2))
    @inbounds for k in axes(raw, 2)
        c = Geometry.spherical_to_cartesian(geo, ntuple(d -> raw[d, k], D))
        pts[1, k] = c.x; pts[2, k] = c.y; pts[3, k] = c.z
    end
    return pts, 1, D ≥ 3 ? ChordEmbedding() : ArcEmbedding(T(Geometry.radius(geo)))
end

function embedded_points(grid::AbstractGrid{G,T}) where {G<:Geometry.AbstractEllipsoidalGeometry,T}
    geo = grid_geometry(grid)
    raw, D = _grid_points(grid)
    pts = Matrix{T}(undef, 3, size(raw, 2))
    @inbounds for k in axes(raw, 2)
        c = Geometry.geodetic_to_cartesian(geo, ntuple(d -> raw[d, k], D))
        pts[1, k] = c.x; pts[2, k] = c.y; pts[3, k] = c.z
    end
    return pts, 1, ChordEmbedding()
end

# ---------------------------------------------------------------------------
# Cell-list index
# ---------------------------------------------------------------------------

"""
    CellListIndex

A uniform-bin spatial index over the embedded cell centres: points are bucketed by which bin of side `h`
they fall in, and a query visits the bins its ball can reach.

Three properties distinguish it from a tree, and all three are why it is the one that runs on a device:

  * it is **arrays only** — bin offsets and point ids — so `Adapt` moves it like any other field;
  * every point lands in **exactly one** bin, because periodicity wraps the bin coordinate rather than
    replicating the point, so a query emits each cell once and needs no candidate buffer to deduplicate
    into. That is what lets [`fold_candidates`](@ref) be a fold rather than a list;
  * bins are hashed into `O(n)` buckets, so the memory does not depend on `h`. A sphere binned at
    100 km would otherwise need `(2R/h)³ ≈ 2×10⁶` mostly empty cells, and far more as `h` shrinks.

Build it for the radius you intend to query at: `h` is that radius, so a query touches `3ᴰ` bins. A much
larger radius still works and costs `(2⌈r/h⌉+1)ᴰ` bins.
"""
struct CellListIndex{D,T<:AbstractFloat,E<:AbstractEmbedding,MT<:AbstractMatrix{T},VI<:AbstractVector{Int}}
    pts::MT                 # D × n embedded centres, one entry per cell, never replicated
    lo::NTuple{D,T}         # bin-lattice origin
    h::NTuple{D,T}          # bin width per direction
    nbins::NTuple{D,Int}    # per-direction bin count where the direction wraps; 0 otherwise
    wrap::NTuple{D,Bool}
    starts::VI              # nbucket + 1, CSR over buckets
    items::VI               # point ids, grouped by bucket
    embedding::E
end

@inline _nbuckets(ix::CellListIndex) = length(ix.starts) - 1

# Bins in a query window, saturating: a radius far wider than the bin side gives a per-direction reach
# whose plain product overflows `Int` and would wrap to a small — or negative — number.
@inline function _window_bins(reach::NTuple{D,Int}) where {D}
    m = 1
    @inbounds for d in 1:D
        v = reach[d]
        v ≤ 0 && return 0
        v > typemax(Int) ÷ m && return typemax(Int)
        m *= v
    end
    return m
end

# Bin coordinate of a point, wrapped where the direction does. A wrapping direction's width divides the
# period exactly, so `mod` lands on a real lattice rather than aliasing a partial bin onto bin zero.
@inline function _bin_of(ix::CellListIndex{D,T}, x::NTuple{D,T}) where {D,T}
    return ntuple(Val(D)) do d
        @inbounds b = Base.unsafe_trunc(Int, floor((x[d] - ix.lo[d]) / ix.h[d]))
        @inbounds ix.wrap[d] ? mod(b, ix.nbins[d]) : b
    end
end

# A cheap integer mix. Only the bucket assignment depends on it: a collision costs a few extra items to
# skip, never a wrong answer, because the bin coordinate is compared before a point is emitted.
@inline function _bin_hash(b::NTuple{D,Int}, nbucket::Int) where {D}
    h = UInt(0x9e3779b97f4a7c15)
    @inbounds for d in 1:D
        h = (h ⊻ (reinterpret(UInt, b[d] * 0x27220a95) + 0x165667b19e3779f9 + (h << 6) + (h >> 2)))
    end
    return Int(h % UInt(nbucket)) + 1
end

@inline function _bin_at(ix::CellListIndex{D,T}, k::Integer) where {D,T}
    return _bin_of(ix, ntuple(d -> @inbounds(ix.pts[d, k]), Val(D)))
end

"""
    cell_list(grid; ball) -> CellListIndex

Build a [`CellListIndex`](@ref) over `grid`'s cell centres, binned at side `ball` — the radius you mean
to query at. Needs no external package.
"""
function cell_list(grid::AbstractGrid{G,T}; ball::Real) where {G,T}
    h = T(ball)
    h > 0 || throw(ArgumentError("the bin side must be positive, got $ball"))
    all_pts, ng, embedding = embedded_points(grid)
    hemb = T(embedded_radius(embedding, h))
    hemb > 0 || throw(ArgumentError("radius $ball is degenerate in this embedding"))
    n = size(all_pts, 2) ÷ ng
    # Originals only. The periodic images `embedded_points` adds are for a tree, which has no way to wrap;
    # a lattice wraps its own bin coordinate, and one entry per cell is what lets a query emit each cell
    # exactly once and so need no buffer.
    pts = ng == 1 ? all_pts : all_pts[:, 1:n]
    # A function barrier on the dimension. `size(pts, 1)` is a runtime value, so building the index type
    # from it inline leaves the whole construction loop dynamically dispatched — measured at 34 MiB and
    # 196 ms for 65k points, against 3 ms once the dimension is a type.
    return _build_cell_list(grid, pts, embedding, hemb, Val(size(pts, 1)))
end

function _build_cell_list(
    grid, pts::AbstractMatrix{T}, embedding::E, hemb::T, ::Val{D},
) where {T,E<:AbstractEmbedding,D}
    n = size(pts, 2)
    # Wrapping is a property of the grid's own directions, and only survives into the embedding when the
    # embedding is the coordinates themselves. A sphere's seam is already closed by the transform.
    wrap, nbins, lo, hd = _cell_lattice(grid, pts, embedding, hemb, Val(D))
    IX = CellListIndex{D,T,E,typeof(pts),Vector{Int}}
    ixp = IX(pts, lo, hd, nbins, wrap, Int[], Int[], embedding)

    nbucket = max(1, nextpow(2, max(n, 1)))
    counts = zeros(Int, nbucket + 1)
    @inbounds for k in 1:n
        counts[_bin_hash(_bin_at(ixp, k), nbucket) + 1] += 1
    end
    starts = Vector{Int}(undef, nbucket + 1)
    starts[1] = 1
    @inbounds for b in 1:nbucket
        starts[b + 1] = starts[b] + counts[b + 1]
    end
    cursor = copy(starts)
    items = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        b = _bin_hash(_bin_at(ixp, k), nbucket)
        items[cursor[b]] = k
        cursor[b] += 1
    end
    return IX(pts, lo, hd, nbins, wrap, starts, items, embedding)
end

# Cartesian directions wrap with the grid; every other embedding is closed by its own transform.
#
# A wrapping direction's width is the period divided by a whole number of bins, not `h` itself: binning
# by `h` and then reducing mod `nbins` would fold a partial bin onto bin zero, and the span a query walks
# would no longer be a lattice neighbourhood.
function _cell_lattice(grid, pts::AbstractMatrix{T}, ::CartesianEmbedding, h::T, ::Val{D}) where {T,D}
    wrap = ntuple(d -> isperiodic(grid, d), Val(D))
    nbins = ntuple(Val(D)) do d
        wrap[d] ? max(1, Base.unsafe_trunc(Int, floor(T(period(grid, d)) / h))) : 0
    end
    hd = ntuple(d -> wrap[d] ? T(period(grid, d)) / nbins[d] : h, Val(D))
    lo = ntuple(d -> wrap[d] ? T(origin(grid, d)) : T(minimum(view(pts, d, :))) - h, Val(D))
    return wrap, nbins, lo, hd
end

function _cell_lattice(_grid, pts::AbstractMatrix{T}, ::AbstractEmbedding, h::T, ::Val{D}) where {T,D}
    lo = ntuple(d -> T(minimum(view(pts, d, :))) - h, Val(D))
    return ntuple(_ -> false, Val(D)), ntuple(_ -> 0, Val(D)), lo, ntuple(_ -> h, Val(D))
end

"""
    fold_candidates(f, acc, index, grid, I, r) -> acc

Thread `acc = f(acc, k)` over every cell `k` the index reports near cell `I`, without building a list. A
**superset** of the ball, each cell exactly once; the caller's exact distance gate decides membership.

A fold rather than a returned list is the whole point: it allocates nothing and needs no per-query
buffer, which is what a kernel requires and what a tree cannot offer, since a tree walk has to
deduplicate the periodic images it searches over.
"""
function fold_candidates end

function fold_candidates(f::F, acc, ix::CellListIndex{D,T}, grid, I, r) where {F,D,T}
    lin = I isa Integer ? Int(I) : LinearIndices(size(mask(grid)))[CartesianIndex(I)]
    return fold_candidates_at(f, acc, ix, ntuple(d -> @inbounds(ix.pts[d, lin]), Val(D)), r, nothing)
end

"""
    fold_candidates_at(f, acc, index, q, r, scratch) -> acc

[`fold_candidates`](@ref) around an arbitrary point `q`, already in the index's embedding, rather than
around a cell. A cell query is this one at the cell's own centre, so there is one traversal.

`scratch` is a candidate buffer for an index that has to materialize one — a tree does, since it must
deduplicate the periodic images it searches over. A cell list folds directly and ignores it.
"""
function fold_candidates_at(f, acc, index, q, r, scratch)
    throw(ArgumentError(
        "$(typeof(index)) cannot be queried at a point; build a `cell_list` for point-seeded queries",
    ))
end

function fold_candidates_at(f::F, acc, ix::CellListIndex{D,T}, q::NTuple{D,T}, r, _scratch) where {F,D,T}
    remb = T(embedded_radius(ix.embedding, r))
    # Per direction, since a wrapping direction's bins are the period divided evenly and so are not
    # exactly `h` wide.
    span = ntuple(d -> max(0, Base.unsafe_trunc(Int, ceil(remb / @inbounds(ix.h[d])))), Val(D))
    b0 = _bin_of(ix, q)
    nbucket = _nbuckets(ix)
    # A wrapping direction is capped at the lattice width, since beyond that the offsets revisit bins
    # already covered.
    reach = ntuple(d -> @inbounds(ix.wrap[d]) ? min(2 * span[d] + 1, ix.nbins[d]) : 2 * span[d] + 1, Val(D))
    # A ball much wider than the bin side walks more bins than the lattice holds cells, and most of them
    # are empty — at which point every cell is a candidate anyway and enumerating them directly is both
    # cheaper and bounded. Without this a query at 10× the bin side costs `10^D` times its own answer,
    # and an index built for one radius and queried at a far larger one degenerates without limit.
    if _window_bins(reach) > nbucket
        @inbounds for k in Base.OneTo(size(ix.pts, 2))
            acc = f(acc, k)
        end
        return acc
    end
    @inbounds for off in CartesianIndices(map(m -> 0:(m - 1), reach))
        b = ntuple(Val(D)) do d
            j = b0[d] - span[d] + off[d]
            ix.wrap[d] ? mod(j, ix.nbins[d]) : j
        end
        bucket = _bin_hash(b, nbucket)
        for t in ix.starts[bucket]:(ix.starts[bucket + 1] - 1)
            k = ix.items[t]
            _bin_at(ix, k) == b || continue     # a hash collision, not a member of this bin
            acc = f(acc, k)
        end
    end
    return acc
end

# The buffered hook, for callers that want a list. No sort and no dedup: every cell sits in exactly one
# bin, so the walk already emits each at most once.
function index_within!(buf::AbstractVector{<:Integer}, ix::CellListIndex, grid, I, r)
    empty!(buf)
    fold_candidates(nothing, ix, grid, I, r) do _, k
        push!(buf, k)
        return nothing
    end
    return buf
end

"""
    locate(grid, p) -> cell index
    locate(grid, p; active_only=false, topology, scratch) -> cell index

The cell of `grid` that `p` belongs to, as an `NTuple` of indices on a rectilinear or curvilinear grid
and a node number on an `UnstructuredGrid`. `p` is a coordinate tuple in the grid's own coordinates.

On a `StructuredGrid` this is [`Discretization.locate`](@ref) per direction, so the answer is the cell
whose faces bracket `p` — an `O(1)` lookup on a uniform axis and a bisection otherwise, with a periodic
direction wrapped first. A direction `p` lies outside reports `0` for that direction.

Elsewhere there are no axes to bracket along and it is the **nearest cell centre**, which is exactly the
containing cell for a node set, whose cells are the Voronoi regions of its nodes. On a curvilinear grid
the two can differ where cells are strongly sheared, so read it as nearest-centre rather than
point-in-quadrilateral. That form takes the keywords: `topology` carrying an index — [`cell_list`](@ref)
— makes it a bin lookup rather than a scan, `scratch` is a `Connectivity.ball_scratch` buffer,
and `active_only` restricts the answer to unmasked cells (`false` here, unlike the ball queries, since
the cell a point falls in is a question about the grid rather than about the active region).
"""
function locate end

function locate(grid::StructuredGrid{G,T,N}, p::NTuple{N,Real}) where {G,T,N}
    return ntuple(Val(N)) do d
        x = coordinates(grid, d)
        v = T(p[d])
        per = isperiodic(grid, d)
        if per
            L = T(period(grid, d))
            lo = axis_stats(grid, d).min_value
            L > 0 && (v = lo + mod(v - lo, L))
        end
        i = Discretization.locate(x, v)
        # The cell straddling the seam has half its extent on each side, so after wrapping into one
        # period part of it lies beyond the outermost face and `locate` reports "outside". It is
        # whichever end cell is nearer across the seam — the comparison, not `1`, so a descending axis
        # is right too.
        if i == 0 && per
            n = length(x)
            L = T(period(grid, d))
            df = abs(rem(v - T(@inbounds x[1]), L, RoundNearest))
            dl = abs(rem(v - T(@inbounds x[n]), L, RoundNearest))
            i = df ≤ dl ? 1 : n
        end
        i
    end
end

"""
    has_spatial_index(grid) -> Bool

Whether the k-d tree behind [`spatial_index`](@ref) can be built for this grid — `false` until the
NearestNeighbors extension is loaded. Answers "is the tree available?" without calling `spatial_index`
speculatively and catching its error.

This is not a test for whether a grid can be indexed at all: [`cell_list`](@ref) needs no extension and
is what the sweeps build.
"""
has_spatial_index(::AbstractGrid) = false

"""
    spatial_index(grid) -> opaque index

Extension hook: a range-queryable spatial index over the grid's cell centres, overridden by the
NearestNeighbors extension. Paired with [`index_within!`](@ref).
"""
function spatial_index(grid::AbstractGrid)
    throw(ArgumentError(
        "a spatial index requires NearestNeighbors.jl — run `using NearestNeighbors`. Without it a ball " *
        "query on this grid scans every cell, which is correct but linear per query.",
    ))
end

"""
    index_within!(buffer, index, grid, I, r) -> candidate cell indices
    index_within(index, grid, I, r) -> candidate cell indices

Extension hook: the cells an index reports near `I`, as linear indices. It must return a **superset** of
the cells within `r`; the caller applies the exact distance gate, so over-returning is safe and
under-returning is not.

`index_within!` overwrites and returns `buffer`, which is how a sweep over many cells avoids one heap
allocation per query — nothing at all, against 480 bytes on a small ball and 6.1 KB on a 310-candidate
one. `index_within` is the same query into a fresh vector.
"""
function index_within!(buffer::AbstractVector{<:Integer}, index, grid, I, r)
    throw(ArgumentError("no `index_within!` method for $(typeof(index)); build one with `spatial_index`"))
end

index_within(index, grid, I, r) = index_within!(Int[], index, grid, I, r)

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
        "Spherical Voronoi-cell areas for an arbitrary point set require Quickhull.jl — run " *
        "`using Quickhull` (or supply `areas` explicitly to the `UnstructuredGrid` constructor).",
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

`periodic`/`period` declare a wrapping domain, and the neighbor search honors it: a node near one
face finds the nodes across the opposite face as genuine neighbors. Spherical longitude wraps by
default; a Cartesian box is opt-in and needs its `period`.
"""
UnstructuredGrid(
    geometry::Geometry.AbstractGeometry, x::AbstractVector, y::AbstractVector,
    mask::AbstractVector{Bool} = AllActive((length(x),)); kwargs...,
) = UnstructuredGrid(geometry, (x, y), mask; kwargs...)

function UnstructuredGrid(
    geometry::Geometry.AbstractGeometry{T}, coords::NTuple{N,AbstractVector},
    mask::AbstractVector{Bool} = AllActive((length(first(coords)),));
    k::Integer = 6, radius::Union{Nothing,Real} = nothing, areas::Union{Nothing,AbstractVector} = nothing,
    periodic = nothing, period = nothing,
) where {N, T<:AbstractFloat}
    c = ntuple(d -> _to_axis(T, coords[d]), Val(N))
    per, prd = _node_periodicity(geometry, Val(N), periodic, period)
    nbrs, ptr = _build_kdtree_neighbors(
        geometry, c; k = k, radius = radius,
        periodic = map(_is_periodic, per), period = prd,
    )
    # Voronoi tessellation is a 2-D algorithm (planar Delaunay / spherical convex hull), so past two
    # directions the control volumes are the caller's to supply — the neighbour search is not, and is
    # built for any `N` above.
    areas_T = if areas !== nothing
        _to_axis(T, areas)
    elseif N == 2
        _voronoi_areas(geometry, c[1], c[2])
    else
        throw(ArgumentError(
            "a $N-direction node grid has no control volumes to derive: the Voronoi tessellation " *
            "behind them is a 2-D algorithm (planar Delaunay / spherical convex hull). Pass `areas`.",
        ))
    end
    return UnstructuredGrid(geometry, c, areas_T, mask, nbrs, ptr; periodic = per, period = prd)
end

"""
    neighbor_nbrs(grid::UnstructuredGrid) -> AbstractVector{<:Integer}
    neighbor_ptr(grid::UnstructuredGrid) -> AbstractVector{<:Integer}

The CSR adjacency arrays: the flat neighbour indices, and the per-node offsets into them.
"""
@inline neighbor_nbrs(grid::UnstructuredGrid) = getfield(grid, :neighbor_nbrs)
@inline neighbor_ptr(grid::UnstructuredGrid) = getfield(grid, :neighbor_ptr)

"""
    neighbors(grid::UnstructuredGrid, idx::Integer) -> AbstractVector{<:Integer}

Neighbor node indices of node `idx`, as a zero-copy view into the CSR-flattened adjacency storage.
"""
@inline function neighbors(grid::UnstructuredGrid, idx::Integer)
    ptr = neighbor_ptr(grid)
    @inbounds lo = ptr[idx]
    @inbounds hi = ptr[idx+1] - 1
    return view(neighbor_nbrs(grid), lo:hi)
end

@inline function _raw_coords(grid::UnstructuredGrid{T,G,N}, idx::Integer) where {T,G,N}
    c = coordinates(grid)
    @boundscheck checkbounds(c[1], idx)
    return ntuple(d -> @inbounds(c[d][idx]), Val(N))
end

# ---------------------------------------------------------------------------
# Minimum image, and the distance between two cells
# ---------------------------------------------------------------------------

"""
    _wrap_lengths(grid, Val(N)) -> NTuple{N,T}

Wrap length per direction, zero where the direction is bounded, so [`_min_image`](@ref) leaves those
components alone.
"""
@inline _wrap_lengths(grid::AbstractGrid, ::Val{N}) where {N} =
    ntuple(d -> isperiodic(grid, d) ? period(grid, d) : zero(period(grid, d)), Val(N))

"""
    _min_image(p0, pt, prd) -> NTuple

`pt` brought to the image nearest `p0`, per component, for each direction with a nonzero wrap length.

For an angular coordinate the geometry's own distance is already `2π`-periodic and this changes nothing
(it also keeps Vincenty inside its `|Δλ| ≤ π` regime); for a periodic Cartesian coordinate it is what
makes the seam invisible. Per-component minimum image is the global minimum for a separable metric,
which the Euclidean one is.
"""
@inline function _min_image(p0::NTuple{N,Any}, pt::NTuple{N,Any}, prd::NTuple{N,Any}) where {N}
    return ntuple(Val(N)) do d
        p = prd[d]
        p > zero(p) ? p0[d] + (pt[d] - p0[d] - p * round((pt[d] - p0[d]) / p)) : pt[d]
    end
end

"""
    displacement(grid, I, J) -> NTuple{N,T}
    displacement(grid::UnstructuredGrid, i, j) -> NTuple{N,T}

The signed per-direction coordinate offset from cell `I` to cell `J`, reduced to the nearest image in
every periodic direction — the offset [`Geometry.distance`](@ref) is taken from.

A coordinate quantity rather than a metric one, which is why it lives here while the distance itself
extends `Geometry.distance`: across a periodic seam the two cells' *stored* coordinates differ by nearly
a full period, and this reports the short way round instead.
"""
function displacement end

@inline function displacement(
    grid::Union{StructuredGrid{G,T,N},CurvilinearGrid{T,G,N}},
    I::NTuple{N,Integer}, J::NTuple{N,Integer},
) where {G,T,N}
    p0 = _raw_coords(grid, I...)
    q = _min_image(p0, _raw_coords(grid, J...), _wrap_lengths(grid, Val(N)))
    return ntuple(d -> q[d] - p0[d], Val(N))
end

@inline function displacement(
    grid::UnstructuredGrid{T,G,N}, i::Integer, j::Integer,
) where {T,G,N}
    p0 = _raw_coords(grid, i)
    q = _min_image(p0, _raw_coords(grid, j), _wrap_lengths(grid, Val(N)))
    return ntuple(d -> q[d] - p0[d], Val(N))
end

"""
    Geometry.distance(grid, I, J) -> T
    Geometry.distance(grid::UnstructuredGrid, i, j) -> T

Distance between the centres of two cells under the grid's own geometry and topology: the coordinates
are resolved from the indices, reduced to the nearest image in every periodic direction, and handed to
the point form of [`Geometry.distance`](@ref).

Across a periodic seam this is the short way round — one spacing between the first and last cell of a
periodic direction, not the full extent, which is what the point form on the raw coordinates would give.
A bounded direction contributes its plain coordinate difference. See [`displacement`](@ref) for the
offset it was taken from.
"""
@inline function Geometry.distance(
    grid::Union{StructuredGrid{G,T,N},CurvilinearGrid{T,G,N}},
    I::NTuple{N,Integer}, J::NTuple{N,Integer},
) where {G,T,N}
    p0 = _raw_coords(grid, I...)
    q = _min_image(p0, _raw_coords(grid, J...), _wrap_lengths(grid, Val(N)))
    return Geometry.distance(grid_geometry(grid), p0, q)
end

@inline function Geometry.distance(
    grid::UnstructuredGrid{T,G,N}, i::Integer, j::Integer,
) where {T,G,N}
    p0 = _raw_coords(grid, i)
    q = _min_image(p0, _raw_coords(grid, j), _wrap_lengths(grid, Val(N)))
    return Geometry.distance(grid_geometry(grid), p0, q)
end

# `CartesianIndex` is what `CartesianIndices` hands a traversal, so accept it directly.
@inline Geometry.distance(grid::AbstractGrid, I::CartesianIndex, J::CartesianIndex) =
    Geometry.distance(grid, Tuple(I), Tuple(J))
@inline displacement(grid::AbstractGrid, I::CartesianIndex, J::CartesianIndex) =
    displacement(grid, Tuple(I), Tuple(J))

end # module
