# ---------------------------------------------------------------------------
# Separable reductions
# ---------------------------------------------------------------------------
#
# A reduction over the outer product factors into a reduction over the axes when the pair `(f, op)`
# permits it, turning `O(∏ Nᵈ)` into `O(∑ Nᵈ)`. Dispatch is on that pair: `mapreduce` is where `sum`,
# `prod`, `maximum`, `minimum`, `sum(f, ·)`, `count` and the `dims` forms all arrive, so one method each
# covers them, and any pair not listed here falls through to the generic dense path.
#
# `f` must be multiplicative — `f(xy) = f(x)f(y)` — for a sum or product to factor. `identity`, `abs`,
# `abs2`, `sqrt` and `inv` are; `exp` and `log` are not, and stay off this path.
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
combinations are formed and the best taken. This holds for factors of either sign, where `∏ maximum`
alone holds only for non-negative ones, and it costs `O(∑ Nᵈ + 2^N)` against the dense `O(∏ Nᵈ)`.
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
# path materializes 4×10⁶ elements from the handful of numbers a separable measure stores. The
# operations that keep the measure a product of per-axis factors stay a measure; everything else falls
# through to Base and materializes, correct and dense.
#
# Non-negative factors are an invariant of this type — widths are `abs`-ed at construction — and
# `findmax`/`findmin` above rely on it, since "largest cell at the per-axis argmax" is a statement
# about non-negative factors only. Scaling by a negative constant therefore materializes.

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
    RingwiseVector(value, offset)

The cell measure of a ring grid, held as its per-ring values.

Every cell of an iso-latitude ring has the same area, so `npoints` entries carry only `nrings` numbers
— `O(√npoints)`. A genuine `AbstractVector`: indexing, broadcasting and `collect` behave as for the
dense equivalent.

`sum` is specialized to `Σᵣ nlonᵣ·valueᵣ`, `O(nrings)` against the dense `O(npoints)`.
"""
struct RingwiseVector{T,V<:AbstractVector{T},O<:AbstractVector{Int}} <: AbstractVector{T}
    value::V     # one per ring
    offset::O    # points before each ring; length nrings + 1
end

@inline Base.size(v::RingwiseVector) = (last(v.offset),)
Base.IndexStyle(::Type{<:RingwiseVector}) = IndexLinear()

@inline function Base.getindex(v::RingwiseVector, i::Int)
    @boundscheck checkbounds(v, i)
    return @inbounds v.value[searchsortedlast(v.offset, i - 1)]
end

@inline function Base.sum(v::RingwiseVector{T}) where {T}
    s = zero(T)
    @inbounds for r in eachindex(v.value)
        s += T(v.offset[r + 1] - v.offset[r]) * v.value[r]
    end
    return s
end

@inline Base.minimum(v::RingwiseVector) = minimum(v.value)
@inline Base.maximum(v::RingwiseVector) = maximum(v.value)
@inline Base.extrema(v::RingwiseVector) = extrema(v.value)

Base.show(io::IO, v::RingwiseVector{T}) where {T} =
    print(io, "RingwiseVector{", T, "}(", length(v.value), " rings, ", length(v), " cells)")

"""
    GridMeasure(grid)

The cell measure of a layout whose measure is a formula, computed per entry and stored nowhere.

A genuine `AbstractVector`: indexing, broadcasting and `collect` behave as for the dense equivalent, and
`measure(grid, k)` is what an entry is. `sum` goes through [`_total_measure`](@ref), which a layout
covering a known region answers in `O(1)`.
"""
struct GridMeasure{T,G} <: AbstractVector{T}
    grid::G
end

GridMeasure(grid::AbstractGrid{G,T}) where {G,T} = GridMeasure{T,typeof(grid)}(grid)

@inline Base.size(m::GridMeasure) = (length(mask(m.grid)),)
Base.IndexStyle(::Type{<:GridMeasure}) = IndexLinear()

@inline function Base.getindex(m::GridMeasure, i::Int)
    @boundscheck checkbounds(m, i)
    return @inbounds measure(m.grid, i)
end

@inline Base.sum(m::GridMeasure) = _total_measure(m.grid)

Base.show(io::IO, m::GridMeasure{T}) where {T} =
    print(io, "GridMeasure{", T, "}(", length(m), " cells of ", nameof(typeof(m.grid)), ")")

"""
    _total_measure(grid) -> T

`sum(measure(grid))`, from whatever the layout knows about the region its cells tile. The fallback adds
the cells up; a layout tiling the whole sphere returns `4πR²` without visiting one.
"""
function _total_measure(grid::AbstractGrid{G,T}) where {G,T}
    s = zero(T)
    @inbounds for c in cells(grid)
        s += measure(grid, _cell_indices(grid, cell_at(grid, c))...)
    end
    return s
end

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
# `f::Function` keeps these more specific than Base's `all(f::Function, ::AbstractArray)`, which is
# otherwise an equally specific match and ambiguous with them.
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

# `@boundscheck` + `@inbounds` body: the check is elided at an `@inbounds` call site, so a hot loop pays
# nothing, and an ordinary call raises on an out-of-range index.
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
    mask(grid) -> AbstractArray{Bool}

Which cells/nodes participate, as an array over the grid's own shape. Ask
[`isactive`](@ref) for one cell; this is the whole array, and it is what fixes the grid's `size`.
"""
@inline mask(grid::AbstractGrid) = getfield(grid, :mask)

"""
    isactive(grid, I...) -> Bool

Whether cell/node `I` participates (`false` = masked out).
"""
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
