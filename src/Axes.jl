module Axes

# Coordinate-axis storage, and the spacing trait everything else dispatches on. An axis is either
# provably uniform — its type carries the spacing — or nonuniform, where the samples are the only
# description. Keeping that in the type is what lets the fast paths be selected at compile time.

# ---------------------------------------------------------------------------
# Spacing trait
# ---------------------------------------------------------------------------

"""
    UniformSpacing()
    NonuniformSpacing()

Whether an axis's spacing is known from its type. [`spacing_trait`](@ref) returns one of these, for
dispatch rather than a runtime branch.
"""
struct UniformSpacing end

"""See [`UniformSpacing`](@ref)."""
struct NonuniformSpacing end

"""
    spacing_trait(x) -> UniformSpacing() | NonuniformSpacing()

The axis's spacing trait. [`UniformAxis`](@ref) and any `AbstractRange` carry a constant step in
their type; every other array is nonuniform.

This is the compile-time question, not the data question: a `Vector` holding an arithmetic sequence is
`NonuniformSpacing()` because its type does not say otherwise, and no code path here inspects values to
decide otherwise — the fast paths are selected by type alone.
"""
spacing_trait(::AbstractArray) = NonuniformSpacing()
spacing_trait(::AbstractRange) = UniformSpacing()

"""
    isuniform(x) -> Bool

Whether `x` has [`UniformSpacing`](@ref). Const-folds, so it is free to branch on in a hot loop.
"""
@inline isuniform(x) = spacing_trait(x) === UniformSpacing()

"""
    spacing(x) -> Number

The constant spacing of a uniform axis, read from its type. Signed, so a descending axis reports a
negative spacing. Raises for an axis that is not [`isuniform`](@ref).
"""
@inline spacing(r::AbstractRange) = step(r)
spacing(x) = throw(ArgumentError(
    "$(typeof(x)) has no single spacing (it is not `isuniform`); use `minimum_spacing` / " *
    "`maximum_spacing` for its range of gaps, or `Grids._cell_width` for one cell's width.",
))

# ---------------------------------------------------------------------------
# UniformAxis
# ---------------------------------------------------------------------------

"""
    UniformAxis(origin, Δ, n)
    UniformAxis{T}(origin, Δ, n)

`n` samples at `origin + (i-1)·Δ`, stored as those three numbers.

Preferred over `range`/`StepRangeLen` for a grid axis:

- `range(0f0; step = 0.1f0, length = n)` is a `StepRangeLen{Float32, Float64, Float64, Int}` — Float32
  elements over a Float64 offset and step. `UniformAxis{T}` computes in `T` throughout.
- `StepRangeLen`'s `TwicePrecision` arithmetic costs measurably for exactness a grid axis does not
  need. Over 2×10⁶ reads it is 24.0 ms against 16.6 ms here for `cos(x[i])`, and 4.40 ms against
  1.82 ms for the adjacent-gap pattern the grid's per-cell width kernel uses.
- `isbits`, so moving an axis to another storage backend is free.

`Δ` may be negative, for a descending axis. `n` must be non-negative.

Unlike `LinRange` this does not pin `last` to a prescribed endpoint: on a grid the spacing is the
primary datum.
"""
struct UniformAxis{T<:AbstractFloat} <: AbstractVector{T}
    origin::T
    Δ::T
    n::Int

    function UniformAxis{T}(origin, Δ, n::Integer) where {T<:AbstractFloat}
        n ≥ 0 || throw(ArgumentError("a UniformAxis needs n ≥ 0, got $n"))
        return new{T}(convert(T, origin), convert(T, Δ), Int(n))
    end
end

function UniformAxis(origin::Real, Δ::Real, n::Integer)
    T = float(promote_type(typeof(origin), typeof(Δ)))
    return UniformAxis{T}(origin, Δ, n)
end

spacing_trait(::UniformAxis) = UniformSpacing()
@inline spacing(a::UniformAxis) = a.Δ

@inline Base.size(a::UniformAxis) = (a.n,)
Base.IndexStyle(::Type{<:UniformAxis}) = IndexLinear()

@inline function Base.getindex(a::UniformAxis{T}, i::Int) where {T}
    @boundscheck checkbounds(a, i)
    return a.origin + T(i - 1) * a.Δ
end

@inline Base.step(a::UniformAxis) = a.Δ
@inline Base.first(a::UniformAxis) = a.origin
@inline Base.last(a::UniformAxis{T}) where {T} = a.origin + T(a.n - 1) * a.Δ

# O(1): a monotone sequence's extremes are its endpoints, and its sum is the arithmetic series.
@inline Base.minimum(a::UniformAxis) = isempty(a) ? _empty_reduce(a, "minimum") :
    (a.Δ ≥ 0 ? first(a) : last(a))
@inline Base.maximum(a::UniformAxis) = isempty(a) ? _empty_reduce(a, "maximum") :
    (a.Δ ≥ 0 ? last(a) : first(a))
@inline Base.extrema(a::UniformAxis) = (minimum(a), maximum(a))
@inline Base.sum(a::UniformAxis{T}) where {T} =
    isempty(a) ? zero(T) : T(a.n) * (first(a) + last(a)) / T(2)

_empty_reduce(a, name) = throw(ArgumentError("$name of an empty UniformAxis is undefined"))

# A slice or reversal of a uniform axis is uniform; keep the proof.
@inline function Base.getindex(a::UniformAxis{T}, r::AbstractUnitRange{<:Integer}) where {T}
    @boundscheck checkbounds(a, r)
    isempty(r) && return UniformAxis{T}(a.origin, a.Δ, 0)
    return UniformAxis{T}(@inbounds(a[first(r)]), a.Δ, length(r))
end

@inline function Base.getindex(a::UniformAxis{T}, r::StepRange{<:Integer}) where {T}
    @boundscheck checkbounds(a, r)
    isempty(r) && return UniformAxis{T}(a.origin, a.Δ * T(step(r)), 0)
    return UniformAxis{T}(@inbounds(a[first(r)]), a.Δ * T(step(r)), length(r))
end

@inline Base.reverse(a::UniformAxis{T}) where {T} =
    isempty(a) ? a : UniformAxis{T}(last(a), -a.Δ, a.n)

Base.show(io::IO, a::UniformAxis{T}) where {T} =
    print(io, "UniformAxis{", T, "}(", a.origin, ", Δ=", a.Δ, ", n=", a.n, ")")

Base.show(io::IO, ::MIME"text/plain", a::UniformAxis{T}) where {T} =
    print(io, a.n, "-element UniformAxis{", T, "}: ", a.origin, " : ", a.Δ, " : ", last(a))

"""
    uniform_axis(x) -> UniformAxis
    uniform_axis(T, x) -> UniformAxis{T}

The [`UniformAxis`](@ref) equal to `x`, for any axis whose spacing is known from its type. A
nonuniform vector raises, because it has no uniform form: replacing its coordinates with a fitted
sequence is a decision only its owner can make, and they can build the axis directly.
"""
uniform_axis(x) = uniform_axis(float(eltype(x)), x)

uniform_axis(::Type{T}, a::UniformAxis) where {T<:AbstractFloat} =
    UniformAxis{T}(a.origin, a.Δ, a.n)
uniform_axis(::Type{T}, r::AbstractRange) where {T<:AbstractFloat} =
    UniformAxis{T}(first(r), step(r), length(r))
uniform_axis(::Type{T}, x::AbstractVector) where {T<:AbstractFloat} = throw(ArgumentError(
    "$(typeof(x)) has no compile-time spacing, so it has no UniformAxis form. Build one directly if " *
    "its spacing is known — `UniformAxis(origin, Δ, n)` — or keep it as it is, since nonuniform axes " *
    "are supported everywhere.",
))


# ---------------------------------------------------------------------------
# ConstantVector
# ---------------------------------------------------------------------------

"""
    ConstantVector(value, n)

`n` copies of `value`, stored as that value and a length — which is what a uniform axis's per-cell
width is.

A genuine `AbstractVector`: indexing, iteration, broadcasting and `collect` behave as for
`fill(value, n)`. `getindex` folds to a constant and every reduction below is closed-form.
"""
struct ConstantVector{T} <: AbstractVector{T}
    value::T
    n::Int

    function ConstantVector{T}(value, n::Integer) where {T}
        n ≥ 0 || throw(ArgumentError("a ConstantVector needs n ≥ 0, got $n"))
        return new{T}(convert(T, value), Int(n))
    end
end

ConstantVector(value::T, n::Integer) where {T} = ConstantVector{T}(value, n)

@inline Base.size(c::ConstantVector) = (c.n,)
Base.IndexStyle(::Type{<:ConstantVector}) = IndexLinear()

@inline function Base.getindex(c::ConstantVector, i::Int)
    @boundscheck checkbounds(c, i)
    return c.value
end

@inline function Base.getindex(c::ConstantVector{T}, r::AbstractUnitRange{<:Integer}) where {T}
    @boundscheck checkbounds(c, r)
    return ConstantVector{T}(c.value, length(r))
end


@inline Base.sum(c::ConstantVector{T}) where {T} = T(c.n) * c.value
@inline Base.prod(c::ConstantVector) = c.value^c.n
@inline Base.minimum(c::ConstantVector) =
    isempty(c) ? _empty_reduce(c, "minimum") : c.value
@inline Base.maximum(c::ConstantVector) =
    isempty(c) ? _empty_reduce(c, "maximum") : c.value
@inline Base.extrema(c::ConstantVector) = (minimum(c), maximum(c))
@inline Base.reverse(c::ConstantVector) = c
@inline Base.all(c::ConstantVector{Bool}) = isempty(c) ? true : c.value
@inline Base.any(c::ConstantVector{Bool}) = isempty(c) ? false : c.value

# `f` is applied once, not n times.
@inline Base.sum(f, c::ConstantVector{T}) where {T} = T(c.n) * f(c.value)
@inline Base.prod(f, c::ConstantVector) = f(c.value)^c.n
@inline Base.minimum(f, c::ConstantVector) =
    isempty(c) ? _empty_reduce(c, "minimum") : f(c.value)
@inline Base.maximum(f, c::ConstantVector) =
    isempty(c) ? _empty_reduce(c, "maximum") : f(c.value)
@inline Base.count(f, c::ConstantVector) = f(c.value) ? c.n : 0

Base.show(io::IO, c::ConstantVector{T}) where {T} =
    print(io, "ConstantVector{", T, "}(", c.value, ", ", c.n, ")")

Base.show(io::IO, ::MIME"text/plain", c::ConstantVector{T}) where {T} =
    print(io, c.n, "-element ConstantVector{", T, "}: all ", c.value)

end # module Axes
