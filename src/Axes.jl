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
    AbstractUniformAxis{T} <: AbstractRange{T}

Supertype for axes whose spacing is constant and known from their type. Default:
[`UniformAxis`](@ref).

To add one, implement the three methods `AbstractRange` already requires — `Base.first`, `Base.step`
and `Base.length` — and everything else here follows: indexing, the `O(1)` reductions, slicing,
reversal, and the affine arithmetic. None of the generic methods touch a field, so a subtype may store
whatever it likes under whatever names.

Implement [`similar_axis`](@ref) as well if derived axes should keep the subtype rather than becoming a
plain `UniformAxis`.
"""
abstract type AbstractUniformAxis{T} <: AbstractRange{T} end

"""
    similar_axis(a, origin, Δ, n) -> AbstractUniformAxis

The axis of `a`'s own kind with the given origin, spacing and length: the hook every derived axis goes
through — a slice, a reversal, `2a`, `a .+ c`.

Defaults to a [`UniformAxis`](@ref), so a subtype that does not define it still gets correct results,
just not its own type back.
"""
similar_axis(::AbstractUniformAxis, origin, Δ, n::Integer) = UniformAxis(origin, Δ, n)

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

An `AbstractRange`: that gets Base's O(1) `searchsorted` (flat 42 ns over `n = 10 … 10⁷`, against
35→69 ns as an `AbstractVector`) and `isa AbstractRange` dispatch from other packages.
"""
struct UniformAxis{T<:AbstractFloat} <: AbstractUniformAxis{T}
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

# The three methods a subtype supplies, for the concrete type.
@inline Base.step(a::UniformAxis) = a.Δ
@inline Base.first(a::UniformAxis) = a.origin
# `AbstractRange` requires `length` directly; it does not derive it from `size`.
@inline Base.length(a::UniformAxis) = a.n
@inline similar_axis(::UniformAxis{T}, origin, Δ, n::Integer) where {T} =
    UniformAxis{T}(origin, Δ, n)

# Everything below is written in terms of `first`/`step`/`length` alone, so it holds for any subtype.
spacing_trait(::AbstractUniformAxis) = UniformSpacing()
@inline spacing(a::AbstractUniformAxis) = step(a)

@inline Base.size(a::AbstractUniformAxis) = (length(a),)
Base.IndexStyle(::Type{<:AbstractUniformAxis}) = IndexLinear()

@inline function Base.getindex(a::AbstractUniformAxis{T}, i::Int) where {T}
    @boundscheck checkbounds(a, i)
    return first(a) + T(i - 1) * step(a)
end

@inline Base.last(a::AbstractUniformAxis{T}) where {T} =
    first(a) + T(length(a) - 1) * step(a)

# O(1): a monotone sequence's extremes are its endpoints, and its sum is the arithmetic series.
@inline Base.minimum(a::AbstractUniformAxis) = isempty(a) ? _empty_reduce(a, "minimum") :
    (step(a) ≥ 0 ? first(a) : last(a))
@inline Base.maximum(a::AbstractUniformAxis) = isempty(a) ? _empty_reduce(a, "maximum") :
    (step(a) ≥ 0 ? last(a) : first(a))
@inline Base.extrema(a::AbstractUniformAxis) = (minimum(a), maximum(a))
@inline Base.sum(a::AbstractUniformAxis{T}) where {T} =
    isempty(a) ? zero(T) : T(length(a)) * (first(a) + last(a)) / T(2)

_empty_reduce(a, name) =
    throw(ArgumentError("$name of an empty $(nameof(typeof(a))) is undefined"))

# A slice or reversal of a uniform axis is uniform; keep the proof, and the subtype.
@inline function Base.getindex(a::AbstractUniformAxis, r::AbstractUnitRange{<:Integer})
    @boundscheck checkbounds(a, r)
    isempty(r) && return similar_axis(a, first(a), step(a), 0)
    return similar_axis(a, @inbounds(a[first(r)]), step(a), length(r))
end

@inline function Base.getindex(a::AbstractUniformAxis{T}, r::StepRange{<:Integer}) where {T}
    @boundscheck checkbounds(a, r)
    Δ = step(a) * T(step(r))
    isempty(r) && return similar_axis(a, first(a), Δ, 0)
    return similar_axis(a, @inbounds(a[first(r)]), Δ, length(r))
end

@inline Base.reverse(a::AbstractUniformAxis) =
    isempty(a) ? a : similar_axis(a, last(a), -step(a), length(a))

Base.show(io::IO, a::AbstractUniformAxis{T}) where {T} =
    print(io, nameof(typeof(a)), "{", T, "}(", first(a), ", Δ=", step(a), ", n=", length(a), ")")

function Base.show(io::IO, ::MIME"text/plain", a::AbstractUniformAxis{T}) where {T}
    print(io, length(a), "-element ", nameof(typeof(a)), "{", T, "}")
    isempty(a) && return
    print(io, ": ", first(a), " : ", step(a), " : ", last(a))
    return
end

# ---------------------------------------------------------------------------
# The AbstractRange protocol
# ---------------------------------------------------------------------------
#
# Base's generic range methods rebuild a range from its endpoints and step, reaching for constructors
# that do not exist here; without these, `x .+ x` recurses until the stack runs out.

Base.copy(a::AbstractUniformAxis) = a   # immutable

# An affine map of an affine sequence is affine, so these keep the spacing guarantee — and, through
# `similar_axis`, the subtype.
Base.:(+)(a::AbstractUniformAxis, c::Number) = similar_axis(a, first(a) + c, step(a), length(a))
Base.:(+)(c::Number, a::AbstractUniformAxis) = a + c
Base.:(-)(a::AbstractUniformAxis, c::Number) = a + (-c)
Base.:(-)(c::Number, a::AbstractUniformAxis) = similar_axis(a, c - first(a), -step(a), length(a))
Base.:(-)(a::AbstractUniformAxis) = similar_axis(a, -first(a), -step(a), length(a))
Base.:(*)(a::AbstractUniformAxis, c::Number) =
    similar_axis(a, first(a) * c, step(a) * c, length(a))
Base.:(*)(c::Number, a::AbstractUniformAxis) = a * c
Base.:(/)(a::AbstractUniformAxis, c::Number) =
    similar_axis(a, first(a) / c, step(a) / c, length(a))

# Intercepted at `broadcasted`; the generic range path is what recurses.
Base.broadcasted(::typeof(+), a::AbstractUniformAxis, c::Number) = a + c
Base.broadcasted(::typeof(+), c::Number, a::AbstractUniformAxis) = a + c
Base.broadcasted(::typeof(-), a::AbstractUniformAxis, c::Number) = a - c
Base.broadcasted(::typeof(-), c::Number, a::AbstractUniformAxis) = c - a
Base.broadcasted(::typeof(-), a::AbstractUniformAxis) = -a
Base.broadcasted(::typeof(*), a::AbstractUniformAxis, c::Number) = a * c
Base.broadcasted(::typeof(*), c::Number, a::AbstractUniformAxis) = a * c
Base.broadcasted(::typeof(/), a::AbstractUniformAxis, c::Number) = a / c

function Base.broadcasted(::typeof(+), a::AbstractUniformAxis, b::AbstractUniformAxis)
    length(a) == length(b) ||
        throw(DimensionMismatch("axis lengths $(length(a)) and $(length(b)) do not match"))
    return similar_axis(a, first(a) + first(b), step(a) + step(b), length(a))
end

function Base.broadcasted(::typeof(-), a::AbstractUniformAxis, b::AbstractUniformAxis)
    length(a) == length(b) ||
        throw(DimensionMismatch("axis lengths $(length(a)) and $(length(b)) do not match"))
    return similar_axis(a, first(a) - first(b), step(a) - step(b), length(a))
end

# Anything else is not affine — `cos.(axis)` must become a plain array, not pretend otherwise.
Base.BroadcastStyle(::Type{<:AbstractUniformAxis}) = Broadcast.DefaultArrayStyle{1}()

# A subtype cannot be reconstructed at a promoted eltype generically, so mixing two different uniform
# axis types lands on the canonical concrete one.
Base.promote_rule(::Type{<:AbstractUniformAxis{T}}, ::Type{<:AbstractUniformAxis{S}}) where {T,S} =
    UniformAxis{promote_type(T, S)}
# Another range has its own spacing, so the affine representation cannot survive the mix.
Base.promote_rule(::Type{UniformAxis{T}}, ::Type{<:AbstractRange{S}}) where {T,S} =
    Vector{promote_type(T, S)}
Base.promote_rule(::Type{UniformAxis{T}}, ::Type{<:AbstractUniformAxis{S}}) where {T,S} =
    UniformAxis{promote_type(T, S)}

Base.:(==)(a::AbstractUniformAxis, b::AbstractUniformAxis) =
    length(a) == length(b) &&
    (isempty(a) || (first(a) == first(b) && (length(a) == 1 || step(a) == step(b))))

"""
    uniform_axis(x) -> UniformAxis
    uniform_axis(T, x) -> UniformAxis{T}

The [`UniformAxis`](@ref) equal to `x`, for any axis whose spacing is known from its type. A
nonuniform vector raises, because it has no uniform form: replacing its coordinates with a fitted
sequence is a decision only its owner can make, and they can build the axis directly.
"""
uniform_axis(x) = uniform_axis(float(eltype(x)), x)

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

"""
    wrap_sign(x) -> ±1

`+1` for an ascending axis and `-1` for a descending one: the sign that turns a period magnitude into
the wrapped neighbour's offset in index order. A descending axis is routine in stored data, and its
wrapped neighbour lies at `x[1] - period`, not `x[1] + period`.
"""
@inline function wrap_sign(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    n < 2 && return one(T)
    return @inbounds(x[n] ≥ x[1]) ? one(T) : -one(T)
end

@inline wrap_sign(x::AbstractRange{T}) where {T<:AbstractFloat} =
    step(x) ≥ 0 ? one(T) : -one(T)

end # module Axes
