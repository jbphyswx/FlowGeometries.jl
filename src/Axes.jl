module Axes

# Coordinate-axis storage, and the spacing trait everything else dispatches on. An axis is either
# provably uniform — its type carries the spacing — or nonuniform, where the samples are the only
# description. The trait is in the type, so the fast paths are selected at compile time.

# ---------------------------------------------------------------------------
# Spacing trait
# ---------------------------------------------------------------------------

"""
    UniformSpacing()
    NonuniformSpacing()

Whether an axis's spacing is known from its type. [`spacing_trait`](@ref) returns one of these, so a
method can dispatch on it.
"""
struct UniformSpacing end

"""See [`UniformSpacing`](@ref)."""
struct NonuniformSpacing end

"""
    spacing_trait(x) -> UniformSpacing() | NonuniformSpacing()

The axis's spacing trait. [`UniformAxis`](@ref) and any `AbstractRange` carry a constant step in
their type; every other array is nonuniform.

The question is answered by the type: a `Vector` holding an arithmetic sequence is
`NonuniformSpacing()`. No code path here inspects values to decide it.
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
    "`maximum_spacing` for its range of gaps, `Discretization.local_spacing` for the gaps at one " *
    "index, or `Discretization.cell_width` for one cell's width.",
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

Implement [`similar_axis`](@ref) as well to have derived axes keep the subtype; the fallback returns a
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
- `StepRangeLen` indexes through `TwicePrecision` arithmetic, buying an exactness a grid axis does not
  need; `UniformAxis` indexes with one multiply and one add.
- `isbits`, so moving an axis to another storage backend is free.

`Δ` may be negative, for a descending axis. `n` must be non-negative.

Unlike `LinRange` this does not pin `last` to a prescribed endpoint: on a grid the spacing is the
primary datum.

An `AbstractRange`: that gets Base's `searchsorted` in closed form, so a lookup is `O(1)` in the axis
length, and `isa AbstractRange` dispatch from other packages.
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

# Any other broadcast leaves the affine family, so `cos.(axis)` becomes a plain array.
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
# Analytic axes
# ---------------------------------------------------------------------------

"""
    AbstractAnalyticAxis{T} <: AbstractVector{T}

An axis whose coordinate is a formula of its index, and whose formula inverts in closed form.

The third axis kind. An [`AbstractUniformAxis`](@ref) carries a constant spacing in its type; a plain
`Vector` has nothing but its samples. A stretched vertical coordinate is neither: the spacing genuinely
varies, so it is not uniform, but the samples are `n` evaluations of two or three parameters, and
storing them stores a formula's output.

A subtype implements three methods:

- `Base.length(a)`
- [`coordinate`](@ref)`(a, ξ)` — the coordinate at continuous index `ξ`, agreeing with `a[i]` at every
  integer `i`
- [`index_at`](@ref)`(a, x)` — its inverse, the continuous index whose coordinate is `x`

and gets indexing, the `O(1)` endpoint reductions, and — through the inverse — `locate`,
`nearest_index` and the interpolation weights without a search. Locating a coordinate on a stretched
`Vector` bisects in `O(log n)`; here the inverse names the cell directly and one comparison against the
neighbouring face settles the rounding.

The coordinate must be strictly monotone in `ξ`, which makes the inverse single-valued. A constructor
checks it once. Where `index_at` cannot answer — a coordinate outside the formula's domain, which a
far-extrapolated face can be — it returns a non-finite value and the caller falls back to the search,
reaching the same answer.
"""
abstract type AbstractAnalyticAxis{T<:AbstractFloat} <: AbstractVector{T} end

"""
    coordinate(a, ξ) -> T

The coordinate of [`AbstractAnalyticAxis`](@ref) `a` at continuous index `ξ`, equal to `a[i]` at every
integer index. A half-integer `ξ` is the midpoint in index space; this package's faces are midpoints in
coordinate space, so the two coincide only on a uniform axis. What is extended here is the formula.
"""
function coordinate end

"""
    index_at(a, x) -> T

The continuous index at which [`AbstractAnalyticAxis`](@ref) `a` has coordinate `x`: the inverse of
[`coordinate`](@ref), and the reason a query on such an axis needs no search.

Returns a non-finite value where `x` lies outside the formula's domain, which callers take as "no
closed-form answer" and fall back to bisection.
"""
function index_at end

spacing_trait(::AbstractAnalyticAxis) = NonuniformSpacing()

@inline Base.size(a::AbstractAnalyticAxis) = (length(a),)
Base.IndexStyle(::Type{<:AbstractAnalyticAxis}) = IndexLinear()

@inline function Base.getindex(a::AbstractAnalyticAxis{T}, i::Int) where {T}
    @boundscheck checkbounds(a, i)
    return coordinate(a, T(i))
end

@inline Base.first(a::AbstractAnalyticAxis{T}) where {T} = coordinate(a, one(T))
@inline Base.last(a::AbstractAnalyticAxis{T}) where {T} = coordinate(a, T(length(a)))

# Strictly monotone by contract, so the extremes are the endpoints and no scan is needed.
@inline Base.minimum(a::AbstractAnalyticAxis) = isempty(a) ? _empty_reduce(a, "minimum") :
    (last(a) ≥ first(a) ? first(a) : last(a))
@inline Base.maximum(a::AbstractAnalyticAxis) = isempty(a) ? _empty_reduce(a, "maximum") :
    (last(a) ≥ first(a) ? last(a) : first(a))
@inline Base.extrema(a::AbstractAnalyticAxis) = (minimum(a), maximum(a))

@inline wrap_sign(a::AbstractAnalyticAxis{T}) where {T} =
    length(a) < 2 ? one(T) : (last(a) ≥ first(a) ? one(T) : -one(T))

Base.show(io::IO, ::MIME"text/plain", a::AbstractAnalyticAxis{T}) where {T} =
    (print(io, length(a), "-element ", nameof(typeof(a)), "{", T, "}: "); show(io, a))

"""
    GeometricAxis(origin, Δ, ratio, n)
    GeometricAxis{T}(origin, Δ, ratio, n)

`n` samples whose successive gaps are `Δ, Δ·r, Δ·r², …` — the stretched grid a boundary layer or a
model's vertical levels are built on — held as four numbers.

    x(ξ) = origin + Δ·(r^(ξ-1) − 1)/(r − 1)

so `x(1) = origin` and `x(i+1) − x(i) = Δ·r^(i-1)`. The inverse is a logarithm, so locating a
coordinate is `O(1)`.

`r > 0` and `r ≠ 1`. At `r == 1` the gaps are constant, which is a [`UniformAxis`](@ref); that type
carries the spacing where this one hides it in a parameter.
"""
struct GeometricAxis{T<:AbstractFloat} <: AbstractAnalyticAxis{T}
    origin::T
    Δ::T
    ratio::T
    n::Int

    function GeometricAxis{T}(origin, Δ, ratio, n::Integer) where {T<:AbstractFloat}
        r = convert(T, ratio)
        d = convert(T, Δ)
        n ≥ 0 || throw(ArgumentError("a GeometricAxis needs n ≥ 0, got $n"))
        r > 0 || throw(ArgumentError("a GeometricAxis needs a positive ratio, got $r"))
        r != one(T) || throw(ArgumentError(
            "a ratio of 1 makes the gaps constant; that axis is a UniformAxis, which says so in its " *
            "type — `UniformAxis(origin, Δ, n)`",
        ))
        iszero(d) || return new{T}(convert(T, origin), d, r, Int(n))
        throw(ArgumentError("a GeometricAxis needs a nonzero first gap, got $d"))
    end
end

function GeometricAxis(origin::Real, Δ::Real, ratio::Real, n::Integer)
    T = float(promote_type(typeof(origin), typeof(Δ), typeof(ratio)))
    return GeometricAxis{T}(origin, Δ, ratio, n)
end

@inline Base.length(a::GeometricAxis) = a.n

@inline coordinate(a::GeometricAxis{T}, ξ::Real) where {T} =
    a.origin + a.Δ * (a.ratio^(T(ξ) - one(T)) - one(T)) / (a.ratio - one(T))

@inline function index_at(a::GeometricAxis{T}, x::Real) where {T}
    u = one(T) + (T(x) - a.origin) * (a.ratio - one(T)) / a.Δ
    # `u ≤ 0` is outside the formula's reach — the gaps shrink geometrically towards a finite limit
    # point, and no index maps beyond it.
    return u > zero(T) ? one(T) + log(u) / log(a.ratio) : T(NaN)
end

Base.show(io::IO, a::GeometricAxis{T}) where {T} = print(
    io, "GeometricAxis{", T, "}(", a.origin, ", Δ=", a.Δ, ", r=", a.ratio, ", n=", a.n, ")",
)

"""
    PowerAxis(origin, extent, exponent, n)
    PowerAxis{T}(origin, extent, exponent, n)

`n` samples spanning `origin` to `origin + extent` with the index mapped through a power:

    x(ξ) = origin + extent·((ξ-1)/(n-1))^p

`p == 1` is uniform, `p > 1` clusters samples near `origin`, and `p < 1` clusters them near the far
end — the shape ocean depth levels and a stretched radial coordinate are usually given. The inverse is
a root, so locating a coordinate on it is `O(1)`.

`n ≥ 2`, `p > 0`, and a nonzero extent.
"""
struct PowerAxis{T<:AbstractFloat} <: AbstractAnalyticAxis{T}
    origin::T
    extent::T
    exponent::T
    n::Int

    function PowerAxis{T}(origin, extent, exponent, n::Integer) where {T<:AbstractFloat}
        L = convert(T, extent)
        p = convert(T, exponent)
        n ≥ 2 || throw(ArgumentError("a PowerAxis needs n ≥ 2, got $n"))
        p > 0 || throw(ArgumentError("a PowerAxis needs a positive exponent, got $p"))
        iszero(L) && throw(ArgumentError("a PowerAxis needs a nonzero extent"))
        return new{T}(convert(T, origin), L, p, Int(n))
    end
end

function PowerAxis(origin::Real, extent::Real, exponent::Real, n::Integer)
    T = float(promote_type(typeof(origin), typeof(extent), typeof(exponent)))
    return PowerAxis{T}(origin, extent, exponent, n)
end

@inline Base.length(a::PowerAxis) = a.n

@inline function coordinate(a::PowerAxis{T}, ξ::Real) where {T}
    s = (T(ξ) - one(T)) / T(a.n - 1)
    # Below the first sample the power is not real; the formula is continued by odd reflection, which
    # keeps it monotone and keeps `index_at` its exact inverse there.
    return a.origin + a.extent * (s ≥ zero(T) ? s^a.exponent : -((-s)^a.exponent))
end

@inline function index_at(a::PowerAxis{T}, x::Real) where {T}
    u = (T(x) - a.origin) / a.extent
    s = u ≥ zero(T) ? u^inv(a.exponent) : -((-u)^inv(a.exponent))
    return one(T) + s * T(a.n - 1)
end

Base.show(io::IO, a::PowerAxis{T}) where {T} = print(
    io, "PowerAxis{", T, "}(", a.origin, ", extent=", a.extent, ", p=", a.exponent, ", n=", a.n, ")",
)

# ---------------------------------------------------------------------------
# ConstantVector
# ---------------------------------------------------------------------------

"""
    ConstantVector(value, n)

`n` copies of `value`, stored as that value and a length. A uniform axis's per-cell width is one.

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

# One call to `f` covers all `n` entries.
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
