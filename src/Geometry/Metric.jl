
"""
    AbstractGeometry{T<:AbstractFloat}

Supertype for coordinate metrics (`T` is the float type).
"""
abstract type AbstractGeometry{T<:AbstractFloat} end

"""
    AbstractCartesianGeometry{T} <: AbstractGeometry{T}

Cartesian metrics.  Default: [`CartesianGeometry`](@ref).
"""
abstract type AbstractCartesianGeometry{T<:AbstractFloat} <: AbstractGeometry{T} end

"""
    AbstractSphericalGeometry{T} <: AbstractGeometry{T}

Spherical metrics.  Default: [`SphericalGeometry`](@ref).

A subtype implements [`radius`](@ref); every other method here is written in terms of it.
"""
abstract type AbstractSphericalGeometry{T<:AbstractFloat} <: AbstractGeometry{T} end

"""
    radius(geo) -> T

The sphere radius. The one method a spherical geometry supplies; defaults to the `R` field.
"""
@inline radius(geo::AbstractSphericalGeometry) = geo.R

"""
    CartesianGeometry()
    CartesianGeometry(T)
    CartesianGeometry{T}()

Flat metric in `T`, of any dimension.

It carries no grid spacing. A cell's extent belongs to the grid's axes, which state it per cell and per
direction, and the dimension likewise lives on the grid.
"""
struct CartesianGeometry{T<:AbstractFloat} <: AbstractCartesianGeometry{T} end

CartesianGeometry() = CartesianGeometry{Float64}()
CartesianGeometry(::Type{T}) where {T<:AbstractFloat} = CartesianGeometry{T}()

"""
    SphericalGeometry{T} <: AbstractSphericalGeometry{T}

Default spherical geometry with sphere radius `R` (meters; default Earth 6.371e6).
"""
struct SphericalGeometry{T<:AbstractFloat} <: AbstractSphericalGeometry{T}
    R::T
end

SphericalGeometry() = SphericalGeometry(6.371e6)

# ---------------------------------------------------------------------------
# Point representation
# ---------------------------------------------------------------------------
# Points are accepted as a `Tuple`, `NamedTuple`, or `AbstractVector` of any `Real` element type,
# and converted to the geometry's float type on entry. They come back as a named `NamedTuple`, or as
# any representation the caller names via a leading type argument — mirroring
# `Grids.coords(grid, I...)` / `Grids.coords(S, grid, I...)`:
#
#     project_to_tangent_plane(geo, c, n)                      # (; x = …, y = …)
#     project_to_tangent_plane(SVector{2,Float64}, geo, c, n)  # SVector{2,Float64}

"""
    point_names(geo, Val(N)) -> NTuple{N,Symbol}

Field names for an `N`-coordinate point in this geometry.
Cartesian: `(:x,)`, `(:x,:y)`, `(:x,:y,:z)`, then `(:x1, …, :xN)`.
Spherical: `(:λ,)`, `(:λ,:φ)`, `(:λ,:φ,:r)`, then `(:λ,:φ,:r,:q4,…)`.
"""
point_names(::AbstractCartesianGeometry, ::Val{1}) = (:x,)
point_names(::AbstractCartesianGeometry, ::Val{2}) = (:x, :y)
point_names(::AbstractCartesianGeometry, ::Val{3}) = (:x, :y, :z)
point_names(::AbstractSphericalGeometry, ::Val{1}) = (:λ,)
point_names(::AbstractSphericalGeometry, ::Val{2}) = (:λ, :φ)
point_names(::AbstractSphericalGeometry, ::Val{3}) = (:λ, :φ, :r)

# Past the named directions the letters run out, so they are numbered from where each convention ends.
# Generated, so the symbols are built once at compile time: `Symbol(:x, d)` goes through `string`, and
# `getproperty` reaches these per cell, where an allocation on every access is felt.
@generated point_names(::AbstractCartesianGeometry, ::Val{N}) where {N} =
    :($(ntuple(d -> Symbol(:x, d), N)))
@generated point_names(::AbstractSphericalGeometry, ::Val{N}) where {N} =
    :($((:λ, :φ, :r, ntuple(d -> Symbol(:q, d + 3), N - 3)...)))

"""
    build_point(S, names::NTuple{N,Symbol}, vals::NTuple{N}) -> S

Assemble a point/vector in representation `S` from positional `vals` and their `names`.
Bare `NamedTuple` takes `names`; a parameterized `NamedTuple` type takes its own. Everything else
is built from `vals` alone, covering `Tuple`, `NTuple{N,T}`, `Vector{T}`, `SVector{N,T}`,
`MVector{N,T}`, and any user type with a positional constructor.
"""
@inline build_point(::Type{NamedTuple}, names::NTuple{N,Symbol}, vals::NTuple{N,Any}) where {N} =
    NamedTuple{names}(vals)
@inline build_point(::Type{S}, ::NTuple{N,Symbol}, vals::NTuple{N,Any}) where {N,S<:NamedTuple} = S(vals)
@inline build_point(::Type{S}, ::NTuple{N,Symbol}, vals::NTuple{N,Any}) where {N,S<:Tuple} = convert(S, vals)
@inline build_point(::Type{Vector{T}}, ::NTuple{N,Symbol}, vals::NTuple{N,Any}) where {N,T} = T[vals...]
@inline build_point(::Type{S}, ::NTuple{N,Symbol}, vals::NTuple{N,Any}) where {N,S} = S(vals...)

"""Component names of an ambient-frame vector (as returned by [`local_tangent_basis`](@ref))."""
@inline _ambient_names(::Val{1}) = (:x,)
@inline _ambient_names(::Val{2}) = (:x, :y)
@inline _ambient_names(::Val{3}) = (:x, :y, :z)

"""
    named_point(geo, vals::NTuple{N}) -> NamedTuple

Build the geometry-appropriate named point from positional values (axis order).
"""
@inline named_point(geo::AbstractGeometry, vals::NTuple{N,Any}) where {N} =
    build_point(NamedTuple, point_names(geo, Val(N)), vals)

# The point spellings other than a `Tuple`. An entry point takes a tuple method plus one forwarding
# method over this union; a forwarder written for `Any` is ambiguous with the tuple method, a `Tuple`
# matching both.
const PointLike = Union{NamedTuple,AbstractVector{<:Real}}

"""
    as_ntuple(p) -> Tuple
    as_ntuple(p, Val(N)) -> NTuple{N}

Normalize a point (`Tuple`, `NamedTuple`, or `AbstractVector` of length 1–3) to a plain `Tuple`.

A vector's length is a runtime value, so for one the first form returns a union of the three tuple
widths. Every entry point here reduces a point to a value whose type does not depend on that width —
`distance` to a scalar, `spherical_to_cartesian` to three components — so the union splits across the
branch and each arm infers concretely and allocates nothing.

[`scale_factors`](@ref) is the exception, its result being one factor per direction. Name the width
with `Val(N)` there, or hand it a tuple.
"""
@inline as_ntuple(p::NamedTuple) = Tuple(p)
@inline as_ntuple(p::Tuple) = p
@inline function as_ntuple(p::AbstractVector)
    n = length(p)
    @inbounds if n == 2
        return (p[1], p[2])
    elseif n == 3
        return (p[1], p[2], p[3])
    elseif n == 1
        return (p[1],)
    end
    throw(ArgumentError("a point vector must have length 1, 2, or 3"))
end

@inline as_ntuple(p::NTuple{N,Any}, ::Val{N}) where {N} = p
@inline as_ntuple(p::Tuple, ::Val{N}) where {N} = throw(ArgumentError(
    "this needs a point with $N component(s); got $(length(p))",
))
@inline as_ntuple(p::NamedTuple, v::Val) = as_ntuple(Tuple(p), v)
@inline function as_ntuple(p::AbstractVector, ::Val{N}) where {N}
    length(p) == N || throw(ArgumentError(
        "this needs a point with $N component(s); got $(length(p))",
    ))
    return ntuple(i -> @inbounds(p[i]), Val(N))
end

"""
    _lonlat(p) -> (λ, φ)
    _xyz(v) -> (x, y, z)

A point's components for a consumer that needs a fixed number of them, with the arity checked.

`λ, φ = as_ntuple(p)` does not check it: on a one-component point the destructure reads `p[2]` and
reports an index out of bounds, when what is wrong is the point. These say so instead, and return a pair
or a triple whose type does not depend on how many components the point had.

The arity is the dispatch, so the accepted width is a method signature and the tuple is indexed only
where its type is known wide enough.
"""
@inline _lonlat(p) = _lonlat(as_ntuple(p))
@inline _lonlat(q::Tuple{Any,Any,Vararg{Any}}) = (q[1], q[2])
@inline _lonlat(q::Tuple) = throw(ArgumentError(
    "this needs a point with a longitude and a latitude; got $(length(q)) component(s)",
))

@inline _xyz(v) = _xyz(as_ntuple(v))
@inline _xyz(q::NTuple{3,Any}) = q
@inline _xyz(q::Tuple) = throw(ArgumentError(
    "this needs three Cartesian components; got $(length(q))",
))

"""Convert positional point values to the geometry's own float type `T`."""
@inline _at(::Type{T}, p::Tuple{Vararg{Real,N}}) where {T<:AbstractFloat,N} = convert(NTuple{N,T}, p)

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

"""
    distance(geo::AbstractGeometry, pt1, pt2)

Distance between two points. Spherical 2D uses great-circle (Haversine); spherical 3D uses the
chord through Cartesian, with `r` the absolute radius from the origin; a lone `(λ,)` is the shorter
arc of its circle.
"""
@inline distance(geo::AbstractGeometry, pt1, pt2) = _distance(geo, as_ntuple(pt1), as_ntuple(pt2))

@inline function _distance(
    ::AbstractCartesianGeometry{T}, pt1::Tuple{Vararg{Real,N}}, pt2::Tuple{Vararg{Real,N}},
) where {T,N}
    p1 = _at(T, pt1)
    p2 = _at(T, pt2)
    s = zero(T)
    @inbounds for i in 1:N
        d = p1[i] - p2[i]
        s += d * d
    end
    return sqrt(s)
end

# One coordinate is a point on the circle of latitude 0: the distance is the shorter arc.
@inline function _distance(
    geo::AbstractSphericalGeometry{T}, coords1::Tuple{Real}, coords2::Tuple{Real},
) where {T}
    Δλ = convert(T, coords2[1]) - convert(T, coords1[1])
    return radius(geo) * abs(rem2pi(Δλ, RoundNearest))
end

@inline function _distance(
    geo::AbstractSphericalGeometry{T}, coords1::Tuple{Real,Real}, coords2::Tuple{Real,Real},
) where {T}
    λ1, φ1 = _at(T, coords1)
    λ2, φ2 = _at(T, coords2)
    dλ = λ2 - λ1
    dφ = φ2 - φ1
    a = sin(dφ / T(2))^2 + cos(φ1) * cos(φ2) * sin(dλ / T(2))^2
    c = T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a)))
    return radius(geo) * c
end

@inline function _distance(
    geo::AbstractSphericalGeometry{T}, coords1::Tuple{Real,Real,Real}, coords2::Tuple{Real,Real,Real},
) where {T}
    p1 = _spherical_to_cartesian(geo, coords1)
    p2 = _spherical_to_cartesian(geo, coords2)
    dx = p1.x - p2.x; dy = p1.y - p2.y; dz = p1.z - p2.z
    return sqrt(dx * dx + dy * dy + dz * dz)
end

"""
    spherical_to_cartesian(geo, coords) -> NamedTuple{(:x,:y,:z)}
    spherical_to_cartesian(S, geo, coords) -> S

`(λ, φ)` on the reference sphere of [`radius`](@ref) `R`, or `(λ, φ, r)` with `r` the absolute radius from
the origin, mapped to Cartesian coordinates of the same space.
"""
@inline spherical_to_cartesian(geo::AbstractSphericalGeometry, coords) =
    _spherical_to_cartesian(geo, as_ntuple(coords))

@inline spherical_to_cartesian(::Type{S}, geo::AbstractSphericalGeometry, coords) where {S} =
    build_point(S, (:x, :y, :z), Tuple(spherical_to_cartesian(geo, coords)))

@inline function _spherical_to_cartesian(
    ::AbstractSphericalGeometry{T}, coords::Tuple{Real,Real,Real},
) where {T}
    λ, φ, r = _at(T, coords)
    sinλ, cosλ = sincos(λ)
    sinφ, cosφ = sincos(φ)
    return (; x = r * cosφ * cosλ, y = r * cosφ * sinλ, z = r * sinφ)
end

@inline function _spherical_to_cartesian(
    geo::AbstractSphericalGeometry{T}, coords::Tuple{Real,Real},
) where {T}
    λ, φ = _at(T, coords)
    sinλ, cosλ = sincos(λ)
    sinφ, cosφ = sincos(φ)
    R = radius(geo)
    return (; x = R * cosφ * cosλ, y = R * cosφ * sinλ, z = R * sinφ)
end

# A lone `(λ,)` is a point on the equator, the circle of radius `R` — the convention [`distance`](@ref)
# uses for one coordinate.
@inline function _spherical_to_cartesian(
    geo::AbstractSphericalGeometry{T}, coords::Tuple{Real},
) where {T}
    sinλ, cosλ = sincos(convert(T, coords[1]))
    R = radius(geo)
    return (; x = R * cosλ, y = R * sinλ, z = zero(T))
end

"""
    cartesian_to_spherical(geo, xyz) -> NamedTuple{(:λ,:φ,:r)}
    cartesian_to_spherical(S, geo, xyz) -> S

Inverse of [`spherical_to_cartesian`](@ref): Cartesian `(x, y, z)` to `(λ, φ, r)` with `λ ∈ (-π, π]`,
`φ ∈ [-π/2, π/2]`, and `r` the absolute radius from the origin. At the origin `λ = φ = 0`.
"""
@inline cartesian_to_spherical(geo::AbstractSphericalGeometry, xyz) =
    _cartesian_to_spherical(geo, as_ntuple(xyz))

@inline cartesian_to_spherical(::Type{S}, geo::AbstractSphericalGeometry, xyz) where {S} =
    build_point(S, (:λ, :φ, :r), Tuple(cartesian_to_spherical(geo, xyz)))

@inline function _cartesian_to_spherical(
    ::AbstractSphericalGeometry{T}, xyz::Tuple{Real,Real,Real},
) where {T}
    x, y, z = _at(T, xyz)
    r = sqrt(x * x + y * y + z * z)
    iszero(r) && return (; λ = zero(T), φ = zero(T), r = r)
    return (; λ = atan(y, x), φ = asin(clamp(z / r, -one(T), one(T))), r = r)
end

# ---------------------------------------------------------------------------
# The ambient Cartesian space
# ---------------------------------------------------------------------------

"""
    embed(geo, coords) -> NTuple{D,T}
    embed(S, geo, coords) -> S

A point's position in the ambient Cartesian space this metric is realized in: the coordinates
themselves on the plane, the point on the sphere of [`radius`](@ref) `R`, the Earth-centred position of
a geodetic `(λ, φ[, h])` on a spheroid.

One of the two primitives the metric-independent constructions here are written against; the other is
[`local_tangent_basis`](@ref). A straight-line displacement, a chord distance, a tangent-plane
projection and a least-squares gradient need a position and a frame and nothing else, so each is
written once and serves every geometry supplying those two.

This is the ambient position, always. What a spatial index searches is a separate question — an index
may scale a sphere's surface coordinates to make an arc comparable to a chord, which is
[`FlowGeometries.Grids.embed_point`](@ref) and its [`FlowGeometries.Grids.AbstractEmbedding`](@ref).
"""
function embed end

@inline embed(geo::AbstractGeometry, coords) = _embed(geo, as_ntuple(coords))

@inline function embed(::Type{S}, geo::AbstractGeometry, coords) where {S}
    v = embed(geo, coords)
    return build_point(S, _ambient_names(Val(length(v))), v)
end

@inline _embed(::AbstractCartesianGeometry{T}, p::Tuple{Vararg{Real,N}}) where {T,N} = _at(T, p)

@inline _embed(geo::AbstractSphericalGeometry, p::Tuple) = Tuple(_spherical_to_cartesian(geo, p))

# ---------------------------------------------------------------------------
# Area Elements
# ---------------------------------------------------------------------------

"""
    area_element(geo::AbstractCartesianGeometry, dx, dy)
    area_element(geo::AbstractSphericalGeometry, φ, dλ, dφ)

Local cell area from the cell's own extents: `dx·dy` on the plane, `R²·cosφ·dλ·dφ` on the sphere.
"""
@inline area_element(::AbstractCartesianGeometry{T}, dx::Real, dy::Real) where {T} =
    convert(T, dx) * convert(T, dy)

@inline function area_element(geo::AbstractSphericalGeometry{T}, φ::T, dλ::T, dφ::T) where {T}
    return radius(geo)^2 * cos(φ) * dλ * dφ
end

"""
    volume_element(geo::AbstractCartesianGeometry, dx, dy, dz)
    volume_element(geo::AbstractSphericalGeometry, r, φ, dλ, dφ, dr)

Local cell volume from the cell's own extents. The spherical form takes the local radius `r` at this
level, giving the shell element `r²·cosφ·dλ·dφ·dr`; the reference [`radius`](@ref) does not enter.
"""
@inline volume_element(::AbstractCartesianGeometry{T}, dx::Real, dy::Real, dz::Real) where {T} =
    convert(T, dx) * convert(T, dy) * convert(T, dz)

@inline function volume_element(::AbstractSphericalGeometry{T}, r::T, φ::T, dλ::T, dφ::T, dr::T) where {T}
    return r^2 * cos(φ) * dλ * dφ * dr
end

# ---------------------------------------------------------------------------
# Nonuniform finite differences
# ---------------------------------------------------------------------------

"""
    nonuniform_first_derivative(f_m, f_0, f_p, h_m, h_p)

Standard 3-point, 2nd-order-accurate centered finite-difference approximation of the first
derivative at the middle node on a possibly *nonuniform* stencil. `f_m`, `f_0`, `f_p` are the
function values at the minus/center/plus nodes, and `h_m = x_0 - x_{-}`, `h_p = x_{+} - x_0 > 0` are
the (physical) left/right spacings.

Reduces exactly to the uniform central difference `(f_p - f_m)/(2h)` when `h_m == h_p == h`, and is
exact for linear and quadratic `f` for any `h_m`, `h_p`.
"""
@inline function nonuniform_first_derivative(f_m::T, f_0::T, f_p::T, h_m::T, h_p::T) where {T<:AbstractFloat}
    return (h_m^2 * f_p + (h_p^2 - h_m^2) * f_0 - h_p^2 * f_m) / (h_m * h_p * (h_m + h_p))
end
