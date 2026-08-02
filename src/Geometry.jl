module Geometry

# Public symbols are reached as `FlowGeometries.Geometry.*` or rebound on `FlowGeometries`
# for `FG.distance`-style calls. Internals (`as_ntuple`, `point_names`, `named_point`, …)
# are intentionally not exported.

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

It carries no grid spacing. A cell's extent belongs to the grid's axes, which already state it per
cell and per direction; a nominal `dx`/`dy`/`dz` on the geometry would be a second, unchecked copy
that could contradict them. The dimension likewise lives on the grid.
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
# and converted to the geometry's float type on entry. Points and vectors are RETURNED as a named
# `NamedTuple`, or as any representation the caller names via a leading type argument — mirroring
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
# these are reached per-cell by `getproperty`, where a runtime symbol build would allocate on every access.
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
@inline _ambient_names(::Val{2}) = (:x, :y)
@inline _ambient_names(::Val{3}) = (:x, :y, :z)

"""
    named_point(geo, vals::NTuple{N}) -> NamedTuple

Build the geometry-appropriate named point from positional values (axis order).
"""
@inline named_point(geo::AbstractGeometry, vals::NTuple{N,Any}) where {N} =
    build_point(NamedTuple, point_names(geo, Val(N)), vals)

"""
    as_ntuple(p) -> Tuple

Normalize a point (`Tuple`, `NamedTuple`, or `AbstractVector` of length 1–3) to a plain `Tuple`.
"""
# The point spellings that are NOT a `Tuple`. An entry point takes a tuple method plus one forwarding
# method over this union: writing the forwarder for `Any` instead makes it ambiguous with the tuple
# method, since a `Tuple` matches both.
const PointLike = Union{NamedTuple,AbstractVector{<:Real}}

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

Local cell volume from the cell's own extents. The spherical form uses the LOCAL radius `r` at this
level, not the reference [`radius`](@ref), so it is the genuine shell element `r²·cosφ·dλ·dφ·dr`.
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

# ---------------------------------------------------------------------------
# Spherical vector fields ↔ Cartesian
# ---------------------------------------------------------------------------
# Orthonormal frame at (λ, φ): ê_λ (∂/∂λ), ê_φ (∂/∂φ), ê_r (radial). Same directions
# as geographic east / north / up — named for the polar coordinates used everywhere else.

"""
    vector_to_cartesian(geo, u_λ, u_φ, u_r, λ, φ) -> NamedTuple{(:x,:y,:z)}
    vector_to_cartesian(geo, u_λ, u_φ, λ, φ) -> NamedTuple{(:x,:y,:z)}
    vector_to_cartesian(S, geo, args...) -> S

Map the components of a vector given in the local spherical basis at `(λ, φ)` into the Cartesian
basis. This transforms VECTOR COMPONENTS at a point, not a position — see
[`spherical_to_cartesian`](@ref) for positions. The 2-component form assumes `u_r = 0`.
"""
@inline function vector_to_cartesian(
    ::AbstractSphericalGeometry{T},
    u_λ::Real, u_φ::Real, u_r::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    uλ = convert(T, u_λ)
    uφ = convert(T, u_φ)
    ur = convert(T, u_r)
    λT = convert(T, λ)
    φT = convert(T, φ)
    sinλ, cosλ = sin(λT), cos(λT)
    sinφ, cosφ = sin(φT), cos(φT)
    return (;
        x = uλ * (-sinλ) + uφ * (-sinφ * cosλ) + ur * (cosφ * cosλ),
        y = uλ * cosλ    + uφ * (-sinφ * sinλ) + ur * (cosφ * sinλ),
        z =                uφ * cosφ           + ur * sinφ,
    )
end

@inline function vector_to_cartesian(
    geo::AbstractSphericalGeometry{T}, u_λ::Real, u_φ::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    return vector_to_cartesian(geo, u_λ, u_φ, zero(T), λ, φ)
end

@inline vector_to_cartesian(::Type{S}, geo::AbstractSphericalGeometry, args::Vararg{Real}) where {S} =
    build_point(S, (:x, :y, :z), Tuple(vector_to_cartesian(geo, args...)))

"""
    vector_from_cartesian(geo, ux, uy, uz, λ, φ) -> NamedTuple{(:λ,:φ,:r)}
    vector_from_cartesian(geo, v, λ, φ) -> NamedTuple{(:λ,:φ,:r)}
    vector_from_cartesian(S, geo, args...) -> S

Map Cartesian velocity `(ux, uy, uz)` at `(λ, φ)` into spherical components
`(λ = u_λ, φ = u_φ, r = u_r)`. `v` may be any 3-component point representation.
"""
@inline function vector_from_cartesian(
    ::AbstractSphericalGeometry{T},
    ux::Real, uy::Real, uz::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    uxT = convert(T, ux)
    uyT = convert(T, uy)
    uzT = convert(T, uz)
    λT = convert(T, λ)
    φT = convert(T, φ)
    sinλ, cosλ = sin(λT), cos(λT)
    sinφ, cosφ = sin(φT), cos(φT)
    return (;
        λ = uxT * (-sinλ) + uyT * cosλ,
        φ = uxT * (-sinφ * cosλ) + uyT * (-sinφ * sinλ) + uzT * cosφ,
        r = uxT * (cosφ * cosλ)  + uyT * (cosφ * sinλ)  + uzT * sinφ,
    )
end

@inline function vector_from_cartesian(geo::AbstractSphericalGeometry, v, λ::Real, φ::Real)
    ux, uy, uz = as_ntuple(v)
    return vector_from_cartesian(geo, ux, uy, uz, λ, φ)
end

@inline vector_from_cartesian(::Type{S}, geo::AbstractSphericalGeometry, args...) where {S} =
    build_point(S, (:λ, :φ, :r), Tuple(vector_from_cartesian(geo, args...)))

# `vᵀ τ w` for a symmetric `τ` given by its six independent components. Written out rather than looped
# over an indexed basis: the loop form measured about 5× slower, which is the reason callers hand-roll
# this instead of calling it, so the flat form belongs here and not in each of them.
@inline _quad(
    τxx::T, τyy::T, τzz::T, τxy::T, τxz::T, τyz::T,
    v1::T, v2::T, v3::T, w1::T, w2::T, w3::T,
) where {T} =
    v1 * w1 * τxx + v2 * w2 * τyy + v3 * w3 * τzz +
    (v1 * w2 + v2 * w1) * τxy + (v1 * w3 + v3 * w1) * τxz + (v2 * w3 + v3 * w2) * τyz

"""
    tensor_to_local(geo, τxx, τyy, τzz, τxy, τxz, τyz, λ, φ) -> NamedTuple
    tensor_to_local(geo, τ, λ, φ) -> NamedTuple
    tensor_to_local(S, geo, args...) -> S

Rotate a **symmetric rank-2 tensor** from the ambient Cartesian frame into the local
`(ê_λ, ê_φ, ê_r)` frame at `(λ, φ)`: `τ' = R τ Rᵀ`, returning
`(; λλ, φφ, rr, λφ, λr, φr)`.

The rank-1 counterpart is [`vector_from_cartesian`](@ref); between them they cover essentially every
field anyone rotates — a stress, a strain rate, a covariance, a subfilter flux. `τ` may be given as the
six independent components or as any 6-component representation, in the order above.

`R` is the same basis [`local_tangent_basis`](@ref) defines, so the two cannot drift apart.
"""
@inline function tensor_to_local(
    ::AbstractSphericalGeometry{T},
    τxx::Real, τyy::Real, τzz::Real, τxy::Real, τxz::Real, τyz::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    xx, yy, zz = convert(T, τxx), convert(T, τyy), convert(T, τzz)
    xy, xz, yz = convert(T, τxy), convert(T, τxz), convert(T, τyz)
    sinλ, cosλ = sincos(convert(T, λ))
    sinφ, cosφ = sincos(convert(T, φ))
    a1, a2, a3 = -sinλ, cosλ, zero(T)                       # ê_λ
    b1, b2, b3 = -sinφ * cosλ, -sinφ * sinλ, cosφ           # ê_φ
    c1, c2, c3 = cosφ * cosλ, cosφ * sinλ, sinφ             # ê_r
    return (;
        λλ = _quad(xx, yy, zz, xy, xz, yz, a1, a2, a3, a1, a2, a3),
        φφ = _quad(xx, yy, zz, xy, xz, yz, b1, b2, b3, b1, b2, b3),
        rr = _quad(xx, yy, zz, xy, xz, yz, c1, c2, c3, c1, c2, c3),
        λφ = _quad(xx, yy, zz, xy, xz, yz, a1, a2, a3, b1, b2, b3),
        λr = _quad(xx, yy, zz, xy, xz, yz, a1, a2, a3, c1, c2, c3),
        φr = _quad(xx, yy, zz, xy, xz, yz, b1, b2, b3, c1, c2, c3),
    )
end

@inline function tensor_to_local(geo::AbstractSphericalGeometry, τ, λ::Real, φ::Real)
    xx, yy, zz, xy, xz, yz = as_tensor6(τ)
    return tensor_to_local(geo, xx, yy, zz, xy, xz, yz, λ, φ)
end

@inline tensor_to_local(::Type{S}, geo::AbstractSphericalGeometry, args...) where {S} =
    build_point(S, (:λλ, :φφ, :rr, :λφ, :λr, :φr), Tuple(tensor_to_local(geo, args...)))

"""
    tensor_from_local(geo, τλλ, τφφ, τrr, τλφ, τλr, τφr, λ, φ) -> NamedTuple
    tensor_from_local(geo, τ, λ, φ) -> NamedTuple
    tensor_from_local(S, geo, args...) -> S

The inverse of [`tensor_to_local`](@ref): `τ = Rᵀ τ' R`, returning the ambient Cartesian components
`(; xx, yy, zz, xy, xz, yz)`. Composing the two in either order is the identity.
"""
@inline function tensor_from_local(
    ::AbstractSphericalGeometry{T},
    τλλ::Real, τφφ::Real, τrr::Real, τλφ::Real, τλr::Real, τφr::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    ll, ff, rr = convert(T, τλλ), convert(T, τφφ), convert(T, τrr)
    lf, lr, fr = convert(T, τλφ), convert(T, τλr), convert(T, τφr)
    sinλ, cosλ = sincos(convert(T, λ))
    sinφ, cosφ = sincos(convert(T, φ))
    # The COLUMNS of `R`, which are the rows of `Rᵀ`, so the same contraction serves both directions.
    u1, u2, u3 = -sinλ, -sinφ * cosλ, cosφ * cosλ
    v1, v2, v3 = cosλ, -sinφ * sinλ, cosφ * sinλ
    w1, w2, w3 = zero(T), cosφ, sinφ
    return (;
        xx = _quad(ll, ff, rr, lf, lr, fr, u1, u2, u3, u1, u2, u3),
        yy = _quad(ll, ff, rr, lf, lr, fr, v1, v2, v3, v1, v2, v3),
        zz = _quad(ll, ff, rr, lf, lr, fr, w1, w2, w3, w1, w2, w3),
        xy = _quad(ll, ff, rr, lf, lr, fr, u1, u2, u3, v1, v2, v3),
        xz = _quad(ll, ff, rr, lf, lr, fr, u1, u2, u3, w1, w2, w3),
        yz = _quad(ll, ff, rr, lf, lr, fr, v1, v2, v3, w1, w2, w3),
    )
end

@inline function tensor_from_local(geo::AbstractSphericalGeometry, τ, λ::Real, φ::Real)
    ll, ff, rr, lf, lr, fr = as_tensor6(τ)
    return tensor_from_local(geo, ll, ff, rr, lf, lr, fr, λ, φ)
end

@inline tensor_from_local(::Type{S}, geo::AbstractSphericalGeometry, args...) where {S} =
    build_point(S, (:xx, :yy, :zz, :xy, :xz, :yz), Tuple(tensor_from_local(geo, args...)))

"""
    as_tensor6(τ) -> NTuple{6}

Normalize a symmetric rank-2 tensor to a plain 6-tuple of its independent components, from a `Tuple`,
`NamedTuple` or `AbstractVector` — the rank-2 counterpart of [`as_ntuple`](@ref).
"""
@inline as_tensor6(τ::NTuple{6,Any}) = τ
@inline as_tensor6(τ::NamedTuple) = Tuple(τ)
@inline function as_tensor6(τ::AbstractVector)
    length(τ) == 6 || throw(ArgumentError(
        "a symmetric rank-2 tensor needs its 6 independent components, got $(length(τ))",
    ))
    @inbounds return (τ[1], τ[2], τ[3], τ[4], τ[5], τ[6])
end

# ---------------------------------------------------------------------------
# Local tangent-plane geometry (curvilinear / unstructured gradient reconstruction)
# ---------------------------------------------------------------------------

"""
    local_tangent_basis(geo, coords) -> NamedTuple
    local_tangent_basis(S, geo, coords) -> NamedTuple of `S`

Coordinate-aligned unit vectors at `coords`, in the ambient Cartesian frame:
- Cartesian: `(; x = ê_x, y = ê_y)`, 2-component
- Spherical: `(; λ = ê_λ, φ = ê_φ)`, 3-component

The basis vectors themselves are `Tuple`s by default; pass a leading `S` to get them as `SVector`s
(or any other representation) so they can be used with vector arithmetic directly.
"""
@inline local_tangent_basis(geo::AbstractGeometry, coords) = _local_tangent_basis(geo, as_ntuple(coords))

@inline function local_tangent_basis(::Type{S}, geo::AbstractGeometry, coords) where {S}
    return map(v -> build_point(S, _ambient_names(Val(length(v))), v), local_tangent_basis(geo, coords))
end

@inline function _local_tangent_basis(::AbstractCartesianGeometry{T}, ::Tuple{Real,Real}) where {T}
    return (; x = (one(T), zero(T)), y = (zero(T), one(T)))
end

@inline function _local_tangent_basis(::AbstractSphericalGeometry{T}, coords::Tuple{Real,Real}) where {T}
    λ, φ = _at(T, coords)
    sinλ, cosλ = sincos(λ)
    sinφ, cosφ = sincos(φ)
    return (;
        λ = (-sinλ, cosλ, zero(T)),
        φ = (-sinφ * cosλ, -sinφ * sinλ, cosφ),
    )
end

"""
    project_to_tangent_plane(geo, center, neighbor) -> NamedTuple
    project_to_tangent_plane(S, geo, center, neighbor) -> S

Tangent-plane displacement of `neighbor` relative to `center` along the local
coordinate basis from [`local_tangent_basis`](@ref).
"""
@inline project_to_tangent_plane(geo::AbstractGeometry, center, neighbor) =
    _project_to_tangent_plane(geo, as_ntuple(center), as_ntuple(neighbor))

@inline function project_to_tangent_plane(::Type{S}, geo::AbstractGeometry, center, neighbor) where {S}
    Δ = project_to_tangent_plane(geo, center, neighbor)
    return build_point(S, keys(Δ), Tuple(Δ))
end

@inline function _project_to_tangent_plane(
    ::AbstractCartesianGeometry{T}, center::Tuple{Real,Real}, neighbor::Tuple{Real,Real},
) where {T}
    cx, cy = _at(T, center)
    nx, ny = _at(T, neighbor)
    return (; x = nx - cx, y = ny - cy)
end

@inline function _project_to_tangent_plane(
    geo::AbstractSphericalGeometry{T}, center::Tuple{Real,Real}, neighbor::Tuple{Real,Real},
) where {T}
    c = _at(T, center)
    Pc = _spherical_to_cartesian(geo, c)
    Pn = _spherical_to_cartesian(geo, _at(T, neighbor))
    dx = Pn.x - Pc.x; dy = Pn.y - Pc.y; dz = Pn.z - Pc.z
    ê = _local_tangent_basis(geo, c)
    êλ = ê.λ
    êφ = ê.φ
    # Written out rather than `sum(dr[i] * ê[i] for i in 1:3)`: the generator indexes the tuple with
    # a loop variable, which does not unroll, and measured ~5× slower than the unrolled contraction.
    return (;
        λ = dx * êλ[1] + dy * êλ[2] + dz * êλ[3],
        φ = dx * êφ[1] + dy * êφ[2] + dz * êφ[3],
    )
end

# ---------------------------------------------------------------------------
# Metric scale factors
# ---------------------------------------------------------------------------

"""
    scale_factors(geo, point) -> NTuple

Physical length of a unit coordinate step in each direction at `point` — see
[`FlowGeometries.Discretization.scale_factors`](@ref).

Cartesian: `1` in every direction, the metric being the identity. Spherical: `(R·cosφ, R)` on the
surface of radius `R`, and `(r·cosφ, r, 1)` where a radius direction is present, `r` being that
point's own radius.
"""
@inline scale_factors(::AbstractCartesianGeometry{T}, p::Tuple{Vararg{Real,N}}) where {T,N} =
    ntuple(_ -> one(T), Val(N))

@inline function scale_factors(geo::AbstractSphericalGeometry{T}, p::Tuple{Real,Real}) where {T}
    φ = convert(T, p[2])
    R = radius(geo)
    return (R * cos(φ), R)
end

@inline function scale_factors(::AbstractSphericalGeometry{T}, p::Tuple{Real,Real,Real}) where {T}
    φ = convert(T, p[2])
    r = convert(T, p[3])
    return (r * cos(φ), r, one(T))
end

@inline scale_factors(geo::AbstractGeometry, p) = scale_factors(geo, as_ntuple(p))

"""
    jacobian(geo, point) -> T

`∏` of [`scale_factors`](@ref): the volume element per unit coordinate volume at `point`.
"""
@inline jacobian(geo::AbstractGeometry, p) = prod(scale_factors(geo, p))


# ---------------------------------------------------------------------------
# Ellipsoidal geometry
# ---------------------------------------------------------------------------

"""
    AbstractEllipsoidalGeometry{T} <: AbstractGeometry{T}

Oblate-spheroid metrics. Default: [`SpheroidGeometry`](@ref). Coordinate names match the spherical
convention, `(λ, φ[, h])`, with `φ` the GEODETIC latitude.

A subtype implements [`semimajor_axis`](@ref) and [`flattening`](@ref); the rest — `semiminor_axis`,
`eccentricity²`, the curvature radii, `distance`, `area_element`, `scale_factors` — follow from those
two.
"""
abstract type AbstractEllipsoidalGeometry{T<:AbstractFloat} <: AbstractGeometry{T} end

"""
    semimajor_axis(geo) -> T

Equatorial radius `a`. Defaults to the `a` field.
"""
@inline semimajor_axis(g::AbstractEllipsoidalGeometry) = g.a

"""
    flattening(geo) -> T

Flattening `f = (a-b)/a`. Defaults to the `f` field.
"""
@inline flattening(g::AbstractEllipsoidalGeometry) = g.f

point_names(::AbstractEllipsoidalGeometry, ::Val{1}) = (:λ,)
point_names(::AbstractEllipsoidalGeometry, ::Val{2}) = (:λ, :φ)
point_names(::AbstractEllipsoidalGeometry, ::Val{3}) = (:λ, :φ, :h)
@generated point_names(::AbstractEllipsoidalGeometry, ::Val{N}) where {N} =
    :($((:λ, :φ, :h, ntuple(d -> Symbol(:q, d + 3), N - 3)...)))

"""
    SpheroidGeometry(a, f)
    SpheroidGeometry()

Oblate spheroid of equatorial radius `a` and flattening `f = (a-b)/a`. The no-argument form is WGS 84,
`a = 6378137.0`, `f = 1/298.257223563`.

`distance`, `area_element`, `volume_element` and `scale_factors` differ from a sphere. Grid directions
are `(λ, φ, h)`, with `h` the height above the ellipsoid — not an absolute radius, which is what the
spherical third direction is.

Rectilinear grids and the index-space connectivity built on them work as they do for a sphere; the
samplings are purely angular and so carry over unchanged. The spherical *area* routines
(`unstructured_grid`'s Voronoi areas, the cell areas behind `spherical_grid`) do not: they are built on
spherical excess and `4πR²/n`, which are sphere identities, so they stay restricted to
[`AbstractSphericalGeometry`](@ref) rather than silently returning sphere areas for an ellipsoid.
"""
struct SpheroidGeometry{T<:AbstractFloat} <: AbstractEllipsoidalGeometry{T}
    a::T
    f::T

    function SpheroidGeometry{T}(a, f) where {T<:AbstractFloat}
        aT, fT = convert(T, a), convert(T, f)
        aT > 0 || throw(ArgumentError("the equatorial radius must be positive, got $aT"))
        zero(T) ≤ fT < one(T) || throw(ArgumentError("the flattening must lie in [0, 1), got $fT"))
        return new{T}(aT, fT)
    end
end

SpheroidGeometry(a::Real, f::Real) = SpheroidGeometry{float(promote_type(typeof(a), typeof(f)))}(a, f)
SpheroidGeometry() = SpheroidGeometry(6378137.0, inv(298.257223563))

"""Semi-minor axis `b = a(1-f)`."""
@inline function semiminor_axis(g::AbstractEllipsoidalGeometry)
    f = flattening(g)
    return semimajor_axis(g) * (one(f) - f)
end

"""First eccentricity squared, `e² = f(2-f)`."""
@inline eccentricity²(g::AbstractEllipsoidalGeometry) = flattening(g) * (2 - flattening(g))

"""
    prime_vertical_radius(geo, φ) -> T

`N(φ) = a / √(1 - e²sin²φ)`: the radius of curvature perpendicular to the meridian. A unit step in
longitude covers `N(φ)cosφ`.
"""
@inline function prime_vertical_radius(g::AbstractEllipsoidalGeometry{T}, φ::Real) where {T}
    s = sin(convert(T, φ))
    return semimajor_axis(g) / sqrt(one(T) - eccentricity²(g) * s * s)
end

"""
    meridional_radius(geo, φ) -> T

`M(φ) = a(1-e²) / (1 - e²sin²φ)^{3/2}`: the radius of curvature along the meridian. A unit step in
latitude covers `M(φ)`.
"""
@inline function meridional_radius(g::AbstractEllipsoidalGeometry{T}, φ::Real) where {T}
    s = sin(convert(T, φ))
    e² = eccentricity²(g)
    w = one(T) - e² * s * s
    return semimajor_axis(g) * (one(T) - e²) / (w * sqrt(w))
end

@inline function scale_factors(g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real,Real}) where {T}
    φ = convert(T, p[2])
    return (prime_vertical_radius(g, φ) * cos(φ), meridional_radius(g, φ))
end

@inline function scale_factors(g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real,Real,Real}) where {T}
    φ = convert(T, p[2])
    h = convert(T, p[3])
    return ((prime_vertical_radius(g, φ) + h) * cos(φ), meridional_radius(g, φ) + h, one(T))
end

"""
    area_element(geo::AbstractEllipsoidalGeometry, φ, dλ, dφ)

`M(φ)·N(φ)cosφ·dλ·dφ`, the exact ellipsoidal surface element — equal to
`a²(1-e²)cosφ / (1-e²sin²φ)²·dλ·dφ`.
"""
@inline function area_element(
    g::AbstractEllipsoidalGeometry{T}, φ::Real, dλ::Real, dφ::Real,
) where {T}
    φT = convert(T, φ)
    return meridional_radius(g, φT) * prime_vertical_radius(g, φT) * cos(φT) *
           convert(T, dλ) * convert(T, dφ)
end

"""
    volume_element(geo::AbstractEllipsoidalGeometry, φ, h, dλ, dφ, dh)

`(N(φ)+h)cosφ · (M(φ)+h) · dλ·dφ·dh`, the geodetic volume element at ellipsoidal height `h`.

Unlike the spherical form this does not factor into a function of `φ` times a function of `h`: both
curvature radii are offset by `h`, so the two directions are coupled.
"""
@inline function volume_element(
    g::AbstractEllipsoidalGeometry{T}, φ::Real, h::Real, dλ::Real, dφ::Real, dh::Real,
) where {T}
    φT, hT = convert(T, φ), convert(T, h)
    return (prime_vertical_radius(g, φT) + hT) * cos(φT) * (meridional_radius(g, φT) + hT) *
           convert(T, dλ) * convert(T, dφ) * convert(T, dh)
end

"""
    distance(geo::AbstractEllipsoidalGeometry, p1, p2)

2-D `(λ, φ)`: geodesic distance by Vincenty's inverse method, iterated to `1e-12` in the auxiliary
longitude, which holds the result to well under a millimetre at Earth scale.

Vincenty's iteration converges slowly for very nearly antipodal point pairs. It is capped, and on
reaching the cap the great-circle distance on a sphere of the mean radius is returned instead of a
half-converged number.

3-D `(λ, φ, h)`: the chord through Cartesian (ECEF), mirroring the spherical 3-D convention. A lone
`(λ,)` is the shorter arc of the equator.
"""
@inline distance(geo::AbstractEllipsoidalGeometry, pt1, pt2) =
    _spheroid_distance(geo, as_ntuple(pt1), as_ntuple(pt2))

"""
    unit_vector(T, p) -> NTuple{3,T}

The direction of the `(λ, φ)` point `p` on the unit sphere, in any accepted point representation. This
is [`spherical_to_cartesian`](@ref) with the radius divided out, as a bare tuple: the form the
spherical-triangle kernels want, computed once per vertex and reused across every triangle sharing it.
"""
@inline function unit_vector(::Type{T}, p) where {T<:AbstractFloat}
    λ, φ = as_ntuple(p)
    sinλ, cosλ = sincos(T(λ))
    sinφ, cosφ = sincos(T(φ))
    return (cosφ * cosλ, cosφ * sinλ, sinφ)
end

"""
    spherical_excess(a, b, c) -> T

Spherical excess of the triangle spanned by three UNIT vectors, via Van Oosterom & Strackee (1983):

    tan(E/2) = |a · (b × c)| / (1 + a·b + b·c + c·a)

One `atan` and no other transcendental, versus L'Huilier's three great-circle distances (each its own
trig) plus four tangents. It takes directions rather than `(λ, φ)` so a mesh can convert each vertex
once — see [`unit_vector`](@ref) — instead of re-deriving them per triangle. Multiply by `R²` for an
area, which is what [`triangle_area`](@ref) does.
"""
@inline function spherical_excess(a::NTuple{3,T}, b::NTuple{3,T}, c::NTuple{3,T}) where {T}
    num = a[1] * (b[2] * c[3] - b[3] * c[2]) +
          a[2] * (b[3] * c[1] - b[1] * c[3]) +
          a[3] * (b[1] * c[2] - b[2] * c[1])
    ab = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    bc = b[1] * c[1] + b[2] * c[2] + b[3] * c[3]
    ca = c[1] * a[1] + c[2] * a[2] + c[3] * a[3]
    return T(2) * abs(atan(num, one(T) + ab + bc + ca))
end

"""
    triangle_area(geo, p1, p2, p3) -> T

Exact area of the spherical triangle through three `(λ, φ)` points, each in any accepted representation.
`R²` times [`spherical_excess`](@ref).

Convenient rather than fast: it converts all three points every call. A mesh that already holds vertex
directions should use `spherical_excess` on those and scale once.
"""
@inline function triangle_area(
    geo::AbstractSphericalGeometry{T}, p1, p2, p3,
) where {T<:AbstractFloat}
    return triangle_area_from_unit_vectors(radius(geo)^2, unit_vector(T, p1), unit_vector(T, p2), unit_vector(T, p3))
end

"""
    triangle_area_from_unit_vectors(geo, u1, u2, u3) -> T
    triangle_area_from_unit_vectors(R², u1, u2, u3) -> T

[`triangle_area`](@ref) for vertices already held as unit vectors — `R²` times
[`spherical_excess`](@ref), with no coordinate conversion.

A separate name rather than a `triangle_area` method because the two cannot be told apart by dispatch:
a unit vector and a 3-D spherical point `(λ, φ, r)` are both 3-tuples.

Pass `R²` itself to walk many triangles: a mesh shares vertices between cells, so the squaring belongs
outside the loop alongside the one-per-vertex [`unit_vector`](@ref) call. That is the form the cell-area
and Voronoi paths here use.
"""
@inline function triangle_area_from_unit_vectors(
    geo::AbstractSphericalGeometry{T}, u1::NTuple{3,T}, u2::NTuple{3,T}, u3::NTuple{3,T},
) where {T<:AbstractFloat}
    return triangle_area_from_unit_vectors(radius(geo)^2, u1, u2, u3)
end

@inline function triangle_area_from_unit_vectors(
    R²::T, u1::NTuple{3,T}, u2::NTuple{3,T}, u3::NTuple{3,T},
) where {T<:AbstractFloat}
    return R² * spherical_excess(u1, u2, u3)
end

"""
    geodetic_to_cartesian(geo, coords) -> (; x, y, z)

`(λ, φ, h)` — geodetic latitude and height above the ellipsoid — to Earth-centred Cartesian:
`x = (N(φ)+h)cosφ·cosλ`, `y = (N(φ)+h)cosφ·sinλ`, `z = (N(φ)(1-e²)+h)sinφ`. The 2-D form takes
`h = 0`, the surface.
"""
@inline geodetic_to_cartesian(geo::AbstractEllipsoidalGeometry, coords) =
    _geodetic_to_cartesian(geo, as_ntuple(coords))

@inline _geodetic_to_cartesian(g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real,Real}) where {T} =
    _geodetic_to_cartesian(g, (p[1], p[2], zero(T)))

@inline function _geodetic_to_cartesian(
    g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real,Real,Real},
) where {T}
    λ, φ, h = _at(T, p)
    sinλ, cosλ = sincos(λ)
    sinφ, cosφ = sincos(φ)
    Nφ = prime_vertical_radius(g, φ)
    return (; x = (Nφ + h) * cosφ * cosλ, y = (Nφ + h) * cosφ * sinλ,
              z = (Nφ * (one(T) - eccentricity²(g)) + h) * sinφ)
end

# One coordinate is a point on the equator, a circle of radius `a`: the distance is the shorter arc —
# the same convention the 1-D grid measure uses.
@inline function _spheroid_distance(
    g::AbstractEllipsoidalGeometry{T}, p1::Tuple{Real}, p2::Tuple{Real},
) where {T}
    Δλ = convert(T, p2[1]) - convert(T, p1[1])
    return semimajor_axis(g) * abs(rem2pi(Δλ, RoundNearest))
end

function _spheroid_distance(
    g::AbstractEllipsoidalGeometry{T}, p1::Tuple{Real,Real,Real}, p2::Tuple{Real,Real,Real},
) where {T}
    c1 = _geodetic_to_cartesian(g, p1)
    c2 = _geodetic_to_cartesian(g, p2)
    return sqrt((c1.x - c2.x)^2 + (c1.y - c2.y)^2 + (c1.z - c2.z)^2)
end

function _spheroid_distance(
    g::AbstractEllipsoidalGeometry{T}, p1::Tuple{Real,Real}, p2::Tuple{Real,Real},
) where {T}
    λ1, φ1 = _at(T, p1)
    λ2, φ2 = _at(T, p2)
    a = semimajor_axis(g)
    b = semiminor_axis(g)
    f = flattening(g)
    L = λ2 - λ1
    U1 = atan((one(T) - f) * tan(φ1))
    U2 = atan((one(T) - f) * tan(φ2))
    sinU1, cosU1 = sincos(U1)
    sinU2, cosU2 = sincos(U2)
    λ = L
    # 1e-12 rad in the auxiliary longitude, Vincenty's own criterion: at Earth scale a looser bound
    # such as sqrt(eps) leaves ~0.1 m of position error, which is far above the method's accuracy.
    tol = max(T(1e-12), eps(T))
    sinσ = cosσ = σ = sinα² = cos2σm = zero(T)
    converged = false
    for _ in 1:200
        sinλ, cosλ = sincos(λ)
        sinσ = sqrt((cosU2 * sinλ)^2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ)^2)
        iszero(sinσ) && return zero(T)                 # coincident points
        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = atan(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        sinα² = sinα * sinα
        cos²α = one(T) - sinα²
        cos2σm = iszero(cos²α) ? zero(T) : cosσ - 2 * sinU1 * sinU2 / cos²α
        C = f / 16 * cos²α * (4 + f * (4 - 3 * cos²α))
        λprev = λ
        λ = L + (one(T) - C) * f * sinα *
            (σ + C * sinσ * (cos2σm + C * cosσ * (-one(T) + 2 * cos2σm^2)))
        if abs(λ - λprev) < tol
            converged = true
            break
        end
    end
    if !converged
        # Near-antipodal: fall back to a sphere of the mean radius rather than report a partial solve.
        R = (2 * a + b) / 3
        dλ = λ2 - λ1
        h = sin((φ2 - φ1) / T(2))^2 + cos(φ1) * cos(φ2) * sin(dλ / T(2))^2
        return R * 2 * atan(sqrt(h), sqrt(max(zero(T), one(T) - h)))
    end
    u² = sinα² == one(T) ? zero(T) : (one(T) - sinα²) * (a * a - b * b) / (b * b)
    A = one(T) + u² / 16384 * (4096 + u² * (-768 + u² * (320 - 175 * u²)))
    B = u² / 1024 * (256 + u² * (-128 + u² * (74 - 47 * u²)))
    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-one(T) + 2 * cos2σm^2) -
         B / 6 * cos2σm * (-3 + 4 * sinσ^2) * (-3 + 4 * cos2σm^2)))
    return b * A * (σ - Δσ)
end

# ---------------------------------------------------------------------------
# Pole rotation
# ---------------------------------------------------------------------------

"""
    PoleRotation(λp, φp)

The frame whose north pole sits at `(λp, φp)` of the original one — a rotated-pole grid's coordinate
change. Apply it with [`rotate`](@ref) and undo it with [`unrotate`](@ref).
"""
struct PoleRotation{T<:AbstractFloat}
    λp::T
    φp::T
end

PoleRotation(λp::Real, φp::Real) =
    PoleRotation{float(promote_type(typeof(λp), typeof(φp)))}(λp, φp)

"""
    rotate(rot, λ, φ) -> (λ′, φ′)

`(λ, φ)` expressed in the rotated frame. The rotation's own pole maps to `φ′ = π/2`.
"""
function rotate(rot::PoleRotation{T}, λ::Real, φ::Real) where {T}
    sinλ, cosλ = sincos(convert(T, λ) - rot.λp)
    sinφ, cosφ = sincos(convert(T, φ))
    # about z by -λp, then about y by φp - π/2
    x = cosφ * cosλ
    y = cosφ * sinλ
    z = sinφ
    sinθ, cosθ = sincos(rot.φp - T(π) / 2)
    xr = cosθ * x + sinθ * z
    zr = -sinθ * x + cosθ * z
    return (mod(atan(y, xr), T(2π)), asin(clamp(zr, -one(T), one(T))))
end

"""
    unrotate(rot, λ′, φ′) -> (λ, φ)

Inverse of [`rotate`](@ref).
"""
function unrotate(rot::PoleRotation{T}, λ::Real, φ::Real) where {T}
    sinλ, cosλ = sincos(convert(T, λ))
    sinφ, cosφ = sincos(convert(T, φ))
    x = cosφ * cosλ
    y = cosφ * sinλ
    z = sinφ
    sinθ, cosθ = sincos(rot.φp - T(π) / 2)
    xr = cosθ * x - sinθ * z
    zr = sinθ * x + cosθ * z
    return (mod(atan(y, xr) + rot.λp, T(2π)), asin(clamp(zr, -one(T), one(T))))
end

"""
    rotate!(λ, φ, rot) -> (λ, φ)
    unrotate!(λ, φ, rot) -> (λ, φ)

Rotate a whole point set in place — the form a sampling's `spherical_points` output takes. `λ` and `φ`
are any arrays of matching shape, so this covers a scattered node set and a grid's 2-D coordinate
fields alike. Allocates nothing.
"""
function rotate! end
function unrotate! end

for (f!, f) in ((:rotate!, :rotate), (:unrotate!, :unrotate))
    @eval function $f!(λ::AbstractArray, φ::AbstractArray, rot::PoleRotation)
        axes(λ) == axes(φ) || throw(DimensionMismatch(
            "λ has axes $(axes(λ)) but φ has $(axes(φ))",
        ))
        @inbounds for i in eachindex(λ, φ)
            λ[i], φ[i] = $f(rot, λ[i], φ[i])
        end
        return (λ, φ)
    end

    @eval function $f(rot::PoleRotation{T}, λ::AbstractArray, φ::AbstractArray) where {T}
        axes(λ) == axes(φ) || throw(DimensionMismatch(
            "λ has axes $(axes(λ)) but φ has $(axes(φ))",
        ))
        out_λ = similar(λ, T)
        out_φ = similar(φ, T)
        @inbounds for i in eachindex(λ, φ)
            out_λ[i], out_φ[i] = $f(rot, λ[i], φ[i])
        end
        return (out_λ, out_φ)
    end
end

end # module
