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
Subtypes should provide `dx`, `dy`, `dz` (or specialize the methods).
"""
abstract type AbstractCartesianGeometry{T<:AbstractFloat} <: AbstractGeometry{T} end

"""
    AbstractSphericalGeometry{T} <: AbstractGeometry{T}

Spherical metrics.  Default: [`SphericalGeometry`](@ref).
Subtypes should provide `R` (or specialize the methods).
"""
abstract type AbstractSphericalGeometry{T<:AbstractFloat} <: AbstractGeometry{T} end

"""
    CartesianGeometry{T} <: AbstractCartesianGeometry{T}

Default Cartesian geometry with spacings `dx`, `dy`, and optional `dz` (zero for 2D).
"""
struct CartesianGeometry{T<:AbstractFloat} <: AbstractCartesianGeometry{T}
    dx::T
    dy::T
    dz::T
end

CartesianGeometry(dx::T, dy::T) where {T<:AbstractFloat} = CartesianGeometry{T}(dx, dy, zero(T))
CartesianGeometry{T}(dx, dy) where {T<:AbstractFloat} = CartesianGeometry{T}(convert(T, dx), convert(T, dy), zero(T))

"""
    SphericalGeometry{T} <: AbstractSphericalGeometry{T}

Default spherical geometry with planet radius `R` (meters; default Earth 6.371e6).
"""
struct SphericalGeometry{T<:AbstractFloat} <: AbstractSphericalGeometry{T}
    R::T
end

SphericalGeometry() = SphericalGeometry(6.371e6)

# ---------------------------------------------------------------------------
# Point naming (small NamedTuples — not data-dimension packing)
# ---------------------------------------------------------------------------

"""
    point_names(geo, Val(N)) -> NTuple{N,Symbol}

Field names for an `N`-coordinate point in this geometry.
Cartesian: `(:x,)`, `(:x,:y)`, `(:x,:y,:z)`.
Spherical: `(:λ,)`, `(:λ,:φ)`, `(:λ,:φ,:r)`.
"""
point_names(::AbstractCartesianGeometry, ::Val{1}) = (:x,)
point_names(::AbstractCartesianGeometry, ::Val{2}) = (:x, :y)
point_names(::AbstractCartesianGeometry, ::Val{3}) = (:x, :y, :z)
point_names(::AbstractSphericalGeometry, ::Val{1}) = (:λ,)
point_names(::AbstractSphericalGeometry, ::Val{2}) = (:λ, :φ)
point_names(::AbstractSphericalGeometry, ::Val{3}) = (:λ, :φ, :r)

"""
    named_point(geo, vals::NTuple{N,T}) -> NamedTuple

Build the geometry-appropriate named point from positional values (axis order).
"""
@inline function named_point(geo::AbstractGeometry, vals::NTuple{N,T}) where {N,T}
    return NamedTuple{point_names(geo, Val(N)), NTuple{N,T}}(vals)
end

"""Normalize a point to `NTuple` (NamedTuple / Tuple / AbstractVector)."""
@inline as_ntuple(p::NamedTuple) = Tuple(p)
@inline as_ntuple(p::Tuple) = p
@inline as_ntuple(p::AbstractVector) = ntuple(i -> @inbounds(p[i]), length(p))

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

"""
    distance(geo::AbstractGeometry, pt1, pt2)

Distance between two points. Accepts `NamedTuple`, `Tuple`, or `AbstractVector`
(length 2/3). Spherical 2D uses great-circle (Haversine); spherical 3D uses the
chord through planetary Cartesian.
"""
@inline function distance(::AbstractCartesianGeometry{T}, pt1::NTuple{N,T}, pt2::NTuple{N,T}) where {N,T}
    s = zero(T)
    @inbounds for i in 1:N
        d = pt1[i] - pt2[i]
        s += d * d
    end
    return sqrt(s)
end

@inline function distance(geo::AbstractSphericalGeometry{T}, coords1::NTuple{2,T}, coords2::NTuple{2,T}) where {T}
    λ1, φ1 = coords1[1], coords1[2]
    λ2, φ2 = coords2[1], coords2[2]
    dλ = λ2 - λ1
    dφ = φ2 - φ1
    a = sin(dφ / T(2))^2 + cos(φ1) * cos(φ2) * sin(dλ / T(2))^2
    c = T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a)))
    return geo.R * c
end

@inline function distance(geo::AbstractSphericalGeometry{T}, coords1::NTuple{3,T}, coords2::NTuple{3,T}) where {T}
    p1 = spherical_to_planetary_position(geo, coords1)
    p2 = spherical_to_planetary_position(geo, coords2)
    dx = p1.x - p2.x; dy = p1.y - p2.y; dz = p1.z - p2.z
    return sqrt(dx * dx + dy * dy + dz * dz)
end

@inline distance(geo::AbstractGeometry, p1, p2) = distance(geo, as_ntuple(p1), as_ntuple(p2))

# Helper: (λ, φ[, r]) → planetary Cartesian. `r` is absolute radius from planet center.
@inline function spherical_to_planetary_position(geo::AbstractSphericalGeometry{T}, coords::NTuple{3,T}) where {T}
    λ, φ, r = coords[1], coords[2], coords[3]
    return (; x = r * cos(φ) * cos(λ), y = r * cos(φ) * sin(λ), z = r * sin(φ))
end

@inline function spherical_to_planetary_position(geo::AbstractSphericalGeometry{T}, coords::NTuple{2,T}) where {T}
    λ, φ = coords[1], coords[2]
    return (; x = geo.R * cos(φ) * cos(λ), y = geo.R * cos(φ) * sin(λ), z = geo.R * sin(φ))
end

@inline spherical_to_planetary_position(geo::AbstractSphericalGeometry, coords) =
    spherical_to_planetary_position(geo, as_ntuple(coords))

# ---------------------------------------------------------------------------
# Area Elements
# ---------------------------------------------------------------------------

"""
    area_element(geo::AbstractCartesianGeometry{T})
    area_element(geo::AbstractSphericalGeometry{T}, φ::T, dλ::T, dφ::T)

Compute local grid cell area.
"""
@inline area_element(geo::AbstractCartesianGeometry{T}) where {T} = geo.dx * geo.dy

@inline function area_element(geo::AbstractSphericalGeometry{T}, φ::T, dλ::T, dφ::T) where {T}
    return geo.R^2 * cos(φ) * dλ * dφ
end

"""
    volume_element(geo::AbstractCartesianGeometry{T})
    volume_element(geo::AbstractSphericalGeometry{T}, r::T, φ::T, dλ::T, dφ::T, dr::T)

Compute local grid cell volume. The spherical form generalizes [`area_element`](@ref) with the
LOCAL radius `r` at this level (not the fixed reference `geo.R`) — a genuine spherical-shell volume
element `r²·cosφ·dλ·dφ·dr`, needed once a grid has real multi-level radial structure instead of a
single reference sphere.
"""
@inline volume_element(geo::AbstractCartesianGeometry{T}) where {T} = geo.dx * geo.dy * geo.dz

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
# Spherical vector fields ↔ planetary Cartesian
# ---------------------------------------------------------------------------
# Orthonormal frame at (λ, φ): ê_λ (∂/∂λ), ê_φ (∂/∂φ), ê_r (radial). Same directions
# as geographic east / north / up — named for the polar coordinates used everywhere else.

"""
    to_planetary_cartesian(geo, u_λ, u_φ, u_r, λ, φ) -> NamedTuple{(:x,:y,:z)}
    to_planetary_cartesian(geo, u_λ, u_φ, λ, φ) -> NamedTuple{(:x,:y,:z)}

Map spherical velocity components `(u_λ, u_φ[, u_r])` at `(λ, φ)` into planetary Cartesian.
The 2-component form assumes `u_r = 0`.
"""
@inline function to_planetary_cartesian(
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

@inline function to_planetary_cartesian(
    geo::AbstractSphericalGeometry{T}, u_λ::Real, u_φ::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    return to_planetary_cartesian(geo, u_λ, u_φ, zero(T), λ, φ)
end

"""
    from_planetary_cartesian(geo, ux, uy, uz, λ, φ) -> NamedTuple{(:λ,:φ,:r)}

Map planetary Cartesian velocity `(ux, uy, uz)` at `(λ, φ)` into spherical components
`(λ = u_λ, φ = u_φ, r = u_r)`.
"""
@inline function from_planetary_cartesian(
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

@inline function from_planetary_cartesian(
    geo::AbstractSphericalGeometry, v::NamedTuple, λ::Real, φ::Real,
)
    return from_planetary_cartesian(geo, v.x, v.y, v.z, λ, φ)
end

# ---------------------------------------------------------------------------
# Local tangent-plane geometry (curvilinear / unstructured gradient reconstruction)
# ---------------------------------------------------------------------------

"""
    local_tangent_basis(geo, coords) -> NamedTuple

Coordinate-aligned unit vectors at `coords`, in the ambient Cartesian frame:
- Cartesian: `(; x = ê_x, y = ê_y)` as 2-tuples
- Spherical: `(; λ = ê_λ, φ = ê_φ)` as 3-tuples

Accepts NamedTuple / Tuple / AbstractVector.
"""
@inline function local_tangent_basis(::AbstractCartesianGeometry{T}, ::NTuple{2,T}) where {T}
    return (; x = (one(T), zero(T)), y = (zero(T), one(T)))
end

@inline function local_tangent_basis(::AbstractSphericalGeometry{T}, coords::NTuple{2,T}) where {T}
    λ, φ = coords[1], coords[2]
    sinλ, cosλ = sin(λ), cos(λ)
    sinφ, cosφ = sin(φ), cos(φ)
    return (;
        λ = (-sinλ, cosλ, zero(T)),
        φ = (-sinφ * cosλ, -sinφ * sinλ, cosφ),
    )
end

@inline local_tangent_basis(geo::AbstractGeometry, coords) =
    local_tangent_basis(geo, as_ntuple(coords))

"""
    project_to_tangent_plane(geo, center, neighbor) -> NamedTuple{(:x,:y)} or NamedTuple{(:λ,:φ)}

Tangent-plane displacement of `neighbor` relative to `center` along the local
coordinate basis from [`local_tangent_basis`](@ref).
"""
@inline function project_to_tangent_plane(
    ::AbstractCartesianGeometry{T}, center::NTuple{2,T}, neighbor::NTuple{2,T},
) where {T}
    return (; x = neighbor[1] - center[1], y = neighbor[2] - center[2])
end

@inline function project_to_tangent_plane(
    geo::AbstractSphericalGeometry{T}, center::NTuple{2,T}, neighbor::NTuple{2,T},
) where {T}
    Pc = spherical_to_planetary_position(geo, center)
    Pn = spherical_to_planetary_position(geo, neighbor)
    chord = (Pn.x - Pc.x, Pn.y - Pc.y, Pn.z - Pc.z)
    ê = local_tangent_basis(geo, center)
    return (;
        λ = sum(chord[i] * ê.λ[i] for i in 1:3),
        φ = sum(chord[i] * ê.φ[i] for i in 1:3),
    )
end

@inline project_to_tangent_plane(geo::AbstractGeometry, center, neighbor) =
    project_to_tangent_plane(geo, as_ntuple(center), as_ntuple(neighbor))

end # module
