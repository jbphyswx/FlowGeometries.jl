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
chord through Cartesian, with `r` the absolute radius from the origin.
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

@inline function _distance(
    geo::AbstractSphericalGeometry{T}, coords1::Tuple{Real,Real}, coords2::Tuple{Real,Real},
) where {T}
    λ1, φ1 = _at(T, coords1)
    λ2, φ2 = _at(T, coords2)
    dλ = λ2 - λ1
    dφ = φ2 - φ1
    a = sin(dφ / T(2))^2 + cos(φ1) * cos(φ2) * sin(dλ / T(2))^2
    c = T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a)))
    return geo.R * c
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

`(λ, φ)` on the reference sphere of radius `geo.R`, or `(λ, φ, r)` with `r` the absolute radius from
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
    return (; x = geo.R * cosφ * cosλ, y = geo.R * cosφ * sinλ, z = geo.R * sinφ)
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

end # module
