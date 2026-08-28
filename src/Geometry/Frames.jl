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
