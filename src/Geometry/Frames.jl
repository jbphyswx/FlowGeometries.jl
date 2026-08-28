# ---------------------------------------------------------------------------
# The local frame at (λ, φ), and the rotations written against it
# ---------------------------------------------------------------------------
# Orthonormal frame at (λ, φ): ê_λ (∂/∂λ), ê_φ (∂/∂φ), ê_r (outward normal). Same directions
# as geographic east / north / up — named for the polar coordinates used everywhere else.

"""
    AbstractLonLatGeometry{T}

The geometries whose coordinates are `(λ, φ, …)` about a polar axis:
[`AbstractSphericalGeometry`](@ref) and [`AbstractEllipsoidalGeometry`](@ref).

They share their local frame. On an oblate spheroid the GEODETIC latitude is defined by the surface
normal, `n̂ = (cosφ·cosλ, cosφ·sinλ, sinφ)`, and the parallel through a point is a circle in a plane of
constant `z`, so `ê_λ = (-sinλ, cosλ, 0)` and `ê_φ = n̂ × ê_λ` — the sphere's three vectors at the same
`(λ, φ)`. Every frame rotation here is therefore one implementation for both hierarchies. What does
differ is the POSITION a `(λ, φ)` sits at, which is [`embed`](@ref)'s business, and the physical length
of a unit coordinate step, which is [`scale_factors`](@ref)'.

A `Union` rather than a common supertype: an ellipsoid is not a sphere, and must not inherit the area
identities — `4πR²/n`, spherical excess — that hold only on one. That is why the two hierarchies are
siblings, and this names exactly the part they do share.
"""
const AbstractLonLatGeometry{T} =
    Union{AbstractSphericalGeometry{T},AbstractEllipsoidalGeometry{T}}

"""
    _enu_frame(geo, λ, φ) -> (ê_λ, ê_φ, ê_r)

The local east/north/up triad at `(λ, φ)` in ambient Cartesian components.

The one definition of the frame, so a vector rotation, a tensor rotation and a tangent-plane projection
cannot drift apart. [`local_tangent_basis`](@ref) is its first two vectors under their coordinate names.
"""
@inline function _enu_frame(::AbstractLonLatGeometry{T}, λ::Real, φ::Real) where {T}
    sinλ, cosλ = sincos(convert(T, λ))
    sinφ, cosφ = sincos(convert(T, φ))
    return ((-sinλ, cosλ, zero(T)),
            (-sinφ * cosλ, -sinφ * sinλ, cosφ),
            (cosφ * cosλ, cosφ * sinλ, sinφ))
end

"""
    vector_to_cartesian(geo, u_λ, u_φ, u_r, λ, φ) -> NamedTuple{(:x,:y,:z)}
    vector_to_cartesian(geo, u_λ, u_φ, λ, φ) -> NamedTuple{(:x,:y,:z)}
    vector_to_cartesian(S, geo, args...) -> S

Map the components of a vector given in the local `(ê_λ, ê_φ, ê_r)` basis at `(λ, φ)` into the ambient
Cartesian basis. This transforms VECTOR COMPONENTS at a point, not a position — see [`embed`](@ref) for
positions. The 2-component form assumes `u_r = 0`.

A rotation of the frame and nothing else, so it is one and the same on a sphere and on a spheroid — see
[`AbstractLonLatGeometry`](@ref).
"""
@inline function vector_to_cartesian(
    geo::AbstractLonLatGeometry{T},
    u_λ::Real, u_φ::Real, u_r::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    uλ = convert(T, u_λ)
    uφ = convert(T, u_φ)
    ur = convert(T, u_r)
    êλ, êφ, êr = _enu_frame(geo, λ, φ)
    # `ê_λ` has no `z` component, the parallel lying in a plane of constant `z`, so that term is
    # omitted rather than multiplied by a zero.
    return (;
        x = uλ * êλ[1] + uφ * êφ[1] + ur * êr[1],
        y = uλ * êλ[2] + uφ * êφ[2] + ur * êr[2],
        z =              uφ * êφ[3] + ur * êr[3],
    )
end

@inline function vector_to_cartesian(
    geo::AbstractLonLatGeometry{T}, u_λ::Real, u_φ::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    return vector_to_cartesian(geo, u_λ, u_φ, zero(T), λ, φ)
end

@inline vector_to_cartesian(::Type{S}, geo::AbstractLonLatGeometry, args::Vararg{Real}) where {S} =
    build_point(S, (:x, :y, :z), Tuple(vector_to_cartesian(geo, args...)))

"""
    vector_from_cartesian(geo, ux, uy, uz, λ, φ) -> NamedTuple{(:λ,:φ,:r)}
    vector_from_cartesian(geo, v, λ, φ) -> NamedTuple{(:λ,:φ,:r)}
    vector_from_cartesian(S, geo, args...) -> S

Map an ambient Cartesian vector `(ux, uy, uz)` at `(λ, φ)` into local components
`(λ = u_λ, φ = u_φ, r = u_r)` — eastward, northward, and along the outward normal. `v` may be any
3-component representation. The inverse of [`vector_to_cartesian`](@ref), and like it the same on a
sphere and on a spheroid.
"""
@inline function vector_from_cartesian(
    geo::AbstractLonLatGeometry{T},
    ux::Real, uy::Real, uz::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    uxT = convert(T, ux)
    uyT = convert(T, uy)
    uzT = convert(T, uz)
    êλ, êφ, êr = _enu_frame(geo, λ, φ)
    # `ê_λ` has no `z` component, so `uz` does not enter the eastward projection.
    return (;
        λ = uxT * êλ[1] + uyT * êλ[2],
        φ = uxT * êφ[1] + uyT * êφ[2] + uzT * êφ[3],
        r = uxT * êr[1] + uyT * êr[2] + uzT * êr[3],
    )
end

@inline function vector_from_cartesian(geo::AbstractLonLatGeometry, v, λ::Real, φ::Real)
    ux, uy, uz = _xyz(v)
    return vector_from_cartesian(geo, ux, uy, uz, λ, φ)
end

@inline vector_from_cartesian(::Type{S}, geo::AbstractLonLatGeometry, args...) where {S} =
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

`R` is the same basis [`local_tangent_basis`](@ref) defines, so the two cannot drift apart — and being
a rotation of the frame alone, it holds on a spheroid as it does on a sphere.
"""
@inline function tensor_to_local(
    geo::AbstractLonLatGeometry{T},
    τxx::Real, τyy::Real, τzz::Real, τxy::Real, τxz::Real, τyz::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    xx, yy, zz = convert(T, τxx), convert(T, τyy), convert(T, τzz)
    xy, xz, yz = convert(T, τxy), convert(T, τxz), convert(T, τyz)
    ê = _enu_frame(geo, λ, φ)
    a1, a2, a3 = ê[1]                                       # ê_λ
    b1, b2, b3 = ê[2]                                       # ê_φ
    c1, c2, c3 = ê[3]                                       # ê_r
    return (;
        λλ = _quad(xx, yy, zz, xy, xz, yz, a1, a2, a3, a1, a2, a3),
        φφ = _quad(xx, yy, zz, xy, xz, yz, b1, b2, b3, b1, b2, b3),
        rr = _quad(xx, yy, zz, xy, xz, yz, c1, c2, c3, c1, c2, c3),
        λφ = _quad(xx, yy, zz, xy, xz, yz, a1, a2, a3, b1, b2, b3),
        λr = _quad(xx, yy, zz, xy, xz, yz, a1, a2, a3, c1, c2, c3),
        φr = _quad(xx, yy, zz, xy, xz, yz, b1, b2, b3, c1, c2, c3),
    )
end

@inline function tensor_to_local(geo::AbstractLonLatGeometry, τ, λ::Real, φ::Real)
    xx, yy, zz, xy, xz, yz = as_tensor6(τ)
    return tensor_to_local(geo, xx, yy, zz, xy, xz, yz, λ, φ)
end

@inline tensor_to_local(::Type{S}, geo::AbstractLonLatGeometry, args...) where {S} =
    build_point(S, (:λλ, :φφ, :rr, :λφ, :λr, :φr), Tuple(tensor_to_local(geo, args...)))

"""
    tensor_from_local(geo, τλλ, τφφ, τrr, τλφ, τλr, τφr, λ, φ) -> NamedTuple
    tensor_from_local(geo, τ, λ, φ) -> NamedTuple
    tensor_from_local(S, geo, args...) -> S

The inverse of [`tensor_to_local`](@ref): `τ = Rᵀ τ' R`, returning the ambient Cartesian components
`(; xx, yy, zz, xy, xz, yz)`. Composing the two in either order is the identity.
"""
@inline function tensor_from_local(
    geo::AbstractLonLatGeometry{T},
    τλλ::Real, τφφ::Real, τrr::Real, τλφ::Real, τλr::Real, τφr::Real, λ::Real, φ::Real,
) where {T<:AbstractFloat}
    ll, ff, rr = convert(T, τλλ), convert(T, τφφ), convert(T, τrr)
    lf, lr, fr = convert(T, τλφ), convert(T, τλr), convert(T, τφr)
    êλ, êφ, êr = _enu_frame(geo, λ, φ)
    # The COLUMNS of `R`, which are the rows of `Rᵀ`, so the same contraction serves both directions.
    u1, u2, u3 = êλ[1], êφ[1], êr[1]
    v1, v2, v3 = êλ[2], êφ[2], êr[2]
    w1, w2, w3 = êλ[3], êφ[3], êr[3]
    return (;
        xx = _quad(ll, ff, rr, lf, lr, fr, u1, u2, u3, u1, u2, u3),
        yy = _quad(ll, ff, rr, lf, lr, fr, v1, v2, v3, v1, v2, v3),
        zz = _quad(ll, ff, rr, lf, lr, fr, w1, w2, w3, w1, w2, w3),
        xy = _quad(ll, ff, rr, lf, lr, fr, u1, u2, u3, v1, v2, v3),
        xz = _quad(ll, ff, rr, lf, lr, fr, u1, u2, u3, w1, w2, w3),
        yz = _quad(ll, ff, rr, lf, lr, fr, v1, v2, v3, w1, w2, w3),
    )
end

@inline function tensor_from_local(geo::AbstractLonLatGeometry, τ, λ::Real, φ::Real)
    ll, ff, rr, lf, lr, fr = as_tensor6(τ)
    return tensor_from_local(geo, ll, ff, rr, lf, lr, fr, λ, φ)
end

@inline tensor_from_local(::Type{S}, geo::AbstractLonLatGeometry, args...) where {S} =
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
- Spherical and spheroidal: `(; λ = ê_λ, φ = ê_φ)`, 3-component — the eastward and northward tangents,
  which are the same triad on either (see [`AbstractLonLatGeometry`](@ref))

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

@inline function _local_tangent_basis(geo::AbstractLonLatGeometry{T}, coords::Tuple{Real,Real}) where {T}
    êλ, êφ, _ = _enu_frame(geo, coords[1], coords[2])
    return (; λ = êλ, φ = êφ)
end

"""
    project_to_tangent_plane(geo, center, neighbor) -> NamedTuple
    project_to_tangent_plane(S, geo, center, neighbor) -> S

Tangent-plane displacement of `neighbor` relative to `center` along the local coordinate basis from
[`local_tangent_basis`](@ref): the ambient chord between the two [`embed`](@ref)ded positions, resolved
on that basis.

This is what a least-squares gradient and a scattered interpolation are built on, so both work on any
geometry supplying those two primitives — a spheroid included, where the chord runs between geodetic
positions and the basis is the same triad.
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

# Written against [`embed`](@ref) and the frame, so it holds on a sphere and on a spheroid alike: the
# chord between the two ambient positions, resolved on the tangent basis at the centre. Only where each
# `(λ, φ)` SITS differs between the two.
@inline function _project_to_tangent_plane(
    geo::AbstractLonLatGeometry{T}, center::Tuple{Real,Real}, neighbor::Tuple{Real,Real},
) where {T}
    c = _at(T, center)
    Pc = _embed(geo, c)
    Pn = _embed(geo, _at(T, neighbor))
    dx = Pn[1] - Pc[1]; dy = Pn[2] - Pc[2]; dz = Pn[3] - Pc[3]
    êλ, êφ, _ = _enu_frame(geo, c[1], c[2])
    # Written out rather than `sum(dr[i] * ê[i] for i in 1:3)`: the generator indexes the tuple with
    # a loop variable, which does not unroll, and measured ~5× slower than the unrolled contraction.
    return (;
        λ = dx * êλ[1] + dy * êλ[2] + dz * êλ[3],
        φ = dx * êφ[1] + dy * êφ[2] + dz * êφ[3],
    )
end

"""
    local_displacement(geo, center, neighbor) -> NamedTuple

`neighbor`'s displacement from `center`, resolved on the local coordinate frame at `center`, one
component per coordinate — the quantity a least-squares fit differences against.

Two coordinates is [`project_to_tangent_plane`](@ref), which this calls: a surface has a tangent plane
and the displacement lies in it. Three is the same ambient chord on the full local frame — `(x, y, z)`
on the plane, and eastward/northward/outward on a sphere or spheroid, the third direction being the
normal [`unit_vector`](@ref) gives.
"""
@inline local_displacement(geo::AbstractGeometry, center, neighbor) =
    _local_displacement(geo, as_ntuple(center), as_ntuple(neighbor))

@inline _local_displacement(geo::AbstractGeometry, c::NTuple{2,Real}, nb::NTuple{2,Real}) =
    _project_to_tangent_plane(geo, c, nb)

@inline function _local_displacement(
    ::AbstractCartesianGeometry{T}, c::NTuple{3,Real}, nb::NTuple{3,Real},
) where {T}
    cx, cy, cz = _at(T, c)
    nx, ny, nz = _at(T, nb)
    return (; x = nx - cx, y = ny - cy, z = nz - cz)
end

@inline function _local_displacement(
    geo::AbstractLonLatGeometry{T}, c::NTuple{3,Real}, nb::NTuple{3,Real},
) where {T}
    cc = _at(T, c)
    Pc = _embed(geo, cc)
    Pn = _embed(geo, _at(T, nb))
    dx = Pn[1] - Pc[1]; dy = Pn[2] - Pc[2]; dz = Pn[3] - Pc[3]
    êλ, êφ, êr = _enu_frame(geo, cc[1], cc[2])
    return build_point(NamedTuple, point_names(geo, Val(3)),
                       (dx * êλ[1] + dy * êλ[2] + dz * êλ[3],
                        dx * êφ[1] + dy * êφ[2] + dz * êφ[3],
                        dx * êr[1] + dy * êr[2] + dz * êr[3]))
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
    metric_invariant_directions(geo) -> NTuple{K,Int}

The coordinate directions along which [`scale_factors`](@ref) does not vary.

A bulk operation that divides by a scale factor can then solve it once per line rather than per cell:
on a sphere no factor depends on longitude, so a whole row shares one value.

The default is `()` — no direction is assumed invariant — and it is declared per CONCRETE geometry
rather than per hierarchy. A subtype of [`AbstractSphericalGeometry`](@ref) may write its own
`scale_factors`, and inheriting a claim about them would hoist a value that in fact varies, giving a
wrong derivative rather than a slow one. Declare the directions for your own geometry to opt in.
"""
function metric_invariant_directions end

metric_invariant_directions(::AbstractGeometry) = ()

# `(R·cosφ, R)` and `(r·cosφ, r, 1)`: longitude enters neither.
metric_invariant_directions(::SphericalGeometry{T}) where {T} = (1,)

"""
    jacobian(geo, point) -> T

`∏` of [`scale_factors`](@ref): the volume element per unit coordinate volume at `point`.
"""
@inline jacobian(geo::AbstractGeometry, p) = prod(scale_factors(geo, p))
