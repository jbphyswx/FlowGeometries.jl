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

# `M` and `N` are functions of latitude, and the height enters additively: longitude enters neither.
metric_invariant_directions(::SpheroidGeometry{T}) where {T} = (1,)

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
