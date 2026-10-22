# ---------------------------------------------------------------------------
# Ellipsoidal geometry
# ---------------------------------------------------------------------------

"""
    AbstractEllipsoidalGeometry{T} <: AbstractGeometry{T}

Oblate-spheroid metrics. Default: [`SpheroidGeometry`](@ref). Coordinate names match the spherical
convention, `(λ, φ[, h])`, with `φ` the *geodetic* latitude.

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
are `(λ, φ, h)`, with `h` the height above the ellipsoid, where a sphere's third direction is an
absolute radius.

Rectilinear grids and the index-space connectivity built on them work as they do for a sphere; the
samplings are purely angular and so carry over unchanged. So does everything written against the two
primitives [`embed`](@ref) and [`local_tangent_basis`](@ref) — the local frame, the vector and tensor
rotations, the tangent-plane projection, and therefore the least-squares gradient and the scattered
interpolation. See [`AbstractLonLatGeometry`](@ref) for why the frame is shared.

The spherical *area* routines do not: `unstructured_grid`'s Voronoi areas and the cell areas behind
`spherical_grid` are built on spherical excess and `4πR²/n`, which are sphere identities. They stay
restricted to [`AbstractSphericalGeometry`](@ref), and raise on an ellipsoid.
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

"""
    meridian_distance(geo, φ) -> T

Distance along a meridian from the equator to geodetic latitude `φ`, negative south of it:

    ∫₀^φ M dφ = a·[E(φ | e²) − e²·sinφ·cosφ / √(1 − e²sin²φ)]

with `E(φ | m) = ∫₀^φ √(1 − m sin²t) dt` the incomplete elliptic integral of the second kind, evaluated
by Carlson's duplication to a relative error below `eps(T)` (Carlson 1995, *Numer. Algorithms* 10,
13–26).
"""
function meridian_distance(g::AbstractEllipsoidalGeometry{T}, φ::Real) where {T}
    φT = convert(T, φ)
    e² = eccentricity²(g)
    # `E(φ + nπ) = E(φ) + 2n·E(π/2)`, and the second term has period π.
    n = round(φT / T(π))
    r = φT - n * T(π)
    E = _elliptic_e(r, e²)
    iszero(n) || (E += 2n * _elliptic_e(T(π) / 2, e²))
    s, c = sincos(r)
    return semimajor_axis(g) * (E - e² * s * c / sqrt(one(T) - e² * s * s))
end

# `E(φ | m)` for `|φ| ≤ π/2`: `sinφ·R_F(cos²φ, q, 1) − (m/3)·sin³φ·R_D(cos²φ, q, 1)`, `q = 1 − m sin²φ`.
@inline function _elliptic_e(φ::T, m::T) where {T<:AbstractFloat}
    s, c = sincos(φ)
    q = one(T) - m * s * s
    return s * _carlson_rf(c * c, q, one(T)) - m * s * s * s * _carlson_rd(c * c, q, one(T)) / 3
end

# Carlson's `R_F(x, y, z)` (Carlson 1995): duplicate until `|X|, |Y|, |Z| ≤ (3 eps)^{1/6}`, then the
# truncated series.
function _carlson_rf(x::T, y::T, z::T) where {T<:AbstractFloat}
    A0 = (x + y + z) / 3
    Q = (3 * eps(T))^(-one(T) / 6) * max(abs(A0 - x), abs(A0 - y), abs(A0 - z))
    xm, ym, zm, Am, s = x, y, z, A0, one(T)
    while s * Q ≥ abs(Am)
        rx, ry, rz = sqrt(xm), sqrt(ym), sqrt(zm)
        λ = rx * ry + rx * rz + ry * rz
        xm, ym, zm, Am = (xm + λ) / 4, (ym + λ) / 4, (zm + λ) / 4, (Am + λ) / 4
        s /= 4
    end
    X = s * (A0 - x) / Am
    Y = s * (A0 - y) / Am
    Z = -(X + Y)
    E2 = X * Y - Z * Z
    E3 = X * Y * Z
    return (one(T) - E2 / 10 + E3 / 14 + E2 * E2 / 24 - 3 * E2 * E3 / 44) / sqrt(Am)
end

# Carlson's `R_D(x, y, z)`, likewise to `(eps/4)^{1/6}`.
function _carlson_rd(x::T, y::T, z::T) where {T<:AbstractFloat}
    A0 = (x + y + 3z) / 5
    Q = (eps(T) / 4)^(-one(T) / 6) * max(abs(A0 - x), abs(A0 - y), abs(A0 - z))
    xm, ym, zm, Am, s, acc = x, y, z, A0, one(T), zero(T)
    while s * Q ≥ abs(Am)
        rx, ry, rz = sqrt(xm), sqrt(ym), sqrt(zm)
        λ = rx * ry + rx * rz + ry * rz
        acc += s / (rz * (zm + λ))
        xm, ym, zm, Am = (xm + λ) / 4, (ym + λ) / 4, (zm + λ) / 4, (Am + λ) / 4
        s /= 4
    end
    X = s * (A0 - x) / Am
    Y = s * (A0 - y) / Am
    Z = -(X + Y) / 3
    XY = X * Y
    E2 = XY - 6 * Z * Z
    E3 = (3 * XY - 8 * Z * Z) * Z
    E4 = 3 * (XY - Z * Z) * Z * Z
    E5 = XY * Z * Z * Z
    series = one(T) - 3 * E2 / 14 + E3 / 6 + 9 * E2 * E2 / 88 - 3 * E4 / 22 - 9 * E2 * E3 / 52 + 3 * E5 / 26
    return s * series / (Am * sqrt(Am)) + 3 * acc
end

# A lone `(λ,)` is a point on the equator, a circle of radius `a`, as `distance` reads it.
@inline scale_factors(g::AbstractEllipsoidalGeometry, ::Tuple{Real}) = (semimajor_axis(g),)

@inline function scale_factors(g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real,Real}) where {T}
    φ = convert(T, p[2])
    return (prime_vertical_radius(g, φ) * cos(φ), meridional_radius(g, φ))
end

@inline function scale_factors(g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real,Real,Real}) where {T}
    φ = convert(T, p[2])
    h = convert(T, p[3])
    return ((prime_vertical_radius(g, φ) + h) * cos(φ), meridional_radius(g, φ) + h, one(T))
end

# `∂(N cos φ)/∂φ = −M sin φ`, so `γ[1][2] = −tan φ/(N + h)`; `h` enters `h_λ` through `cos φ` and `h_φ`
# with unit slope.
@inline log_scale_gradients(::AbstractEllipsoidalGeometry{T}, ::Tuple{Real}) where {T} = ((zero(T),),)

@inline function log_scale_gradients(g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real,Real}) where {T}
    φ = convert(T, p[2])
    z = zero(T)
    return ((z, -tan(φ) / prime_vertical_radius(g, φ)), (z, z))
end

@inline function log_scale_gradients(g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real,Real,Real}) where {T}
    φ = convert(T, p[2])
    h = convert(T, p[3])
    z = zero(T)
    iN = inv(prime_vertical_radius(g, φ) + h)
    return ((z, -tan(φ) * iN, iN), (z, z, inv(meridional_radius(g, φ) + h)), (z, z, z))
end

# `M` and `N` are functions of latitude, and the height enters additively: longitude enters neither.
metric_invariant_directions(::SpheroidGeometry{T}) where {T} = (1,)

# `h_λ = (N(φ) + h)·cosφ` has no λ and `h_h = 1`; `h_φ = M(φ) + h` varies along φ.
scale_invariant_directions(::SpheroidGeometry{T}) where {T} = (1, 3)

# Along φ, `∫(M + h) dφ`: the meridian distance plus the height times the latitude step.
function line_distance(g::SpheroidGeometry{T}, p::Tuple, d::Int, a::Real, b::Real) where {T}
    aT, bT = convert(T, a), convert(T, b)
    d == 2 || return scale_factors(g, p)[d] * (bT - aT)
    return meridian_distance(g, bT) - meridian_distance(g, aT) + _height(T, p) * (bT - aT)
end

@inline _height(::Type{T}, p::Tuple{Any,Any,Any}) where {T} = convert(T, p[3])
@inline _height(::Type{T}, ::Tuple) where {T} = zero(T)

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
reaching the cap the result is the great-circle distance on a sphere of the mean radius.

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
    λ, φ = _lonlat(p)
    sinλ, cosλ = sincos(T(λ))
    sinφ, cosφ = sincos(T(φ))
    return (cosφ * cosλ, cosφ * sinλ, sinφ)
end

"""
    spherical_excess(a, b, c) -> T

Spherical excess of the triangle spanned by three unit vectors, via Van Oosterom & Strackee (1983):

    tan(E/2) = |a · (b × c)| / (1 + a·b + b·c + c·a)

One `atan` and no other transcendental. Taking directions lets a mesh convert each vertex once — see
[`unit_vector`](@ref) — and share it across every triangle. Multiply by `R²` for an area, as
[`triangle_area`](@ref) does.

The numerator is evaluated as `a · (u × v)` for `u = b - a` and `v = c - a`, an identity because
`a · (a × x)` vanishes for every `x`. Given unit vectors, `u` and `v` are differences of nearly equal
ones, so the excess carries a relative error of about `eps(T)·R/side`. [`triangle_area`](@ref), given
coordinates, forms `u` and `v` from them directly and keeps `eps(T)`.
"""
@inline spherical_excess(a::NTuple{3,T}, b::NTuple{3,T}, c::NTuple{3,T}) where {T} =
    _excess(a, b, c, (b[1] - a[1], b[2] - a[2], b[3] - a[3]), (c[1] - a[1], c[2] - a[2], c[3] - a[3]))

# The excess with the two edges from `a` given: `u = b − a`, `v = c − a`.
@inline function _excess(
    a::NTuple{3,T}, b::NTuple{3,T}, c::NTuple{3,T}, u::NTuple{3,T}, v::NTuple{3,T},
) where {T}
    num = a[1] * (u[2] * v[3] - u[3] * v[2]) +
          a[2] * (u[3] * v[1] - u[1] * v[3]) +
          a[3] * (u[1] * v[2] - u[2] * v[1])
    ab = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    bc = b[1] * c[1] + b[2] * c[2] + b[3] * c[3]
    ca = c[1] * a[1] + c[2] * a[2] + c[3] * a[3]
    return T(2) * abs(atan(num, one(T) + ab + bc + ca))
end

# `x(q) − x(p)` for the unit vectors at two `(λ, φ)` points, from half-angle sums and differences:
# `cos λq − cos λp = −2 sin λm sin δλ`, `sin λq − sin λp = 2 cos λm sin δλ`, and likewise in `φ`, with
# `λm, δλ` the half sum and half difference. A short chord keeps the relative precision of its
# endpoints' coordinates.
@inline function _lonlat_chord(p::NTuple{2,T}, q::NTuple{2,T}) where {T<:AbstractFloat}
    δλ = _lon_diff(p[1], q[1]) / 2
    δφ = (q[2] - p[2]) / 2
    sλm, cλm = sincos(p[1] + δλ)
    sφm, cφm = sincos(p[2] + δφ)
    sδλ, sδφ = sin(δλ), sin(δφ)
    sλp, cλp = sincos(p[1])
    cφq = cos(q[2])
    dcosφ = -2 * sφm * sδφ
    return (cφq * (-2 * sλm * sδλ) + cλp * dcosφ, cφq * (2 * cλm * sδλ) + sλp * dcosφ, 2 * cφm * sδφ)
end

@inline _dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
@inline _cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
@inline _axpy3(s, x, y) = (s * x[1] + y[1], s * x[2] + y[2], s * x[3] + y[3])

# `M − V` for `M` the midpoint direction of the arc from unit `V` to `V + e`: `|2V + e|² = 4 − |e|²`.
@inline function _midpoint_offset(V::NTuple{3,T}, e::NTuple{3,T}) where {T}
    e² = _dot3(e, e)
    s = sqrt(4 - e²)
    return _axpy3(e² / ((2 + s) * s), V, map(c -> c / s, e))
end

# `O − V` for `O` the circumcentre direction of the spherical triangle `(V, V + e₁, V + e₂)`: the
# planar circumcentre `V + x`, `x = ((|e₁|²e₂ − |e₂|²e₁) × n) / (2|n|²)` with `n = e₁ × e₂`, is the
# centre of the circle through the three points, so `O = (V + x)/|V + x|`.
@inline function _circumcenter_offset(V::NTuple{3,T}, e1::NTuple{3,T}, e2::NTuple{3,T}) where {T}
    n = _cross3(e1, e2)
    w = _axpy3(-_dot3(e2, e2), e1, map(c -> _dot3(e1, e1) * c, e2))
    x = map(c -> c / (2 * _dot3(n, n)), _cross3(w, n))
    q = 2 * _dot3(V, x) + _dot3(x, x)       # |V + x|² − 1
    t = sqrt(1 + q)
    return _axpy3(-q / (t * (1 + t)), V, map(c -> c / t, x))
end

# Vertex `V`'s Voronoi share of the spherical triangle `(V, V + e₁, V + e₂)`: the quadrilateral from `V`
# through the two edge midpoints and the circumcentre, as two spherical excesses. Those arcs are the
# perpendicular bisectors, so the three vertices' shares tile the triangle.
@inline function _dual_share(V::NTuple{3,T}, e1::NTuple{3,T}, e2::NTuple{3,T}) where {T}
    m1, m2 = _midpoint_offset(V, e1), _midpoint_offset(V, e2)
    o = _circumcenter_offset(V, e1, e2)
    M1, M2, O = _axpy3(one(T), m1, V), _axpy3(one(T), m2, V), _axpy3(one(T), o, V)
    return _excess(V, M1, O, m1, o) + _excess(V, O, M2, o, m2)
end

"""
    triangle_area(geo, p1, p2, p3) -> T

Exact area of the spherical triangle through three `(λ, φ)` points, each in any accepted representation.
`R²` times the spherical excess, its edges formed from the coordinates themselves, so the area is
accurate to `eps(T)` relative whatever the triangle's size.

This converts all three points on every call. A mesh that already holds vertex directions should call
[`spherical_excess`](@ref) on those and scale once.
"""
@inline function triangle_area(
    geo::AbstractSphericalGeometry{T}, p1, p2, p3,
) where {T<:AbstractFloat}
    q1, q2, q3 = map(p -> map(T, _lonlat(p)), (p1, p2, p3))
    a, b, c = unit_vector(T, q1), unit_vector(T, q2), unit_vector(T, q3)
    return radius(geo)^2 * _excess(a, b, c, _lonlat_chord(q1, q2), _lonlat_chord(q1, q3))
end

"""
    triangle_area_from_unit_vectors(geo, u1, u2, u3) -> T
    triangle_area_from_unit_vectors(R², u1, u2, u3) -> T

[`triangle_area`](@ref) for vertices already held as unit vectors — `R²` times
[`spherical_excess`](@ref), with no coordinate conversion.

It carries its own name because dispatch cannot tell the two apart: a unit vector and a 3-D spherical
point `(λ, φ, r)` are both 3-tuples.

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

# A lone `(λ,)` is a point on the equator, the circle of radius `a` — the convention [`distance`](@ref)
# uses for one coordinate.
@inline function _geodetic_to_cartesian(
    g::AbstractEllipsoidalGeometry{T}, p::Tuple{Real},
) where {T}
    sinλ, cosλ = sincos(convert(T, p[1]))
    a = semimajor_axis(g)
    return (; x = a * cosλ, y = a * sinλ, z = zero(T))
end

@inline _embed(g::AbstractEllipsoidalGeometry, p::Tuple) = Tuple(_geodetic_to_cartesian(g, p))

"""
    cartesian_to_geodetic(geo, xyz) -> NamedTuple{(:λ,:φ,:h)}
    cartesian_to_geodetic(S, geo, xyz) -> S

Inverse of [`geodetic_to_cartesian`](@ref): Earth-centred Cartesian to geodetic `(λ, φ, h)` with
`λ ∈ [0, 2π)` (see [`wrap_longitude`](@ref)), `φ` the geodetic latitude in `[-π/2, π/2]`, and `h` the
height above the ellipsoid.

Unlike the spherical inverse this is not closed-form — geodetic latitude satisfies a quartic — so it is
solved by Newton's method on

    tan φ = z(N+h) / (p(N(1-e²)+h)),   p = √(x²+y²)

with `h = p·cosφ + z·sinφ − a√(1-e²sin²φ)` substituted each step, which is exact for any `φ` and stays
finite at the poles where `p/cosφ − N` does not. Convergence is quadratic and reaches `eps(T)` in three
or four steps at terrestrial flattening. On the polar axis `λ = 0` and the latitude is `±π/2` exactly.

The foot of the normal is unique for any point outside the ellipsoid's evolute — every position on or
above the surface, and far below it. Deep inside, near the centre, a point has several feet and no
preferred one; the centre itself is reported as `(0, 0, -a)`, the same convention
[`cartesian_to_spherical`](@ref) uses at the origin.
"""
@inline cartesian_to_geodetic(geo::AbstractEllipsoidalGeometry, xyz) =
    _cartesian_to_geodetic(geo, _xyz(xyz))

@inline cartesian_to_geodetic(::Type{S}, geo::AbstractEllipsoidalGeometry, xyz) where {S} =
    build_point(S, (:λ, :φ, :h), Tuple(cartesian_to_geodetic(geo, xyz)))

function _cartesian_to_geodetic(
    g::AbstractEllipsoidalGeometry{T}, xyz::Tuple{Real,Real,Real},
    tol = max(T(1e-14), eps(T))
) where {T}
    x, y, z = _at(T, xyz)
    a = semimajor_axis(g)
    e² = eccentricity²(g)
    p = sqrt(x * x + y * y)
    λ = wrap_longitude(atan(y, x))
    # The centre has no normal to stand on. It also has to be taken out before the loop: there
    # `N(1-e²)+h` is negative, so `p·(N(1-e²)+h)` is a negative zero, and `atan(+0, -0)` is `π`.
    (iszero(p) && iszero(z)) && return (; λ = zero(T), φ = zero(T), h = -a)
    # `atan(z, (1-e²)p)` is the latitude of the point's reduced (parametric) position, which is within
    # `f` of the geodetic one — one Newton step of the loop below then already carries most of the way.
    φ = atan(z, (one(T) - e²) * p)
    h = zero(T)
   
    converged = false
    for _ in 1:20
        sinφ, cosφ = sincos(φ)
        W = sqrt(one(T) - e² * sinφ * sinφ)
        N = a / W
        h = p * cosφ + z * sinφ - a * W
        φnext = atan(z * (N + h), p * (N * (one(T) - e²) + h))
        if abs(φnext - φ) < tol
            φ = φnext
            converged = true
            break
        end
        φ = φnext
    end
    if !converged
        throw(ErrorException(
            "the geodetic latitude of ($x, $y, $z) did not converge on a spheroid of a = $a, " *
            "f = $(flattening(g))",
        ))
    end
    # One last evaluation, at the `φ` being returned, so `h` matches it.
    sinφ, cosφ = sincos(φ)
    h = p * cosφ + z * sinφ - a * sqrt(one(T) - e² * sinφ * sinφ)
    return (; λ = λ, φ = φ, h = h)
end

"""
    geocentric_latitude(geo, φ) -> T

The latitude of the direction from the centre to the surface point at latitude `φ`. On a sphere that is
`φ`. On a spheroid `φ` is geodetic, the angle of the surface normal, and the surface point
`(N cos φ, N(1 − e²) sin φ)` gives `atan((1 − e²) sin φ, cos φ)`, `N` cancelling.
"""
@inline geocentric_latitude(::AbstractSphericalGeometry{T}, φ::Real) where {T} = convert(T, φ)

@inline function geocentric_latitude(geo::AbstractEllipsoidalGeometry{T}, φ::Real) where {T}
    s, c = sincos(convert(T, φ))
    return atan((one(T) - eccentricity²(geo)) * s, c)
end

# One coordinate is a point on the equator, a circle of radius `a`: the distance is the shorter arc —
# the same convention the 1-D grid measure uses.
@inline function _spheroid_distance(
    g::AbstractEllipsoidalGeometry{T}, p1::Tuple{Real}, p2::Tuple{Real},
) where {T}
    return semimajor_axis(g) * abs(_lon_diff(convert(T, p1[1]), convert(T, p2[1])))
end

function _spheroid_distance(
    g::AbstractEllipsoidalGeometry{T}, p1::Tuple{Real,Real,Real}, p2::Tuple{Real,Real,Real},
) where {T}
    c1 = _geodetic_to_cartesian(g, p1)
    c2 = _geodetic_to_cartesian(g, p2)
    return sqrt((c1.x - c2.x)^2 + (c1.y - c2.y)^2 + (c1.z - c2.z)^2)
end

@inline _spheroid_distance(
    g::AbstractEllipsoidalGeometry, p1::Tuple{Real,Real}, p2::Tuple{Real,Real},
) = first(_vincenty_inverse(g, p1, p2))

# `(s, sin α₁, cos α₁)`: the geodesic length from `p1` to `p2` by Vincenty's inverse method, and its
# azimuth at `p1`, east of north.
#
# The reduced latitudes enter as `(sin U, cos U) = ((1 − f) sin φ, cos φ)/Q`, `Q = √(cos²φ + (1 − f)²sin²φ)`,
# and their difference as `(sin ΔU, cos ΔU) = ((1 − f) sin Δφ, cos φ₁cos φ₂ + (1 − f)²sin φ₁sin φ₂)/(Q₁Q₂)`.
# With `h = sin²(λ/2)`, `cos U₁ sin U₂ − sin U₁ cos U₂ cos λ = sin ΔU + 2 sin U₁ cos U₂ h` and
# `cos σ = cos ΔU − 2 cos U₁ cos U₂ h`, so no term is a difference of nearly equal ones.
function _vincenty_inverse(
    g::AbstractEllipsoidalGeometry{T}, p1::Tuple{Real,Real}, p2::Tuple{Real,Real},
) where {T}
    λ1, φ1 = _at(T, p1)
    λ2, φ2 = _at(T, p2)
    a = semimajor_axis(g)
    b = semiminor_axis(g)
    f = flattening(g)
    L = _lon_diff(λ1, λ2)
    m = one(T) - f
    sφ1, cφ1 = sincos(φ1)
    sφ2, cφ2 = sincos(φ2)
    Q1 = hypot(cφ1, m * sφ1)
    Q2 = hypot(cφ2, m * sφ2)
    sinU1, cosU1 = m * sφ1 / Q1, cφ1 / Q1
    sinU2, cosU2 = m * sφ2 / Q2, cφ2 / Q2
    sinΔU = m * sin(φ2 - φ1) / (Q1 * Q2)
    cosΔU = (cφ1 * cφ2 + m * m * sφ1 * sφ2) / (Q1 * Q2)
    λ = L
    # Converged once the auxiliary longitude moves less than 1e-12 rad, Vincenty's own criterion, or
    # less than its own resolution in `T` where that is coarser.
    tol = T(1e-12)
    sinσ = cosσ = σ = sinα² = cos2σm = zero(T)
    converged = false
    for _ in 1:200
        sinλ = sin(λ)
        h = sin(λ / T(2))^2
        sinσ = hypot(cosU2 * sinλ, sinΔU + T(2) * sinU1 * cosU2 * h)
        iszero(sinσ) && return (zero(T), zero(T), one(T))   # coincident points
        cosσ = cosΔU - T(2) * cosU1 * cosU2 * h
        σ = atan(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        sinα² = sinα * sinα
        cos²α = one(T) - sinα²
        cos2σm = iszero(cos²α) ? zero(T) : cosσ - 2 * sinU1 * sinU2 / cos²α
        C = f / 16 * cos²α * (4 + f * (4 - 3 * cos²α))
        λprev = λ
        λ = L + (one(T) - C) * f * sinα *
            (σ + C * sinσ * (cos2σm + C * cosσ * (-one(T) + 2 * cos2σm^2)))
        if abs(λ - λprev) <= max(tol, eps(T) * abs(λ))
            converged = true
            break
        end
    end
    if !converged
        # Near-antipodal: the great circle on a sphere of the mean radius.
        R = (2 * a + b) / 3
        pe, pn, hav = _unit_on_frame(T, p1, p2)
        s = R * 2 * atan(sqrt(hav), sqrt(max(zero(T), one(T) - hav)))
        n = hypot(pe, pn)
        return iszero(n) ? (s, zero(T), one(T)) : (s, pe / n, pn / n)
    end
    u² = sinα² == one(T) ? zero(T) : (one(T) - sinα²) * (a * a - b * b) / (b * b)
    A = one(T) + u² / 16384 * (4096 + u² * (-768 + u² * (320 - 175 * u²)))
    B = u² / 1024 * (256 + u² * (-128 + u² * (74 - 47 * u²)))
    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-one(T) + 2 * cos2σm^2) -
         B / 6 * cos2σm * (-3 + 4 * sinσ^2) * (-3 + 4 * cos2σm^2)))
    Y = cosU2 * sin(λ)
    X = sinΔU + T(2) * sinU1 * cosU2 * sin(λ / T(2))^2
    n = hypot(Y, X)
    return (b * A * (σ - Δσ), Y / n, X / n)
end
