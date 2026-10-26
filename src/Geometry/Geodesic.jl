# ---------------------------------------------------------------------------
# The inverse geodesic problem on an oblate ellipsoid
# ---------------------------------------------------------------------------
#
# Karney, C. F. F., Algorithms for geodesics, J. Geodesy 87, 43–55 (2013),
# https://doi.org/10.1007/s00190-012-0578-z. The series run to sixth order in the third flattening `n`
# The solution is accurate to round-off at every separation, nearly antipodal points included.
#
# Every routine is scalar arithmetic on tuples, so the solution runs inside a device kernel.

# `Σ_{l=1}^{K-1} c[l+1] sin(2lx)` by Clenshaw summation; `c[1]` is unused.
@inline function _sin_series(sinx::T, cosx::T, c::NTuple{K,T}) where {K,T}
    n = K - 1
    ar = 2 * (cosx - sinx) * (cosx + sinx)
    y0 = zero(T)
    y1 = zero(T)
    j = K
    if isodd(n)
        y0 = c[j]
        j -= 1
    end
    for _ in 1:(n ÷ 2)
        y1 = ar * y0 - y1 + c[j]
        j -= 1
        y0 = ar * y1 - y0 + c[j]
        j -= 1
    end
    return 2 * sinx * cosx * y0
end

# `Σ_{l=1}^{K-1} c[l+1] (sin 2lσ₂ − sin 2lσ₁)` for `σ₂ = σ₁ + σ₁₂`, summed as
# `Σ 2c[l+1] cos(l(σ₁ + σ₂)) sin(lσ₁₂)`, so its relative error is a few `eps` however short `σ₁₂` is. Both
# angles arrive as unit (sine, cosine) pairs.
@inline function _sin_series_diff(ssig1::T, csig1::T, ssig12::T, csig12::T, c::NTuple{K,T}) where {K,T}
    s2 = 2 * ssig1 * csig1
    c2 = (csig1 - ssig1) * (csig1 + ssig1)
    sθ = s2 * csig12 + c2 * ssig12
    cθ = c2 * csig12 - s2 * ssig12
    sa, ca = sθ, cθ
    sb, cb = ssig12, csig12
    acc = zero(T)
    for l in 1:(K - 1)
        acc += c[l + 1] * ca * sb
        sa, ca = sa * cθ + ca * sθ, ca * cθ - sa * sθ
        sb, cb = sb * csig12 + cb * ssig12, cb * csig12 - sb * ssig12
    end
    return 2 * acc
end

# The degree-`N` polynomial in `x` whose coefficients, highest first, start at 0-based offset `s` of `p`.
@inline function _polyval(N::Int, p::Tuple, s::Int, x::T) where {T}
    y = N < 0 ? zero(T) : T(p[s + 1])
    for i in 1:N
        y = y * x + T(p[s + 1 + i])
    end
    return y
end

@inline function _norm2(x::T, y::T) where {T}
    r = hypot(x, y)
    return x / r, y / r
end

const _GEOD_ORDER = 6

const _A1_COEFF = (1, 4, 64, 0, 256)
const _C1_COEFF = (-1, 6, -16, 32, -9, 64, -128, 2048, 9, -16, 768, 3, -5, 512, -7, 1280, -7, 2048)
const _A2_COEFF = (-11, -28, -192, 0, 256)
const _C2_COEFF = (1, 2, 16, 32, 35, 64, 384, 2048, 15, 80, 768, 7, 35, 512, 63, 1280, 77, 2048)
const _A3_COEFF = (-3, 128, -2, -3, 64, -1, -3, -1, 16, 3, -1, -2, 8, 1, -1, 2, 1, 1)
const _C3_COEFF = (3, 128, 2, 5, 128, -1, 3, 3, 64, -1, 0, 1, 8, -1, 1, 4, 5, 256, 1, 3, 128,
                   -3, -2, 3, 64, 1, -3, 2, 32, 7, 512, -10, 9, 384, 5, -9, 5, 192, 7, 512,
                   -14, 7, 512, 21, 2560)

# `A₁ − 1` and `A₂ − 1`, Karney eqs. (17) and (42).
@inline function _a1m1(ε::T) where {T}
    t = _polyval(3, _A1_COEFF, 0, ε * ε) / T(_A1_COEFF[5])
    return (t + ε) / (1 - ε)
end

@inline function _a2m1(ε::T) where {T}
    t = _polyval(3, _A2_COEFF, 0, ε * ε) / T(_A2_COEFF[5])
    return (t - ε) / (1 + ε)
end

# `C₁ₗ` or `C₂ₗ`, `l = 1:6`, at `c[l+1]` of a 7-tuple.
@inline function _c12(coeff::Tuple, ε::T) where {T}
    ε2 = ε * ε
    c = ntuple(_ -> zero(T), Val(_GEOD_ORDER + 1))
    d = ε
    o = 0
    for l in 1:_GEOD_ORDER
        m = (_GEOD_ORDER - l) ÷ 2
        c = Base.setindex(c, d * _polyval(m, coeff, o, ε2) / T(coeff[o + m + 2]), l + 1)
        o += m + 2
        d *= ε
    end
    return c
end

# The coefficients of `A₃` and `C₃` as polynomials in `ε`, which depend on `n` alone.
@inline function _a3x(n::T) where {T}
    a = ntuple(_ -> zero(T), Val(_GEOD_ORDER))
    o = 0
    k = 1
    for j in (_GEOD_ORDER - 1):-1:0
        m = min(_GEOD_ORDER - j - 1, j)
        a = Base.setindex(a, _polyval(m, _A3_COEFF, o, n) / T(_A3_COEFF[o + m + 2]), k)
        k += 1
        o += m + 2
    end
    return a
end

@inline function _c3x(n::T) where {T}
    c = ntuple(_ -> zero(T), Val((_GEOD_ORDER * (_GEOD_ORDER - 1)) ÷ 2))
    o = 0
    k = 1
    for l in 1:(_GEOD_ORDER - 1), j in (_GEOD_ORDER - 1):-1:l
        m = min(_GEOD_ORDER - j - 1, j)
        c = Base.setindex(c, _polyval(m, _C3_COEFF, o, n) / T(_C3_COEFF[o + m + 2]), k)
        k += 1
        o += m + 2
    end
    return c
end

@inline _a3(a3x::NTuple{6,T}, ε::T) where {T} = _polyval(_GEOD_ORDER - 1, a3x, 0, ε)

# `C₃ₗ`, `l = 1:5`, at `c[l+1]` of a 6-tuple.
@inline function _c3(c3x::NTuple{15,T}, ε::T) where {T}
    c = ntuple(_ -> zero(T), Val(_GEOD_ORDER))
    mult = one(T)
    o = 0
    for l in 1:(_GEOD_ORDER - 1)
        m = _GEOD_ORDER - l - 1
        mult *= ε
        c = Base.setindex(c, mult * _polyval(m, c3x, o, ε), l + 1)
        o += m + 1
    end
    return c
end

"""
    _GeodesicConstants(a, f)

What the inverse solution reads of an ellipsoid of equatorial radius `a` and flattening `0 ≤ f < 1`: the
derived axes and eccentricity, the `A₃` and `C₃` coefficient polynomials in `ε`, and the short-line
threshold.
"""
struct _GeodesicConstants{T<:AbstractFloat}
    a::T
    f::T
    f1::T
    ep2::T
    n::T
    b::T
    etol2::T
    a3x::NTuple{6,T}
    c3x::NTuple{15,T}
end

@inline function _GeodesicConstants(a::T, f::T) where {T<:AbstractFloat}
    f1 = 1 - f
    ep2 = f * (2 - f) / (f1 * f1)
    n = f / (2 - f)
    etol2 = T(0.1) * sqrt(eps(T)) / sqrt(max(T(0.001), abs(f)) * min(one(T), 1 - f / 2) / 2)
    return _GeodesicConstants{T}(a, f, f1, ep2, n, a * f1, etol2, _a3x(n), _c3x(n))
end

@inline _tiny(::Type{T}) where {T} = sqrt(floatmin(T))

# A latitude within `eps(T)²` of the equator is on it. The move is below the rounding of any separation
# of more than `eps(T)·a`, and no square of a latitude underflows.
@inline _lat_flush(x::T) where {T} = abs(x) < eps(T)^2 ? zero(T) : x

# `sin(x + y)` for latitudes `x` and `y`: the sine of the sum where `|x + y| ≤ π/2`, and the expansion
# otherwise, where its two terms share a sign. Either keeps the relative precision of the inputs.
@inline _sin_sum(x::T, sx::T, cx::T, y::T, sy::T, cy::T) where {T} =
    abs(x + y) ≤ T(π) / 2 ? sin(x + y) : sx * cy + cx * sy

# The arc of the auxiliary sphere from `(β₁, 0)` to `(β₂, ω₁₂)`, as `sin σ₁₂·(sin α₁, cos α₁)`,
# `sin σ₁₂·(sin α₂, cos α₂)` and `cos σ₁₂`. Each is formed from `sin(β₂ − β₁)` and `1 − cos ω₁₂`, or from
# `sin(β₁ + β₂)` and `1 + cos ω₁₂` where `cos ω₁₂ < 0`, and no term is a difference of nearly equal ones.
@inline function _aux_arc(
    sbet1::T, cbet1::T, sbet2::T, cbet2::T, sbet12::T, cbet12::T, sbet12a::T, cbet12a::T, somg12::T, comg12::T,
) where {T}
    if comg12 ≥ 0
        h = somg12^2 / (1 + comg12)
        x1 = sbet12 + sbet1 * cbet2 * h
        x2 = sbet12 - cbet1 * sbet2 * h
        csig12 = cbet12 - cbet1 * cbet2 * h
    else
        k = somg12^2 / (1 - comg12)
        x1 = sbet12a - sbet1 * cbet2 * k
        x2 = cbet1 * sbet2 * k - sbet12a
        csig12 = cbet1 * cbet2 * k - cbet12a
    end
    return cbet2 * somg12, x1, cbet1 * somg12, x2, csig12
end

# `s/b` along the arc from `σ₁` to `σ₁ + σ₁₂` of a geodesic of parameter `ε`, Karney eq. (7), from unit
# (sine, cosine) pairs.
@inline function _geod_arc_length(ε::T, sig12::T, ssig1::T, csig1::T, ssig12::T, csig12::T) where {T}
    return (1 + _a1m1(ε)) * (sig12 + _sin_series_diff(ssig1, csig1, ssig12, csig12, _c12(_C1_COEFF, ε)))
end

# The ε of the geodesic whose equatorial azimuth has cosine `calp0`, Karney eq. (16).
@inline function _geod_eps(calp0::T, ep2::T) where {T}
    k2 = calp0 * calp0 * ep2
    return k2 / (2 * (1 + sqrt(1 + k2)) + k2)
end

# The distance `s/b` from `σ₁` to `σ₂` along a geodesic of parameter `ε`, Karney eq. (7).
@inline function _geod_distance(ε::T, sig12::T, ssig1::T, csig1::T, ssig2::T, csig2::T) where {T}
    c1 = _c12(_C1_COEFF, ε)
    b1 = _sin_series(ssig2, csig2, c1) - _sin_series(ssig1, csig1, c1)
    return (1 + _a1m1(ε)) * (sig12 + b1)
end

# The reduced length `m/b`, Karney eqs. (38)–(41).
@inline function _geod_reduced(
    ε::T, sig12::T, ssig1::T, csig1::T, dn1::T, ssig2::T, csig2::T, dn2::T,
) where {T}
    a1m1 = _a1m1(ε)
    a2m1 = _a2m1(ε)
    a1 = 1 + a1m1
    a2 = 1 + a2m1
    c1 = _c12(_C1_COEFF, ε)
    c2 = _c12(_C2_COEFF, ε)
    c = ntuple(l -> l == 1 ? zero(T) : a1 * c1[l] - a2 * c2[l], Val(_GEOD_ORDER + 1))
    j12 = (a1m1 - a2m1) * sig12 + (_sin_series(ssig2, csig2, c) - _sin_series(ssig1, csig1, c))
    return dn2 * (csig1 * ssig2) - dn1 * (ssig1 * csig2) - csig1 * csig2 * j12
end

# The positive root `k` of `k⁴ + 2k³ − (x² + y² − 1)k² − 2y²k − y² = 0`, Karney eq. (55).
@inline function _astroid(x::T, y::T) where {T}
    p = x * x
    q = y * y
    r = (p + q - 1) / 6
    (q == 0 && r ≤ 0) && return zero(T)
    S = p * q / 4
    r2 = r * r
    r3 = r * r2
    disc = S * (S + 2 * r3)
    u = r
    if disc ≥ 0
        T3 = S + r3
        T3 += T3 < 0 ? -sqrt(disc) : sqrt(disc)
        Tc = cbrt(T3)
        u += Tc + (Tc != 0 ? r2 / Tc : zero(T))
    else
        ang = atan(sqrt(-disc), -(S + r3))
        u += 2 * r * cos(ang / 3)
    end
    v = sqrt(u * u + q)
    uv = u < 0 ? q / (v - u) : u + v
    w = (uv - q) / (2 * v)
    return uv / (sqrt(uv + w * w) + w)
end

# The starting azimuth for Newton's method, and for a short line the whole answer:
# `(σ₁₂, sin α₁, cos α₁, sin α₂, cos α₂, dnm)`, `σ₁₂ ≥ 0` where no iteration is needed.
@inline function _inverse_start(
    g::_GeodesicConstants{T}, sbet1::T, cbet1::T, sbet2::T, cbet2::T, sbet12::T, cbet12::T, sbet12a::T,
    lam12::T, slam12::T, clam12::T,
) where {T}
    sig12 = -one(T)
    salp2 = calp2 = dnm = T(NaN)
    shortline = cbet12 ≥ 0 && sbet12 < T(0.5) && cbet2 * lam12 < T(0.5)
    if shortline
        sbetm2 = (sbet1 + sbet2)^2
        sbetm2 /= sbetm2 + (cbet1 + cbet2)^2
        dnm = sqrt(1 + g.ep2 * sbetm2)
        somg12, comg12 = sincos(lam12 / (g.f1 * dnm))
    else
        somg12, comg12 = slam12, clam12
    end
    salp1 = cbet2 * somg12
    calp1 = comg12 ≥ 0 ? sbet12 + cbet2 * sbet1 * somg12^2 / (1 + comg12) :
                         sbet12a - cbet2 * sbet1 * somg12^2 / (1 - comg12)
    ssig12 = hypot(salp1, calp1)
    csig12 = sbet1 * sbet2 + cbet1 * cbet2 * comg12
    if shortline && ssig12 < g.etol2
        salp2, calp2 = _norm2(cbet1 * somg12,
                              sbet12 - cbet1 * sbet2 * (comg12 ≥ 0 ? somg12^2 / (1 + comg12) : 1 - comg12))
        sig12 = atan(ssig12, csig12)
    elseif !(abs(g.n) ≥ T(0.1) || csig12 ≥ 0 || ssig12 ≥ 6 * abs(g.n) * T(π) * cbet1^2)
        # Nearly antipodal: coordinates with the antipode at the origin and the singular point at
        # `x = −1, y = 0`.
        lam12x = atan(-slam12, -clam12)
        lamscale = g.f * cbet1 * _a3(g.a3x, _geod_eps(sbet1, g.ep2)) * T(π)
        x = lam12x / lamscale
        y = sbet12a / (lamscale * cbet1)
        if y > -200 * eps(T) && x > -1 - 1000 * sqrt(eps(T))
            salp1 = min(one(T), -x)
            calp1 = -sqrt(1 - salp1^2)
        else
            k = _astroid(x, y)
            somg12, comg12 = sincos(lamscale * (-x * k / (1 + k)))
            comg12 = -comg12
            salp1 = cbet2 * somg12
            calp1 = sbet12a - cbet2 * sbet1 * somg12^2 / (1 - comg12)
        end
    end
    if salp1 > 0 || isnan(salp1)
        salp1, calp1 = _norm2(salp1, calp1)
    else
        salp1, calp1 = one(T), zero(T)
    end
    return sig12, salp1, calp1, salp2, calp2, dnm
end

# The longitude difference `λ₁₂` the geodesic leaving `β₁` at azimuth `α₁` reaches at `β₂`, less the
# target, and its derivative in `α₁` where `diffp`, with what the distance then reads and the longitude
# `ω₁₂` on the auxiliary sphere: Karney eqs. (5)–(11) and (54).
@inline function _lambda12(
    g::_GeodesicConstants{T}, sbet1::T, cbet1::T, dn1::T, sbet2::T, cbet2::T, dn2::T,
    salp1::T, calp1::T, slam120::T, clam120::T, diffp::Bool,
) where {T}
    (sbet1 == 0 && calp1 == 0) && (calp1 = -_tiny(T))
    salp0 = salp1 * cbet1
    calp0 = hypot(calp1, salp1 * sbet1)
    somg1 = salp0 * sbet1
    comg1 = calp1 * cbet1
    ssig1, csig1 = _norm2(sbet1, comg1)
    salp2 = cbet2 != cbet1 ? salp0 / cbet2 : salp1
    calp2 = (cbet2 != cbet1 || abs(sbet2) != -sbet1) ?
        sqrt((calp1 * cbet1)^2 + (cbet1 < -sbet1 ? (cbet2 - cbet1) * (cbet1 + cbet2) :
                                                   (sbet1 - sbet2) * (sbet1 + sbet2))) / cbet2 :
        abs(calp1)
    somg2 = salp0 * sbet2
    comg2 = calp2 * cbet2
    ssig2, csig2 = _norm2(sbet2, comg2)
    sig12 = atan(max(zero(T), csig1 * ssig2 - ssig1 * csig2) + zero(T), csig1 * csig2 + ssig1 * ssig2)
    somg12 = max(zero(T), comg1 * somg2 - somg1 * comg2) + zero(T)
    comg12 = comg1 * comg2 + somg1 * somg2
    eta = atan(somg12 * clam120 - comg12 * slam120, comg12 * clam120 + somg12 * slam120)
    ε = _geod_eps(calp0, g.ep2)
    c3 = _c3(g.c3x, ε)
    b312 = _sin_series(ssig2, csig2, c3) - _sin_series(ssig1, csig1, c3)
    lam12 = eta - g.f * _a3(g.a3x, ε) * salp0 * (sig12 + b312)
    dlam12 = if !diffp
        T(NaN)
    elseif calp2 == 0
        -2 * g.f1 * dn1 / sbet1
    else
        _geod_reduced(ε, sig12, ssig1, csig1, dn1, ssig2, csig2, dn2) * g.f1 / (calp2 * cbet2)
    end
    return lam12, salp2, calp2, sig12, ssig1, csig1, ssig2, csig2, ε, dlam12, atan(somg12, comg12)
end

# Karney's relation `ω₁₂ = λ₁₂ + f·A₃·sin α₀·(σ₁₂ + B₃(σ₂) − B₃(σ₁))`, eq. (8), iterated from `ω` with each
# arc from `_aux_arc`. Where `σ₁₂ ≤ π/2` the map's slope is at most `f·A₃·(1 + π)`, so a few steps from a
# root found to absolute precision reach it to relative precision. Returns `(s/b, sin α₁, cos α₁, sin α₂,
# cos α₂)` at the root.
@inline function _polish(
    g::_GeodesicConstants{T}, lam12::T, sbet1::T, cbet1::T, sbet2::T, cbet2::T,
    sbet12::T, cbet12::T, sbet12a::T, cbet12a::T, ω::T,
) where {T}
    sig12 = ssig1 = ssig12 = ε = zero(T)
    csig1 = csig12 = one(T)
    salp1 = calp1 = salp2 = calp2 = zero(T)
    for _ in 1:10
        somg12, comg12 = sincos(ω)
        a1, x1, a2, x2, c = _aux_arc(sbet1, cbet1, sbet2, cbet2, sbet12, cbet12, sbet12a, cbet12a, somg12, comg12)
        s = hypot(a1, x1)
        salp1, calp1 = a1 / s, x1 / s
        salp2, calp2 = _norm2(a2, x2)
        sig12 = atan(s, c)
        ssig12, csig12 = _norm2(s, c)
        ssig1, csig1 = _norm2(sbet1, calp1 * cbet1)
        salp0 = salp1 * cbet1
        ε = _geod_eps(hypot(calp1, salp1 * sbet1), g.ep2)
        b312 = _sin_series_diff(ssig1, csig1, ssig12, csig12, _c3(g.c3x, ε))
        ωn = lam12 + g.f * _a3(g.a3x, ε) * salp0 * (sig12 + b312)
        done = abs(ωn - ω) ≤ 2 * eps(T) * ωn
        ω = ωn
        done && break
    end
    return _geod_arc_length(ε, sig12, ssig1, csig1, ssig12, csig12), salp1, calp1, salp2, calp2
end

"""
    _geodesic_inverse(g, (λ₁, φ₁), (λ₂, φ₂)) -> (s, sin α₁, cos α₁)

The geodesic distance between two geodetic points on the ellipsoid `g`, and its azimuth at the first,
east of north, by Karney's (2013) equations. The arc is formed from latitude and longitude differences
with no cancelling terms, so the relative error is a few `eps(T)` at every separation.
"""
function _geodesic_inverse(g::_GeodesicConstants{T}, p1::NTuple{2,T}, p2::NTuple{2,T}) where {T}
    tiny = _tiny(T)
    lon12 = _lon_diff(p1[1], p2[1])
    lonsign = copysign(one(T), lon12)
    lam12 = lonsign * lon12
    slam12, clam12 = sincos(lam12)
    lon12s = T(π) - lam12
    lat1 = _lat_flush(p1[2])
    lat2 = _lat_flush(p2[2])
    # The canonical form `−π/2 ≤ φ₁ ≤ 0`, `φ₁ ≤ φ₂ ≤ −φ₁`, `0 ≤ λ₁₂ ≤ π`; the three signs undo it.
    swapp = (abs(lat1) < abs(lat2) || isnan(lat2)) ? -one(T) : one(T)
    if swapp < 0
        lonsign = -lonsign
        lat1, lat2 = lat2, lat1
    end
    latsign = copysign(one(T), -lat1)
    lat1 *= latsign
    lat2 *= latsign
    sphi1, cphi1 = sin(lat1), _cos_latitude(lat1)
    sphi2, cphi2 = sin(lat2), _cos_latitude(lat2)
    q1 = hypot(cphi1, g.f1 * sphi1)
    q2 = hypot(cphi2, g.f1 * sphi2)
    sbet1, cbet1 = g.f1 * sphi1 / q1, max(tiny, cphi1 / q1)
    sbet2, cbet2 = g.f1 * sphi2 / q2, max(tiny, cphi2 / q2)
    if cbet1 < -sbet1
        cbet2 == cbet1 && (sbet2 = copysign(sbet1, sbet2))
    else
        abs(sbet2) == -sbet1 && (cbet2 = cbet1)
    end
    # `sin(β₂ ∓ β₁) = (1 − f)·sin(φ₂ ∓ φ₁)/(q₁q₂)`.
    sbet12 = g.f1 * _sin_sum(lat2, sphi2, cphi2, -lat1, -sphi1, cphi1) / (q1 * q2)
    sbet12a = g.f1 * _sin_sum(lat2, sphi2, cphi2, lat1, sphi1, cphi1) / (q1 * q2)
    cbet12 = cbet2 * cbet1 + sbet2 * sbet1
    cbet12a = cbet2 * cbet1 - sbet2 * sbet1
    dn1 = sqrt(1 + g.ep2 * sbet1^2)
    dn2 = sqrt(1 + g.ep2 * sbet2^2)

    s12 = zero(T)
    salp1, calp1 = zero(T), one(T)
    salp2, calp2 = zero(T), one(T)
    meridian = lat1 == -T(π) / 2 || slam12 == 0
    if meridian
        # Both points on one meridian: head for the target longitude, arriving northward.
        salp1, calp1 = slam12, clam12
        salp2, calp2 = zero(T), one(T)
        a1, x1, _, _, c = _aux_arc(sbet1, cbet1, sbet2, cbet2, sbet12, cbet12, sbet12a, cbet12a, slam12, clam12)
        s = hypot(a1, x1)
        sig12 = atan(s, c)
        ssig12, csig12 = _norm2(s, c)
        ssig1, csig1 = _norm2(sbet1, calp1 * cbet1)
        m12x = _geod_reduced(g.n, sig12, ssig1, csig1, dn1, sbet2, cbet2, dn2)
        if sig12 < sqrt(eps(T)) || m12x ≥ 0
            s12 = sig12 < 3 * tiny ? zero(T) : _geod_arc_length(g.n, sig12, ssig1, csig1, ssig12, csig12) * g.b
        else
            meridian = false
        end
    end

    if !meridian && sbet1 == 0 && (g.f ≤ 0 || lon12s ≥ g.f * T(π))
        # Along the equator.
        salp1, calp1 = one(T), zero(T)
        salp2, calp2 = one(T), zero(T)
        s12 = g.a * lam12
    elseif !meridian
        sig12, salp1, calp1, salp2, calp2, dnm =
            _inverse_start(g, sbet1, cbet1, sbet2, cbet2, sbet12, cbet12, sbet12a, lam12, slam12, clam12)
        if sig12 ≥ 0
            s12 = sig12 * g.b * dnm
        else
            # Newton's method on `λ₁₂(α₁) − λ₁₂ = 0`, whose one root in `(0, π)` it keeps bracketed,
            # taking the bracket's midpoint where a step leaves it or the slope is not positive.
            maxit1 = 20
            maxit2 = maxit1 + precision(T) + 10
            numit = 0
            tripn = false
            tripb = false
            salp1a, calp1a = tiny, one(T)
            salp1b, calp1b = tiny, -one(T)
            ssig1 = csig1 = ssig2 = csig2 = ε = omg12 = zero(T)
            while true
                v, salp2, calp2, sig12, ssig1, csig1, ssig2, csig2, ε, dv, omg12 = _lambda12(
                    g, sbet1, cbet1, dn1, sbet2, cbet2, dn2, salp1, calp1, slam12, clam12, numit < maxit1,
                )
                (tripb || !(abs(v) ≥ (tripn ? 8 : 1) * eps(T)) || numit == maxit2) && break
                if v > 0 && (numit > maxit1 || calp1 / salp1 > calp1b / salp1b)
                    salp1b, calp1b = salp1, calp1
                elseif v < 0 && (numit > maxit1 || calp1 / salp1 < calp1a / salp1a)
                    salp1a, calp1a = salp1, calp1
                end
                numit += 1
                if numit < maxit1 && dv > 0
                    dalp1 = -v / dv
                    if abs(dalp1) < T(π)
                        sdalp1, cdalp1 = sincos(dalp1)
                        nsalp1 = salp1 * cdalp1 + calp1 * sdalp1
                        if nsalp1 > 0
                            salp1, calp1 = _norm2(nsalp1, calp1 * cdalp1 - salp1 * sdalp1)
                            tripn = abs(v) ≤ 16 * eps(T)
                            continue
                        end
                    end
                end
                salp1, calp1 = _norm2((salp1a + salp1b) / 2, (calp1a + calp1b) / 2)
                tripn = false
                tripb = abs(salp1a - salp1) + (calp1a - calp1) < eps(T) ||
                        abs(salp1 - salp1b) + (calp1 - calp1b) < eps(T)
            end
            if sig12 ≤ T(π) / 2
                sb, salp1, calp1, salp2, calp2 =
                    _polish(g, lam12, sbet1, cbet1, sbet2, cbet2, sbet12, cbet12, sbet12a, cbet12a, omg12)
                s12 = sb * g.b
            else
                s12 = _geod_distance(ε, sig12, ssig1, csig1, ssig2, csig2) * g.b
            end
        end
    end
    # Back from the canonical form: a swap makes the second point's azimuth the first's.
    if swapp < 0
        salp1, salp2 = salp2, salp1
        calp1, calp2 = calp2, calp1
    end
    return zero(T) + s12, salp1 * swapp * lonsign, calp1 * swapp * latsign
end

@inline _geodesic_inverse(g::AbstractEllipsoidalGeometry{T}, p1::Tuple{Real,Real}, p2::Tuple{Real,Real}) where {T} =
    _geodesic_inverse(_GeodesicConstants(T(semimajor_axis(g)), T(flattening(g))), _at(T, p1), _at(T, p2))
