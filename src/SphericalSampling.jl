module SphericalSampling

using ..Execution: Execution

# Public API via `FlowGeometries.SphericalSampling.*` or parent rebinds. No exports.

"""
    AbstractSphericalSampling

How points are placed on the sphere. Orthogonal to
[`FlowGeometries.Geometry.AbstractSphericalGeometry`](@ref)
(the metric / radius) and to grid architecture (`StructuredGrid` vs unstructured).
"""
abstract type AbstractSphericalSampling end

"""
    AbstractTensorProductSphericalSampling <: AbstractSphericalSampling

Iso-latitude × equispaced-longitude (or arbitrary lon/lat) layouts that fit a 2D
`StructuredGrid` with axes `(λ, φ)`.
"""
abstract type AbstractTensorProductSphericalSampling <: AbstractSphericalSampling end

"""
    AbstractSpectralQuadratureSampling <: AbstractTensorProductSphericalSampling

Samplings with a known exact quadrature for band-limited spherical harmonics (up to a
documented `lmax` ↔ grid-size relation).
"""
abstract type AbstractSpectralQuadratureSampling <: AbstractTensorProductSphericalSampling end

abstract type AbstractGaussLegendreSampling <: AbstractSpectralQuadratureSampling end
abstract type AbstractDriscollHealySampling <: AbstractSpectralQuadratureSampling end
abstract type AbstractClenshawCurtisSampling <: AbstractSpectralQuadratureSampling end
abstract type AbstractMcEwenWiauxSampling <: AbstractSpectralQuadratureSampling end

"""
    AbstractLatLonSampling <: AbstractTensorProductSphericalSampling

Geophysical / model latitude–longitude grids. No exact band-limited SHT claim.
"""
abstract type AbstractLatLonSampling <: AbstractTensorProductSphericalSampling end

abstract type AbstractEqualAreaSphericalSampling <: AbstractSphericalSampling end
abstract type AbstractHEALPixSampling <: AbstractEqualAreaSphericalSampling end
abstract type AbstractCubedSphereSampling <: AbstractSphericalSampling end
abstract type AbstractIcosahedralSampling <: AbstractSphericalSampling end
abstract type AbstractYinYangSampling <: AbstractSphericalSampling end
abstract type AbstractScatteredSphericalSampling <: AbstractSphericalSampling end

# ---------------------------------------------------------------------------
# Concrete defaults
# ---------------------------------------------------------------------------

"""
    GaussLegendreSampling <: AbstractGaussLegendreSampling

Gauss–Legendre (Gauss–Neumann) latitudes: ``μ = \\cosθ`` at the ``N_θ`` roots of
``P_{N_θ}``, with ``N_λ = 2N_θ − 1`` equispaced longitudes.
Exact for band-limit ``l_{\\max} = N_θ − 1``.
"""
struct GaussLegendreSampling <: AbstractGaussLegendreSampling end

"""
    DriscollHealySampling <: AbstractDriscollHealySampling

Driscoll–Healy equiangular grid, rectangular (**DH2**) layout:
``N_θ × 2N_θ`` with ``N_θ = 2(l_{\\max}+1)``, ``Δθ = π/N_θ``, ``Δλ = π/N_θ``.
Includes the north pole band, excludes the south (weight at the north pole is zero).

See also [`DriscollHealyEqualSampling`](@ref) for the square **DH1** ``N_θ × N_θ`` layout.
"""
struct DriscollHealySampling <: AbstractDriscollHealySampling end

"""
    DriscollHealyEqualSampling <: AbstractDriscollHealySampling

Driscoll–Healy **DH1** layout: ``N_θ × N_θ`` with ``N_θ = 2(l_{\\max}+1)``,
``Δθ = π/N_θ``, ``Δλ = 2π/N_θ``.
"""
struct DriscollHealyEqualSampling <: AbstractDriscollHealySampling end

"""
    ClenshawCurtisSampling <: AbstractClenshawCurtisSampling

Open equiangular Clenshaw–Curtis grid: ``θ_i = π(i−1/2)/N_θ`` (no poles),
``N_λ = 2N_θ − 1``, ``l_{\\max} = N_θ − 1``.

This is *not* the same as Driscoll–Healy: different ``N_θ(l_{\\max})``, open vs polar-cap
nodes, and a different quadrature.
"""
struct ClenshawCurtisSampling <: AbstractClenshawCurtisSampling end

"""
    McEwenWiauxSampling <: AbstractMcEwenWiauxSampling

McEwen–Wiaux (2011) equiangular sampling: ``θ_t = π(2t+1)/(2L−1)`` for
``t = 0,…,L−1``, ``λ_p = 2π p/(2L−1)`` for ``p = 0,…,2L−2``, with ``L = l_{\\max}+1``.
Requires asymptotically ``∼ 2 L^2`` samples (about half of classical DH).
"""
struct McEwenWiauxSampling <: AbstractMcEwenWiauxSampling end

"""
    LatLonSampling <: AbstractLatLonSampling

Arbitrary model lat–lon (regional or global, poles optional). No spectral exactness claim.
"""
struct LatLonSampling <: AbstractLatLonSampling end

"""
    HEALPixSampling <: AbstractHEALPixSampling

Hierarchical Equal Area isoLatitude Pixelisation (Górski et al.).
Equal-area pixels on ``4 N_{\\mathrm{side}} − 1`` iso-latitude rings; not a tensor-product
``N_λ × N_φ`` array (``n_λ`` varies by ring).
"""
struct HEALPixSampling <: AbstractHEALPixSampling
    nside::Int
    function HEALPixSampling(nside::Integer)
        nside ≥ 1 || throw(ArgumentError("HEALPix nside must be ≥ 1, got $nside"))
        return new(Int(nside))
    end
end

"""
    CubedSphereSampling <: AbstractCubedSphereSampling

Inscribed-cube projection: six logically rectangular panels (gnomonic by default).
Quasi-uniform alternative to global lat–lon (e.g. FV3).
"""
struct CubedSphereSampling <: AbstractCubedSphereSampling end

"""
    IcosahedralSampling <: AbstractIcosahedralSampling

Icosahedral geodesic discretization (triangular faces; hexagonal dual is the Voronoi mesh).
Quasi-uniform; unstructured connectivity.
"""
struct IcosahedralSampling <: AbstractIcosahedralSampling
    frequency::Int  # geodesic frequency ν (ν=1 → 12 vertices of the icosahedron)
    function IcosahedralSampling(frequency::Integer = 1)
        frequency ≥ 1 || throw(ArgumentError("icosahedral frequency must be ≥ 1"))
        return new(Int(frequency))
    end
end

"""
    YinYangSampling <: AbstractYinYangSampling

Overset of two low-latitude lat–lon patches (Kageyama–Sato), avoiding polar singularities
while keeping structured coordinates on each panel.
"""
struct YinYangSampling <: AbstractYinYangSampling end

"""
    ScatteredSphericalSampling <: AbstractScatteredSphericalSampling

Arbitrary ``(λ, φ)`` point set (NUFFT / NUFSHT paths).
"""
struct ScatteredSphericalSampling <: AbstractScatteredSphericalSampling end

# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------

is_tensor_product(::AbstractSphericalSampling) = false
is_tensor_product(::AbstractTensorProductSphericalSampling) = true
is_tensor_product(::AbstractYinYangSampling) = false  # two structured panels, not one TP grid

is_iso_latitude(::AbstractSphericalSampling) = false
is_iso_latitude(::AbstractTensorProductSphericalSampling) = true
is_iso_latitude(::AbstractHEALPixSampling) = true

is_equal_area(::AbstractSphericalSampling) = false
is_equal_area(::AbstractEqualAreaSphericalSampling) = true

"""
    admits_exact_bandlimited_quadrature(sampling) -> Bool

Whether this sampling's [`latitude_weights`](@ref) integrate the PRODUCTS that spectral analysis
actually forms — two degree-`lmax` functions, hence degree `2·lmax` — exactly at the sampling's own
[`bandlimit`](@ref).

That is a stronger statement than "the quadrature integrates a single `P_l` up to `lmax`", and the
distinction decides the answer here. Measured exactness of the weights in this package:

| sampling | exact for a single `P_l` up to | `bandlimit` | needs `2·lmax` | exact? |
|---|---|---|---|---|
| Gauss–Legendre | `2N−1` | `N−1` | `2N−2` | yes |
| Driscoll–Healy | `N−1` | `N/2−1` | `N−2` | yes |
| Clenshaw–Curtis | `N−1` | `N−1` | `2N−2` | **no** |

Clenshaw–Curtis's band limit describes what its grid can REPRESENT; its quadrature supports
quadrature-based analysis only to `lmax ≈ (N−1)/2`. Use `GaussLegendreSampling` when analysis must
be exact at the stated band limit.
"""
admits_exact_bandlimited_quadrature(::AbstractSphericalSampling) = false
admits_exact_bandlimited_quadrature(::AbstractSpectralQuadratureSampling) = true
admits_exact_bandlimited_quadrature(::AbstractClenshawCurtisSampling) = false
admits_exact_bandlimited_quadrature(::AbstractMcEwenWiauxSampling) = false

# ---------------------------------------------------------------------------
# Angle convention helpers
# ---------------------------------------------------------------------------

"""Colatitude ``θ ∈ [0, π]`` from geographic latitude ``φ ∈ [-π/2, π/2]``."""
@inline colatitude(φ::Real) = oftype(float(φ), π) / 2 - float(φ)

"""Geographic latitude ``φ`` from colatitude ``θ``."""
@inline geographic_latitude(θ::Real) = oftype(float(θ), π) / 2 - float(θ)

# ---------------------------------------------------------------------------
# Band-limit ↔ grid size (ℓ = 0 … lmax)
# ---------------------------------------------------------------------------

"""
    bandlimit(sampling, nlat) -> lmax

Maximum spherical-harmonic degree supported by `nlat` latitude samples under `sampling`.
"""
bandlimit(::AbstractGaussLegendreSampling, nlat::Integer) = Int(nlat) - 1
bandlimit(::AbstractClenshawCurtisSampling, nlat::Integer) = Int(nlat) - 1
bandlimit(::AbstractMcEwenWiauxSampling, nlat::Integer) = Int(nlat) - 1
function bandlimit(::AbstractDriscollHealySampling, nlat::Integer)
    nlat ≥ 2 && iseven(Int(nlat)) || throw(ArgumentError("DH nlat must be even and ≥ 2"))
    return Int(nlat) ÷ 2 - 1
end
bandlimit(::AbstractLatLonSampling, ::Integer) =
    throw(ArgumentError("LatLonSampling has no exact band-limit relation"))

"""
    nlat_for_bandlimit(sampling, lmax) -> nlat
"""
nlat_for_bandlimit(::AbstractGaussLegendreSampling, lmax::Integer) = Int(lmax) + 1
nlat_for_bandlimit(::AbstractClenshawCurtisSampling, lmax::Integer) = Int(lmax) + 1
nlat_for_bandlimit(::AbstractMcEwenWiauxSampling, lmax::Integer) = Int(lmax) + 1
nlat_for_bandlimit(::AbstractDriscollHealySampling, lmax::Integer) = 2 * (Int(lmax) + 1)

"""
    nlon_for_nlat(sampling, nlat) -> nlon
"""
nlon_for_nlat(::AbstractGaussLegendreSampling, nlat::Integer) = 2 * Int(nlat) - 1
nlon_for_nlat(::AbstractClenshawCurtisSampling, nlat::Integer) = 2 * Int(nlat) - 1
nlon_for_nlat(::AbstractMcEwenWiauxSampling, nlat::Integer) = 2 * Int(nlat) - 1
nlon_for_nlat(::DriscollHealySampling, nlat::Integer) = 2 * Int(nlat)
nlon_for_nlat(::DriscollHealyEqualSampling, nlat::Integer) = Int(nlat)
nlon_for_nlat(::AbstractLatLonSampling, nlat::Integer) =
    throw(ArgumentError("LatLonSampling longitude count is user-chosen; pass nlon explicitly"))


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------

"""
    axes_lengths(sampling, nlat; nlon=nothing) -> NamedTuple{(:nlon,:nlat)}

Buffer lengths required by [`spherical_axes!`](@ref): `(; nlon, nlat)`.
"""
function axes_lengths(s::AbstractTensorProductSphericalSampling, nlat::Integer; nlon::Union{Nothing,Integer} = nothing)
    nlat = Int(nlat)
    nlon_eff = nlon === nothing ? nlon_for_nlat(s, nlat) : Int(nlon)
    return (; nlon = nlon_eff, nlat)
end

function axes_lengths(::AbstractLatLonSampling, nlat::Integer; nlon::Integer)
    return (; nlon = Int(nlon), nlat = Int(nlat))
end

"""
    npoints(sampling, args...) -> Int

Number of geographic samples produced by [`spherical_points!`](@ref) / [`spherical_points`](@ref).
"""
npoints(s::AbstractTensorProductSphericalSampling, nlat::Integer; kwargs...) =
    let sz = axes_lengths(s, nlat; kwargs...)
        sz.nlon * sz.nlat
    end
npoints(s::HEALPixSampling) = healpix_npix(s)
npoints(::CubedSphereSampling, n::Integer) = 6 * Int(n)^2
npoints(::YinYangSampling, nlon::Integer, nlat::Integer) = 2 * Int(nlon) * Int(nlat)
icosahedral_nvertices(frequency::Integer) = 10 * Int(frequency)^2 + 2
npoints(s::IcosahedralSampling) = icosahedral_nvertices(s.frequency)

# ---------------------------------------------------------------------------
# Gauss–Legendre nodes (Golub–Welsch)
# ---------------------------------------------------------------------------

"""
    _gauss_legendre_μ!(μ, w) -> NamedTuple{(:μ,:w)}

Write `n = length(μ)`-point Gauss–Legendre nodes/weights on ``μ ∈ (-1, 1)`` into the provided
buffers, ascending in `μ` (`length(μ) == length(w)`).

Newton's method on ``Pₙ``, evaluated by the Bonnet recurrence, from a Tricomi starting estimate
accurate to ``O(n⁻³)`` — so 2–4 iterations reach machine precision. Roots come in ``±`` pairs, so
only the upper half is solved.

Needs ``O(1)`` scratch. Time is still ``O(n²)`` — each of the ``n/2`` roots costs an ``O(n)``
recurrence; an ``O(n)`` total needs Bogaert-style asymptotic expansions instead of Newton.
"""
function _gauss_legendre_μ!(μ::AbstractVector{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
    length(w) == length(μ) || throw(DimensionMismatch("μ and w must have the same length"))
    _gauss_legendre!(T, μ, w, length(μ))
    return (; μ, w)
end

@inline _put!(::Nothing, ::Int, _) = nothing
@inline _put!(v::AbstractVector, i::Int, x) = (@inbounds v[i] = x; nothing)

# ---------------------------------------------------------------------------
# Gauss–Legendre by asymptotic expansion (Bogaert, SIAM J. Sci. Comput. 36(3), 2014;
# Hale & Townsend, SIAM J. Sci. Comput. 35(2), 2013)
# ---------------------------------------------------------------------------
#
# Each node sits near `j_k/(n+½)` for `j_k` the k-th zero of `J₀`, with corrections in powers of
# `(n+½)⁻²`. Nothing is iterated and no root depends on its neighbours, so this is `O(1)` per node
# with no error accumulation along the sequence — which is what separates it from a march.

# The first 20 zeros of J₀ to full Float64 precision; beyond that McMahon's expansion (DLMF 10.21.19)
# is already accurate, with fewer terms needed as k grows.
const _J0_ZEROS = (
    2.4048255576957728, 5.5200781102863106, 8.6537279129110122, 11.791534439014281,
    14.930917708487785, 18.071063967910922, 21.211636629879258, 24.352471530749302,
    27.493479132040254, 30.634606468431975, 33.775820213573568, 36.917098353664044,
    40.058425764628239, 43.199791713176730, 46.341188371661814, 49.482609897397817,
    52.624051841114996, 55.765510755019979, 58.906983926080942, 62.048469190227170,
)
# J₁(j_k)², likewise tabulated then continued asymptotically (DLMF 10.17.3).
const _J1SQ_AT_J0_ZEROS = (
    0.2695141239419169, 0.1157801385822037, 0.07368635113640822, 0.05403757319811628,
    0.04266142901724309, 0.03524210349099610, 0.03002107010305467, 0.02614739149530809,
    0.02315912182469139, 0.02078382912226786,
)

@inline function _j0_zero(k::Int)
    k ≤ 20 && return @inbounds _J0_ZEROS[k]
    ak = π * (k - 0.25)
    q = (0.125 / ak)^2
    k ≤ 47 && return ak + 0.125 / ak * evalpoly(q, (1.0, -124 / 3, 120928 / 15, -401743168 / 105))
    k ≤ 344 && return ak + 0.125 / ak * evalpoly(q, (1.0, -124 / 3, 120928 / 15))
    k ≤ 13191 && return ak + 0.125 / ak * muladd(q, -124 / 3, 1.0)
    return ak + 0.125 / ak
end

@inline function _j1sq_at_j0_zero(k::Int)
    k ≤ 10 && return @inbounds _J1SQ_AT_J0_ZEROS[k]
    ak = π * (k - 0.25)
    q = (1 / ak)^2
    s = 1 / (π * ak)
    c1 = -171497088497 / 15206400; c2 = 461797 / 1152; c3 = -172913 / 8064
    c4 = 151 / 80; c5 = -7 / 24
    k ≤ 15 && return s * muladd(evalpoly(q, (c5, c4, c3, c2, c1)), q^2, 2.0)
    k ≤ 21 && return s * muladd(evalpoly(q, (c5, c4, c3, c2)), q^2, 2.0)
    k ≤ 55 && return s * muladd(evalpoly(q, (c5, c4, c3)), q^2, 2.0)
    k ≤ 279 && return s * muladd(muladd(q, c4, c5), q^2, 2.0)
    k ≤ 2279 && return s * muladd(q^2, c5, 2.0)
    return s * 2.0
end

function _gauss_legendre_asy!(
    ::Type{T}, μ::Union{Nothing,AbstractVector}, w::Union{Nothing,AbstractVector},
    n::Int, m::Int,
) where {T<:AbstractFloat}
    vn = 1 / (n + 0.5)
    vn² = vn * vn
    vn⁴ = vn² * vn²
    vn⁶ = vn⁴ * vn²
    @inbounds for i in 1:m
        ai = _j0_zero(i) * vn
        u = cot(ai)
        u² = u * u
        ai² = ai * ai; ai³ = ai² * ai; ai⁵ = ai² * ai³
        # Node: leading term aᵢ, then successive powers of vn². Fewer terms are needed as n grows,
        # because vn shrinks; past n = 3950 the second-order term already reaches Float64 precision.
        node = ai + (u - 1 / ai) / 8 * vn²
        if n ≤ 3950
            v1 = (6 * (1 + u²) / ai + 25 / ai³ - u * muladd(31, u², 33)) / 384
            node = muladd(v1, vn⁴, node)
            if n ≤ 255
                v2 = u * evalpoly(u², (2595 / 15360, 6350 / 15360, 3779 / 15360))
                v3 = (1 + u²) *
                     (-muladd(31 / 1024, u², 11 / 1024) / ai + u / 512 / ai² - 25 / 3072 / ai³)
                node = muladd(v2 - 1073 / 5120 / ai⁵ + v3, vn⁶, node)
            end
        end
        xt = T(cos(node))
        ua = u * ai
        W1 = muladd(ua - 1, 1 / ai², 1.0) / 8
        W = 1 / vn² + W1
        if n ≤ 1500
            W2 = evalpoly(1 / ai², (
                evalpoly(u², (-27.0, -84.0, -56.0)),
                muladd(-3.0, muladd(u², -2.0, 1.0), 6 * ua),
                muladd(ua, -31.0, 81.0),
            )) / 384
            if n ≤ 170
                W3 = evalpoly(1 / ai, (
                    evalpoly(u², (153 / 1024, 295 / 256, 187 / 96, 151 / 160)),
                    evalpoly(u², (-65 / 1024, -119 / 768, -35 / 384)) * u,
                    evalpoly(u², (5 / 512, 15 / 512, 7 / 384)),
                    muladd(u², 1 / 512, -13 / 1536) * u,
                    muladd(u², -7 / 384, 53 / 3072),
                    3749 / 15360 * u, -1125 / 1024,
                ))
                W = evalpoly(vn², (1 / vn² + W1, W2, W3))
            else
                W = muladd(vn², W2, 1 / vn² + W1)
            end
        end
        wi = T(2 / (_j1sq_at_j0_zero(i) * (ai / sin(ai)) * W))
        # Index 1 is the root nearest +1, so it lands at the top of the ascending output.
        _put!(μ, n + 1 - i, xt); _put!(w, n + 1 - i, wi)
        _put!(μ, i, -xt);        _put!(w, i, wi)
    end
    isodd(n) && _put!(μ, m, zero(T))
    return nothing
end

"""
    _gauss_legendre!(T, μ, w, n)

Core solve. Either output may be `nothing`, so a caller that wants only the nodes or only the
weights needs no scratch vector for the other half.
"""
function _gauss_legendre!(
    ::Type{T}, μ::Union{Nothing,AbstractVector}, w::Union{Nothing,AbstractVector}, n::Int,
) where {T<:AbstractFloat}
    n ≥ 1 || throw(ArgumentError("need n ≥ 1"))
    μ === nothing || length(μ) == n || throw(DimensionMismatch("μ must have length n"))
    w === nothing || length(w) == n || throw(DimensionMismatch("w must have length n"))
    if n == 1
        _put!(μ, 1, zero(T))
        _put!(w, 1, T(2))
        return nothing
    end
    m = (n + 1) ÷ 2                       # roots are symmetric about 0; solve the upper half only
    # Work at no less than Float64 and round once: the series below sums ~30 terms, which at Float32
    # would leave the root short of that type's own precision.
    TW = promote_type(T, Float64)
    # The asymptotic expansions are Float64 coefficient sets truncated at a fixed order, so they are
    # used only where they are actually the better answer: at Float64 width and above n = 60. Wider
    # types and small n fall through to Newton, which is exact arithmetic converging to eps(TW).
    # Measured relative weight error at Float64 against a high-precision reference:
    #   n       8      33      64      1024     4096
    #   Newton  8e-16  3e-14   5e-14   3e-12    2e-10
    #   asy     5e-9   2e-13   1e-15   3e-15    1e-14
    if TW === Float64 && n > 60
        _gauss_legendre_asy!(T, μ, w, n, m)
    else
        _gauss_legendre_newton!(T, TW, μ, w, n, m)
    end
    return nothing
end

"""
    _gauss_legendre_newton!(T, TW, μ, w, n, m)

Nodes and weights by Newton on `Pₙ`, evaluated by the Bonnet recurrence from a Tricomi start.

`O(n)` per root, so `O(n²)` overall — but it is exact arithmetic converging to `eps(TW)`, which the
asymptotic expansion's fixed Float64 coefficient set cannot do. That makes it the right method for
wide element types at any `n`, and for small `n`, where the expansion is inaccurate and the quadratic
cost is microseconds.
"""
function _gauss_legendre_newton!(
    ::Type{T}, ::Type{TW}, μ::Union{Nothing,AbstractVector}, w::Union{Nothing,AbstractVector},
    n::Int, m::Int,
) where {T<:AbstractFloat, TW<:AbstractFloat}
    tol = 4 * eps(TW)
    Tn = TW(n)
    @inbounds for i in 1:m
        θ = TW(π) * (4 * TW(i) - 1) / (4 * Tn + 2)
        x = (1 - (Tn - 1) / (8 * Tn^3)) * cos(θ)
        dp = zero(TW)
        for _ in 1:100
            # Bonnet: p1 ends as Pₙ(x), p2 as Pₙ₋₁(x); P′ₙ follows from both.
            p1 = one(TW); p2 = zero(TW)
            for k in 1:n
                p3 = p2; p2 = p1
                p1 = ((2 * TW(k) - 1) * x * p2 - (TW(k) - 1) * p3) / TW(k)
            end
            dp = Tn * (x * p1 - p2) / (x * x - 1)
            δ = p1 / dp
            x -= δ
            abs(δ) ≤ tol * (abs(x) + 1) && break
        end
        wi = T(TW(2) / ((1 - x * x) * dp * dp))
        xt = T(x)
        _put!(μ, i, -xt);        _put!(w, i, wi)
        _put!(μ, n + 1 - i, xt); _put!(w, n + 1 - i, wi)
    end
    # For odd n the central root is exactly zero; the ± assignment above leaves it as -0.0.
    isodd(n) && _put!(μ, m, zero(T))
    return nothing
end

function _gauss_legendre_μ(n::Integer; T::Type{<:AbstractFloat} = Float64)
    μ = Vector{T}(undef, Int(n))
    w = Vector{T}(undef, Int(n))
    _gauss_legendre_μ!(μ, w)
    return (; μ, w)
end

# ---------------------------------------------------------------------------
# spherical_axes! / spherical_axes
# ---------------------------------------------------------------------------

"""
    spherical_axes!(λ, φ, sampling, nlat; nlon=…) -> NamedTuple{(:λ,:φ)}

Fill preallocated longitude / geographic-latitude axes. Requires
`length(λ) == sz.nlon` and `length(φ) == sz.nlat` for `sz = axes_lengths(...)`.
"""
function spherical_axes! end

function spherical_axes!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, ::AbstractGaussLegendreSampling, nlat::Integer;
    nlon::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    nlat = Int(nlat)
    sz = axes_lengths(GaussLegendreSampling(), nlat; nlon)
    length(λ) == sz.nlon && length(φ) == sz.nlat || throw(DimensionMismatch("λ/φ lengths must match axes_lengths"))
    nlon_eff = sz.nlon
    # φ receives μ and is converted in place; the weights are not requested, so there is no scratch
    # at all. Callers that want both should use `spherical_quadrature!`, which solves once.
    _gauss_legendre!(T, φ, nothing, nlat)
    @inbounds for i in 1:nlat
        φ[i] = asin(φ[i])
    end
    dλ = T(2π) / T(nlon_eff)
    @inbounds for i in 1:nlon_eff
        λ[i] = T(i - 1) * dλ
    end
    return (; λ, φ)
end

function spherical_axes!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, ::AbstractClenshawCurtisSampling, nlat::Integer;
    nlon::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    nlat = Int(nlat)
    sz = axes_lengths(ClenshawCurtisSampling(), nlat; nlon)
    length(λ) == sz.nlon && length(φ) == sz.nlat || throw(DimensionMismatch("λ/φ lengths must match axes_lengths"))
    nlon_eff = sz.nlon
    @inbounds for i in 1:nlat
        θ = T(π) * (T(i) - T(0.5)) / T(nlat)
        φ[i] = geographic_latitude(θ)
    end
    dλ = T(2π) / T(nlon_eff)
    @inbounds for i in 1:nlon_eff
        λ[i] = T(i - 1) * dλ
    end
    return (; λ, φ)
end

function spherical_axes!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, ::AbstractMcEwenWiauxSampling, nlat::Integer;
    nlon::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    L = Int(nlat)
    L ≥ 1 || throw(ArgumentError("MW nlat (= L) must be ≥ 1"))
    sz = axes_lengths(McEwenWiauxSampling(), L; nlon)
    length(λ) == sz.nlon && length(φ) == sz.nlat || throw(DimensionMismatch("λ/φ lengths must match axes_lengths"))
    nlon_eff = sz.nlon
    den = T(2 * L - 1)
    @inbounds for t in 0:(L - 1)
        θ = T(π) * T(2t + 1) / den
        φ[t + 1] = geographic_latitude(θ)
    end
    dλ = T(2π) / T(nlon_eff)
    @inbounds for i in 1:nlon_eff
        λ[i] = T(i - 1) * dλ
    end
    return (; λ, φ)
end

function spherical_axes!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, ::DriscollHealySampling, nlat::Integer;
    nlon::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    nlat = Int(nlat)
    iseven(nlat) || throw(ArgumentError("DH nlat must be even (N_θ = 2(l_max+1))"))
    sz = axes_lengths(DriscollHealySampling(), nlat; nlon)
    length(λ) == sz.nlon && length(φ) == sz.nlat || throw(DimensionMismatch("λ/φ lengths must match axes_lengths"))
    nlon_eff = sz.nlon
    @inbounds for i in 0:(nlat - 1)
        θ = T(i) * T(π) / T(nlat)
        φ[i + 1] = geographic_latitude(θ)
    end
    dλ = T(π) / T(nlat)   # DH2
    @inbounds for i in 1:nlon_eff
        λ[i] = T(i - 1) * dλ
    end
    return (; λ, φ)
end

function spherical_axes!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, ::DriscollHealyEqualSampling, nlat::Integer;
    nlon::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    nlat = Int(nlat)
    iseven(nlat) || throw(ArgumentError("DH nlat must be even"))
    sz = axes_lengths(DriscollHealyEqualSampling(), nlat; nlon)
    length(λ) == sz.nlon && length(φ) == sz.nlat || throw(DimensionMismatch("λ/φ lengths must match axes_lengths"))
    nlon_eff = sz.nlon
    @inbounds for i in 0:(nlat - 1)
        θ = T(i) * T(π) / T(nlat)
        φ[i + 1] = geographic_latitude(θ)
    end
    dλ = T(2π) / T(nlon_eff)  # DH1
    @inbounds for i in 1:nlon_eff
        λ[i] = T(i - 1) * dλ
    end
    return (; λ, φ)
end

function spherical_axes!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, ::AbstractLatLonSampling, nlat::Integer;
    nlon::Integer,
    lat_range::Tuple{<:Real,<:Real} = (-π / 2, π / 2),
    lon_range::Tuple{<:Real,<:Real} = (0, 2π),
) where {T<:AbstractFloat}
    nlat = Int(nlat); nlon = Int(nlon)
    length(λ) == nlon && length(φ) == nlat || throw(DimensionMismatch("λ/φ lengths must be (nlon, nlat)"))
    φ0, φ1 = T(lat_range[1]), T(lat_range[2])
    λ0, λ1 = T(lon_range[1]), T(lon_range[2])
    dφ = nlat == 1 ? zero(T) : (φ1 - φ0) / T(nlat - 1)
    dλ = (λ1 - λ0) / T(nlon)
    @inbounds for j in 1:nlat
        φ[j] = φ0 + T(j - 1) * dφ
    end
    @inbounds for i in 1:nlon
        λ[i] = λ0 + T(i - 1) * dλ
    end
    return (; λ, φ)
end

"""
    spherical_axes(sampling, nlat; nlon=…, T=Float64) -> NamedTuple{(:λ,:φ)}

Allocating wrapper around [`spherical_axes!`](@ref).
"""
function spherical_axes(s::AbstractTensorProductSphericalSampling, nlat::Integer; nlon::Union{Nothing,Integer} = nothing, T::Type{<:AbstractFloat} = Float64)
    sz = axes_lengths(s, nlat; nlon)
    λ = Vector{T}(undef, sz.nlon)
    φ = Vector{T}(undef, sz.nlat)
    return spherical_axes!(λ, φ, s, nlat; nlon)
end

function spherical_axes(
    s::AbstractLatLonSampling, nlat::Integer;
    nlon::Integer,
    lat_range::Tuple{<:Real,<:Real} = (-π / 2, π / 2),
    lon_range::Tuple{<:Real,<:Real} = (0, 2π),
    T::Type{<:AbstractFloat} = Float64,
)
    λ = Vector{T}(undef, Int(nlon))
    φ = Vector{T}(undef, Int(nlat))
    return spherical_axes!(λ, φ, s, nlat; nlon, lat_range, lon_range)
end

# ---------------------------------------------------------------------------
# spherical_quadrature! / spherical_quadrature
# ---------------------------------------------------------------------------

"""
    spherical_quadrature!(λ, φ, w, sampling, nlat; nlon=…) -> NamedTuple{(:λ,:φ,:w)}

Fill preallocated axes *and* latitude weights together. Buffer lengths are as for
[`spherical_axes!`](@ref), with `length(w) == sz.nlat`.

This is the entry point to use whenever both are needed. Gauss–Legendre nodes and weights fall out
of a single root solve, but [`spherical_axes!`](@ref) keeps only the nodes and
[`latitude_weights!`](@ref) only the weights — so calling them in sequence pays the ``O(n²)`` solve
twice. Every other sampling has independent closed forms for the two, and gets the generic method.
"""
function spherical_quadrature! end

function spherical_quadrature!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, w::AbstractVector{T},
    ::AbstractGaussLegendreSampling, nlat::Integer; nlon::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    nlat = Int(nlat)
    sz = axes_lengths(GaussLegendreSampling(), nlat; nlon)
    length(λ) == sz.nlon && length(φ) == sz.nlat && length(w) == sz.nlat ||
        throw(DimensionMismatch("λ/φ/w lengths must match axes_lengths"))
    # One solve, no scratch: φ receives μ and is converted to latitude in place.
    _gauss_legendre!(T, φ, w, nlat)
    @inbounds for i in 1:nlat
        φ[i] = asin(φ[i])
    end
    dλ = T(2π) / T(sz.nlon)
    @inbounds for i in 1:sz.nlon
        λ[i] = T(i - 1) * dλ
    end
    return (; λ, φ, w)
end

function spherical_quadrature!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, w::AbstractVector{T},
    s::AbstractTensorProductSphericalSampling, nlat::Integer; nlon::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
    spherical_axes!(λ, φ, s, nlat; nlon)
    latitude_weights!(w, s, nlat)
    return (; λ, φ, w)
end

"""
    spherical_quadrature(sampling, nlat; nlon=…, T=Float64) -> NamedTuple{(:λ,:φ,:w)}

Allocating wrapper around [`spherical_quadrature!`](@ref).
"""
function spherical_quadrature(
    s::AbstractTensorProductSphericalSampling, nlat::Integer;
    nlon::Union{Nothing,Integer} = nothing, T::Type{<:AbstractFloat} = Float64,
)
    sz = axes_lengths(s, nlat; nlon)
    λ = Vector{T}(undef, sz.nlon)
    φ = Vector{T}(undef, sz.nlat)
    w = Vector{T}(undef, sz.nlat)
    return spherical_quadrature!(λ, φ, w, s, nlat; nlon)
end

# ---------------------------------------------------------------------------
# latitude_weights! / latitude_weights
# ---------------------------------------------------------------------------

"""
    latitude_weights!(w, sampling, nlat) -> w

Fill preallocated latitude quadrature weights (`length(w) == nlat`).
"""
function latitude_weights! end

function latitude_weights!(w::AbstractVector{T}, ::AbstractGaussLegendreSampling, nlat::Integer) where {T<:AbstractFloat}
    nlat = Int(nlat)
    length(w) == nlat || throw(DimensionMismatch("w length must equal nlat"))
    _gauss_legendre!(T, nothing, w, nlat)
    return w
end

"""
    OpenNodes(), ClosedNodes()

The two equiangular colatitude families: open `θᵢ = π(i−½)/N` (Clenshaw–Curtis) and closed
`θᵢ = π(i−1)/N` (Driscoll–Healy). Named rather than passed as a `θ(i)` closure so the sum below can
dispatch on which one it is.
"""
struct OpenNodes end
struct ClosedNodes end

@inline _node_theta(::OpenNodes, i::Int, nlat::Int, ::Type{T}) where {T} =
    T(π) * (T(i) - T(0.5)) / T(nlat)
@inline _node_theta(::ClosedNodes, i::Int, nlat::Int, ::Type{T}) where {T} =
    T(π) * T(i - 1) / T(nlat)

"""
    _equiangular_sums!(s, family, nlat, nterm) -> s

`s[i] = Σ_{k=0}^{nterm-1} sin((2k+1)·θᵢ)/(2k+1)` for the given node family.

Evaluated by the angle-addition recurrence
`sin((2k+1)θ) = 2cos(2θ)·sin((2k−1)θ) − sin((2k−3)θ)`, seeded with `s₋₁ = −sin θ`, `s₀ = sin θ`, so
each term costs two multiplies rather than a `sin`. This is `O(nlat·nterm)`; loading an FFT
implementation replaces it with one length-`nlat` transform (see the `AbstractFFTs` extension).
"""
function _equiangular_sums!(s::AbstractVector{T}, family, nlat::Int, nterm::Int) where {T<:AbstractFloat}
    @inbounds for i in 1:nlat
        θi = _node_theta(family, i, nlat, T)
        sθ = sin(θi)
        u = 2 * cos(2 * θi)
        skm1 = -sθ
        sk = sθ
        acc = zero(T)
        for k in 0:(nterm - 1)
            acc += sk / T(2k + 1)
            skm1, sk = sk, u * sk - skm1
        end
        s[i] = acc
    end
    return s
end

"""
    _equiangular_weights!(w, family, nlat) -> w

Latitude quadrature weights for an equiangular node set, from the sine-series expansion of the
`sinθ` Jacobian:

    wᵢ = (4/N)·sinθᵢ·Σ_{k=0}^{⌊N/2⌋} sin((2k+1)θᵢ)/(2k+1)

`family` is [`OpenNodes`](@ref) or `ClosedNodes`, so one construction serves both. Weights sum to
`∫₀^π sinθ dθ = 2` — see [`latitude_weights`](@ref) for why every sampling here uses that
normalization.
"""
function _equiangular_weights!(w::AbstractVector{T}, family, nlat::Int) where {T<:AbstractFloat}
    nterm = (nlat + 1) ÷ 2
    _equiangular_sums!(w, family, nlat, nterm)      # w doubles as the sum buffer
    @inbounds for i in 1:nlat
        w[i] *= (T(4) / T(nlat)) * sin(_node_theta(family, i, nlat, T))
    end
    return w
end

function latitude_weights!(w::AbstractVector{T}, ::AbstractDriscollHealySampling, nlat::Integer) where {T<:AbstractFloat}
    nlat = Int(nlat)
    iseven(nlat) || throw(ArgumentError("DH nlat must be even"))
    length(w) == nlat || throw(DimensionMismatch("w length must equal nlat"))
    return _equiangular_weights!(w, ClosedNodes(), nlat)
end

"""
    latitude_weights!(w, ::AbstractClenshawCurtisSampling, nlat)

Weights for the open nodes `θᵢ = π(i−½)/N`, from the same sine-series rule as the closed
equiangular families.

These integrate a single `P_l` exactly for `l ≤ N−1`, which is weaker than
[`bandlimit`](@ref)`(ClenshawCurtisSampling(), N) = N−1` suggests: spectral analysis integrates
PRODUCTS of two degree-`lmax` functions, so a quadrature exact to `l ≤ N−1` supports
quadrature-based analysis only up to `lmax ≈ (N−1)/2`. The reported band limit describes what the
grid can represent, not what this quadrature can integrate. Use `GaussLegendreSampling` (exact to
`2N−1`) when analysis must be exact at the stated band limit.
"""
function latitude_weights!(w::AbstractVector{T}, ::AbstractClenshawCurtisSampling, nlat::Integer) where {T<:AbstractFloat}
    nlat = Int(nlat)
    length(w) == nlat || throw(DimensionMismatch("w length must equal nlat"))
    return _equiangular_weights!(w, OpenNodes(), nlat)
end

latitude_weights!(::AbstractVector, ::AbstractMcEwenWiauxSampling, ::Integer) = throw(ArgumentError(
    "McEwen–Wiaux latitude weights are not implemented: the MW quadrature is not the sine-series " *
    "rule the other equiangular samplings use (it is built on an extension of the sphere to a torus), " *
    "and applying that rule to MW nodes is not exact even for l = 0. Use `GaussLegendreSampling`, " *
    "`DriscollHealySampling`, or `ClenshawCurtisSampling` if you need quadrature weights.",
))
latitude_weights!(::AbstractVector, ::AbstractLatLonSampling, ::Integer) =
    throw(ArgumentError("LatLonSampling is an arbitrary lat–lon layout with no spectral quadrature weights"))

"""
    latitude_weights(s, nlat; T=Float64) -> Vector{T}
    latitude_weights!(w, s, nlat) -> w

Latitude quadrature weights `wⱼ` for sampling `s`, normalized so that

    Σⱼ wⱼ = ∫₀^π sinθ dθ = 2

for EVERY sampling that provides them. The weights therefore carry the `sinθ` Jacobian and nothing
else; the longitude factor is the caller's, so a full-sphere integral is always

    ∫ f dΩ ≈ (2π/nlon) · Σⱼ wⱼ Σᵢ f(λᵢ, φⱼ)

regardless of which sampling produced the weights. Not every sampling has them — `LatLonSampling`
has no spectral quadrature at all, and `McEwenWiauxSampling`'s is a different construction.
"""
function latitude_weights(s::AbstractSphericalSampling, nlat::Integer; T::Type{<:AbstractFloat} = Float64)
    w = Vector{T}(undef, Int(nlat))
    return latitude_weights!(w, s, nlat)
end

# ---------------------------------------------------------------------------
# spherical_points! / spherical_points
# ---------------------------------------------------------------------------

"""
    spherical_points!(λ, φ, sampling, args...) -> NamedTuple{(:λ,:φ)}

Fill preallocated point-coordinate buffers. See [`npoints`](@ref) for required lengths.
"""
function spherical_points! end

function spherical_points!(
    Λ::AbstractVector{T}, Φ::AbstractVector{T}, s::AbstractTensorProductSphericalSampling, nlat::Integer; kwargs...,
) where {T<:AbstractFloat}
    sz = axes_lengths(s, nlat; kwargs...)
    length(Λ) == sz.nlon * sz.nlat && length(Φ) == sz.nlon * sz.nlat || throw(DimensionMismatch("buffers must have length nlon*nlat"))
    λ = Vector{T}(undef, sz.nlon)
    φ = Vector{T}(undef, sz.nlat)
    spherical_axes!(λ, φ, s, nlat; kwargs...)
    k = 1
    @inbounds for j in 1:sz.nlat, i in 1:sz.nlon
        Λ[k] = λ[i]
        Φ[k] = φ[j]
        k += 1
    end
    return (; λ = Λ, φ = Φ)
end

"""
    spherical_points(sampling, args...; T=Float64) -> NamedTuple{(:λ,:φ)}

Every point of the sampling, flattened, as longitude/latitude vectors. Allocating wrapper around
[`spherical_points!`](@ref); use [`npoints`](@ref) to size buffers for the in-place form.

For a tensor-product sampling this is the outer product of its axes, so prefer
[`spherical_axes`](@ref) when the separable form will do.
"""
function spherical_points(s::AbstractTensorProductSphericalSampling, nlat::Integer; T::Type{<:AbstractFloat} = Float64, kwargs...)
    n = npoints(s, nlat; kwargs...)
    Λ = Vector{T}(undef, n)
    Φ = Vector{T}(undef, n)
    return spherical_points!(Λ, Φ, s, nlat; kwargs...)
end

# ---- HEALPix ----------------------------------------------------------------

healpix_npix(nside::Integer) = 12 * Int(nside)^2
healpix_npix(s::HEALPixSampling) = healpix_npix(s.nside)
healpix_nring(nside::Integer) = 4 * Int(nside) - 1
healpix_nring(s::HEALPixSampling) = healpix_nring(s.nside)
healpix_pixel_area(nside::Integer) = 4π / healpix_npix(nside)
healpix_pixel_area(s::HEALPixSampling) = healpix_pixel_area(s.nside)

function spherical_points!(λ::AbstractVector{T}, φ::AbstractVector{T}, s::HEALPixSampling) where {T<:AbstractFloat}
    npix = healpix_npix(s)
    length(λ) == npix && length(φ) == npix || throw(DimensionMismatch("buffers must have length healpix_npix"))
    nside = s.nside
    @inbounds for ipix in 0:(npix - 1)
        θ, ϕ = _healpix_pix2ang_ring(nside, ipix, T)
        λ[ipix + 1] = ϕ
        φ[ipix + 1] = geographic_latitude(θ)
    end
    return (; λ, φ)
end

function spherical_points(s::HEALPixSampling; T::Type{<:AbstractFloat} = Float64)
    n = npoints(s)
    return spherical_points!(Vector{T}(undef, n), Vector{T}(undef, n), s)
end

function _healpix_pix2ang_ring(nside::Int, ipix::Int, ::Type{T}) where {T<:AbstractFloat}
    fn = T(nside)
    nl2 = 2 * nside
    nl4 = 4 * nside
    npix = 12 * nside * nside
    # Pixels in the north polar cap, i.e. rings 1 … nside-1, which hold 4, 8, … 4(nside-1) pixels:
    # 2·nside·(nside-1). Getting this wrong routes equatorial pixels through the cap branch.
    ncap = 2 * nside * (nside - 1)
    fact1 = T(1.5) * fn
    fact2 = T(3) * fn * fn
    if ipix < ncap
        hip = (ipix + 1) / T(2)
        fihip = floor(hip)
        iring = Int(floor(sqrt(hip - sqrt(fihip))) + 1)
        iphi = ipix + 1 - 2 * iring * (iring - 1)
        z = one(T) - T(iring * iring) / fact2
        ϕ = (T(iphi) - T(0.5)) * T(π) / (T(2) * T(iring))
    elseif ipix < (npix - ncap)
        # Every equatorial ring holds 4·nside pixels, so the ring index advances per nl4 — not per
        # nl2, which would invent twice as many half-width rings. `fodd` staggers alternate rings by
        # half a pixel: 1 when (iring+nside) is odd, 1/2 when even.
        ip = ipix - ncap
        tmp = ip ÷ nl4
        iring = tmp + nside
        iphi = ip - tmp * nl4 + 1
        fodd = isodd(iring + nside) ? one(T) : T(0.5)
        z = (T(nl2) - T(iring)) / fact1
        ϕ = (T(iphi) - fodd) * T(π) / (T(2) * fn)
    else
        ip = npix - ipix
        hip = ip / T(2)
        fihip = floor(hip)
        iring = Int(floor(sqrt(hip - sqrt(fihip))) + 1)
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        z = -one(T) + T(iring * iring) / fact2
        ϕ = (T(iphi) - T(0.5)) * T(π) / (T(2) * T(iring))
    end
    θ = acos(clamp(z, -one(T), one(T)))
    return θ, mod(ϕ, T(2π))
end

# ---- Cubed sphere -----------------------------------------------------------

"""
    cubed_sphere_points!(λ, φ, panel, n; backend=nothing) -> NamedTuple{(:λ,:φ,:panel)}

Gnomonic cubed-sphere CELL CENTRES into caller-owned buffers of length `6n²`, plus each point's panel
index. Pass `panel = nothing` when the panel id is not wanted, and it is not computed.

See [`cubed_sphere_points`](@ref) for the allocating form.
"""
function cubed_sphere_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T},
    panel::Union{Nothing,AbstractVector{<:Integer}}, n::Integer; backend = nothing,
) where {T<:AbstractFloat}
    n = Int(n)
    n ≥ 1 || throw(ArgumentError("cubed-sphere n must be ≥ 1, got $n"))
    N = 6 * n * n
    length(λ) == N && length(φ) == N || throw(DimensionMismatch("buffers must have length 6n²"))
    panel === nothing || length(panel) == N || throw(DimensionMismatch("buffers must have length 6n²"))
    # CELL CENTRES, not panel vertices: ξ_i = -π/4 + (i-½)·(π/2)/n.
    #
    # An endpoint-inclusive `range(-π/4, π/4; length=n)` puts nodes ON the panel edges, so adjacent
    # panels emit coincident points — measured, exactly 12(n-2)+16 duplicates — while
    # `_cubed_neighbor` simultaneously treats those edges as folding onto a *different* panel's
    # cells. Points and connectivity would then disagree, and any grid built from them carries
    # coincident nodes (degenerate tessellation, zero-area cells). Cell centres give 6n² genuinely
    # distinct points that match the connectivity, and make n=1 (one cell per face, at the face
    # centre) fall out of the formula instead of needing a special case.
    Δ = T(π) / 2 / T(n)
    a = range(-T(π) / 4 + Δ / 2; step = Δ, length = n)
    # Each output slot is a pure function of its own linear index, so chunks are independent.
    Execution.run_chunks(N, backend) do rng
        @inbounds for k in rng
            q, r0 = divrem(k - 1, n * n)
            j, i = divrem(r0, n)
            f = q + 1
            p = _cubed_face_to_xyz(f, tan(a[i + 1]), tan(a[j + 1]), T)
            r = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            x = p.x / r; y = p.y / r; z = p.z / r
            θ = acos(clamp(z, -one(T), one(T)))
            ϕ = atan(y, x)
            ϕ < 0 && (ϕ += T(2π))
            λ[k] = ϕ
            φ[k] = geographic_latitude(θ)
            _put!(panel, k, f)
        end
    end
    return (; λ, φ, panel)
end

"""
    cubed_sphere_points(n; T=Float64, backend=nothing) -> NamedTuple{(:λ,:φ,:panel)}

Gnomonic cubed-sphere CELL CENTRES: `6n²` distinct points, plus each point's panel index.

Allocating wrapper around [`cubed_sphere_points!`](@ref). Use
[`spherical_points`](@ref)`(CubedSphereSampling(), n)` when the panel id is not needed.
"""
function cubed_sphere_points(n::Integer; T::Type{<:AbstractFloat} = Float64, backend = nothing)
    N = npoints(CubedSphereSampling(), n)
    return cubed_sphere_points!(
        Vector{T}(undef, N), Vector{T}(undef, N), Vector{Int}(undef, N), n; backend = backend,
    )
end

function spherical_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, ::CubedSphereSampling, n::Integer; backend = nothing,
) where {T<:AbstractFloat}
    cubed_sphere_points!(λ, φ, nothing, n; backend = backend)   # panel id is not part of this result
    return (; λ, φ)
end

function spherical_points(
    ::CubedSphereSampling, n::Integer; T::Type{<:AbstractFloat} = Float64, backend = nothing,
)
    N = npoints(CubedSphereSampling(), n)
    return spherical_points!(
        Vector{T}(undef, N), Vector{T}(undef, N), CubedSphereSampling(), n; backend = backend,
    )
end

@inline function _cubed_face_to_xyz(face::Int, X::T, Y::T, ::Type{T}) where {T}
    if face == 1
        return (; x = X, y = Y, z = one(T))
    elseif face == 2
        return (; x = X, y = one(T), z = -Y)
    elseif face == 3
        return (; x = one(T), y = -X, z = -Y)
    elseif face == 4
        return (; x = -X, y = -one(T), z = -Y)
    elseif face == 5
        return (; x = -one(T), y = X, z = -Y)
    elseif face == 6
        return (; x = -Y, y = X, z = -one(T))
    else
        throw(ArgumentError("cubed-sphere face must be 1:6"))
    end
end

# ---- Yin–Yang ---------------------------------------------------------------

"""
    yin_yang_panels!(λyin, φyin, λyang, φyang, nlon, nlat) -> (; yin, yang)

The two Kageyama–Sato panels. `yin` is a pair of AXES (`nlon` and `nlat` long): in its own frame the
panel is a separable lat–lon patch. `yang` is that panel rotated onto the sphere, which is no longer
separable in global lon/lat, so it is a pair of `nlon × nlat` FIELDS — one `(λ, φ)` per cell.

The shapes differ because the geometry does, not by convention; the argument types say so.
"""
function yin_yang_panels!(
    λyin::AbstractVector{T}, φyin::AbstractVector{T},
    λyang::AbstractMatrix{T}, φyang::AbstractMatrix{T},
    nlon::Integer, nlat::Integer,
) where {T<:AbstractFloat}
    nlon = Int(nlon); nlat = Int(nlat)
    length(λyin) == nlon && length(φyin) == nlat ||
        throw(DimensionMismatch("yin axes must be nlon and nlat long"))
    size(λyang) == (nlon, nlat) && size(φyang) == (nlon, nlat) ||
        throw(DimensionMismatch("yang fields must be nlon × nlat"))
    # Cell centres, not panel edges: each node carries one cell, so the nlon×nlat cells tile
    # [-3π/4, 3π/4] × [-π/4, π/4] exactly. Sampling the endpoints instead would give the two
    # boundary columns/rows half-width cells while the connectivity still counts them whole.
    Δλ = (T(3π) / 2) / T(nlon)
    Δφ = (T(π) / 2) / T(nlat)
    @inbounds for i in 1:nlon
        λyin[i] = -T(3π) / 4 + (T(i) - T(0.5)) * Δλ
    end
    @inbounds for j in 1:nlat
        φyin[j] = -T(π) / 4 + (T(j) - T(0.5)) * Δφ
    end
    @inbounds for j in 1:nlat
        sinφ, cosφ = sincos(φyin[j])   # constant along the row; hoisted out of the longitude loop
        for i in 1:nlon
            sinλ, cosλ = sincos(λyin[i])
            X = -sinφ; Y = cosφ * cosλ; Z = -cosφ * sinλ
            θ = acos(clamp(Z, -one(T), one(T)))
            ϕ = atan(Y, X)
            ϕ < 0 && (ϕ += T(2π))
            λyang[i, j] = ϕ
            φyang[i, j] = geographic_latitude(θ)
        end
    end
    return (; yin = (; λ = λyin, φ = φyin), yang = (; λ = λyang, φ = φyang))
end

"""
    yin_yang_panels(nlon, nlat; T=Float64) -> (; yin, yang)

Allocating wrapper around [`yin_yang_panels!`](@ref).
"""
function yin_yang_panels(nlon::Integer, nlat::Integer; T::Type{<:AbstractFloat} = Float64)
    nlon = Int(nlon); nlat = Int(nlat)
    return yin_yang_panels!(
        Vector{T}(undef, nlon), Vector{T}(undef, nlat),
        Matrix{T}(undef, nlon, nlat), Matrix{T}(undef, nlon, nlat),
        nlon, nlat,
    )
end

function spherical_points!(Λ::AbstractVector{T}, Φ::AbstractVector{T}, ::YinYangSampling, nlon::Integer, nlat::Integer) where {T<:AbstractFloat}
    nlon = Int(nlon); nlat = Int(nlat)
    n = 2 * nlon * nlat
    length(Λ) == n && length(Φ) == n || throw(DimensionMismatch("buffers must have length 2*nlon*nlat"))
    λyin = Vector{T}(undef, nlon)
    φyin = Vector{T}(undef, nlat)
    λyang = Matrix{T}(undef, nlon, nlat)
    φyang = Matrix{T}(undef, nlon, nlat)
    yin_yang_panels!(λyin, φyin, λyang, φyang, nlon, nlat)
    k = 1
    @inbounds for φ in φyin, λ in λyin
        Λ[k] = λ; Φ[k] = φ; k += 1
    end
    @inbounds for i in eachindex(λyang)   # column-major, so this is the same (i, j) order as yin
        Λ[k] = λyang[i]; Φ[k] = φyang[i]; k += 1
    end
    return (; λ = Λ, φ = Φ)
end

function spherical_points(::YinYangSampling, nlon::Integer, nlat::Integer; T::Type{<:AbstractFloat} = Float64)
    n = npoints(YinYangSampling(), nlon, nlat)
    return spherical_points!(Vector{T}(undef, n), Vector{T}(undef, n), YinYangSampling(), nlon, nlat)
end

# ---- Icosahedral ------------------------------------------------------------

function _xyz_to_lonlat!(λ::AbstractVector{T}, φ::AbstractVector{T}, verts) where {T}
    n = length(verts)
    length(λ) == n && length(φ) == n || throw(DimensionMismatch("buffers must match vertex count"))
    @inbounds for i in 1:n
        x, y, z = verts[i]
        θ = acos(clamp(T(z), -one(T), one(T)))
        ϕ = atan(T(y), T(x))
        ϕ < 0 && (ϕ += T(2π))
        λ[i] = ϕ
        φ[i] = geographic_latitude(θ)
    end
    return (; λ, φ)
end

"""
    icosahedral_mesh(frequency; T=Float64) -> (; λ, φ, edges, triangles, verts)

Geodesic vertices at frequency `ν` as both lon/lat (`λ`, `φ`) and unit vectors (`verts`), plus the
`10ν²+2` mesh's undirected edges `(i,j)` with `i < j` and its `20ν²` triangles `(i,j,k)` (1-based).
Vertex numbering is topological — corners, then macro-edge interiors, then face interiors — so it is
deterministic and every vertex is generated exactly once.
"""
function icosahedral_mesh(
    frequency::Integer = 1; T::Type{<:AbstractFloat} = Float64, topology::Bool = true,
)
    ν = Int(frequency)
    nexp = icosahedral_nvertices(ν)
    φg = (one(T) + sqrt(T(5))) / T(2)
    raw = NTuple{3,T}[
        (zero(T), one(T), φg), (zero(T), -one(T), φg),
        (zero(T), one(T), -φg), (zero(T), -one(T), -φg),
        (one(T), φg, zero(T)), (-one(T), φg, zero(T)),
        (one(T), -φg, zero(T)), (-one(T), -φg, zero(T)),
        (φg, zero(T), one(T)), (φg, zero(T), -one(T)),
        (-φg, zero(T), one(T)), (-φg, zero(T), -one(T)),
    ]
    base = map(raw) do (x, y, z)
        r = sqrt(x * x + y * y + z * z)
        (x / r, y / r, z / r)
    end
    faces = _ICOSAHEDRON_FACES
    macro_edges = _icosahedron_edges(faces)     # the 30 canonical (lo, hi) corner pairs
    edge_index = zeros(Int, 12, 12)
    for (e, (lo, hi)) in enumerate(macro_edges)
        edge_index[lo, hi] = e
    end

    # Vertices are numbered by TOPOLOGY, not by hashing their coordinates: the 12 corners, then the
    # ν-1 interior points of each of the 30 macro-edges, then the (ν-1)(ν-2)/2 interior points of
    # each of the 20 faces — which sums to exactly 10ν²+2. Every vertex therefore has one owner and
    # is generated once, so no dedup dictionary, no quantized keys, and no per-vertex hashing.
    nint = ((ν - 1) * (ν - 2)) ÷ 2
    verts = Vector{NTuple{3,T}}(undef, nexp)
    @inline norm3(p) = (r = sqrt(p[1]^2 + p[2]^2 + p[3]^2); (p[1] / r, p[2] / r, p[3] / r))
    @inline lerp3(P, Q, a, b) = (a * P[1] + b * Q[1], a * P[2] + b * Q[2], a * P[3] + b * Q[3])

    @inbounds for v in 1:12
        verts[v] = base[v]
    end
    @inbounds for (e, (lo, hi)) in enumerate(macro_edges), t in 1:(ν - 1)
        verts[12 + (e - 1) * (ν - 1) + t] =
            norm3(lerp3(base[lo], base[hi], T(ν - t) / T(ν), T(t) / T(ν)))
    end
    face_base = 12 + 30 * (ν - 1)
    @inbounds for (f, (ia, ib, ic)) in enumerate(faces)
        A, B, C = base[ia], base[ib], base[ic]
        k = 0
        for i in 1:(ν - 1), j in 1:(ν - 1 - i)
            k += 1
            w = T(ν - i - j) / T(ν); u = T(i) / T(ν); v = T(j) / T(ν)
            verts[face_base + (f - 1) * nint + k] = norm3((
                w * A[1] + u * B[1] + v * C[1],
                w * A[2] + u * B[2] + v * C[2],
                w * A[3] + u * B[3] + v * C[3],
            ))
        end
    end

    # Triangles and edges from ONE walk of the three lattice directions on each face. A
    # face-boundary edge is generated by both adjacent faces, so the edge list is canonicalized and
    # deduped by sorting — a single sort of ~30ν² pairs, rather than hashing every edge into a Set.
    # Triangles need no dedup: each belongs to exactly one face. The two lattice orientations give
    # ν(ν+1)/2 upward plus ν(ν-1)/2 downward per face, i.e. 20ν² in total.
    per_face = 3 * ((ν * (ν + 1)) ÷ 2)
    edges = Vector{NTuple{2,Int}}(undef, topology ? 20 * per_face : 0)
    triangles = Vector{NTuple{3,Int}}(undef, topology ? 20 * ν * ν : 0)
    ne = 0
    nt = 0
    @inbounds topology && for (f, face) in enumerate(faces)
        for i in 0:ν, j in 0:(ν - i)
            v0 = _ico_node_id(f, face, i, j, ν, edge_index, nint, face_base)
            if i + j < ν
                vi = _ico_node_id(f, face, i + 1, j, ν, edge_index, nint, face_base)
                vj = _ico_node_id(f, face, i, j + 1, ν, edge_index, nint, face_base)
                ne += 1; edges[ne] = minmax(v0, vi)
                ne += 1; edges[ne] = minmax(v0, vj)
                nt += 1; triangles[nt] = (v0, vi, vj)
                if i + j < ν - 1
                    nt += 1
                    triangles[nt] =
                        (vi, vj, _ico_node_id(f, face, i + 1, j + 1, ν, edge_index, nint, face_base))
                end
            end
            if i ≥ 1
                ne += 1; edges[ne] = minmax(v0, _ico_node_id(f, face, i - 1, j + 1, ν, edge_index, nint, face_base))
            end
        end
    end
    if topology
        nt == 20 * ν * ν || throw(AssertionError("icosahedral triangle count $nt ≠ $(20 * ν * ν)"))
        resize!(edges, ne)
        sort!(edges)
        unique!(edges)
    end

    λ = Vector{T}(undef, nexp)
    φ = Vector{T}(undef, nexp)
    _xyz_to_lonlat!(λ, φ, verts)
    return (; λ, φ, edges, triangles, verts)
end

# The 30 canonical (lo, hi) corner pairs of the base icosahedron, in a deterministic order.
function _icosahedron_edges(faces)
    edges = NTuple{2,Int}[]
    for (a, b, c) in faces, (u, v) in ((a, b), (b, c), (c, a))
        push!(edges, minmax(u, v))
    end
    sort!(edges)
    unique!(edges)
    length(edges) == 30 || throw(ArgumentError("icosahedron edge recovery failed (got $(length(edges)))"))
    return edges
end

"""
    _ico_node_id(f, face, i, j, ν, edge_index, nint, face_base) -> Int

Global vertex id of barycentric lattice node `(i, j)` (with `i + j ≤ ν`, weights `ν-i-j`, `i`, `j`
on the face's corners `A`, `B`, `C`) of face `f`. Nodes on a corner or a macro-edge resolve to that
shared entity's id, which is what makes the two faces meeting at an edge agree without any lookup
table.
"""
@inline function _ico_node_id(
    f::Int, face::NTuple{3,Int}, i::Int, j::Int, ν::Int,
    edge_index::AbstractMatrix{Int}, nint::Int, face_base::Int,
)
    A, B, C = face
    # Corners.
    (i == 0 && j == 0) && return A
    (i == ν) && return B
    (j == ν) && return C
    # Macro-edge interiors: walk from `u` toward `v` with parameter `t`.
    if j == 0
        return _ico_edge_id(A, B, i, ν, edge_index)
    elseif i == 0
        return _ico_edge_id(A, C, j, ν, edge_index)
    elseif i + j == ν
        return _ico_edge_id(B, C, j, ν, edge_index)
    end
    # Face interior. Closed form for the position of (i, j) in the same `for ii, jj` order the
    # vertex pass used, so this stays O(1) instead of rescanning the lattice per lookup.
    k = (i - 1) * (ν - 1) - ((i - 1) * i) ÷ 2 + j
    return face_base + (f - 1) * nint + k
end

@inline function _ico_edge_id(u::Int, v::Int, t::Int, ν::Int, edge_index::AbstractMatrix{Int})
    lo, hi = minmax(u, v)
    e = @inbounds edge_index[lo, hi]
    # `t` counts from `u`; the stored interior points count from `lo`.
    s = (u == lo) ? t : (ν - t)
    return 12 + (e - 1) * (ν - 1) + s
end

function icosahedral_vertices!(λ::AbstractVector{T}, φ::AbstractVector{T}, frequency::Integer = 1) where {T<:AbstractFloat}
    mesh = icosahedral_mesh(frequency; T = T, topology = false)
    length(λ) == length(mesh.λ) && length(φ) == length(mesh.φ) || throw(DimensionMismatch("buffers must have length 10ν²+2"))
    copyto!(λ, mesh.λ)
    copyto!(φ, mesh.φ)
    return (; λ, φ)
end

"""
    icosahedral_vertices(frequency=1; T=Float64) -> NamedTuple{(:λ,:φ)}

The `10ν²+2` geodesic vertices as longitude/latitude, without building the mesh topology — several
times faster than [`icosahedral_mesh`](@ref) at large `ν`, which also returns edges and triangles.
"""
function icosahedral_vertices(frequency::Integer = 1; T::Type{<:AbstractFloat} = Float64)
    mesh = icosahedral_mesh(frequency; T = T, topology = false)
    return (; λ = mesh.λ, φ = mesh.φ)
end

# The 20 faces, as index triples into the 12 base vertices in the order `icosahedral_mesh` lists them.
const _ICOSAHEDRON_FACES = (
    (1, 2, 9), (1, 2, 11), (1, 5, 6), (1, 5, 9), (1, 6, 11),
    (2, 7, 8), (2, 7, 9), (2, 8, 11), (3, 4, 10), (3, 4, 12),
    (3, 5, 6), (3, 5, 10), (3, 6, 12), (4, 7, 8), (4, 7, 10),
    (4, 8, 12), (5, 9, 10), (6, 11, 12), (7, 9, 10), (8, 11, 12),
)

function spherical_points!(λ::AbstractVector{T}, φ::AbstractVector{T}, s::IcosahedralSampling) where {T<:AbstractFloat}
    return icosahedral_vertices!(λ, φ, s.frequency)
end

function spherical_points(s::IcosahedralSampling; T::Type{<:AbstractFloat} = Float64)
    return icosahedral_vertices(s.frequency; T = T)
end

spherical_points(::ScatteredSphericalSampling, λ::AbstractVector, φ::AbstractVector) = (; λ, φ)
spherical_points!(λ::AbstractVector, φ::AbstractVector, ::ScatteredSphericalSampling) = (; λ, φ)

end # module
