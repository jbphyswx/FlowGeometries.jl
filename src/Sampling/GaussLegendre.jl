# ---------------------------------------------------------------------------
# Gauss–Legendre nodes and weights
# ---------------------------------------------------------------------------

"""
    _gauss_legendre_μ!(μ, w) -> NamedTuple{(:μ,:w)}

Write `n = length(μ)`-point Gauss–Legendre nodes/weights on ``μ ∈ (-1, 1)`` into the provided
buffers, ascending in `μ` (`length(μ) == length(w)`). At `Float64` width above `n = 60` they come from
the asymptotic expansions, `O(1)` per node; otherwise from Newton on ``Pₙ``, `O(n)` per root. See
[`_gauss_legendre!`](@ref).
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
# `(n+½)⁻²`. Nothing is iterated and no root depends on its neighbours, so this is `O(1)` per node and
# carries no error along the sequence.

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
    lat::Union{Nothing,AbstractVector}, n::Int, m::Int,
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
        φt = T(π / 2 - node)                  # `node` is the colatitude
        # Index 1 is the root nearest +1, so it lands at the top of the ascending output.
        _put!(μ, n + 1 - i, xt); _put!(w, n + 1 - i, wi); _put!(lat, n + 1 - i, φt)
        _put!(μ, i, -xt);        _put!(w, i, wi);        _put!(lat, i, -φt)
    end
    isodd(n) && (_put!(μ, m, zero(T)); _put!(lat, m, zero(T)))
    return nothing
end

"""
    _gauss_legendre!(T, μ, w, n; lat = nothing)

Core solve, ascending in `μ`. Any output may be `nothing`, so a caller that wants only some of the
nodes, the weights and the latitudes needs no scratch vector for the rest.

`lat` receives the latitudes `asin(μ)`, formed as `π/2 − θ` from the colatitude `θ` the solve works in.
Near a pole `μ` rounds next to `±1`, where `asin` amplifies that rounding by `1/sin θ`; the colatitude
carries no such loss.
"""
function _gauss_legendre!(
    ::Type{T}, μ::Union{Nothing,AbstractVector}, w::Union{Nothing,AbstractVector}, n::Int;
    lat::Union{Nothing,AbstractVector} = nothing,
) where {T<:AbstractFloat}
    n ≥ 1 || throw(ArgumentError("need n ≥ 1"))
    μ === nothing || length(μ) == n || throw(DimensionMismatch("μ must have length n"))
    w === nothing || length(w) == n || throw(DimensionMismatch("w must have length n"))
    lat === nothing || length(lat) == n || throw(DimensionMismatch("lat must have length n"))
    if n == 1
        _put!(μ, 1, zero(T))
        _put!(w, 1, T(2))
        _put!(lat, 1, zero(T))
        return nothing
    end
    m = (n + 1) ÷ 2                       # roots are symmetric about 0; solve the upper half only
    # Work at no less than Float64 and round once: the series below sums ~30 terms, so accumulating it
    # at Float32 leaves the root short of Float32's own precision.
    TW = promote_type(T, Float64)
    # The asymptotic expansions are Float64 coefficient sets truncated at a fixed order, accurate to
    # Float64 above n = 60. Wider types and smaller n take Newton, which converges to eps(TW).
    if TW === Float64 && n > 60
        _gauss_legendre_asy!(T, μ, w, lat, n, m)
    else
        _gauss_legendre_newton!(T, TW, μ, w, lat, n, m)
    end
    return nothing
end

"""
    _gauss_legendre_newton!(T, TW, μ, w, lat, n, m)

Nodes and weights by Newton on `Pₙ`, from a Tricomi start, each root iterated in the angle that
resolves it to a relative `eps(TW)`: a root within `π/4` of the pole in its colatitude `θ`, any other in
its latitude `φ`. See [`_gl_newton_colatitude`](@ref) and [`_gl_newton_latitude`](@ref).

`O(n)` per root, `O(n²)` overall, converging to `eps(TW)`: the method for element types wider than
`Float64` at any `n`, and for `n ≤ 60`.
"""
function _gauss_legendre_newton!(
    ::Type{T}, ::Type{TW}, μ::Union{Nothing,AbstractVector}, w::Union{Nothing,AbstractVector},
    lat::Union{Nothing,AbstractVector}, n::Int, m::Int,
) where {T<:AbstractFloat, TW<:AbstractFloat}
    tol = 4 * eps(TW)
    Tn = TW(n)
    @inbounds for i in 1:m
        θ0 = TW(π) * (4 * TW(i) - 1) / (4 * Tn + 2)
        θ = acos((1 - (Tn - 1) / (8 * Tn^3)) * cos(θ0))
        if θ < TW(π) / 4
            θ, dp = _gl_newton_colatitude(θ, n, tol)
            x, φ = cos(θ), TW(π) / 2 - θ
        else
            φ, dp = _gl_newton_latitude(TW(π) / 2 - θ, n, tol)
            x = sin(φ)
        end
        # wᵢ = 2 / ((1 − x²) P′ₙ(x)²), which is 2 over the squared derivative in either angle
        wi = T(TW(2) / (dp * dp))
        xt = T(x)
        φt = T(φ)
        _put!(μ, i, -xt);        _put!(w, i, wi);        _put!(lat, i, -φt)
        _put!(μ, n + 1 - i, xt); _put!(w, n + 1 - i, wi); _put!(lat, n + 1 - i, φt)
    end
    # For odd n the central root is exactly zero; the ± assignment above leaves it at round-off.
    isodd(n) && (_put!(μ, m, zero(T)); _put!(lat, m, zero(T)))
    return nothing
end

"""
    _gl_newton_colatitude(θ, n, tol) -> (θ, dPₙ/dθ)

Newton on `Pₙ(cos θ)` in `θ`. `Pₙ` is evaluated in `d = 1 − cos θ = 2sin²(θ/2)`, which holds a small
colatitude to full precision where `cos θ` rounds next to 1, by the Bonnet recurrence written for the
differences `qₖ = Pₖ − Pₖ₋₁`:

    k qₖ = (k − 1) qₖ₋₁ − (2k − 1) d Pₖ₋₁
"""
@inline function _gl_newton_colatitude(θ::TW, n::Int, tol::TW) where {TW<:AbstractFloat}
    Tn = TW(n)
    dp = zero(TW)
    for _ in 1:100
        sh, ch = sincos(θ / 2)
        d = 2 * sh * sh
        p = one(TW); q = zero(TW)                      # P₀ and q₀; they end as Pₙ and qₙ
        for k in 1:n
            q = ((TW(k) - 1) * q - (2 * TW(k) - 1) * d * p) / TW(k)
            p += q
        end
        # dPₙ(cos θ)/dθ = −sin θ P′ₙ(x) = n (x Pₙ − Pₙ₋₁) / sin θ, and x Pₙ − Pₙ₋₁ = qₙ − d Pₙ
        dp = Tn * (q - d * p) / (2 * sh * ch)
        δ = p / dp
        θ -= δ
        abs(δ) ≤ tol * θ && break
    end
    return θ, dp
end

"""
    _gl_newton_latitude(φ, n, tol) -> (φ, dPₙ/dφ)

Newton on `Pₙ(sin φ)` in `φ`, `Pₙ` by the Bonnet recurrence in `x = sin φ`, which holds a latitude to
full precision away from the poles.
"""
@inline function _gl_newton_latitude(φ::TW, n::Int, tol::TW) where {TW<:AbstractFloat}
    Tn = TW(n)
    dp = zero(TW)
    for _ in 1:100
        x, c = sincos(φ)
        p1 = one(TW); p2 = zero(TW)                    # they end as Pₙ(x) and Pₙ₋₁(x)
        for k in 1:n
            p3 = p2; p2 = p1
            p1 = ((2 * TW(k) - 1) * x * p2 - (TW(k) - 1) * p3) / TW(k)
        end
        # dPₙ(sin φ)/dφ = cos φ P′ₙ(x) = −n (x Pₙ − Pₙ₋₁) / cos φ
        dp = -Tn * (x * p1 - p2) / c
        δ = p1 / dp
        φ -= δ
        # Relative to the root, floored at the root spacing's scale `1/n`: odd `n` has a root at 0.
        abs(δ) ≤ tol * max(abs(φ), inv(Tn)) && break
    end
    return φ, dp
end

_gauss_legendre_μ(n::Integer) = _gauss_legendre_μ(Float64, n)
function _gauss_legendre_μ(::Type{T}, n::Integer) where {T<:AbstractFloat}
    μ = Vector{T}(undef, Int(n))
    w = Vector{T}(undef, Int(n))
    _gauss_legendre_μ!(μ, w)
    return (; μ, w)
end
