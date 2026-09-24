# ---- HEALPix ----------------------------------------------------------------

healpix_npix(nside::Integer) = 12 * Int(nside)^2
healpix_npix(s::HEALPixSampling) = healpix_npix(s.nside)
healpix_nring(nside::Integer) = 4 * Int(nside) - 1
healpix_nring(s::HEALPixSampling) = healpix_nring(s.nside)
healpix_pixel_area(nside::Integer) = 4π / healpix_npix(nside)
healpix_pixel_area(s::HEALPixSampling) = healpix_pixel_area(s.nside)

# A ring walk: latitude is constant along a ring, so the ring's trigonometry runs `4·nside − 1` times
# over the `12·nside²` pixels, and a pixel's longitude is `(j − shift)·Δϕ` with `Δϕ` the ring's.
# `ringpix` is `4·nr` in every regime, so one expression for `Δϕ` reproduces all three exactly.
function spherical_points!(λ::AbstractVector{T}, φ::AbstractVector{T}, s::HEALPixSampling) where {T<:AbstractFloat}
    npix = healpix_npix(s)
    length(λ) == npix && length(φ) == npix || throw(DimensionMismatch("buffers must have length healpix_npix"))
    ns = s.nside
    @inbounds for r in 1:healpix_nring(ns)
        info = ring_info(T, ns, r)
        m = info.ringpix
        Δϕ = T(π) / (T(2) * T(m ÷ 4))
        shift = info.shifted ? T(0.5) : one(T)
        for j in 1:m
            k = info.startpix + j
            λ[k] = mod((T(j) - shift) * Δϕ, T(2π))
            φ[k] = info.latitude
        end
    end
    return (; λ, φ)
end

spherical_points(s::HEALPixSampling) = spherical_points(Float64, s)

function spherical_points(::Type{T}, s::HEALPixSampling) where {T<:AbstractFloat}
    n = npoints(s)
    return spherical_points!(Vector{T}(undef, n), Vector{T}(undef, n), s)
end

"""
    _healpix_nested_cloud!(λ, φ, nside) -> (; λ, φ)

The whole HEALPix cloud in NESTED pixel order.

The ring quantities are tabulated once, `4·nside − 1` of them, so the ring's trigonometry runs per ring
here too. A nested pixel's ring and position along it come from its face coordinates
(`_hp_xyf2ringj`), and the coordinates are then the ring walk's own expressions, so the two
orderings hold the same numbers to the bit. Both writes are sequential.
"""
function _healpix_nested_cloud!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, nside::Integer,
) where {T<:AbstractFloat}
    ns = Int(nside)
    _require_nested_nside(ns)
    npix = healpix_npix(ns)
    length(λ) == npix && length(φ) == npix ||
        throw(DimensionMismatch("buffers must have length healpix_npix"))
    nring = healpix_nring(ns)
    Δϕ = Vector{T}(undef, nring)
    shift = Vector{T}(undef, nring)
    lat = Vector{T}(undef, nring)
    @inbounds for r in 1:nring
        info = ring_info(T, ns, r)
        Δϕ[r] = T(π) / (T(2) * T(info.ringpix ÷ 4))
        shift[r] = info.shifted ? T(0.5) : one(T)
        lat[r] = info.latitude
    end
    @inbounds for p in 0:(npix - 1)
        ix, iy, f = _hp_nest2xyf(ns, p)
        jr, jp, _ = _hp_xyf2ringj(ns, ix, iy, f)
        λ[p + 1] = mod((T(jp) - shift[jr]) * Δϕ[jr], T(2π))
        φ[p + 1] = lat[jr]
    end
    return (; λ, φ)
end

"""
    _hp_ring_angles(T, nside, ring) -> (θ, φ)

Colatitude and latitude of HEALPix ring `ring`, counted from the north pole, each formed directly. In a
polar cap `cos θ = ±(1 − δ)` with `δ = r²/(3·nside²)`, `r` the ring's count from its own pole, and the
angle from that pole is `2asin(r/(√6·nside))`, accurate to the last bit however small `r/nside` is. In
the belt `|cos θ| ≤ 2/3`, where `acos` and `asin` are well conditioned.
"""
@inline function _hp_ring_angles(::Type{T}, nside::Int, ring::Int) where {T<:AbstractFloat}
    fn = T(nside)
    if nside ≤ ring ≤ 3 * nside
        z = (T(2 * nside) - T(ring)) / (T(1.5) * fn)
        return acos(z), asin(z)
    end
    north = ring < nside
    r = north ? ring : 4 * nside - ring
    a = 2 * asin(T(r) / (sqrt(T(6)) * fn))
    return north ? (a, T(π) / 2 - a) : (T(π) - a, a - T(π) / 2)
end

"""
    _healpix_ring_phi(nside, ipix, T) -> (ring, ϕ)

The ring (counted from the north pole) and longitude of 0-based RING pixel `ipix`.
"""
function _healpix_ring_phi(nside::Int, ipix::Int, ::Type{T}) where {T<:AbstractFloat}
    fn = T(nside)
    nl4 = 4 * nside
    npix = 12 * nside * nside
    # Pixels in the north polar cap, i.e. rings 1 … nside-1, which hold 4, 8, … 4(nside-1) pixels:
    # 2·nside·(nside-1). Getting this wrong routes equatorial pixels through the cap branch.
    ncap = 2 * nside * (nside - 1)
    if ipix < ncap
        hip = (ipix + 1) / T(2)
        fihip = floor(hip)
        iring = Int(floor(sqrt(hip - sqrt(fihip))) + 1)
        iphi = ipix + 1 - 2 * iring * (iring - 1)
        ring = iring
        ϕ = (T(iphi) - T(0.5)) * T(π) / (T(2) * T(iring))
    elseif ipix < (npix - ncap)
        # Every equatorial ring holds 4·nside pixels, so the ring index advances per nl4. `fodd`
        # staggers alternate rings by half a pixel: 1 when (iring+nside) is odd, 1/2 when even.
        ip = ipix - ncap
        tmp = ip ÷ nl4
        ring = tmp + nside
        iphi = ip - tmp * nl4 + 1
        fodd = isodd(ring + nside) ? one(T) : T(0.5)
        ϕ = (T(iphi) - fodd) * T(π) / (T(2) * fn)
    else
        ip = npix - ipix
        hip = ip / T(2)
        fihip = floor(hip)
        iring = Int(floor(sqrt(hip - sqrt(fihip))) + 1)
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        ring = nl4 - iring
        ϕ = (T(iphi) - T(0.5)) * T(π) / (T(2) * T(iring))
    end
    return ring, mod(ϕ, T(2π))
end

function _healpix_pix2ang_ring(nside::Int, ipix::Int, ::Type{T}) where {T<:AbstractFloat}
    ring, ϕ = _healpix_ring_phi(nside, ipix, T)
    return _hp_ring_angles(T, nside, ring)[1], ϕ
end


# ---------------------------------------------------------------------------
# HEALPix pixel geometry: RING <-> face-local (ix, iy, face)
# ---------------------------------------------------------------------------
#
# Follows Górski et al. (2005) and Reinecke (2003). The face-local form is the hinge both orderings and
# the neighbour walk go through.

const _HP_JRLL = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
const _HP_JPLL = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)
@inline _hp_special_div(a::Int, b::Int) = (t = Int(a ≥ (b << 1)); a2 = a - t * (b << 1); (t << 1) + Int(a2 ≥ b))

@inline function _hp_ncap(nside::Int)
    # Pixels in the north polar cap above the ring at iring == nside, as the RING↔XYF conversion
    # needs it. Distinct from the classic 2 nside (nside+1) cap count used for pixel centers.
    return 2 * nside * (nside - 1)
end

@inline function _hp_get_ring_info_small(nside::Int, ring::Int)
    npix = 12 * nside * nside
    ncap = _hp_ncap(nside)
    if ring < nside
        return (startpix = 2 * ring * (ring - 1), ringpix = 4 * ring, shifted = true)
    elseif ring < 3 * nside
        ringpix = 4 * nside
        return (startpix = ncap + (ring - nside) * ringpix, ringpix = ringpix, shifted = ((ring - nside) & 1) == 0)
    else
        nr = 4 * nside - ring
        return (startpix = npix - 2 * nr * (nr + 1), ringpix = 4 * nr, shifted = true)
    end
end

function _hp_ring2xyf(nside::Int, pix::Int)
    # pix 0-based RING
    ncap = _hp_ncap(nside)
    npix = 12 * nside * nside
    nl2 = 2 * nside
    iring = 0
    iphi = 0
    kshift = 0
    nr = 0
    face_num = 0
    if pix < ncap
        iring = (1 + isqrt(1 + 2 * pix)) >> 1
        iphi = (pix + 1) - 2 * iring * (iring - 1)
        kshift = 0
        nr = iring
        face_num = _hp_special_div(iphi - 1, nr)
    elseif pix < (npix - ncap)
        ip = pix - ncap
        tmp = ip ÷ (4 * nside)
        iring = tmp + nside
        iphi = ip - tmp * 4 * nside + 1
        kshift = (iring + nside) & 1
        nr = nside
        ire = tmp + 1
        irm = nl2 + 1 - tmp
        ifm = (iphi - (ire >> 1) + nside - 1) ÷ nside
        ifp = (iphi - (irm >> 1) + nside - 1) ÷ nside
        face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8))
    else
        ip = npix - pix
        iring = (1 + isqrt(2 * ip - 1)) >> 1
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        kshift = 0
        nr = iring
        iring = 2 * nl2 - iring
        face_num = _hp_special_div(iphi - 1, nr) + 8
    end
    irt = iring - ((2 + (face_num >> 2)) * nside) + 1
    ipt = 2 * iphi - _HP_JPLL[face_num + 1] * nr - kshift - 1
    ipt ≥ nl2 && (ipt -= 8 * nside)
    ix = (ipt - irt) >> 1
    iy = (-ipt - irt) >> 1
    return ix, iy, face_num
end

# A pixel's ring `jr` and its 1-based position `jp` along that ring, from its face coordinates. The ring
# index is what the ring walk indexes and `jp` the `j` it writes at, so a caller holding a per-ring table
# reaches a pixel's coordinates from these two without forming its ring index at all.
@inline function _hp_xyf2ringj(nside::Int, ix::Int, iy::Int, face_num::Int)
    nl4 = 4 * nside
    jr = (_HP_JRLL[face_num + 1] * nside) - ix - iy - 1
    info = _hp_get_ring_info_small(nside, jr)
    nr = info.ringpix >> 2
    kshift = 1 - Int(info.shifted)
    jp = (_HP_JPLL[face_num + 1] * nr + ix - iy + 1 + kshift) ÷ 2
    jp < 1 && (jp += nl4)
    return jr, jp, info.startpix
end

function _hp_xyf2ring(nside::Int, ix::Int, iy::Int, face_num::Int)
    _jr, jp, startpix = _hp_xyf2ringj(nside, ix, iy, face_num)
    return startpix + jp - 1
end


"""
    RingScheme

Which HEALPix pixel ordering is meant: [`Ring`](@ref) or [`Nested`](@ref).
"""
abstract type RingScheme end

"""
    Ring()

Pixels numbered along iso-latitude rings, north to south and east within a ring. The ordering that
makes a ring contiguous, so a longitude transform per ring is possible.
"""
struct Ring <: RingScheme end

"""
    Nested()

Pixels numbered so that each is subdivided into four contiguous children — a quadtree per base face.
The ordering that makes a neighbourhood contiguous. Requires `nside` to be a power of two.
"""
struct Nested <: RingScheme end

# Bit interleaving: the nested scheme packs the two face-local coordinates into one index by placing
# `ix` on the even bit positions and `iy` on the odd ones, which puts a pixel's four children at
# consecutive indices. The cascade below doubles the gap between bits five times, spreading all 32
# input bits with no branch and no loop.
@inline function _spread_bits(v::Int)
    x = UInt64(v) & 0x00000000ffffffff
    x = (x | (x << 16)) & 0x0000ffff0000ffff
    x = (x | (x <<  8)) & 0x00ff00ff00ff00ff
    x = (x | (x <<  4)) & 0x0f0f0f0f0f0f0f0f
    x = (x | (x <<  2)) & 0x3333333333333333
    x = (x | (x <<  1)) & 0x5555555555555555
    return Int(x)
end

@inline function _compress_bits(v::Int)
    x = UInt64(v) & 0x5555555555555555
    x = (x | (x >>  1)) & 0x3333333333333333
    x = (x | (x >>  2)) & 0x0f0f0f0f0f0f0f0f
    x = (x | (x >>  4)) & 0x00ff00ff00ff00ff
    x = (x | (x >>  8)) & 0x0000ffff0000ffff
    x = (x | (x >> 16)) & 0x00000000ffffffff
    return Int(x)
end

@inline _is_power_of_two(n::Int) = n > 0 && (n & (n - 1)) == 0

_require_nested_nside(nside::Int) = _is_power_of_two(nside) || throw(ArgumentError(
    "the NESTED scheme needs nside to be a power of two, got $nside",
))

@inline function _hp_xyf2nest(nside::Int, ix::Int, iy::Int, face::Int)
    return face * nside * nside + _spread_bits(ix) + 2 * _spread_bits(iy)
end

@inline function _hp_nest2xyf(nside::Int, pix::Int)
    npface = nside * nside
    face, p = divrem(pix, npface)
    return _compress_bits(p), _compress_bits(p >> 1), face
end

"""
    _hp_ang2xyf(nside, θ, ϕ) -> (ix, iy, face)

Face-local coordinates of the pixel containing colatitude `θ`, longitude `ϕ`, by the HEALPix projection
(Górski et al. 2005). Division and remainder, so it holds for any `nside`, including one that is not a
power of two.
"""
function _hp_ang2xyf(nside::Int, θ::T, ϕ::T) where {T<:AbstractFloat}
    z = cos(θ)
    za = abs(z)
    tt = mod(ϕ / (T(π) / 2), T(4))
    if za ≤ T(2) / 3
        # Equatorial belt: the pixel sits at the crossing of an ascending and a descending edge line.
        temp1 = T(nside) * (T(0.5) + tt)
        temp2 = T(nside) * z * T(0.75)
        jp = Int(floor(temp1 - temp2))
        jm = Int(floor(temp1 + temp2))
        ifp = jp ÷ nside
        ifm = jm ÷ nside
        face = ifp == ifm ? (ifp | 4) : (ifp < ifm ? ifp : ifm + 8)
        return (mod(jm, nside), nside - mod(jp, nside) - 1, face)
    else
        # Polar caps: within one of the four base faces of that hemisphere. `3(1 − |z|)` is
        # `6sin²(θ/2)` in the north and `6cos²(θ/2)` in the south, both accurate at the pole.
        ntt = min(3, Int(floor(tt)))
        tp = tt - T(ntt)
        tmp = T(nside) * sqrt(T(6)) * (z ≥ 0 ? sin(θ / 2) : cos(θ / 2))
        jp = min(Int(floor(tp * tmp)), nside - 1)
        jm = min(Int(floor((one(T) - tp) * tmp)), nside - 1)
        return z ≥ 0 ? (nside - jm - 1, nside - jp - 1, ntt) : (jp, jm, ntt + 8)
    end
end

"""
    ring_info([T = Float64], nside, ring) -> NamedTuple

What HEALPix ring `ring ∈ 1:(4·nside-1)` contains, counted from the north pole: `startpix` (the 0-based
RING index of its first pixel, matching [`ang2pix`](@ref)), `ringpix` (how many pixels it holds),
`colatitude`, `latitude`, and `shifted` — whether its pixel centres are offset half a pixel in `ϕ`.

Ring width grows `4, 8, …` through the polar cap, is `4·nside` across the equatorial belt, and shrinks
again symmetrically, so this is how to walk a HEALPix map ring by ring without decoding every pixel.
"""
ring_info(nside::Integer, ring::Integer) = ring_info(Float64, nside, ring)

function ring_info(::Type{T}, nside::Integer, ring::Integer) where {T<:AbstractFloat}
    ns = Int(nside)
    ns ≥ 1 || throw(ArgumentError("HEALPix nside must be ≥ 1, got $ns"))
    r = Int(ring)
    1 ≤ r ≤ 4 * ns - 1 || throw(ArgumentError(
        "ring must lie in 1:$(4 * ns - 1) for nside = $ns, got $r",
    ))
    info = _hp_get_ring_info_small(ns, r)
    θ, φ = _hp_ring_angles(T, ns, r)
    return (; startpix = info.startpix, ringpix = info.ringpix,
              colatitude = θ, latitude = φ, shifted = info.shifted)
end

"""
    ang2pix(nside, θ, ϕ; scheme = Ring()) -> Int

The 0-based index of the pixel containing colatitude `θ ∈ [0, π]` and longitude `ϕ`.

`θ` is a *colatitude*, matching the HEALPix convention throughout this section; use
[`colatitude`](@ref) to convert a geographic latitude.
"""
function ang2pix(nside::Integer, θ::Real, ϕ::Real; scheme::RingScheme = Ring())
    ns = Int(nside)
    ns ≥ 1 || throw(ArgumentError("HEALPix nside must be ≥ 1, got $ns"))
    T = float(promote_type(typeof(θ), typeof(ϕ)))
    ix, iy, f = _hp_ang2xyf(ns, T(θ), T(ϕ))
    return _xyf2pix(ns, ix, iy, f, scheme)
end

@inline _xyf2pix(ns::Int, ix::Int, iy::Int, f::Int, ::Ring) = _hp_xyf2ring(ns, ix, iy, f)
@inline function _xyf2pix(ns::Int, ix::Int, iy::Int, f::Int, ::Nested)
    _require_nested_nside(ns)
    return _hp_xyf2nest(ns, ix, iy, f)
end

"""
    pix2ang([T = Float64], nside, pix; scheme = Ring()) -> (θ, ϕ)

Colatitude and longitude of pixel `pix`'s centre (0-based index).
"""
pix2ang(nside::Integer, pix::Integer; kwargs...) = pix2ang(Float64, nside, pix; kwargs...)

pix2ang(::Type{T}, nside::Integer, pix::Integer; scheme::RingScheme = Ring()) where {T<:AbstractFloat} =
    _pix2ang(Int(nside), Int(pix), scheme, T)

@inline function _pix2ang(ns::Int, p::Int, scheme::RingScheme, ::Type{T}) where {T<:AbstractFloat}
    npix = healpix_npix(ns)
    0 ≤ p < npix || throw(ArgumentError("HEALPix pixel $p out of range 0:$(npix - 1)"))
    ring = _pix2ring_index(ns, p, scheme)
    return _healpix_pix2ang_ring(ns, ring, T)
end

"""
    _pix2lonlat(T, nside, pix, scheme) -> (λ, φ)

Pixel `pix`'s centre as longitude and geographic latitude, the latitude taken straight from the ring
([`_hp_ring_angles`](@ref)) so it matches [`ring_info`](@ref)'s to the bit and carries no `π/2 − θ`
rounding near a pole.
"""
@inline function _pix2lonlat(::Type{T}, ns::Int, p::Int, scheme::RingScheme) where {T<:AbstractFloat}
    npix = healpix_npix(ns)
    0 ≤ p < npix || throw(ArgumentError("HEALPix pixel $p out of range 0:$(npix - 1)"))
    ring, ϕ = _healpix_ring_phi(ns, _pix2ring_index(ns, p, scheme), T)
    return ϕ, _hp_ring_angles(T, ns, ring)[2]
end

@inline _pix2ring_index(::Int, p::Int, ::Ring) = p
@inline function _pix2ring_index(ns::Int, p::Int, ::Nested)
    _require_nested_nside(ns)
    ix, iy, f = _hp_nest2xyf(ns, p)
    return _hp_xyf2ring(ns, ix, iy, f)
end

"""
    ring2nest(nside, pix) -> Int
    nest2ring(nside, pix) -> Int

Convert a 0-based pixel index between the two orderings. Both need `nside` to be a power of two, which
is the condition for the nested quadtree to exist.
"""
function ring2nest(nside::Integer, pix::Integer)
    ns = Int(nside)
    _require_nested_nside(ns)
    ix, iy, f = _hp_ring2xyf(ns, Int(pix))
    return _hp_xyf2nest(ns, ix, iy, f)
end

function nest2ring(nside::Integer, pix::Integer)
    ns = Int(nside)
    _require_nested_nside(ns)
    ix, iy, f = _hp_nest2xyf(ns, Int(pix))
    return _hp_xyf2ring(ns, ix, iy, f)
end

"""
    pix2vec([T = Float64], nside, pix; scheme = Ring()) -> NTuple{3}

Unit vector to pixel `pix`'s centre.
"""
pix2vec(nside::Integer, pix::Integer; kwargs...) = pix2vec(Float64, nside, pix; kwargs...)

function pix2vec(::Type{T}, nside::Integer, pix::Integer;
                 scheme::RingScheme = Ring()) where {T<:AbstractFloat}
    θ, ϕ = _pix2ang(Int(nside), Int(pix), scheme, T)
    sinθ, cosθ = sincos(θ)
    sinϕ, cosϕ = sincos(ϕ)
    return (sinθ * cosϕ, sinθ * sinϕ, cosθ)
end

"""
    vec2pix(nside, v; scheme = Ring()) -> Int

The 0-based index of the pixel containing direction `v`, which need not be normalized.
"""
function vec2pix(nside::Integer, v; scheme::RingScheme = Ring())
    x, y, z = v[1], v[2], v[3]
    T = float(promote_type(typeof(x), typeof(y), typeof(z)))
    ρ = hypot(T(x), T(y))
    iszero(ρ) && iszero(z) && throw(ArgumentError("the zero vector has no direction"))
    θ = atan(ρ, T(z))              # accurate relative to θ at the poles
    ϕ = mod(atan(T(y), T(x)), T(2π))
    return ang2pix(nside, θ, ϕ; scheme = scheme)
end
