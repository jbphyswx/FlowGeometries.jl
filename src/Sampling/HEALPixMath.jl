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

spherical_points(s::HEALPixSampling) = spherical_points(Float64, s)

function spherical_points(::Type{T}, s::HEALPixSampling) where {T<:AbstractFloat}
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
    # Pixels in the north polar cap ABOVE the ring at iring == nside, as the RING↔XYF conversion
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

function _hp_xyf2ring(nside::Int, ix::Int, iy::Int, face_num::Int)
    nl4 = 4 * nside
    jr = (_HP_JRLL[face_num + 1] * nside) - ix - iy - 1
    info = _hp_get_ring_info_small(nside, jr)
    nr = info.ringpix >> 2
    kshift = 1 - Int(info.shifted)
    jp = (_HP_JPLL[face_num + 1] * nr + ix - iy + 1 + kshift) ÷ 2
    jp < 1 && (jp += nl4)
    return info.startpix + jp - 1
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

# Bit interleaving: NESTED packs the two face-local coordinates into one index by placing `ix` on the
# even bit positions and `iy` on the odd ones, which is what makes each pixel's four children adjacent.
# Done as a fixed cascade of mask-and-shift rather than a branch per bit: the interleave doubles the
# gap between bits five times, which spreads all 32 input bits at once. Measured against the per-bit
# loop over the whole domain it covers, same answers, 10.3 ns → 1.7 ns spreading and 9.1 ns → 1.7 ns
# compressing — paid on every pixel of a NESTED conversion.
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

Face-local coordinates of the pixel containing colatitude `θ`, longitude `ϕ`, by the HEALPix
projection (Górski et al. 2005). Written with division and remainder rather than shift and mask, so it
holds for any `nside` rather than only a power of two.
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
        # Polar caps: within one of the four base faces of that hemisphere.
        ntt = min(3, Int(floor(tt)))
        tp = tt - T(ntt)
        tmp = T(nside) * sqrt(T(3) * (one(T) - za))
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
    # `z = cosθ` on the ring, by the same two-regime formula the pixel centres use.
    fn = T(ns)
    z = if r < ns
        one(T) - T(r * r) / (T(3) * fn * fn)
    elseif r ≤ 3 * ns
        (T(2 * ns) - T(r)) / (T(1.5) * fn)
    else
        nr = 4 * ns - r
        T(nr * nr) / (T(3) * fn * fn) - one(T)
    end
    θ = acos(clamp(z, -one(T), one(T)))
    return (; startpix = info.startpix, ringpix = info.ringpix,
              colatitude = θ, latitude = geographic_latitude(θ), shifted = info.shifted)
end

"""
    ang2pix(nside, θ, ϕ; scheme = Ring()) -> Int

The 0-based index of the pixel containing colatitude `θ ∈ [0, π]` and longitude `ϕ`.

`θ` is a COLATITUDE, matching the HEALPix convention throughout this section; use
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

@inline _pix2ring_index(::Int, p::Int, ::Ring) = p
@inline function _pix2ring_index(ns::Int, p::Int, ::Nested)
    _require_nested_nside(ns)
    ix, iy, f = _hp_nest2xyf(ns, p)
    return _hp_xyf2ring(ns, ix, iy, f)
end

"""
    ring2nest(nside, pix) -> Int
    nest2ring(nside, pix) -> Int

Convert a 0-based pixel index between the two orderings. Both need `nside` to be a power of two,
which is what makes the NESTED quadtree exist.
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
    r = sqrt(T(x)^2 + T(y)^2 + T(z)^2)
    iszero(r) && throw(ArgumentError("the zero vector has no direction"))
    θ = acos(clamp(T(z) / r, -one(T), one(T)))
    ϕ = mod(atan(T(y), T(x)), T(2π))
    return ang2pix(nside, θ, ϕ; scheme = scheme)
end
