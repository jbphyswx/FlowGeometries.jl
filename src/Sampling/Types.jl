
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

"""
    AbstractRingSampling <: AbstractSphericalSampling

Iso-latitude rings whose longitude count VARIES by ring, so the layout is not a tensor product. A
reduced Gaussian grid is the canonical case: latitudes are the Gaussian ones, but each ring carries
only as many longitudes as its circumference warrants.
"""
abstract type AbstractRingSampling <: AbstractSphericalSampling end

"""
    AbstractReducedGaussianSampling <: AbstractRingSampling

Ring samplings on Gaussian latitudes. Default: [`OctahedralGaussianSampling`](@ref).
"""
abstract type AbstractReducedGaussianSampling <: AbstractRingSampling end

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

"""
    OctahedralGaussianSampling(N)

Octahedral reduced Gaussian grid with `N` latitude rings between pole and equator (`2N` rings in all).

Ring `i` counted from either pole carries `4i + 16` longitudes — 20 on the ring nearest the pole and
four more on each successive ring — which totals `4N(N+9)` points. Latitudes are the `2N` Gaussian
latitudes, so the latitude quadrature is the Gauss–Legendre one.
"""
struct OctahedralGaussianSampling <: AbstractReducedGaussianSampling
    nlat_half::Int
    function OctahedralGaussianSampling(nlat_half::Integer)
        nlat_half ≥ 1 || throw(ArgumentError("octahedral N must be ≥ 1, got $nlat_half"))
        return new(Int(nlat_half))
    end
end

"""
    ReducedGaussianSampling(nlon_per_ring)

Reduced Gaussian grid with an explicit longitude count per ring, north to south.

The classical reduced grids are published as tables — the count holds constant across blocks of
latitudes and jumps between them — so the table is the input. Use
[`OctahedralGaussianSampling`](@ref) for the octahedral rule, which is a formula.
"""
struct ReducedGaussianSampling{V<:AbstractVector{Int}} <: AbstractReducedGaussianSampling
    nlon_per_ring::V
    # Cumulative counts with a leading zero, so a ring's slice of the flattened point vector is two
    # reads. Built once here, the table being arbitrary; the octahedral rule has a closed form and
    # stores nothing.
    ring_offset::V
    function ReducedGaussianSampling(nlon::AbstractVector{<:Integer})
        isempty(nlon) && throw(ArgumentError("a reduced Gaussian grid needs at least one ring"))
        all(>(0), nlon) || throw(ArgumentError("every ring needs at least one longitude"))
        v = collect(Int, nlon)
        off = similar(v, length(v) + 1)
        off[1] = 0
        cumsum!(view(off, 2:length(off)), v)
        return new{typeof(v)}(v, off)
    end
end

"""
    FibonacciSampling(n)

`n` points on the spherical Fibonacci (golden-spiral) lattice: `z_k = (2k+1)/n - 1` with
`λ_k = 2πk/φ mod 2π`, `φ` the golden ratio.

`z` advances in equal steps, so the points are spread one per equal-area band, and the golden angle in
longitude — the most irrational rotation there is — keeps them from lining up into visible spokes at
any `n`. Quasi-uniform with no polar clustering and no panel seams.
"""
struct FibonacciSampling <: AbstractEqualAreaSphericalSampling
    n::Int
    function FibonacciSampling(n::Integer)
        n ≥ 1 || throw(ArgumentError("FibonacciSampling needs n ≥ 1, got $n"))
        return new(Int(n))
    end
end

# ---------------------------------------------------------------------------
# Traits
# ---------------------------------------------------------------------------

is_tensor_product(::AbstractSphericalSampling) = false
is_tensor_product(::AbstractTensorProductSphericalSampling) = true
is_tensor_product(::AbstractYinYangSampling) = false  # two structured panels, each its own patch

is_iso_latitude(::AbstractSphericalSampling) = false
is_iso_latitude(::AbstractTensorProductSphericalSampling) = true
is_iso_latitude(::AbstractHEALPixSampling) = true
is_iso_latitude(::AbstractRingSampling) = true

# The longitude count varies by ring, so the layout is not an (nlon x nlat) array.
is_tensor_product(::AbstractRingSampling) = false

# The latitudes are the Gaussian ones, so the latitude rule is exact exactly as Gauss-Legendre's is.
admits_exact_bandlimited_quadrature(::AbstractReducedGaussianSampling) = true

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

# Extra keywords are ignored so this can take the same keyword set as `spherical_axes!`, whose
# `lat_range`/`lon_range` do not change a count.
function axes_lengths(::AbstractLatLonSampling, nlat::Integer; nlon::Integer, _...)
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

"""
    nlon_per_ring(sampling) -> Vector{Int}
    nlon_per_ring(sampling, nlat; nlon=nothing) -> Vector{Int}

Longitudes on each iso-latitude ring, north to south.

Defined for **every** sampling laid out in rings, so a caller walking a map ring by ring — a per-ring
longitude transform, a zonal reduction, a row-wise sweep — writes one loop for all of them. A sampling
that carries its own size (`HEALPixSampling`, the reduced Gaussians) answers from itself; a
tensor-product one takes the `nlat` that fixes its shape, as [`npoints`](@ref) does.
"""
function nlon_per_ring(s::OctahedralGaussianSampling)
    N = s.nlat_half
    return [4 * min(j, 2N + 1 - j) + 16 for j in 1:(2N)]
end
nlon_per_ring(s::ReducedGaussianSampling) = copy(s.nlon_per_ring)

# HEALPix rings widen 4, 8, … through the polar cap, hold 4·nside across the belt and shrink back.
# The counts are the same ones `ring_info` reports, taken here without decoding a pixel.
function nlon_per_ring(s::HEALPixSampling)
    ns = s.nside
    return [_hp_get_ring_info_small(ns, r).ringpix for r in 1:(4 * ns - 1)]
end

# A tensor-product sampling is a rectangle: every ring is the same width.
nlon_per_ring(s::AbstractTensorProductSphericalSampling, nlat::Integer; kwargs...) =
    fill(axes_lengths(s, nlat; kwargs...).nlon, Int(nlat))

"""
    nlon_in_ring(sampling, ring) -> Int
    nlon_in_ring(sampling, nlat, ring) -> Int

Longitudes on ONE iso-latitude ring, counted from the north pole, in `O(1)` and allocating nothing.

The per-ring form of [`nlon_per_ring`](@ref), and the one a ring-by-ring loop wants: the table costs
an allocation and an `O(nrings)` build per call, which a loop over rings pays again on every
iteration if it asks for it there.
"""
function nlon_in_ring end

@inline _check_ring(r::Int, n::Int) =
    (1 ≤ r ≤ n || throw(ArgumentError("ring must lie in 1:$n, got $r")); r)

@inline function nlon_in_ring(s::OctahedralGaussianSampling, ring::Integer)
    N = s.nlat_half
    j = _check_ring(Int(ring), 2N)
    return 4 * min(j, 2N + 1 - j) + 16
end

@inline nlon_in_ring(s::ReducedGaussianSampling, ring::Integer) =
    @inbounds s.nlon_per_ring[_check_ring(Int(ring), length(s.nlon_per_ring))]

@inline function nlon_in_ring(s::HEALPixSampling, ring::Integer)
    ns = s.nside
    return _hp_get_ring_info_small(ns, _check_ring(Int(ring), 4 * ns - 1)).ringpix
end

@inline function nlon_in_ring(s::AbstractTensorProductSphericalSampling, nlat::Integer,
                              ring::Integer; kwargs...)
    _check_ring(Int(ring), Int(nlat))
    return axes_lengths(s, nlat; kwargs...).nlon
end

"""
    ring_range(sampling, ring) -> UnitRange{Int}
    ring_range(sampling, nlat, ring) -> UnitRange{Int}

The 1-based slice of the flattened point vector holding one iso-latitude ring, north to south — the
indices [`spherical_points`](@ref) writes that ring into.

`O(1)` for every sampling, so a ring-by-ring pass is a loop over slices carrying no running offset.
The octahedral rule's offset is a closed form in the ring index; the tabulated reduced grid carries
cumulative counts, built once with the sampling.
"""
function ring_range end

# Rings widen by four to the equator and mirror, so the count before ring `j` is a triangular number
# on either side of it.
@inline function _octahedral_before(N::Int, j::Int)
    k = j - 1
    k ≤ N && return 2 * k * (k + 1) + 16 * k
    m = 2N - k
    return 4 * N * (N + 9) - (2 * m * (m + 1) + 16 * m)
end

@inline function ring_range(s::OctahedralGaussianSampling, ring::Integer)
    N = s.nlat_half
    j = _check_ring(Int(ring), 2N)
    off = _octahedral_before(N, j)
    return (off + 1):(off + 4 * min(j, 2N + 1 - j) + 16)
end

@inline function ring_range(s::ReducedGaussianSampling, ring::Integer)
    r = _check_ring(Int(ring), length(s.nlon_per_ring))
    @inbounds return (s.ring_offset[r] + 1):s.ring_offset[r + 1]
end

@inline function ring_range(s::HEALPixSampling, ring::Integer)
    info = _hp_get_ring_info_small(s.nside, _check_ring(Int(ring), 4 * s.nside - 1))
    return (info.startpix + 1):(info.startpix + info.ringpix)
end

@inline function ring_range(s::AbstractTensorProductSphericalSampling, nlat::Integer,
                            ring::Integer; kwargs...)
    r = _check_ring(Int(ring), Int(nlat))
    m = axes_lengths(s, nlat; kwargs...).nlon
    return ((r - 1) * m + 1):(r * m)
end

"""
    nrings(sampling) -> Int
    nrings(sampling, nlat) -> Int

Number of iso-latitude rings, for any sampling laid out in them — the loop bound that goes with
[`nlon_per_ring`](@ref).
"""
nrings(s::OctahedralGaussianSampling) = 2 * s.nlat_half
nrings(s::ReducedGaussianSampling) = length(s.nlon_per_ring)
nrings(s::HEALPixSampling) = 4 * s.nside - 1
nrings(::AbstractTensorProductSphericalSampling, nlat::Integer) = Int(nlat)

npoints(s::OctahedralGaussianSampling) = 4 * s.nlat_half * (s.nlat_half + 9)
npoints(s::ReducedGaussianSampling) = @inbounds s.ring_offset[end]
npoints(s::FibonacciSampling) = s.n

bandlimit(::AbstractReducedGaussianSampling, nlat::Integer) = Int(nlat) - 1
nlat_for_bandlimit(::AbstractReducedGaussianSampling, lmax::Integer) = Int(lmax) + 1
npoints(s::IcosahedralSampling) = icosahedral_nvertices(s.frequency)
