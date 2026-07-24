module SphericalSampling

using LinearAlgebra: LinearAlgebra as LA

# Public API via `FlowGeometries.SphericalSampling.*` or parent rebinds. No exports.

"""
    AbstractSphericalSampling

How points are placed on the sphere. Orthogonal to [`Geometry.AbstractSphericalGeometry`](@ref)
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
Exact for band-limit ``l_{\\max} = N_θ − 1`` (SHTOOLS GLQ; SHTns).
"""
struct GaussLegendreSampling <: AbstractGaussLegendreSampling end

"""
    DriscollHealySampling <: AbstractDriscollHealySampling

Driscoll–Healy equiangular grid in the SHTOOLS **DH2** layout:
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

Open equiangular Clenshaw–Curtis / FastTransforms grid (FastSphericalHarmonics `sph_points`):
``θ_i = π(i−1/2)/N_θ`` (no poles), ``N_λ = 2N_θ − 1``, ``l_{\\max} = N_θ − 1``.

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

admits_exact_bandlimited_quadrature(::AbstractSphericalSampling) = false
admits_exact_bandlimited_quadrature(::AbstractSpectralQuadratureSampling) = true

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

Golub–Welsch: write `n = length(μ)`-point Gauss–Legendre nodes/weights on
``μ ∈ (-1, 1)`` into the provided buffers (`length(μ) == length(w)`).

The bang only guarantees that the *result* lands in `μ`/`w`. The Jacobi eigen-
decomposition still needs ``O(n²)`` scratch for eigenvectors (LAPACK); that is
inherent to this algorithm, not a missing in-place path.
"""
function _gauss_legendre_μ!(μ::AbstractVector{T}, w::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(μ)
    length(w) == n || throw(DimensionMismatch("μ and w must have the same length"))
    n ≥ 1 || throw(ArgumentError("need n ≥ 1"))
    if n == 1
        μ[1] = zero(T)
        w[1] = T(2)
        return (; μ, w)
    end
    # Subdiagonal βᵢ = i / √(4i²−1). Diagonal is zero (reuse `μ` as SymTridiagonal.dv).
    β = Vector{T}(undef, n - 1)
    @inbounds for i in 1:(n - 1)
        β[i] = T(i) / sqrt(T(4 * i * i - 1))
    end
    fill!(μ, zero(T))
    # LAPACK stegr returns eigenvalues ascending — no post-sort / permute needed.
    F = LA.eigen!(LA.SymTridiagonal(μ, β))
    @inbounds for i in 1:n
        μ[i] = F.values[i]
        w[i] = T(2) * abs2(F.vectors[1, i])
    end
    return (; μ, w)
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
    μ = similar(φ)
    wscratch = similar(φ)
    _gauss_legendre_μ!(μ, wscratch)
    @inbounds for i in 1:nlat
        φ[i] = asin(μ[i])
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
    iseven(nlat) || throw(ArgumentError("DH nlat must be even (SHTOOLS N = 2(L+1))"))
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
    μ = similar(w)
    _gauss_legendre_μ!(μ, w)
    return w
end

function latitude_weights!(w::AbstractVector{T}, ::AbstractDriscollHealySampling, nlat::Integer) where {T<:AbstractFloat}
    nlat = Int(nlat)
    iseven(nlat) || throw(ArgumentError("DH nlat must be even"))
    length(w) == nlat || throw(DimensionMismatch("w length must equal nlat"))
    L = nlat ÷ 2
    @inbounds for t in 0:(nlat - 1)
        θ = T(π) * T(t) / T(nlat)
        s = zero(T)
        for k in 0:(L - 1)
            s += sin(T(2k + 1) * θ) / T(2k + 1)
        end
        w[t + 1] = (T(2π) / T(L)^2) * sin(θ) * s
    end
    return w
end

latitude_weights!(::AbstractVector, ::AbstractClenshawCurtisSampling, ::Integer) =
    throw(ArgumentError("Clenshaw–Curtis sphere weights come from the FastTransforms plan, not a closed q(θ)"))
latitude_weights!(::AbstractVector, ::AbstractMcEwenWiauxSampling, ::Integer) =
    throw(ArgumentError("MW weights come from the SSHT / MW quadrature construction"))
latitude_weights!(::AbstractVector, ::AbstractLatLonSampling, ::Integer) =
    throw(ArgumentError("LatLonSampling has no spectral quadrature weights"))

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
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside + 1)
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
        ip = ipix - ncap
        iring = ip ÷ nl2 + nside
        iphi = (ip % nl2) + 1
        fodd = T(0.5) * ((iring + nside) % 2)
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

function cubed_sphere_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, panel::AbstractVector{<:Integer}, n::Integer,
) where {T<:AbstractFloat}
    n = Int(n)
    N = 6 * n * n
    length(λ) == N && length(φ) == N && length(panel) == N || throw(DimensionMismatch("buffers must have length 6n²"))
    a = range(-T(π) / 4, T(π) / 4; length = n)
    k = 1
    @inbounds for f in 1:6, j in 1:n, i in 1:n
        ξ, η = a[i], a[j]
        X, Y = tan(ξ), tan(η)
        p = _cubed_face_to_xyz(f, X, Y, T)
        r = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
        x = p.x / r; y = p.y / r; z = p.z / r
        θ = acos(clamp(z, -one(T), one(T)))
        ϕ = atan(y, x)
        ϕ < 0 && (ϕ += T(2π))
        λ[k] = ϕ
        φ[k] = geographic_latitude(θ)
        panel[k] = f
        k += 1
    end
    return (; λ, φ, panel)
end

function cubed_sphere_points(n::Integer; T::Type{<:AbstractFloat} = Float64)
    N = npoints(CubedSphereSampling(), n)
    return cubed_sphere_points!(Vector{T}(undef, N), Vector{T}(undef, N), Vector{Int}(undef, N), n)
end

function spherical_points!(λ::AbstractVector{T}, φ::AbstractVector{T}, ::CubedSphereSampling, n::Integer) where {T<:AbstractFloat}
    panel = Vector{Int}(undef, length(λ))
    cubed_sphere_points!(λ, φ, panel, n)
    return (; λ, φ)
end

function spherical_points(::CubedSphereSampling, n::Integer; T::Type{<:AbstractFloat} = Float64)
    N = npoints(CubedSphereSampling(), n)
    return spherical_points!(Vector{T}(undef, N), Vector{T}(undef, N), CubedSphereSampling(), n)
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

function yin_yang_axes!(
    λyin::AbstractVector{T}, φyin::AbstractVector{T},
    λyang::AbstractVector{T}, φyang::AbstractVector{T},
    nlon::Integer, nlat::Integer,
) where {T<:AbstractFloat}
    nlon = Int(nlon); nlat = Int(nlat)
    length(λyin) == nlon && length(φyin) == nlat || throw(DimensionMismatch("yin axes sizes"))
    length(λyang) == nlon * nlat && length(φyang) == nlon * nlat || throw(DimensionMismatch("yang point sizes"))
    if nlon == 1
        λyin[1] = zero(T)
    else
        @inbounds for i in 1:nlon
            λyin[i] = -T(3π) / 4 + (T(3π) / 2) * T(i - 1) / T(nlon - 1)
        end
    end
    if nlat == 1
        φyin[1] = zero(T)
    else
        @inbounds for j in 1:nlat
            φyin[j] = -T(π) / 4 + (T(π) / 2) * T(j - 1) / T(nlat - 1)
        end
    end
    k = 1
    @inbounds for φ in φyin, λ in λyin
        x = cos(φ) * cos(λ)
        y = cos(φ) * sin(λ)
        z = sin(φ)
        X = -z; Y = x; Z = -y
        θ = acos(clamp(Z, -one(T), one(T)))
        ϕ = atan(Y, X)
        ϕ < 0 && (ϕ += T(2π))
        λyang[k] = ϕ
        φyang[k] = geographic_latitude(θ)
        k += 1
    end
    return (; yin = (; λ = λyin, φ = φyin), yang = (; λ = λyang, φ = φyang))
end

function yin_yang_axes(nlon::Integer, nlat::Integer; T::Type{<:AbstractFloat} = Float64)
    nlon = Int(nlon); nlat = Int(nlat)
    return yin_yang_axes!(
        Vector{T}(undef, nlon), Vector{T}(undef, nlat),
        Vector{T}(undef, nlon * nlat), Vector{T}(undef, nlon * nlat),
        nlon, nlat,
    )
end

function spherical_points!(Λ::AbstractVector{T}, Φ::AbstractVector{T}, ::YinYangSampling, nlon::Integer, nlat::Integer) where {T<:AbstractFloat}
    nlon = Int(nlon); nlat = Int(nlat)
    n = 2 * nlon * nlat
    length(Λ) == n && length(Φ) == n || throw(DimensionMismatch("buffers must have length 2*nlon*nlat"))
    λyin = Vector{T}(undef, nlon)
    φyin = Vector{T}(undef, nlat)
    λyang = Vector{T}(undef, nlon * nlat)
    φyang = Vector{T}(undef, nlon * nlat)
    yin_yang_axes!(λyin, φyin, λyang, φyang, nlon, nlat)
    k = 1
    @inbounds for φ in φyin, λ in λyin
        Λ[k] = λ; Φ[k] = φ; k += 1
    end
    @inbounds for i in eachindex(λyang)
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
    icosahedral_mesh(frequency; T=Float64) -> (; λ, φ, edges)

Geodesic vertices at frequency `ν` plus undirected triangular-mesh edges
`(i,j)` with `i < j` (1-based). Vertex ordering is deterministic (sorted by
quantized XYZ key).
"""
function icosahedral_mesh(frequency::Integer = 1; T::Type{<:AbstractFloat} = Float64)
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
    key(p) = (round(Int, p[1] * 1_000_000), round(Int, p[2] * 1_000_000), round(Int, p[3] * 1_000_000))
    if ν == 1
        # Stable order by key, then recover base-icosahedron edges via distance.
        order = sortperm(1:12; by = i -> key(base[i]))
        verts = [base[i] for i in order]
        inv = Dict(key(verts[i]) => i for i in 1:12)
        faces = _icosahedron_faces(base)
        edgeset = Set{NTuple{2,Int}}()
        for (a, b, c) in faces
            for (u, v) in ((a, b), (b, c), (c, a))
                iu, iv = inv[key(base[u])], inv[key(base[v])]
                push!(edgeset, iu < iv ? (iu, iv) : (iv, iu))
            end
        end
        λ = Vector{T}(undef, 12)
        φ = Vector{T}(undef, 12)
        _xyz_to_lonlat!(λ, φ, verts)
        return (; λ, φ, edges = collect(edgeset))
    end
    faces = _icosahedron_faces(base)
    # Map quantized key → vertex; also accumulate edges during subdivision.
    seen = Dict{NTuple{3,Int},NTuple{3,T}}()
    function add_vert!(p)
        r = sqrt(p[1]^2 + p[2]^2 + p[3]^2)
        q = (p[1] / r, p[2] / r, p[3] / r)
        k = key(q)
        seen[k] = q
        return k
    end
    edgeset = Set{NTuple{2,NTuple{3,Int}}}()
    @inline function add_edge!(ka, kb)
        ka == kb && return nothing
        push!(edgeset, ka < kb ? (ka, kb) : (kb, ka))
        return nothing
    end
    for (ia, ib, ic) in faces
        A, B, C = base[ia], base[ib], base[ic]
        # Barycentric lattice on the face: nodes (i,j) with i+j ≤ ν.
        nodekey = Dict{NTuple{2,Int},NTuple{3,Int}}()
        for i in 0:ν, j in 0:(ν - i)
            u = T(i) / T(ν); v = T(j) / T(ν); w = one(T) - u - v
            nodekey[(i, j)] = add_vert!((
                w * A[1] + u * B[1] + v * C[1],
                w * A[2] + u * B[2] + v * C[2],
                w * A[3] + u * B[3] + v * C[3],
            ))
        end
        for i in 0:ν, j in 0:(ν - i)
            k0 = nodekey[(i, j)]
            if i + j < ν
                # edge toward B (increase i)
                add_edge!(k0, nodekey[(i + 1, j)])
                # edge toward C (increase j)
                add_edge!(k0, nodekey[(i, j + 1)])
            end
            if i ≥ 1 && (i - 1) + (j + 1) ≤ ν
                # hypotenuse of the up-pointing micro-triangle
                add_edge!(nodekey[(i, j)], nodekey[(i - 1, j + 1)])
            end
        end
    end
    length(seen) == nexp || throw(ArgumentError("icosahedral subdivision produced $(length(seen)) ≠ $nexp vertices"))
    keys_sorted = sort!(collect(keys(seen)))
    id_of = Dict{NTuple{3,Int},Int}(keys_sorted[i] => i for i in eachindex(keys_sorted))
    verts = [seen[k] for k in keys_sorted]
    edges = NTuple{2,Int}[(id_of[a], id_of[b]) for (a, b) in edgeset]
    λ = Vector{T}(undef, nexp)
    φ = Vector{T}(undef, nexp)
    _xyz_to_lonlat!(λ, φ, verts)
    return (; λ, φ, edges)
end

function icosahedral_vertices!(λ::AbstractVector{T}, φ::AbstractVector{T}, frequency::Integer = 1) where {T<:AbstractFloat}
    mesh = icosahedral_mesh(frequency; T = T)
    length(λ) == length(mesh.λ) && length(φ) == length(mesh.φ) || throw(DimensionMismatch("buffers must have length 10ν²+2"))
    copyto!(λ, mesh.λ)
    copyto!(φ, mesh.φ)
    return (; λ, φ)
end

function icosahedral_vertices(frequency::Integer = 1; T::Type{<:AbstractFloat} = Float64)
    mesh = icosahedral_mesh(frequency; T = T)
    return (; λ = mesh.λ, φ = mesh.φ)
end

function _icosahedron_faces(verts::Vector{NTuple{3,T}}) where {T}
    n = length(verts)
    n == 12 || throw(ArgumentError("expected 12 icosahedron vertices"))
    dmin = T(Inf)
    for i in 1:n, j in (i + 1):n
        d = _xyz_dist(verts[i], verts[j])
        d < dmin && (dmin = d)
    end
    tol = dmin * T(1.01)
    nbrs = [Int[] for _ in 1:n]
    for i in 1:n, j in (i + 1):n
        if _xyz_dist(verts[i], verts[j]) ≤ tol
            push!(nbrs[i], j); push!(nbrs[j], i)
        end
    end
    faces = NTuple{3,Int}[]
    seen = Set{NTuple{3,Int}}()
    for a in 1:n, b in nbrs[a], c in nbrs[a]
        b < c || continue
        c in nbrs[b] || continue
        key = (min(a, b, c), sum((a, b, c)) - min(a, b, c) - max(a, b, c), max(a, b, c))
        if !(key in seen)
            push!(seen, key)
            push!(faces, (a, b, c))
        end
    end
    length(faces) == 20 || throw(ArgumentError("icosahedron face recovery failed (got $(length(faces)))"))
    return faces
end

@inline function _xyz_dist(a::NTuple{3,T}, b::NTuple{3,T}) where {T}
    return sqrt((a[1] - b[1])^2 + (a[2] - b[2])^2 + (a[3] - b[3])^2)
end

function spherical_points!(λ::AbstractVector{T}, φ::AbstractVector{T}, s::IcosahedralSampling) where {T<:AbstractFloat}
    return icosahedral_vertices!(λ, φ, s.frequency)
end

function spherical_points(s::IcosahedralSampling; T::Type{<:AbstractFloat} = Float64)
    return icosahedral_vertices(s.frequency; T = T)
end

spherical_points(::ScatteredSphericalSampling, λ::AbstractVector, φ::AbstractVector) = (; λ, φ)
spherical_points!(λ::AbstractVector, φ::AbstractVector, ::ScatteredSphericalSampling) = (; λ, φ)

end # module
