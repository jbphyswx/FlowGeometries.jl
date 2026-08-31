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
    spherical_axes([T = Float64], sampling, nlat; nlon=…) -> NamedTuple{(:λ,:φ)}

Allocating wrapper around [`spherical_axes!`](@ref).

The element type leads, as it does for `zeros` and `rand`, so it takes part in dispatch and the
returned eltype is known from the signature.
"""
function spherical_axes end

spherical_axes(s::AbstractTensorProductSphericalSampling, nlat::Integer; kwargs...) =
    spherical_axes(Float64, s, nlat; kwargs...)

function spherical_axes(::Type{T}, s::AbstractTensorProductSphericalSampling, nlat::Integer;
                        nlon::Union{Nothing,Integer} = nothing) where {T<:AbstractFloat}
    sz = axes_lengths(s, nlat; nlon)
    λ = Vector{T}(undef, sz.nlon)
    φ = Vector{T}(undef, sz.nlat)
    return spherical_axes!(λ, φ, s, nlat; nlon)
end

spherical_axes(s::AbstractLatLonSampling, nlat::Integer; kwargs...) =
    spherical_axes(Float64, s, nlat; kwargs...)

function spherical_axes(
    ::Type{T}, s::AbstractLatLonSampling, nlat::Integer;
    nlon::Integer,
    lat_range::Tuple{<:Real,<:Real} = (-π / 2, π / 2),
    lon_range::Tuple{<:Real,<:Real} = (0, 2π),
) where {T<:AbstractFloat}
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
    spherical_quadrature([T = Float64], sampling, nlat; nlon=…) -> NamedTuple{(:λ,:φ,:w)}

Allocating wrapper around [`spherical_quadrature!`](@ref).
"""
spherical_quadrature(s::AbstractTensorProductSphericalSampling, nlat::Integer; kwargs...) =
    spherical_quadrature(Float64, s, nlat; kwargs...)

function spherical_quadrature(
    ::Type{T}, s::AbstractTensorProductSphericalSampling, nlat::Integer;
    nlon::Union{Nothing,Integer} = nothing,
) where {T<:AbstractFloat}
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
    AbstractEquiangularAlgorithm

Which construction computes an equiangular node family's sine sums: [`Recurrence`](@ref), always
available, or [`Transform`](@ref), which needs an FFT planner.

A type, as the mask policies are. The two agree to round-off, so naming one pins a result across
machines that differ in whether an FFT backend is installed.
"""
abstract type AbstractEquiangularAlgorithm end

"""
    Recurrence()

The angle-addition recurrence: `O(nlat·nterm)`, no dependency, defined for every element type.
"""
struct Recurrence <: AbstractEquiangularAlgorithm end

"""
    Transform()

One length-`nlat` backward transform: `O(nlat·log nlat)`. Implemented by the `AbstractFFTs` extension,
and raises without one — a planner has to exist for this to mean anything.
"""
struct Transform <: AbstractEquiangularAlgorithm end

"""
    _equiangular_algorithm(T) -> AbstractEquiangularAlgorithm

The default for element type `T`: [`Recurrence`](@ref) unless a loaded extension can plan a transform
for it. The single place availability is consulted, so no method below branches on it.
"""
_equiangular_algorithm(::Type) = Recurrence()

"""
    latitude_weights!(w, sampling, nlat) -> w

Fill preallocated latitude quadrature weights (`length(w) == nlat`). The equiangular families also
take `algorithm` — see [`latitude_weights`](@ref).
"""
function latitude_weights! end

# The keyword is valid for the equiangular families, so it is accepted here and answered with a message
# about this sampling.
function latitude_weights!(
    w::AbstractVector{T}, ::AbstractGaussLegendreSampling, nlat::Integer;
    algorithm::Union{Nothing,AbstractEquiangularAlgorithm} = nothing,
) where {T<:AbstractFloat}
    nlat = Int(nlat)
    algorithm === nothing || throw(ArgumentError(
        "Gauss–Legendre weights come from the Bogaert root solve, so there is no $(algorithm) to " *
        "select. `algorithm` picks the equiangular sine-series construction, and applies to " *
        "DriscollHealySampling and ClenshawCurtisSampling.",
    ))
    length(w) == nlat || throw(DimensionMismatch("w length must equal nlat"))
    _gauss_legendre!(T, nothing, w, nlat)
    return w
end

"""
    OpenNodes(), ClosedNodes()

The two equiangular colatitude families: open `θᵢ = π(i−½)/N` (Clenshaw–Curtis) and closed
`θᵢ = π(i−1)/N` (Driscoll–Healy). They are types, so the sum below dispatches on which one it has.
"""
struct OpenNodes end
struct ClosedNodes end

@inline _node_theta(::OpenNodes, i::Int, nlat::Int, ::Type{T}) where {T} =
    T(π) * (T(i) - T(0.5)) / T(nlat)
@inline _node_theta(::ClosedNodes, i::Int, nlat::Int, ::Type{T}) where {T} =
    T(π) * T(i - 1) / T(nlat)

"""
    _equiangular_sums!(s, family, nlat, nterm[, algorithm]) -> s

`s[i] = Σ_{k=0}^{nterm-1} sin((2k+1)·θᵢ)/(2k+1)` for the given node family.

With no `algorithm`, [`_equiangular_algorithm`](@ref) chooses one for the element type.
"""
@inline _equiangular_sums!(
    s::AbstractVector{T}, family, nlat::Int, nterm::Int,
) where {T<:AbstractFloat} =
    _equiangular_sums!(s, family, nlat, nterm, _equiangular_algorithm(T))

# `Transform` names a capability an extension supplies. This method resolves without the extension and
# says which package to load.
_equiangular_sums!(::AbstractVector{T}, _, ::Int, ::Int, ::Transform) where {T<:AbstractFloat} =
    throw(ArgumentError(
        "Transform() needs an FFT planner for $T; load an AbstractFFTs implementation (FFTW, for " *
        "one), or ask for Recurrence()",
    ))

"""
The recurrence: `sin((2k+1)θ) = 2cos(2θ)·sin((2k−1)θ) − sin((2k−3)θ)`, seeded with `s₋₁ = −sin θ`,
`s₀ = sin θ`, so each term costs two multiplies and no transcendental.
"""
function _equiangular_sums!(
    s::AbstractVector{T}, family, nlat::Int, nterm::Int, ::Recurrence,
) where {T<:AbstractFloat}
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
    _equiangular_weights!(w, family, nlat[, algorithm]) -> w

Latitude quadrature weights for an equiangular node set, from the sine-series expansion of the
`sinθ` Jacobian:

    wᵢ = (4/N)·sinθᵢ·Σ_{k=0}^{⌊N/2⌋} sin((2k+1)θᵢ)/(2k+1)

`family` is [`OpenNodes`](@ref) or `ClosedNodes`, so one construction serves both. Weights sum to
`∫₀^π sinθ dθ = 2` — see [`latitude_weights`](@ref) for why every sampling here uses that
normalization.

`algorithm` selects which construction computes the sums; omitted, the element type's default does.
"""
function _equiangular_weights!(
    w::AbstractVector{T}, family, nlat::Int,
    algorithm::AbstractEquiangularAlgorithm = _equiangular_algorithm(T),
) where {T<:AbstractFloat}
    nterm = (nlat + 1) ÷ 2
    _equiangular_sums!(w, family, nlat, nterm, algorithm)   # w doubles as the sum buffer
    @inbounds for i in 1:nlat
        w[i] *= (T(4) / T(nlat)) * sin(_node_theta(family, i, nlat, T))
    end
    return w
end

function latitude_weights!(
    w::AbstractVector{T}, ::AbstractDriscollHealySampling, nlat::Integer;
    algorithm::AbstractEquiangularAlgorithm = _equiangular_algorithm(T),
) where {T<:AbstractFloat}
    nlat = Int(nlat)
    iseven(nlat) || throw(ArgumentError("DH nlat must be even"))
    length(w) == nlat || throw(DimensionMismatch("w length must equal nlat"))
    return _equiangular_weights!(w, ClosedNodes(), nlat, algorithm)
end

"""
    latitude_weights!(w, ::AbstractClenshawCurtisSampling, nlat)

Weights for the open nodes `θᵢ = π(i−½)/N`, from the same sine-series rule as the closed
equiangular families.

These integrate a single `P_l` exactly for `l ≤ N−1`, which is weaker than
[`bandlimit`](@ref)`(ClenshawCurtisSampling(), N) = N−1` suggests: spectral analysis integrates
products of two degree-`lmax` functions, so a quadrature exact to `l ≤ N−1` supports quadrature-based
analysis only up to `lmax ≈ (N−1)/2`. The reported band limit describes what the grid represents; this
quadrature integrates to half of it. Use `GaussLegendreSampling` (exact to `2N−1`) where analysis must
be exact at the stated band limit.
"""
function latitude_weights!(
    w::AbstractVector{T}, ::AbstractClenshawCurtisSampling, nlat::Integer;
    algorithm::AbstractEquiangularAlgorithm = _equiangular_algorithm(T),
) where {T<:AbstractFloat}
    nlat = Int(nlat)
    length(w) == nlat || throw(DimensionMismatch("w length must equal nlat"))
    return _equiangular_weights!(w, OpenNodes(), nlat, algorithm)
end

latitude_weights!(::AbstractVector, ::AbstractMcEwenWiauxSampling, ::Integer; _...) = throw(ArgumentError(
    "McEwen–Wiaux latitude weights are not implemented: the MW quadrature is not the sine-series " *
    "rule the other equiangular samplings use (it is built on an extension of the sphere to a torus), " *
    "and applying that rule to MW nodes is not exact even for l = 0. Use `GaussLegendreSampling`, " *
    "`DriscollHealySampling`, or `ClenshawCurtisSampling` if you need quadrature weights.",
))
latitude_weights!(::AbstractVector, ::AbstractLatLonSampling, ::Integer; _...) =
    throw(ArgumentError("LatLonSampling is an arbitrary lat–lon layout with no spectral quadrature weights"))

"""
    latitude_weights([T = Float64], s, nlat) -> Vector{T}
    latitude_weights!(w, s, nlat) -> w

Latitude quadrature weights `wⱼ` for sampling `s`, normalized so that

    Σⱼ wⱼ = ∫₀^π sinθ dθ = 2

for every sampling that provides them. The weights therefore carry the `sinθ` Jacobian and nothing
else; the longitude factor is the caller's, so a full-sphere integral is always

    ∫ f dΩ ≈ (2π/nlon) · Σⱼ wⱼ Σᵢ f(λᵢ, φⱼ)

regardless of which sampling produced the weights. Not every sampling has them — `LatLonSampling`
has no spectral quadrature at all, and `McEwenWiauxSampling`'s is a different construction.

The equiangular families — Driscoll–Healy and Clenshaw–Curtis — additionally take
`algorithm::`[`AbstractEquiangularAlgorithm`](@ref), which pins the construction of their sine sums.
The two constructions agree only to round-off, so naming one fixes the result across machines that
differ in whether an FFT backend is installed. Gauss–Legendre's weights come from a root solve, so it
takes no `algorithm`.
"""
latitude_weights(s::AbstractSphericalSampling, nlat::Integer; kwargs...) =
    latitude_weights(Float64, s, nlat; kwargs...)

function latitude_weights(
    ::Type{T}, s::AbstractSphericalSampling, nlat::Integer; kwargs...,
) where {T<:AbstractFloat}
    w = Vector{T}(undef, Int(nlat))
    return latitude_weights!(w, s, nlat; kwargs...)
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
    n = sz.nlon * sz.nlat
    length(Λ) == n && length(Φ) == n ||
        throw(DimensionMismatch("buffers must have length nlon*nlat = $n"))
    n == 0 && return (; λ = Λ, φ = Φ)
    # The axes are built into the leading elements of the outputs and expanded in place, so no scratch
    # is needed. The expansion runs backwards, and cell (i, j) lands at k = (j-1)·nlon + i ≥ max(i, j),
    # so a write never lands on an axis element still to be read.
    spherical_axes!(view(Λ, 1:sz.nlon), view(Φ, 1:sz.nlat), s, nlat; kwargs...)
    @inbounds for j in sz.nlat:-1:1
        φj = Φ[j]
        base = (j - 1) * sz.nlon
        for i in sz.nlon:-1:1
            Λ[base + i] = Λ[i]
            Φ[base + i] = φj
        end
    end
    return (; λ = Λ, φ = Φ)
end

"""
    spherical_points([T = Float64], sampling, args...) -> NamedTuple{(:λ,:φ)}

Every point of the sampling, flattened, as longitude/latitude vectors. Allocating wrapper around
[`spherical_points!`](@ref); use [`npoints`](@ref) to size buffers for the in-place form.

For a tensor-product sampling this is the outer product of its axes, so prefer
[`spherical_axes`](@ref) when the separable form will do.
"""
spherical_points(s::AbstractTensorProductSphericalSampling, nlat::Integer; kwargs...) =
    spherical_points(Float64, s, nlat; kwargs...)

function spherical_points(::Type{T}, s::AbstractTensorProductSphericalSampling, nlat::Integer;
                          kwargs...) where {T<:AbstractFloat}
    n = npoints(s, nlat; kwargs...)
    Λ = Vector{T}(undef, n)
    Φ = Vector{T}(undef, n)
    return spherical_points!(Λ, Φ, s, nlat; kwargs...)
end
