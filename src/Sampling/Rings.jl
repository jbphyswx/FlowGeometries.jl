# ---- Reduced Gaussian / octahedral ------------------------------------------

"""
    spherical_points!(λ, φ, sampling::AbstractReducedGaussianSampling; scratch = nothing) -> NamedTuple

Ring-by-ring points of a reduced Gaussian grid, north to south, longitudes equispaced within each ring
starting at zero. Buffer length is [`npoints`](@ref).

The Gaussian latitudes need one value per ring, which cannot overlap the output here (see the note in
the body), so `scratch` — any vector of at least `nrings(sampling)` elements — makes the fill
allocation-free for a caller filling many grids.
"""
function spherical_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, s::AbstractReducedGaussianSampling;
    scratch::Union{Nothing,AbstractVector{T}} = nothing,
) where {T<:AbstractFloat}
    nring = nrings(s)
    n = npoints(s)
    length(λ) == n && length(φ) == n ||
        throw(DimensionMismatch("buffers must have length npoints = $n"))
    # The Gaussian latitudes come from the same solve every other spectral sampling uses. They cannot
    # live in the output's leading elements the way a tensor-product grid's axes do: those are
    # expanded BACKWARDS, so a write never lands on an axis element still to be read, whereas the
    # rings here are written forwards and ring 1's block would overwrite nodes belonging to rings near
    # the south pole. Hence a buffer — `O(nrings) = O(√n)` against an `O(n)` output, and the caller
    # can supply one to make the fill allocation-free outright.
    scratch === nothing && return _reduced_gaussian_points!(λ, φ, s, Vector{T}(undef, nring), nring)
    length(scratch) ≥ nring ||
        throw(DimensionMismatch("scratch must hold nrings = $nring latitudes"))
    return _reduced_gaussian_points!(λ, φ, s, scratch, nring)
end

# Behind a function barrier so the buffer's type is concrete in the loop: resolved once here rather
# than left as a `Union{Nothing,…}` for every ring to re-dispatch on.
function _reduced_gaussian_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, s::AbstractReducedGaussianSampling,
    μ::AbstractVector{T}, nring::Int,
) where {T<:AbstractFloat}
    _gauss_legendre!(T, view(μ, 1:nring), nothing, nring)
    @inbounds for j in 1:nring
        # `μ` ascends, so ring 1 (north) is the LAST entry.
        φj = asin(μ[nring + 1 - j])
        rng = ring_range(s, j)
        dλ = T(2π) / T(length(rng))
        for (i, k) in enumerate(rng)
            λ[k] = T(i - 1) * dλ
            φ[k] = φj
        end
    end
    return (; λ, φ)
end

spherical_points(s::AbstractReducedGaussianSampling) = spherical_points(Float64, s)

function spherical_points(::Type{T}, s::AbstractReducedGaussianSampling) where {T<:AbstractFloat}
    n = npoints(s)
    return spherical_points!(Vector{T}(undef, n), Vector{T}(undef, n), s)
end

"""
    ring_latitudes([T = Float64], sampling) -> Vector{T}

The Gaussian latitudes of a reduced Gaussian grid's rings, north to south.
"""
ring_latitudes(s::AbstractReducedGaussianSampling) = ring_latitudes(Float64, s)

function ring_latitudes(::Type{T}, s::AbstractReducedGaussianSampling) where {T<:AbstractFloat}
    nring = nrings(s)
    μ = Vector{T}(undef, nring)
    _gauss_legendre!(T, μ, nothing, nring)
    return [asin(μ[nring + 1 - j]) for j in 1:nring]
end

"""
    latitude_weights([T = Float64], sampling::AbstractReducedGaussianSampling) -> Vector{T}

Gauss–Legendre weights for the grid's rings, north to south, normalized as everywhere else in this
module so that `Σw = 2`. A full-sphere integral is then `Σⱼ wⱼ (2π/nlonⱼ) Σᵢ f`, the longitude factor
varying by ring because the ring populations do.
"""
latitude_weights(s::AbstractReducedGaussianSampling) = latitude_weights(Float64, s)

function latitude_weights(::Type{T}, s::AbstractReducedGaussianSampling) where {T<:AbstractFloat}
    nring = nrings(s)
    w = Vector{T}(undef, nring)
    _gauss_legendre!(T, nothing, w, nring)
    return reverse!(w)
end

# ---- Fibonacci lattice ------------------------------------------------------

"""
    spherical_points!(λ, φ, sampling::FibonacciSampling) -> NamedTuple

The `n` golden-spiral points, allocating nothing.
"""
function spherical_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, s::FibonacciSampling,
) where {T<:AbstractFloat}
    n = s.n
    length(λ) == n && length(φ) == n ||
        throw(DimensionMismatch("buffers must have length n = $n"))
    golden = (one(T) + sqrt(T(5))) / T(2)
    dλ = T(2π) / golden
    @inbounds for k in 0:(n - 1)
        z = T(2k + 1) / T(n) - one(T)
        φ[k + 1] = asin(clamp(z, -one(T), one(T)))
        λ[k + 1] = mod(T(k) * dλ, T(2π))
    end
    return (; λ, φ)
end

spherical_points(s::FibonacciSampling) = spherical_points(Float64, s)

function spherical_points(::Type{T}, s::FibonacciSampling) where {T<:AbstractFloat}
    return spherical_points!(Vector{T}(undef, s.n), Vector{T}(undef, s.n), s)
end
