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
    # The Gaussian latitudes come from the same solve every other spectral sampling uses. The rings are
    # filled forwards, so ring 1's block lands on output elements holding latitudes that rings near the
    # south pole have yet to read: the latitudes need a buffer of their own, `O(nrings) = O(√n)`
    # against an `O(n)` output. A caller can supply one to make the fill allocation-free outright.
    scratch === nothing && return _reduced_gaussian_points!(λ, φ, s, Vector{T}(undef, nring), nring)
    length(scratch) ≥ nring ||
        throw(DimensionMismatch("scratch must hold nrings = $nring latitudes"))
    return _reduced_gaussian_points!(λ, φ, s, scratch, nring)
end

# Behind a function barrier so the buffer's type is concrete in the loop, resolved once here for every
# ring that reads it.
function _reduced_gaussian_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, s::AbstractReducedGaussianSampling,
    lat::AbstractVector{T}, nring::Int,
) where {T<:AbstractFloat}
    _gauss_legendre!(T, nothing, nothing, nring; lat = view(lat, 1:nring))
    @inbounds for j in 1:nring
        # The latitudes ascend, so ring 1 (north) is the last entry.
        φj = lat[nring + 1 - j]
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
    lat = Vector{T}(undef, nring)
    _gauss_legendre!(T, nothing, nothing, nring; lat = lat)
    return reverse!(lat)
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
        # Point `k` sits at `z = (2k+1)/n − 1`, and `1 ∓ z` is a ratio of integers. The angle from the
        # nearer pole is formed from it as `2asin(√((1 ∓ z)/2))`, which keeps a small angle to full
        # precision.
        south = 2k + 1 ≤ n
        θp = 2 * asin(sqrt(T(south ? 2k + 1 : 2n - 2k - 1) / T(2n)))
        φ[k + 1] = south ? θp - T(π) / 2 : T(π) / 2 - θp
        λ[k + 1] = mod(T(k) * dλ, T(2π))
    end
    return (; λ, φ)
end

spherical_points(s::FibonacciSampling) = spherical_points(Float64, s)

function spherical_points(::Type{T}, s::FibonacciSampling) where {T<:AbstractFloat}
    return spherical_points!(Vector{T}(undef, s.n), Vector{T}(undef, s.n), s)
end
