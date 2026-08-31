module FlowGeometriesAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using FlowGeometries.SphericalSampling: SphericalSampling, OpenNodes, ClosedNodes

# Equiangular latitude weights need
#
#     s[i] = Σ_{k=0}^{nterm-1} sin((2k+1)·θᵢ)/(2k+1) = Im[ e^{iθᵢ} · Σ_k cₖ e^{2ikθᵢ} ],  cₖ = 1/(2k+1)
#
# and on both node families 2θᵢ is an integer multiple of 2π/N, so the inner sum is a DFT:
#
#   closed, θᵢ = π(i−1)/N : 2θᵢ = 2π(i−1)/N          ⇒ Σ_k cₖ e^{2πik(i−1)/N}
#   open,   θᵢ = π(i−½)/N : 2θᵢ = π(2i−1)/N          ⇒ Σ_k (cₖ e^{iπk/N}) e^{2πik(i−1)/N}
#
# i.e. one length-N transform of the zero-padded coefficients, with the open family pre-twiddled.
# `bfft` is the unnormalized `exp(+2πi·)` direction, which is the sign these sums carry.

@inline _twiddle(::ClosedNodes, k::Int, nlat::Int, ::Type{T}) where {T} = complex(one(T))
@inline _twiddle(::OpenNodes, k::Int, nlat::Int, ::Type{T}) where {T} =
    cis(T(π) * T(k) / T(nlat))

# `AbstractFFTs` is an interface, so it can be loaded with nothing able to plan a transform. This tests
# the two-argument `plan_bfft!(x, dims)` that `bfft!` dispatches to; the one-argument form is defined
# generically and answers yes with no backend present.
@inline _has_fft(::Type{T}) where {T} =
    hasmethod(AbstractFFTs.plan_bfft!, Tuple{Vector{Complex{T}},UnitRange{Int}})

# The only place availability is consulted; the methods below implement one algorithm each.

SphericalSampling._equiangular_algorithm(::Type{T}) where {T<:AbstractFloat} =
    _has_fft(T) ? SphericalSampling.Transform() : SphericalSampling.Recurrence()

"""
    _sum_plan(T, nlat) -> (plan, buffer)

The in-place length-`nlat` backward transform, and the buffer it was planned against, held across calls.

`bfft!` plans afresh every time, and planning dominates a transform this small, while a set of latitude
weights is built once per grid, so the same `(T, nlat)` recurs. The buffer is cached alongside the plan
because an FFTW plan records the alignment of the array it was planned for and refuses another.

The cache is task-local: a plan and its buffer are mutable state, and two tasks weighting the same
`nlat` concurrently need one each.
"""
function _sum_plan(::Type{T}, nlat::Int) where {T<:AbstractFloat}
    store = get!(() -> Dict{Tuple{DataType,Int},Any}(), task_local_storage(),
                 :flowgeometries_equiangular_plans)::Dict{Tuple{DataType,Int},Any}
    entry = get!(store, (T, nlat)) do
        buf = zeros(Complex{T}, nlat)
        return (AbstractFFTs.plan_bfft!(buf, 1:1), buf)
    end
    return entry[1], entry[2]::Vector{Complex{T}}
end

function SphericalSampling._equiangular_sums!(
    s::AbstractVector{T}, family::Union{OpenNodes,ClosedNodes}, nlat::Int, nterm::Int,
    ::SphericalSampling.Transform,
) where {T<:AbstractFloat}
    length(s) == nlat || throw(DimensionMismatch("s must have length nlat"))
    plan, d = _sum_plan(T, nlat)
    fill!(d, zero(Complex{T}))
    @inbounds for k in 0:(nterm - 1)
        d[k + 1] = (one(T) / T(2k + 1)) * _twiddle(family, k, nlat, T)
    end
    D = (plan * d)::Vector{Complex{T}}
    @inbounds for i in 1:nlat
        θi = SphericalSampling._node_theta(family, i, nlat, T)
        s[i] = imag(cis(θi) * D[i])
    end
    return s
end

end # module
