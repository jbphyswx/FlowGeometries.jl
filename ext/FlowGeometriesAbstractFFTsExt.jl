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

# `AbstractFFTs` is an INTERFACE, not an implementation: loading it does not mean a transform can
# actually be planned. Plenty of packages (CairoMakie among them) pull it in without a backend, and
# this extension fires on any of them. Without this guard the fast path would replace working
# weights with a `MethodError` — an extension must only ever add capability.
# Check the two-argument `plan_bfft!(x, dims)` that `bfft!` actually dispatches to. The one-argument
# form is a generic fallback that exists with no backend at all, so testing it would answer "yes"
# and then fail.
@inline _has_fft(::Type{T}) where {T} =
    hasmethod(AbstractFFTs.plan_bfft!, Tuple{Vector{Complex{T}},UnitRange{Int}})

function SphericalSampling._equiangular_sums!(
    s::AbstractVector{T}, family::Union{OpenNodes,ClosedNodes}, nlat::Int, nterm::Int,
) where {T<:AbstractFloat}
    length(s) == nlat || throw(DimensionMismatch("s must have length nlat"))
    _has_fft(T) || return @invoke SphericalSampling._equiangular_sums!(
        s::AbstractVector{T}, family::Any, nlat::Int, nterm::Int,
    )
    d = zeros(Complex{T}, nlat)
    @inbounds for k in 0:(nterm - 1)
        d[k + 1] = (one(T) / T(2k + 1)) * _twiddle(family, k, nlat, T)
    end
    D = AbstractFFTs.bfft!(d)
    @inbounds for i in 1:nlat
        θi = SphericalSampling._node_theta(family, i, nlat, T)
        s[i] = imag(cis(θi) * D[i])
    end
    return s
end

end # module
