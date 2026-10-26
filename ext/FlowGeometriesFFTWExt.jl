module FlowGeometriesFFTWExt

using FlowGeometries.SphericalSampling: SphericalSampling

# FFTW plans these element types, and the AbstractFFTs extension implements `Transform` through it.
# TODO :: This coopts the entire AbstractFFTs extension interface for it self, as if it's the only FFT packaage... This needs to be fixed since we will need to support any FFT package on the AbstractFFTs interface.
SphericalSampling._equiangular_algorithm(::Type{<:Union{Float32,Float64}}) = SphericalSampling.Transform()

end # module
