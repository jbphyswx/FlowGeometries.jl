module SphericalSampling

using ComputationalBackends: ComputationalBackends as CB
using ..Execution: Execution

# Public API via `FlowGeometries.SphericalSampling.*`. No exports and no top-level rebind.

include("Sampling/Types.jl")
include("Sampling/GaussLegendre.jl")
include("Sampling/TensorProduct.jl")
include("Sampling/HEALPixMath.jl")
include("Sampling/Rings.jl")
include("Sampling/Meshes.jl")

end # module
