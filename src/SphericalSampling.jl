module SphericalSampling

using ..Execution: Execution

# Public API via `FlowGeometries.SphericalSampling.*` or parent rebinds. No exports.

include("Sampling/Types.jl")
include("Sampling/GaussLegendre.jl")
include("Sampling/TensorProduct.jl")
include("Sampling/HEALPixMath.jl")
include("Sampling/Rings.jl")
include("Sampling/Meshes.jl")

end # module
