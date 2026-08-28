module Grids

using ..Execution: Execution
using ..Axes: Axes
using ..Geometry: Geometry
using ..Discretization: Discretization
using ..SphericalSampling: SphericalSampling

# Public API via `FlowGeometries.Grids.*` or `FlowGeometries.coords` (parent rebind). No exports.

include("Grids/Interface.jl")
include("Grids/Measure.jl")
include("Grids/Structured.jl")
include("Grids/Curvilinear.jl")
include("Grids/Unstructured.jl")
include("Grids/HEALPix.jl")
include("Grids/Ring.jl")
include("Grids/Embedding.jl")
include("Grids/SpatialIndex.jl")
include("Grids/Distance.jl")

end # module
