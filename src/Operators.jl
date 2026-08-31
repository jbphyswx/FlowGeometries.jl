module Operators

using ..Axes: Axes
using ..Execution: Execution
using ..Geometry: Geometry
using ..Stencils: Stencils
using ..Discretization: Discretization
using ..Grids: Grids
using ..Connectivity: Connectivity

# Everything that reads or writes a field.
#
# `Discretization` turns geometry into numbers — where a point falls, the gaps around a sample, the
# weights of a stencil — and never touches data. Applying those weights does, and so does interpolating,
# differentiating and taking a gradient. The separation keeps the weights computable, cacheable and
# checkable with no field in hand, and it fixes the load order: a field operation needs the grid types,
# and the grid modules need the weights.
#

include("Operators/Policies.jl")
include("Operators/StencilApply.jl")
include("Operators/PlanApply.jl")
include("Operators/Derivative.jl")
include("Operators/Interpolate.jl")
include("Operators/Gradient.jl")
include("Operators/Structured.jl")
include("Operators/Scattered.jl")
include("Operators/Staggered.jl")

end # module Operators
