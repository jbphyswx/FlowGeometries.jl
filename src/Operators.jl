module Operators

using ..Axes: Axes
using ..Execution: Execution
using ..Geometry: Geometry
using ..Stencils: Stencils
using ..Discretization: Discretization
using ..Grids: Grids
using ..Connectivity: Connectivity

# Everything that reads or writes a FIELD.
#
# `Discretization` turns geometry into numbers — where a point falls, the gaps around a sample, the
# weights of a stencil — and never touches data. Applying those weights does, and so does interpolating,
# differentiating and taking a gradient. Separating the two is what lets the weights be computed, cached
# and reasoned about without a field in hand, and it is why this module loads last: a field operation
# needs the grid types, and the grid modules need the weights.
#
# What is deliberately NOT here: an operator that requires a result location and a boundary-condition
# policy — a staggered difference, a divergence, a curl — which the caller assembles from these weights
# and the metric factors, because only they can choose those conventions.

include("Operators/Policies.jl")
include("Operators/StencilApply.jl")
include("Operators/Derivative.jl")
include("Operators/Interpolate.jl")
include("Operators/Gradient.jl")
include("Operators/Structured.jl")
include("Operators/Scattered.jl")

end # module Operators
