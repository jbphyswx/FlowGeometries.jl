module Discretization

using ..Axes: Axes
using ..Execution: Execution
using ..Geometry: Geometry

# The geometric inputs a numerical method needs from a grid: where a point falls, the weights that
# interpolate to it, where a cell's faces are and how wide it is, the metric factors of the coordinate
# system, and the finite-difference weights and stencil plans of any stencil. All of it is a function of
# coordinates alone; the operators that apply it to a field are in `Operators`.

include("Discretization/Staggering.jl")
include("Discretization/Spacing.jl")
include("Discretization/Locate.jl")
include("Discretization/Weights.jl")
include("Discretization/Plan.jl")

end # module Discretization
