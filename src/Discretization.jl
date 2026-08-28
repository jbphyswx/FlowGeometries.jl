module Discretization

using ..Axes: Axes
using ..Execution: Execution
using ..Geometry: Geometry

# The geometric inputs a numerical method needs from a grid: where a point falls, the weights that
# interpolate to it, where a cell's faces are, the metric factors of the coordinate system, and the
# finite-difference weights of an arbitrary stencil.
#
# Nearly all of it is a function of coordinates alone. The one exception is `apply_stencil!`, which
# applies a weight set along ONE direction leaving the result where the input was — a case in which
# every convention is already fixed (no staggering to pick, `fd_weights`' inward shift at a bounded end,
# wrapping on a periodic one, hence no halo). Operators that genuinely do need a result location and a
# boundary-condition policy — a staggered difference, a divergence, a curl — are the caller's to
# assemble, from these weights and the metric factors below.

include("Discretization/Staggering.jl")
include("Discretization/Spacing.jl")
include("Discretization/Locate.jl")
include("Discretization/Weights.jl")
include("Discretization/Plan.jl")

end # module Discretization
