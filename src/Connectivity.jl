module Connectivity

using ..Execution: Execution
using ..Geometry: Geometry
using ..Stencils: Stencils
using ..Discretization: Discretization
using ..Grids: Grids

# Connectivity is a property of the *grid architecture* or of a *spherical sampling*
# that defines its own mesh topology:
#   Structured / Curvilinear → index-topology stencil (periodicity + mask)
#   Unstructured            → stored CSR on the grid
#   Cubed-sphere / Yin–Yang / HEALPix / icosahedral / tensor-product samplings
#                           → `build_connectivity(sampling, …)` (see `Connectivity/Spherical.jl`)
#
# Primary API dispatches on `grid` or `sampling`. `CSRConnectivity` is the sparse storage
# format from `build_connectivity` when you need a flat graph.

include("Connectivity/CSR.jl")
include("Connectivity/IndexStencil.jl")
include("Connectivity/Metric.jl")
include("Connectivity/Reach.jl")
include("Connectivity/PointQueries.jl")
include("Connectivity/KNearest.jl")
include("Connectivity/Sweeps.jl")
include("Connectivity/Build.jl")
include("Connectivity/Adjacency.jl")
include("Connectivity/MaskTopology.jl")


include("Connectivity/Spherical.jl")

end # module
