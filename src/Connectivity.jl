module Connectivity

using ComputationalBackends: ComputationalBackends as CB
using ..Axes: Axes
using ..Execution: Execution
using ..Geometry: Geometry
using ..Stencils: Stencils
using ..Discretization: Discretization
using ..SphericalSampling: SphericalSampling
using ..Grids: Grids

# A grid's neighbours come from its `Grids.adjacency_source`:
#   Structured / Curvilinear / Rotated                  → index-topology stencil (periodicity + mask)
#   HEALPix / Ring / Cubed-sphere / Yin–Yang / icosahedral → arithmetic on the cell id (`Formula.jl`)
#   Unstructured                                         → stored CSR on the grid
# A spherical sampling that defines a mesh builds the same graph with no grid, through
# `build_connectivity(sampling, …)` (`Connectivity/Spherical.jl`). `CSRConnectivity` is the flat form
# `build_connectivity` returns.

include("Connectivity/CSR.jl")
include("Connectivity/IndexStencil.jl")
include("Connectivity/Formula.jl")
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
