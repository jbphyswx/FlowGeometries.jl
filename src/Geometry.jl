module Geometry

# Public symbols are reached as `FlowGeometries.Geometry.*` or rebound on `FlowGeometries`
# for `FG.distance`-style calls. Internals (`as_ntuple`, `point_names`, `named_point`, …)
# are intentionally not exported.

include("Geometry/Metric.jl")
include("Geometry/Frames.jl")
include("Geometry/Spheroid.jl")
include("Geometry/Rotation.jl")
include("Geometry/Width.jl")

end # module
