module Geometry

# Public symbols are reached as `FlowGeometries.Geometry.*` or rebound on `FlowGeometries`
# for `FG.distance`-style calls. Internals (`as_ntuple`, `point_names`, `named_point`, …)
# are intentionally not exported.

# Spheroid before Frames: the local frame is shared by the spherical and the ellipsoidal hierarchy, and
# the `const` naming that union is evaluated at its definition, so both hierarchies must exist by then.
include("Geometry/Metric.jl")
include("Geometry/Spheroid.jl")
include("Geometry/Frames.jl")
include("Geometry/Rotation.jl")
include("Geometry/Width.jl")

end # module
