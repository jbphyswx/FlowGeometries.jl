module Geometry

# Public symbols are reached as `FlowGeometries.Geometry.*` or rebound on `FlowGeometries`
# for `FG.distance`-style calls. Internals (`as_ntuple`, `point_names`, `named_point`, …)
# are intentionally not exported.

# Spheroid before Frames: the local frame is shared by the spherical and the ellipsoidal hierarchy, and
# the type naming that union is evaluated where it is written rather than when it is first called.
include("Geometry/Metric.jl")
include("Geometry/Spheroid.jl")
include("Geometry/Frames.jl")
include("Geometry/Rotation.jl")
include("Geometry/Width.jl")

end # module
