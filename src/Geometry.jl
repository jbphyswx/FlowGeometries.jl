module Geometry

# Public symbols are reached as `FlowGeometries.Geometry.*`. No exports and no top-level rebind.
# Internals (`as_ntuple`, `point_names`, `named_point`, …) are not part of that surface.

# Spheroid before Frames: the local frame is shared by the spherical and the ellipsoidal hierarchy, and
# the `const` naming that union is evaluated at its definition, so both hierarchies must exist by then.
include("Geometry/Metric.jl")
include("Geometry/Spheroid.jl")
include("Geometry/Frames.jl")
include("Geometry/Rotation.jl")
include("Geometry/Width.jl")

end # module
