"""
    FlowGeometries

Coordinate metrics, spherical samplings, and grid types.

```
AbstractGeometry
├── AbstractCartesianGeometry  ← CartesianGeometry
└── AbstractSphericalGeometry  ← SphericalGeometry

AbstractSphericalSampling
├── … spectral / lat-lon / HEALPix / cubed-sphere / icosahedral / Yin–Yang / scattered …

AbstractGrid
├── AbstractStructuredGrid     ← StructuredGrid
├── AbstractCurvilinearGrid    ← CurvilinearGrid
└── AbstractUnstructuredGrid   ← UnstructuredGrid
```

Nothing is exported and nothing is rebound at the top level: every name is reached through the
submodule that defines it.

```julia
using FlowGeometries: FlowGeometries as FG

geo  = FG.Geometry.SphericalGeometry()
grid = FG.Connectivity.structured_grid(FG.SphericalSampling.GaussLegendreSampling(), 64)
FG.Grids.coords(grid, 2, 3)                  # (λ=, φ=)
FG.Geometry.distance(geo, p1, p2)
```

Submodules: `Axes` (coordinate axes and the spacing trait), `Geometry` (metrics), `Stencils`
(neighbourhood shapes), `Discretization` (location, interpolation, staggering, derivative weights),
`SphericalSampling` (where the points go), `Grids` (storage), `Connectivity` (topology), `Operators`
(everything that reads or writes a field), `Execution` (how bulk loops run).
"""
module FlowGeometries

include("Execution.jl")
using .Execution: Execution

include("Axes.jl")
using .Axes: Axes

include("Stencils.jl")
using .Stencils: Stencils

include("Geometry.jl")
using .Geometry: Geometry

include("Discretization.jl")
using .Discretization: Discretization

include("SphericalSampling.jl")
using .SphericalSampling: SphericalSampling

include("Grids.jl")
using .Grids: Grids

include("Connectivity.jl")
using .Connectivity: Connectivity

include("Operators.jl")
using .Operators: Operators

end # module FlowGeometries
