# FlowGeometries.jl

Coordinate metrics, spherical samplings, and grid types.

**API style:** `using FlowGeometries: FlowGeometries as FG`, then qualified calls (`FG.coords`,
`FG.Geometry.distance`, …). Nothing is bare-exported into your namespace.

Geometry (metric), sampling (point placement on the sphere), and grid architecture are
separate. Dispatch on the abstracts; use the concrete defaults as instances.

Core point type from `coords` is a geometry-named `NamedTuple`: `(x=, y=)` or `(λ=, φ=)`.
Escapes: `coords!(out, …)`, `coords(NTuple{2,T}, …)`, `coords(SVector{2,T}, …)` with StaticArrays.

Spherical vector / tangent helpers use the same polar frame (`λ`, `φ`, `r`), not east/north:
`local_tangent_basis` → `(; λ=ê_λ, φ=ê_φ)`, `to_planetary_cartesian` / `from_planetary_cartesian`
↔ `(; x,y,z)` / `(; λ,φ,r)`. Sampling allocators return NamedTuples too (`(; λ, φ)`, …).

Sampling constructors are allocating + in-place (`spherical_axes!`, `spherical_points!`,
`latitude_weights!`, …). Use `axes_lengths` / `npoints` to size buffers.

| Abstract | Default |
|----------|---------|
| `AbstractCartesianGeometry` | `CartesianGeometry` |
| `AbstractSphericalGeometry` | `SphericalGeometry` |
| `AbstractGaussLegendreSampling` | `GaussLegendreSampling` |
| `AbstractDriscollHealySampling` | `DriscollHealySampling` / `DriscollHealyEqualSampling` |
| `AbstractClenshawCurtisSampling` | `ClenshawCurtisSampling` |
| `AbstractMcEwenWiauxSampling` | `McEwenWiauxSampling` |
| `AbstractLatLonSampling` | `LatLonSampling` |
| `AbstractHEALPixSampling` | `HEALPixSampling` |
| `AbstractCubedSphereSampling` | `CubedSphereSampling` |
| `AbstractIcosahedralSampling` | `IcosahedralSampling` |
| `AbstractYinYangSampling` | `YinYangSampling` |
| `AbstractScatteredSphericalSampling` | `ScatteredSphericalSampling` |
| `AbstractStructuredGrid` | `StructuredGrid` |
| `AbstractCurvilinearGrid` | `CurvilinearGrid` |
| `AbstractUnstructuredGrid` | `UnstructuredGrid` |

Optional extensions (weakdeps): `StaticArrays`, `NearestNeighbors` (unstructured k-d-tree
neighbors), `DelaunayTriangulation` (Cartesian Voronoi areas), `Quickhull` (spherical Voronoi
areas).

```julia
using FlowGeometries: FlowGeometries as FG

geom = FG.CartesianGeometry(1000.0, 1000.0)
grid = FG.StructuredGrid(geom, 0.0:1000.0:10_000.0, 0.0:1000.0:5000.0, trues(11, 6))
FG.coords(grid, 2, 3)  # (x = …, y = …)

sz = FG.axes_lengths(FG.ClenshawCurtisSampling(), 32)
λ = Vector{Float64}(undef, sz.nlon); φ = Vector{Float64}(undef, sz.nlat)
FG.spherical_axes!(λ, φ, FG.ClenshawCurtisSampling(), 32)
```
