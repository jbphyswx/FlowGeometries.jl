```@meta
CurrentModule = FlowGeometries
```

# Extensions

The package itself has **no dependencies**. Every optional capability is a package extension that
loads when — and only when — you load its trigger.

| load this | and you get |
|---|---|
| `NearestNeighbors` | k-d-tree neighbour construction for `UnstructuredGrid` |
| `Quickhull` | spherical Voronoi dual areas for arbitrary point sets |
| `DelaunayTriangulation` | planar Voronoi areas |
| `SparseArrays` | `sparse_adjacency_matrix` |
| `StaticArrays` | `SVector`/`MVector` points and returns, in vector form end to end |
| `AbstractFFTs` | O(n log n) equiangular quadrature weights |
| `Adapt` | move a grid to another storage backend (GPU arrays, wrappers) |
| `ComputationalBackends` | opt-in threading through the execution tags |

## NearestNeighbors — spatial search

Required to build an `UnstructuredGrid` from a bare point set:

```julia
using NearestNeighbors
g = FG.UnstructuredGrid(geo, λ, φ, mask; k = 6, areas = areas)          # k nearest
g = FG.UnstructuredGrid(geo, x, y, mask; radius = 0.02, areas = areas,  # or by radius
                        periodic = (true, true), period = (Lx, Ly))
```

On a sphere the tree is built on the unit-sphere embedding, where nearest-by-chord is exactly
nearest-by-great-circle — which also makes longitude wrap for free, since `λ` and `λ+2π` embed to the
same point. Cartesian domains wrap by replicating the point set at the periodic images.

## Quickhull / DelaunayTriangulation — Voronoi areas

Only needed for genuinely arbitrary point sets. Every built-in sampling has a closed-form cell area
and does not touch these.

```julia
using Quickhull
g = FG.unstructured_grid(FG.ScatteredSphericalSampling(), λ, φ)
```

The spherical dual cell of a node is the polygon through the circumcentres of its incident
triangles — the dual of the convex hull on the sphere.

## SparseArrays

```julia
using SparseArrays
A = FG.sparse_adjacency_matrix(grid)      # SparseMatrixCSC{Bool,Int}
A = FG.sparse_adjacency_matrix(conn; Ti = Int32, Tv = Float64)
```

## StaticArrays

Adds `SVector`/`MVector` methods that stay in vector form end to end rather than round-tripping
through tuples:

```julia
using StaticArrays
p = FG.coords(SVector, grid, 3, 5)
FG.distance(geo, p1, p2)
FG.project_to_tangent_plane(SVector{2,Float64}, geo, centre, neighbour)
```

Measured over 10⁶ points against the generic path: `distance` is a wash (0.96–1.02×),
`project_to_tangent_plane` is 1.36×.

## AbstractFFTs — fast equiangular weights

Driscoll–Healy and Clenshaw–Curtis weights come from a sine series that costs O(n²) directly. Loading
any FFT implementation turns it into one length-`nlat` transform:

```julia
using FFTW      # or any AbstractFFTs backend
FG.latitude_weights(FG.DriscollHealySampling(), 4096)   # ~106× faster than the fallback
```

The trigger is `AbstractFFTs`, not FFTW, so any backend serves. Without one, an angle-addition
recurrence gives the same weights to 1.4e-14 — correctness never depends on the extension.

## Adapt — device transfer

```julia
using Adapt, CUDA
dev = adapt(CuArray, grid)
```

Handles all three grid types plus `CSRConnectivity` and `IndexTopology`. A `SeparableMeasure` moves
its *factors*, not a materialized outer product — materializing onto a device is exactly what the
factored form exists to avoid — and `AllActive` carries only its size, so there is nothing to move.

## ComputationalBackends — threading

```julia
using ComputationalBackends: ThreadedBackend
FG.cubed_sphere_points(512; backend = ThreadedBackend())
FG.CurvilinearGrid(geo, λ, φ, mask; backend = ThreadedBackend())
FG.build_connectivity(grid; backend = ThreadedBackend())
```

Serial by default. Results are bit-identical to serial. See [Performance](@ref performance-page) for
the measured speedups and for which kernels are threaded.
