```@meta
CurrentModule = FlowGeometries
```

# Extensions

The package itself has **no dependencies**. Every optional capability is a package extension that
loads when — and only when — you load its trigger.

| load this | and you get |
|---|---|
| `NearestNeighbors` | k-d-tree neighbour construction for `UnstructuredGrid`, and indexed ball queries |
| `Quickhull` | spherical Voronoi dual areas for arbitrary point sets |
| `DelaunayTriangulation` | planar Voronoi areas |
| `SparseArrays` | `sparse_adjacency_matrix` |
| `StaticArrays` | `SVector`/`MVector` points and returns, in vector form end to end |
| `AbstractFFTs` | O(n log n) equiangular quadrature weights |
| `Adapt` | move a grid to another storage backend (GPU arrays, wrappers) |
| `KernelAbstractions` | run the index-parallel loops as device kernels |
| `ComputationalBackends` | opt-in threading through the execution tags |

## NearestNeighbors — spatial search

Required to build an `UnstructuredGrid` from a bare point set:

```julia
using NearestNeighbors
g = FG.Grids.UnstructuredGrid(geo, λ, φ, mask; k = 6, areas = areas)          # k nearest
g = FG.Grids.UnstructuredGrid(geo, x, y, mask; radius = 0.02, areas = areas,  # or by radius
                        periodic = (true, true), period = (Lx, Ly))
```

On a sphere the tree is built on the unit-sphere embedding, where nearest-by-chord is exactly
nearest-by-great-circle — which also makes longitude wrap for free, since `λ` and `λ+2π` embed to the
same point. Cartesian domains wrap by replicating the point set at the periodic images.

It also supplies the reusable index behind [`Connectivity.indexed`](@ref), which is what makes a ball
query on a curvilinear or node grid a range search rather than a scan of every cell:

```julia
using NearestNeighbors
top = FG.Connectivity.indexed(grid)                    # holds a k-d tree over the cell centres
buf = FG.Connectivity.ball_scratch()                   # candidate buffer, one per task
FG.Connectivity.neighbors_within(grid, i, j; ball = r, topology = top, scratch = buf)
```

The same embedding serves construction and queries, and the index only ever returns a superset of the
ball — the exact distance gate still decides membership — so the result is the same set of cells the
scan gives. On a spheroid that superset is genuine: the ECEF chord the tree searches is a lower bound on
the Vincenty geodesic, so the query over-returns and the gate trims.

## Quickhull / DelaunayTriangulation — Voronoi areas

Only needed for genuinely arbitrary point sets. Every built-in sampling has a closed-form cell area
and does not touch these.

```julia
using Quickhull
g = FG.Connectivity.unstructured_grid(FG.SphericalSampling.ScatteredSphericalSampling(), λ, φ)
```

The spherical dual cell of a node is the polygon through the circumcentres of its incident
triangles — the dual of the convex hull on the sphere.

## KernelAbstractions — device execution

Bulk loops here come in two shapes, and only one maps to a kernel. `run_indices` applies a body to one
index at a time with nothing carried across indices; `run_chunks` hands a contiguous range to a body that
accumulates across it. Loading `KernelAbstractions` makes the first launchable on any backend it
supports, and makes the second raise on a device backend rather than quietly running on the host.

```julia
using KernelAbstractions
backend = KernelAbstractions.CPU()          # or a vendor backend
FG.Operators.apply_stencil!(out, field, x, 1; order = 2, backend = backend)
FG.Connectivity.build_connectivity(grid; stencil = FG.Stencils.Moore(1), backend = backend)
```

Both are bit-identical to the serial result, which the suite checks on `KernelAbstractions.CPU()` — no
GPU needed to verify that the code is device-generic. There is no per-vendor code in the package: a
backend arrives from the caller and the kernel is compiled for it.

The precondition is that a kernel cannot allocate, which is why the allocation gate over every per-cell
entry point is what makes this possible at all rather than an aspiration.

Ball queries follow from the same rule. Without an index a query reads only coordinates and the mask and
allocates nothing, so it runs inside a launch — including through `foreach_within`, which becomes one
body per cell:

```julia
FG.Connectivity.foreach_within(grid; ball = r,
                               topology = FG.Connectivity.MetricTopology(grid),
                               backend = backend) do I, J, d
    ...
end
```

What stays on the host is the *indexed* form, for a reason rather than as pending work: the index is a
k-d tree, a host structure, so `Adapt` refuses to move a topology carrying one instead of dropping it
silently and leaving the device scanning every cell. That is no loss — the tree exists to spare a single
thread an `O(n)` scan, and a device has a thread per cell instead.

## SparseArrays

```julia
using SparseArrays
A = FG.Connectivity.sparse_adjacency_matrix(grid)      # SparseMatrixCSC{Bool,Int}
A = FG.Connectivity.sparse_adjacency_matrix(conn; Ti = Int32, Tv = Float64)
```

## StaticArrays

Adds `SVector`/`MVector` methods that stay in vector form end to end rather than round-tripping
through tuples:

```julia
using StaticArrays
p = FG.Grids.coords(SVector, grid, 3, 5)
FG.Geometry.distance(geo, p1, p2)
FG.Geometry.project_to_tangent_plane(SVector{2,Float64}, geo, centre, neighbour)
```

Measured over 10⁶ points against the generic path: `distance` is a wash (0.96–1.02×),
`project_to_tangent_plane` is 1.36×.

## AbstractFFTs — fast equiangular weights

Driscoll–Healy and Clenshaw–Curtis weights come from a sine series that costs O(n²) directly. Loading
any FFT implementation turns it into one length-`nlat` transform:

```julia
using FFTW      # or any AbstractFFTs backend
FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), 4096)   # ~106× faster than the fallback
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
FG.SphericalSampling.cubed_sphere_points(512; backend = ThreadedBackend())
FG.Grids.CurvilinearGrid(geo, λ, φ, mask; backend = ThreadedBackend())
FG.Connectivity.build_connectivity(grid; backend = ThreadedBackend())
```

Serial by default. Results are bit-identical to serial. See [Performance](@ref performance-page) for
the measured speedups and for which kernels are threaded.
