```@meta
CurrentModule = FlowGeometries
```

# Extensions

The package depends only on ComputationalBackends, whose tags name how a bulk loop runs. Every optional
capability is a package extension that loads when — and only when — you load its trigger.

| load this | and you get |
|---|---|
| `NearestNeighbors` | k-d-tree neighbour construction for `UnstructuredGrid`, and indexed ball queries |
| `Quickhull` | spherical Voronoi dual areas for arbitrary point sets |
| `DelaunayTriangulation` | planar Voronoi areas |
| `SparseArrays` | `sparse_adjacency_matrix` |
| `StaticArrays` | `SVector`/`MVector` points and returns, in vector form end to end |
| `AbstractFFTs` | O(n log n) equiangular quadrature weights |
| `Adapt` | move a grid to another storage backend (GPU arrays, wrappers) |
| `KernelAbstractions` | run the index-parallel loops as device kernels, under `GPUBackend` |
| `Distributed` | run the bulk loops as one share per worker, under `DistributedBackend`; split a grid into a block per worker |
| `MPI` | run the bulk loops as one share per rank, under `MPIBackend`; split a grid into a block per rank |
| `PencilArrays` (with `MPI`) | transpose a field between two splits over ranks; a split as a `Pencil`, a field as a `PencilArray` |

## NearestNeighbors — spatial search

Required to build an `UnstructuredGrid` from a bare point set:

```julia
using NearestNeighbors
g = FG.Grids.UnstructuredGrid(geo, λ, φ, mask; k = 6, areas = areas)          # k nearest
g = FG.Grids.UnstructuredGrid(geo, x, y, mask; radius = 0.02, areas = areas,  # or by radius
                        topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (Lx, Ly))
```

On a sphere the tree is built on the unit-sphere embedding, where nearest-by-chord is exactly
nearest-by-great-circle — which also makes longitude wrap for free, since `λ` and `λ+2π` embed to the
same point. Cartesian domains wrap by replicating the point set at the periodic images.

It also supplies the reusable index behind [`Connectivity.indexed`](@ref), which brings a ball query on
a curvilinear or node grid down to a range search:

```julia
using NearestNeighbors
top = FG.Connectivity.indexed(grid)                    # holds a k-d tree over the cell centres
buf = FG.Connectivity.ball_scratch()                   # candidate buffer, one per task
FG.Connectivity.neighbors_within(grid, i, j; ball = r, topology = top, scratch = buf)
```

The same embedding serves construction and queries, and the index only ever returns a superset of the
ball — the exact distance gate still decides membership — so the result is the same set of cells the
scan gives. On a spheroid that superset is genuine: the ECEF chord the tree searches is a lower bound on
the geodesic, so the query over-returns and the gate trims.

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
supports; the second raises on a device backend.

```julia
using KernelAbstractions
using ComputationalBackends: GPUBackend
execution = GPUBackend(KernelAbstractions.CPU())        # or a vendor backend
FG.Operators.apply_stencil!(out, field, x, 1; order = 2, execution = execution)
FG.Connectivity.build_connectivity(grid; stencil = FG.Stencils.Moore(1), execution = execution)
```

Both are bit-identical to the serial result, which the suite checks on
`GPUBackend(KernelAbstractions.CPU())`, where the launched kernels run in host memory. There is no
per-vendor code in the package: a backend arrives from the caller and the kernel is compiled for it.

The field operators — `apply_stencil!`, `derivative!`, `gradient!`, `strain_rate!`, the collocated and
mimetic `curl!` and `divergence!`, the frame rotations, the other mimetic operators and the Laplacian's —
read the grid, metrics and plans they are given in the backend's memory. One held on the host is moved
there for the call; one moved once with `Execution.on_backend(execution, x)` is read where it is, at no
cost. Build metrics and plans (`face_metrics`, `laplacian_coefficients`, `stencil_plans`,
`gradient_plan`) from a host grid and move them; `derivative!` builds its own weights and metric factors
on the host, from host copies of the grid's axes.

A kernel cannot allocate, and the suite gates every per-cell entry point at zero bytes.

Ball queries follow from the same rule. Without an index a query reads only coordinates and the mask and
allocates nothing, so it runs inside a launch — including through `foreach_within`, which becomes one
body per cell:

```julia
FG.Connectivity.foreach_within(grid; ball = r,
                               topology = FG.Connectivity.MetricTopology(grid),
                               execution = execution) do I, J, d
    ...
end
```

The *indexed* form stays on the host: the index is a k-d tree, a host structure, so `Adapt` raises on a
topology carrying one. Little is lost — the tree spares a single thread an `O(n)` scan, and a device
has a thread per cell.

## SparseArrays

```julia
using SparseArrays
A = FG.Connectivity.sparse_adjacency_matrix(grid)                 # Bool entries, the graph's id type
A = FG.Connectivity.sparse_adjacency_matrix(Float64, Int32, conn)  # entry type, then index type
```

## StaticArrays

Adds `SVector`/`MVector` methods that stay in vector form end to end, with no tuple round-trip:

```julia
using StaticArrays
p = FG.Grids.coords(SVector, grid, 3, 5)
FG.Geometry.distance(geo, p1, p2)
FG.Geometry.project_to_tangent_plane(SVector{2,Float64}, geo, centre, neighbour)
```

A caller already working in `SVector`s keeps that type through both calls, with no conversion at
either end. See [Performance](@ref performance-page) for the measured effect.

## AbstractFFTs — fast equiangular weights

Driscoll–Healy and Fejér first-rule weights come from a sine series that costs O(n²) directly. Loading
any FFT implementation turns it into one length-`nlat` transform:

```julia
using FFTW      # or any AbstractFFTs backend
FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), 4096)   # one length-4096 transform
```

The trigger is `AbstractFFTs`, so any backend serves. With no backend loaded, an angle-addition
recurrence gives the same weights to 1.4e-14 — correctness never depends on the extension.

## Adapt — device transfer

```julia
using Adapt, CUDA
dev = adapt(CuArray, grid)
```

Handles every grid layout (a rotated grid moves its base grid) and a `StaggeredGrid`, plus
`CSRConnectivity`, `IndexTopology`, a cell list and the `MetricTopology` carrying one, stencil, masked
stencil and gradient plans, `FaceMetrics` and `LaplacianCoefficients`. A `MetricTopology` carrying a k-d
tree raises. A `SeparableMeasure` moves its *factors*, so the device receives `O(∑ Nᵈ)` numbers, and
`AllActive` carries only its size. A `BitArray` mask moved to a KernelAbstractions backend other than
`KernelAbstractions.CPU()` is unpacked to one `Bool` per cell and lands in that backend's array type;
every other target moves it by its own rule.

## Execution backends

Every bulk entry point takes a ComputationalBackends tag as `execution`, `SerialBackend()` by default:

| tag | runs a loop | needs |
|---|---|---|
| `SerialBackend()` | in the calling task | — |
| `ThreadedBackend()` | as one contiguous chunk per Julia thread | — |
| `AutoBackend()` | threaded when Julia runs more than one thread, else serially | — |
| `GPUBackend(b)` | as a launch on the KernelAbstractions backend `b` (index-parallel loops) | `KernelAbstractions` |
| `DistributedBackend(inner)` | as one share of the indices per worker, each under `inner` | `Distributed` |
| `MPIBackend(inner, comm)` | as one share per rank of `comm`, each under `inner` | `MPI` |

```julia
using ComputationalBackends: ThreadedBackend
FG.SphericalSampling.cubed_sphere_points(512; execution = ThreadedBackend())
FG.Grids.CurvilinearGrid(geo, λ, φ, mask; execution = ThreadedBackend())
FG.Connectivity.build_connectivity(grid; execution = ThreadedBackend())
```

Every write a loop makes is fixed by its index, so a result equals the serial one. A reduction combines
its partials in index order; a floating-point sum can then differ from the serial sum in its last bits.
See [Performance](@ref performance-page) for the measured speedups and for which kernels are threaded.

A worker or a rank computes on its own copy of what a loop reads, so a loop hands back what it writes
through the arrays it declares: [`Execution.Written`](@ref)`(array, span)`, where `span` maps a range of
indices to the positions of `array` they write ([`Execution.ByIndex`](@ref),
[`Execution.ByBlock`](@ref), [`Execution.ByOffsets`](@ref)). The package's own loops declare theirs. A
`foreach_within` body declares its writes with `outputs`:

```julia
using Distributed
addprocs(4)
@everywhere using FlowGeometries
using ComputationalBackends: DistributedBackend

count = zeros(Int, FG.Grids.size_tuple(grid))
FG.Connectivity.foreach_within(grid; ball = r, execution = DistributedBackend(),
                               outputs = (FG.Execution.Written(count),)) do I, J, d
    count[I...] += 1
end
```

Under `DistributedBackend` the declared arrays are filled on the calling process. Under `MPIBackend`
every rank makes the call, after `MPI.Init()`, and every rank ends with the whole of each array.

## Split grids — a field too large for one process

Under a process backend each worker or rank reads its own copy of what a loop captures, which suits work
over many points, pairs or cells of a grid every process can hold. A field too large to copy to every
process is split instead: [`Decomposition.split_grid`](@ref) cuts a `StructuredGrid` into blocks, one per
rank or worker, and each process holds its block of every field, with a halo of its neighbours' cells
along every direction that has a neighbour.

```julia
using MPI: MPI
MPI.Init()
D = FG.Decomposition
s = D.split_grid(grid, MPIBackend(); dims = (2,), halo = 3)   # blocks along the second direction
f = D.distribute(s, A)                                        # each rank keeps its block of A
o = D.split_field(s, Float64)
FG.Operators.derivative!(o, f, 2; nodes = 3)                  # halo exchange, then each block
whole = D.gather(o)                                           # on rank 0
```

Each block's own cells take the values the whole grid gives them. A stencil along a direction with
halos needs a halo at least as wide as its node count, and a narrower one is refused.
[`Decomposition.foreach_block`](@ref) and [`Decomposition.map_blocks`](@ref) run any computation on the
blocks, each process on its own, and [`Decomposition.exchange_halo!`](@ref) refreshes the halos. Under
`DistributedBackend` the same calls run from the caller, each worker computing on the block it holds.

[`Decomposition.redistribute!`](@ref) moves a field to another split of the same grid, the distributed
transpose: under MPI it is PencilArrays' `transpose!`, loaded with `using PencilArrays`, on splits made
one from the other with `split_grid(split; dims)`. The blocks are the ones PencilArrays places on the
same communicator, so [`Decomposition.pencil`](@ref) and [`Decomposition.pencil_array`](@ref) hand a split
to PencilFFTs.
