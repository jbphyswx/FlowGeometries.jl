```@meta
CurrentModule = FlowGeometries.Connectivity
```

```@setup connectivity
using FlowGeometries: FlowGeometries as FG
geo   = FG.Geometry.SphericalGeometry()
grid  = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 16)
conn  = FG.Connectivity.build_connectivity(grid)
i, j  = 3, 5
nlon, nlat = size(grid)
n     = 4
nside = 4
out   = Vector{Int}(undef, 64)
A     = falses(length(grid), length(grid))
topo  = FG.Connectivity.IndexTopology((nlon, nlat), (true, false), nothing)
lin   = 7
```

# [Stencils & Connectivity](@id connectivity-page)

Connectivity answers: **which cells are neighbours?**

## Stencil shapes

A stencil is a neighbourhood *shape* in index space. It carries no dimensionality — the same shape
applies to a 2-D and a 5-D grid — and its shape and radius live in its type, so the offset set is
built at compile time and a loop over it unrolls.

```@example conn
using FlowGeometries: FlowGeometries as FG
S = FG.Stencils

S.offsets(S.Axial(1), Val(2))
```

```@example conn
S.offsets(S.Moore(1), Val(2))
```

| shape | offsets | meaning |
|---|---|---|
| `Axial(r)` | `2·N·r` | axis-aligned only, `±k·êᵈ` for `k = 1…r` |
| `VonNeumann(r)` | — | the `L¹` ball, `0 < ‖δ‖₁ ≤ r` |
| `Moore(r)` (alias `Vertex`) | `(2r+1)^N − 1` | the `L^∞` ball, the surrounding box |
| `Diagonal(r)` | `2^N·r` | pure diagonals only |
| `Anisotropic(radii)` | — | a box with its own radius per direction |
| `Custom(offsets)` | as given | an explicit offset set |

`Axial(1)` and `VonNeumann(1)` are the same four-in-2-D set; beyond radius 1 they differ, because
`VonNeumann` admits diagonal combinations whose step count still fits.

```@example conn
S.nstencil(S.Axial(2), Val(2)), S.nstencil(S.VonNeumann(2), Val(2)), S.nstencil(S.Moore(2), Val(2))
```

Any dimension:

```@example conn
S.nstencil(S.Axial(1), Val(5)), S.nstencil(S.Moore(1), Val(5))
```

```@example conn
S.reach(S.Anisotropic((3, 1)), Val(2))     # halo width per direction
```

!!! note "A stencil is named by its type, never by a symbol"
    A symbol could only be resolved at run time, so the neighbour iterator built from it would not be
    concretely typed and every cell of a traversal would allocate. `stencil = S.Moore(2)` is free;
    there is no symbol form to reach for.

Two different things get called a radius, and they are separate types.
[`Stencils.CellRadius`](@ref) counts cells in index space; [`Stencils.MetricBall`](@ref) is a physical
distance measured through the geometry. On a stretched or spherical grid the number of cells within a
given distance varies across the grid, so the two cannot be collapsed into one.

The observation the whole module rests on is that a neighbour computation never looks at a
coordinate. It reads three things — extent per dimension, wrapping per dimension, and which cells are
active — and nothing else. That triple is `IndexTopology`, and it is why a curvilinear grid needs no
separate implementation from a structured one: it is the `N = 2` case of the same algorithm.

## Querying neighbours

![HEALPix and icosahedral connectivity](assets/connectivity.png)


Three forms, in increasing order of how much they allocate:

```@example connectivity
out = Vector{Int}(undef, 8)
n = FG.Connectivity.neighbors!(out, grid, i, j; stencil = FG.Stencils.Moore(1))   # writes n indices
k = FG.Connectivity.nneighbors(grid, i, j)                            # just the count
it = FG.Grids.neighbors(grid, i, j)                            # lazy iterator, no array
```

`stencil` is any [`Stencils`](@ref) shape — `Axial(r)`, `VonNeumann(r)`, `Moore(r)` (alias `Vertex`),
`Diagonal(r)`, `Anisotropic(radii)` or `Custom(offsets)` — at any radius and in any number of
dimensions. It is named by its TYPE, not by a symbol: a symbol could only be resolved at run time, and
the neighbour iterator built from it would then allocate once per cell. `active_only = true`
excludes masked-out cells from both ends of an edge.

Prefer `neighbors!` in hot loops. `neighbors` returns an iterator rather than a freshly built vector
per cell, which would cost two heap allocations for every cell visited.

## CSR

For a whole graph at once:

```@example connectivity
conn = FG.Connectivity.build_connectivity(grid)                   # or (sampling, args...)
FG.Connectivity.nnodes(conn), FG.Connectivity.nedges(conn)
FG.Grids.neighbors(conn, i)                                # a view into the flat array
```

`CSRConnectivity` stores one flat neighbour array plus offsets — never a vector per node, which costs
`nnodes` allocations and turns every later traversal into pointer chasing. Both buffers are typed
independently and accept any `Integer`, so a large mesh can carry `Int32` indices.

Building from a sampling never constructs a grid:

```@example connectivity
FG.Connectivity.build_connectivity(FG.SphericalSampling.GaussLegendreSampling(), 1000)
```

The neighbour graph of a tensor-product sampling is fixed by its axis *lengths* and longitude
wrapping alone, so the axes are never evaluated — for Gauss–Legendre that alone would be an O(n²)
root solve, to produce numbers the answer does not depend on.

## Index topology directly

```@example connectivity
topo = FG.Connectivity.IndexTopology((nlon, nlat), (true, false), nothing)   # mask = nothing → all active
conn = FG.Connectivity.build_connectivity(topo; stencil = FG.Stencils.Axial(1))
```

Useful when you have a shape and a wrapping rule but no grid, and enough to drive every stencil query
in the package.

## Dense and sparse adjacency

```@example connectivity
FG.Connectivity.adjacency_matrix(conn)                       # Matrix{Bool}
FG.Connectivity.adjacency_matrix!(A, conn)                   # into your buffer
# needs SparseArrays; see the Extensions page
# FG.Connectivity.sparse_adjacency_matrix(grid)
```

!!! warning "The dense form is N × N"
    `adjacency_matrix` on a 1000² grid asks for a 10⁶ × 10⁶ `Matrix{Bool}` — 10¹² bytes. It exists for
    small grids and for testing. Use `sparse_adjacency_matrix` for anything real.

The sparse path builds CSC straight from the neighbour list by one counting pass and one placement
pass — no coordinate triples, no sort, no permutation vector, and row indices come out ascending for
free because the placement walks nodes in order.

For a **symmetric** adjacency with sorted rows the CSR and CSC arrays are the *same* arrays, so
`sparse_adjacency_matrix(grid)` on a structured or curvilinear grid hands the connectivity's own
buffers to the matrix rather than transposing into a second copy.

```@example connectivity
FG.Connectivity.is_symmetric_adjacency(conn)   # what licenses that shortcut
FG.Connectivity.sort_neighbors!(conn)          # order each node's block ascending, in place
```

## Threading

Every builder takes an opt-in `backend`; see [Performance](@ref performance-page).

```julia
using ComputationalBackends: ThreadedBackend
FG.Connectivity.build_connectivity(grid; backend = ThreadedBackend())
```

Results are bit-identical to serial, which the test suite asserts with `==` rather than a tolerance.

## Sampling-specific topology

```@example connectivity
FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(64))              # RING face-table adjacency
FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), n)         # with the gnomonic seam fold
FG.Connectivity.build_connectivity(FG.SphericalSampling.YinYangSampling(), nlon, nlat)
FG.Connectivity.healpix_neighbors!(out, nside, 0)             # one pixel; 0-based pixel ids
```

HEALPix RING neighbours follow the standard face-table algorithm (Górski et al. 2005; Reinecke 2003).
The offsets are walked in ring order, so the emitted ids arrive already ascending for ~99% of pixels
and the dedup pass has almost nothing to move.

## Mask topology

The mask's own shape is a grid property, so it is answered here.

```@example conn
C = FG.Connectivity
geo = FG.Geometry.CartesianGeometry()
m = trues(9, 9)
m[4:6, 4:6] .= false          # an enclosed block
g = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0, m)

C.count_holes(g), C.connected_components(g)[2]
```

```@example conn
count(C.interior(g)), count(C.boundary_cells(g)), count(m)
```

`interior` is the active cells whose whole stencil is active and in range; `boundary_cells` is the
rest of the active set, and the two partition it. `count_holes` counts the connected inactive regions
fully enclosed by active cells — an estimate of the active region's first Betti number. A region that
reaches a non-wrapping edge is outside rather than enclosed, so wrapping a direction can turn one into
the other:

```@example conn
gp = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0, m;
                             periodic = (true, true), period = (9.0, 9.0))
C.count_holes(gp)
```

All of it is dimension-generic:

```@example conn
m3 = trues(7, 7, 7); m3[3:5, 3:5, 3:5] .= false
C.count_holes(FG.Grids.StructuredGrid(geo, 0.0:1.0:6.0, 0.0:1.0:6.0, 0.0:1.0:6.0, m3))
```

## Index helpers

```@example connectivity
FG.Connectivity.linear_index(grid, i, j)      # column-major, i fastest
FG.Connectivity.cartesian_index(grid, lin)    # the inverse
```
