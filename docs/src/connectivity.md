```@meta
CurrentModule = FlowGeometries.Connectivity
```

# [Connectivity](@id connectivity-page)

Connectivity answers: **which cells are neighbours?**

The observation the whole module rests on is that a neighbour computation never looks at a
coordinate. It reads three things — extent per dimension, wrapping per dimension, and which cells are
active — and nothing else. That triple is `IndexTopology`, and it is why a curvilinear grid needs no
separate implementation from a structured one: it is the `N = 2` case of the same algorithm.

## Querying neighbours

Three forms, in increasing order of how much they allocate:

```julia
out = Vector{Int}(undef, 8)
n = FG.neighbors!(out, grid, i, j; stencil = :vertex)   # writes n indices, allocates nothing
k = FG.nneighbors(grid, i, j)                            # just the count
it = FG.neighbors(grid, i, j)                            # lazy iterator, no array
```

`stencil` is `:face` (4 in 2-D, 6 in 3-D) or `:vertex` (8 in 2-D, 26 in 3-D). `active_only = true`
excludes masked-out cells from both ends of an edge.

Prefer `neighbors!` in hot loops. `neighbors` returns an iterator rather than a freshly built vector
per cell, which would cost two heap allocations for every cell visited.

## CSR

For a whole graph at once:

```julia
conn = FG.build_connectivity(grid)                   # or (sampling, args...)
FG.nnodes(conn), FG.nedges(conn)
FG.neighbors(conn, i)                                # a view into the flat array
```

`CSRConnectivity` stores one flat neighbour array plus offsets — never a vector per node, which costs
`nnodes` allocations and turns every later traversal into pointer chasing. Both buffers are typed
independently and accept any `Integer`, so a large mesh can carry `Int32` indices.

Building from a sampling never constructs a grid:

```julia
FG.build_connectivity(FG.GaussLegendreSampling(), 1000)
```

The neighbour graph of a tensor-product sampling is fixed by its axis *lengths* and longitude
wrapping alone, so the axes are never evaluated — for Gauss–Legendre that alone would be an O(n²)
root solve, to produce numbers the answer does not depend on.

## Index topology directly

```julia
topo = FG.IndexTopology((nlon, nlat), (true, false), nothing)   # mask = nothing → all active
conn = FG.build_connectivity(topo; stencil = :face)
```

Useful when you have a shape and a wrapping rule but no grid, and enough to drive every stencil query
in the package.

## Dense and sparse adjacency

```julia
FG.adjacency_matrix(conn)                       # Matrix{Bool}
FG.adjacency_matrix!(A, conn)                   # into your buffer
FG.sparse_adjacency_matrix(grid_or_conn)        # SparseMatrixCSC, needs SparseArrays
FG.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)
FG.sparse_adjacency_csc!(colptr, rowval, conn)  # structure only
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

```julia
FG.is_symmetric_adjacency(conn)   # what licenses that shortcut
FG.sort_neighbors!(conn)          # order each node's block ascending, in place
```

## Threading

Every builder takes an opt-in `backend`; see [Performance](@ref performance-page).

```julia
using ComputationalBackends: ThreadedBackend
FG.build_connectivity(grid; backend = ThreadedBackend())
```

Results are bit-identical to serial, which the test suite asserts with `==` rather than a tolerance.

## Sampling-specific topology

```julia
FG.build_connectivity(FG.HEALPixSampling(64))              # RING face-table adjacency
FG.build_connectivity(FG.CubedSphereSampling(), n)         # with the gnomonic seam fold
FG.build_connectivity(FG.YinYangSampling(), nlon, nlat)
FG.healpix_neighbors!(out, nside, ipix0)                   # one pixel; 0-based pixel ids
```

HEALPix RING neighbours follow the standard face-table algorithm (Górski et al. 2005; Reinecke 2003).
The offsets are walked in ring order, so the emitted ids arrive already ascending for ~99% of pixels
and the dedup pass has almost nothing to move.

## Index helpers

```julia
FG.linear_index(grid, i, j)      # column-major, i fastest
FG.cartesian_index(grid, lin)    # the inverse
```
