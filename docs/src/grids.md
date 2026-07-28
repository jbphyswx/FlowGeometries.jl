```@meta
CurrentModule = FlowGeometries.Grids
```

# [Grids](@id grids-page)

A grid answers: **how is the data stored?** It pairs a geometry with coordinates, a per-cell measure
and an activity mask.

| type | coordinates | use |
|---|---|---|
| `StructuredGrid` | one 1-D axis per direction | rectilinear: lat–lon, any tensor-product sampling |
| `CurvilinearGrid` | an `Nx × Ny` array per direction | logically rectangular, geometrically warped |
| `UnstructuredGrid` | one value per node, plus CSR neighbours | HEALPix, icosahedral, cubed sphere, scattered |

## Building one

```julia
using FlowGeometries: FlowGeometries as FG

# From a sampling — the usual route
grid = FG.structured_grid(FG.ClenshawCurtisSampling(), 64)
g    = FG.unstructured_grid(FG.HEALPixSampling(16))

# Or directly
sg = FG.StructuredGrid(FG.SphericalGeometry(), λaxis, φaxis, mask)
cg = FG.CurvilinearGrid(FG.SphericalGeometry(), λ2d, φ2d, mask)
```

## The common interface

Everything below works on all three architectures:

```julia
FG.coords(grid, I...)          # (λ = …, φ = …) or (x = …, y = …)
FG.coords!(out, grid, I...)    # into your buffer
FG.coords(NTuple{2,Float64}, grid, I...)   # a specific storage type
FG.measure(grid, I...)         # cell area / volume / control-volume size
FG.isactive(grid, I...)        # participates in the domain?
FG.neighbors(grid, I...)       # lazy iterator over neighbour indices
size(grid), length(grid), FG.size_tuple(grid)
FG.isperiodic(grid, d), FG.period(grid, d)
```

Coordinates are also reachable by their geometry-correct names — `grid.λ`, `grid.φ` on a sphere;
`grid.x`, `grid.y` on a plane.

## Cell measure is stored factored

On a rectilinear grid every measure this package supports is a product of one factor per axis:
Cartesian `Δx·Δy·Δz`, spherical `R²cosφ·Δλ·Δφ = (Δλ)·(R²cosφ·Δφ)`. So a `StructuredGrid` stores the
factors, not the `∏ Nᵈ` products.

```julia
m = FG.measure(grid)           # a SeparableMeasure — a real AbstractArray
m[3, 5]                        # indexes exactly like the dense outer product
sum(m)                         # ∏ᵈ ∑ᵢ wᵈᵢ — O(∑ Nᵈ), not O(∏ Nᵈ)
FG.measure_factors(grid)       # the per-axis factors, or `nothing`
FG.measure_array(grid)         # materialize densely, if you really need it
```

Indexing, broadcasting and `collect` behave exactly as for the dense array — only the storage
differs, and the values are bit-identical. At 2000² that is **61.0 MiB → 0.046 MiB**, and it is
*faster* on every access pattern measured, because the factors stay in cache while a 61 MiB array is
DRAM-bound. See [Performance](@ref performance-page).

`sum` being O(∑Nᵈ) is also what stops `show` from adding eight million numbers to print one line.

## Masks

```julia
FG.mask(grid)                  # the mask array
FG.isactive(grid, i, j)        # false = excluded (land, say)
```

When every cell participates the mask is `AllActive`, which stores its size and nothing else —
`getindex` folds to a constant and `count` is `length` without a scan. Pass a real `BitArray` or
`Array{Bool}` when you need to exclude cells.

## Curvilinear corners

`CurvilinearGrid` stores cell-vertex arrays as well as centres, and computes exact quadrilateral
areas from them — the spherical excess of the two triangles through the four corner directions on a
sphere, the shoelace area on a plane.

```julia
cg = FG.CurvilinearGrid(geo, λ2d, φ2d, mask; x_corner = λc, y_corner = φc)
FG.corners(cg), FG.corner_coords(cg, i, j)
```

Supply `x_corner`/`y_corner` when your source model ships its own cell-vertex grid; otherwise they
are reconstructed from the centres, which needs at least a 2×2 grid.

## Cell areas on unstructured grids

`unstructured_grid` computes exact cell areas in closed form, dispatched on the sampling — never a
uniform `4πR²/N` fallback, which is right only for equal-area samplings and silently wrong elsewhere
(icosahedral dual cells span a min/max ratio of 0.52).

| sampling | default areas |
|---|---|
| HEALPix | uniform `4πR²/N` — exact by construction |
| cubed sphere | spherical excess of each cell's own panel quadrilateral |
| icosahedral | true dual-cell areas, from the mesh's own triangulation |
| Yin–Yang | lat–lon patch area, `R²Δλ·2sin(Δφ/2)·cosφ` |
| arbitrary points | Voronoi areas, via the Quickhull or DelaunayTriangulation extension |

![Cell area relative to the mean](assets/cell_areas.png)

Only HEALPix is one flat colour. The dark spots on the icosahedral panel are its twelve pentagons —
the smallest cells on the mesh, and the reason a uniform default is off by nearly a factor of two
between the largest and smallest cell there.

All four closed forms sum to `4πR²` to machine precision and need no optional dependency. Pass
`areas = …` to override any of them.

![Yin–Yang overlap](assets/yinyang.png)

!!! note "Yin–Yang panels overlap"
    The two panels overlap by construction, so their cell areas sum to `3√2πR²` — 6.07% more than the
    sphere, at *every* resolution. That excess is the grid's real geometry, not a discretisation
    error. Integrating over both panels needs a partition-of-unity weight for the shared region,
    which is a modelling choice made on top of these areas.
