```@meta
CurrentModule = FlowGeometries.Grids
```

```@setup grids
using FlowGeometries: FlowGeometries as FG
geo    = FG.Geometry.SphericalGeometry()
ax     = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.ClenshawCurtisSampling(), 32)
λaxis, φaxis = ax.λ, ax.φ
mask   = trues(length(λaxis), length(φaxis))
grid   = FG.Grids.StructuredGrid(geo, λaxis, φaxis, mask)
λ2d    = [λaxis[i] for i in eachindex(λaxis), j in eachindex(φaxis)]
φ2d    = [φaxis[j] for i in eachindex(λaxis), j in eachindex(φaxis)]
λc, φc = FG.Grids._centers_to_corners(λ2d), FG.Grids._centers_to_corners(φ2d)
cg     = FG.Grids.CurvilinearGrid(geo, λ2d, φ2d, mask)
I      = (3, 5)
i, j   = 3, 5
```

# [Grids](@id grids-page)

A grid answers: **how is the data stored?** It pairs a geometry with coordinates, a per-cell measure
and an activity mask.

| type | coordinates | use |
|---|---|---|
| `StructuredGrid` | one 1-D axis per direction | rectilinear: lat–lon, any tensor-product sampling |
| `CurvilinearGrid` | one `N`-D array per direction | logically rectangular, geometrically warped |
| `HEALPixGrid` | none — `(nside, pixel)` arithmetic | the HEALPix pixelization |
| `RingGrid` | one value per *ring* | reduced Gaussian, octahedral |
| `UnstructuredGrid` | one value per node, plus CSR neighbours | icosahedral, cubed sphere, scattered |

The first two store coordinates because a warped mesh's are arbitrary; the last stores them because a
node set's are the caller's. The middle two store none: a cell's position, its neighbours and its area
are closed-form in the layout's parameters, so they hold those parameters instead. `HEALPixGrid` is the
same size at `nside = 1024` — 12.6 million pixels — as at `nside = 1`. Ask for the dense cloud with
[`Grids.materialize`](@ref) when something outside needs it.

## Building one

```@example grids
using FlowGeometries: FlowGeometries as FG

# From a sampling — the usual route
grid = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 64)
g    = FG.Grids.HEALPixGrid(16)
rg   = FG.Grids.RingGrid(FG.SphericalSampling.OctahedralGaussianSampling(64))

# Or directly, in any number of dimensions; the mask is optional
sg  = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(), λaxis, φaxis, mask)
cg  = FG.Grids.CurvilinearGrid(FG.Geometry.SphericalGeometry(), λ2d, φ2d, mask)
g4  = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(),
                              0:1.0:3, 0:1.0:3, 0:1.0:3, 0:1.0:3)  # 4-D, all active
size(g4)
```

A formula layout's storage does not depend on its resolution, and the arithmetic is what answers every
per-cell question:

```@example grids
g1024 = FG.Grids.HEALPixGrid(1024)                 # 12 582 912 pixels
Base.summarysize(g) == Base.summarysize(g1024), length(g1024)
```

```@example grids
FG.Grids.coords(g, 100), FG.Grids.measure(g, 100), collect(FG.Grids.neighbors(g, 100))
```

```@example grids
λ, φ = FG.Grids.materialize(g)                     # the dense cloud, asked for explicitly
length(λ)
```

## The common interface

Everything below works on every architecture:

```@example grids
out = zeros(2)
FG.Grids.coords(grid, i, j)                        # (λ = …, φ = …) or (x = …, y = …)
FG.Grids.coords!(out, grid, i, j)                  # into your buffer
FG.Grids.coords(NTuple{2,Float64}, grid, i, j)     # a specific storage type
FG.Grids.measure(grid, i, j)                       # cell area / volume / control-volume size
FG.Grids.isactive(grid, i, j)                      # participates in the domain?
collect(FG.Grids.neighbors(grid, i, j))            # lazy iterator over neighbour indices
size(grid), length(grid), FG.Grids.size_tuple(grid)
FG.Grids.isperiodic(grid, 1), FG.Grids.period(grid, 1)
```

Coordinates are also reachable by their geometry-correct names — `grid.λ`, `grid.φ` on a sphere;
`grid.x`, `grid.y` on a plane.

## Cell measure is stored factored

On a rectilinear grid every measure this package supports is a product of one factor per axis:
Cartesian `Δx·Δy·Δz`, spherical `R²cosφ·Δλ·Δφ = (Δλ)·(R²cosφ·Δφ)`. So a `StructuredGrid` stores the
factors, not the `∏ Nᵈ` products.

```@example grids
m = FG.Grids.measure(grid)           # a SeparableMeasure — a real AbstractArray
m[3, 5]                        # indexes exactly like the dense outer product
sum(m)                         # ∏ᵈ ∑ᵢ wᵈᵢ — O(∑ Nᵈ), not O(∏ Nᵈ)
FG.Grids.measure_factors(grid)       # the per-axis factors, or `nothing`
FG.Grids.measure_array(grid)         # materialize densely, if you really need it
```

Indexing, broadcasting and `collect` behave exactly as for the dense array — only the storage
differs, and the values are bit-identical. At 2000² that is **61.0 MiB → 0.046 MiB**, and it is
*faster* on every access pattern measured, because the factors stay in cache while a 61 MiB array is
DRAM-bound. See [Performance](@ref performance-page).

Operations that keep the measure a *product* stay factored, so a unit conversion does not undo the
saving: scaling, a multiplicative map (`abs`, `abs2`, `sqrt`, `inv`), and a factor-wise product or
quotient of two measures.

```@example grids
km² = m ./ 1e6
typeof(km²).name.name, Base.summarysize(km²), sum(km²) ≈ sum(m) / 1e6
```

Anything else materializes — correct, just dense. `exp` is the clean example: it is not multiplicative,
so `exp(∏wᵈ) ≠ ∏exp(wᵈ)` and there is no factored form to keep. A *negative* scale also materializes
deliberately: non-negative factors are an invariant the `findmax`/`findmin` shortcut relies on.

```@example grids
typeof(exp.(m)).name.name, typeof((-1.0) .* m).name.name
```

`sum` being O(∑Nᵈ) is also what stops `show` from adding eight million numbers to print one line.

## Masks

```@example grids
FG.Grids.mask(grid)                  # the mask array
FG.Grids.isactive(grid, i, j)        # false = excluded (land, say)
```

When every cell participates the mask is `AllActive`, which stores its size and nothing else —
`getindex` folds to a constant and `count` is `length` without a scan. Pass a real `BitArray` or
`Array{Bool}` when you need to exclude cells.

## Curvilinear corners

`CurvilinearGrid` stores cell-vertex arrays as well as centres, and computes exact quadrilateral
areas from them — the spherical excess of the two triangles through the four corner directions on a
sphere, the shoelace area on a plane.

```@example grids
cg2 = FG.Grids.CurvilinearGrid(geo, λ2d, φ2d, mask; x_corner = λc, y_corner = φc)
size(FG.Grids.corners(cg2, 1)), FG.Grids.corner_coords(cg2, i, j)
```

Supply `x_corner`/`y_corner` when your source model ships its own cell-vertex grid; otherwise they
are reconstructed from the centres, which needs at least a 2×2 grid.

A curvilinear grid takes any number of directions — one `N`-D array each, `mask` last. Beyond 2-D the
cell measure is yours to pass: the corner-area kernel is an exact-quadrilateral algorithm, not the 2-D
case of an N-D one, so asking for a 3-D measure it cannot compute is an error rather than a number from
the wrong formula.

```@example grids
cart = FG.Geometry.CartesianGeometry()
X = [x for x in 0.0:1.0:3.0, _ in 1:3, _ in 1:2]
Y = [y for _ in 1:4, y in 0.0:2.0:4.0, _ in 1:2]
Z = [z for _ in 1:4, _ in 1:3, z in 0.0:0.5:0.5]
cg3 = FG.Grids.CurvilinearGrid(cart, X, Y, Z, fill(1.0, 4, 3, 2), trues(4, 3, 2))
size(cg3), FG.Grids.coordinate_names(cg3), FG.Grids.coords(cg3, 2, 3, 2)
```

The measure goes either before the mask, as above, or as the `measure` keyword — the same array
either way:

```@example grids
FG.Grids.measure(FG.Grids.CurvilinearGrid(cart, X, Y, Z, trues(4, 3, 2);
                                          measure = fill(1.0, 4, 3, 2))) ==
    FG.Grids.measure(cg3)
```

Node grids generalize the same way, with the coordinates as a tuple — a run of vectors cannot say how
many of them are coordinates, where a curvilinear grid counts them from `ndims(mask)`:

```@example grids
nodes = FG.Grids.UnstructuredGrid(cart, (rand(6), rand(6), rand(6)), ones(6), trues(6))
FG.Grids.coordinate_names(nodes), FG.Grids.coords(nodes, 4)
```

## Cell areas on the spherical node sets

Cell areas are an exact closed form, dispatched on the sampling — never a uniform `4πR²/N` fallback,
which is right only for equal-area samplings and silently wrong elsewhere (icosahedral dual cells span
a min/max ratio of 0.52).

| layout / sampling | default areas |
|---|---|
| `HEALPixGrid` | uniform `4πR²/N` — exact by construction, and stored as one number |
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
