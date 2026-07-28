# FlowGeometries.jl

Coordinate metrics, spherical samplings, and grid types — with a **dependency-free core**.

![Six spherical samplings](docs/src/assets/samplings.png)

Eleven samplings, three grid architectures, two metrics — chosen independently of each other. Spatial
search, tessellation, sparse output, static vectors, FFT-based quadrature, device transfer and
threading all arrive through package extensions, so you pay for exactly what you load.

**API style:** `using FlowGeometries: FlowGeometries as FG`, then qualified calls (`FG.coords`,
`FG.Geometry.distance`, …). Nothing is bare-exported into your namespace.

```julia
using FlowGeometries: FlowGeometries as FG

geo  = FG.SphericalGeometry()
grid = FG.structured_grid(FG.GaussLegendreSampling(), 64)   # 127 × 64

FG.coords(grid, 3, 5)                                    # (λ = …, φ = …)
FG.measure(grid, 3, 5)                                   # that cell's area
sum(FG.measure(grid)) / (4π * geo.R^2)                   # ≈ 1
```

## Three orthogonal choices

Geometry (the metric), sampling (where the points are), and grid architecture (how data is stored and
connected) are separate. Dispatch on the abstracts; use the concrete defaults as instances.

## Cell areas are exact, in closed form

![Cell area relative to the mean](docs/src/assets/cell_areas.png)

Every built-in sampling gets true cell areas with **no optional dependency** — spherical excess for
the cubed sphere, dual-cell areas from the mesh's own triangulation for icosahedral, lat–lon patches
for Yin–Yang. A uniform `4πR²/N` is exact only for HEALPix (flat colour above); on an icosahedral
geodesic the largest cell is nearly twice the smallest, so that default would silently corrupt every
area-weighted integral. The dark spots are the twelve pentagons.

```julia
g = FG.unstructured_grid(FG.IcosahedralSampling(16))     # 2562 nodes, Σarea = 4πR² exactly
```

Yin–Yang is the exception worth knowing about: its two panels **overlap by construction**, so the
cell areas sum to 6.07% more than the sphere — at *every* resolution. That is real geometry, not a
discretisation error.

![Yin–Yang overlap](docs/src/assets/yinyang.png)

## Quadrature

![Quadrature exactness and cost](docs/src/assets/quadrature.png)

Nodes and weights come from a single solve — ask for both if you need both:

```julia
q = FG.spherical_quadrature(FG.GaussLegendreSampling(), 1024)   # (; λ, φ, w), Σw = 2
```

Gauss–Legendre holds machine precision out past degree `2N−1`; Driscoll–Healy and Clenshaw–Curtis
lose exactness just after `N−1`. That distinction is what `admits_exact_bandlimited_quadrature`
encodes — spectral analysis forms *products*, so it needs `2·lmax`, not `lmax`.

Gauss–Legendre is `O(n)` by asymptotic expansion (Bogaert 2014; Hale & Townsend 2013), with Newton on
the Bonnet recurrence for small `n` and for element types wider than `Float64`. Equiangular weights
are `O(n log n)` with any FFT backend loaded, and fall back to an exact recurrence without one.

## Connectivity

![HEALPix and icosahedral connectivity](docs/src/assets/connectivity.png)

Topology of a *grid* or of a *sampling that defines a mesh*. A neighbour computation reads only
extent, wrapping and activity — never a coordinate — which is what `IndexTopology` captures, and why
a curvilinear grid needs no separate implementation from a structured one.

| Layout | How connectivity is obtained |
|--------|------------------------------|
| `StructuredGrid` / `CurvilinearGrid` | Index stencil (`:face` / `:vertex`) + per-axis `periodic` |
| `UnstructuredGrid` | CSR stored on the grid |
| Tensor-product samplings (GL, CC, DH, MW, lat–lon) | `build_connectivity(sampling, nlat)` — no grid is built |
| Cubed sphere | `build_connectivity(CubedSphereSampling(), n)` — 6 panels + gnomonic seams |
| Yin–Yang | `build_connectivity(YinYangSampling(), nlon, nlat)` — panel-local |
| HEALPix | `build_connectivity(HEALPixSampling(nside))` — RING topological neighbors |
| Icosahedral | `build_connectivity(IcosahedralSampling(ν))` — geodesic triangulation edges |

`unstructured_grid(sampling, …)` builds points + that CSR in one step. Prefer bang forms on hot paths
(`neighbors!`, `adjacency_matrix!`, `sparse_adjacency_csc!`, `healpix_neighbors!`) — they write into
your buffers and allocate nothing beyond their return value.

## Points and frames

The point type from `coords` is a geometry-named `NamedTuple`: `(x=, y=)` or `(λ=, φ=)`. Asking for
the wrong name is an error, never a silent alias for the other quantity. Escapes:
`coords!(out, …)`, `coords(NTuple{2,T}, …)`, `coords(SVector{2,T}, …)` with StaticArrays.

Spherical vector / tangent helpers use the polar frame (`λ`, `φ`, `r`), not east/north:
`local_tangent_basis` → `(; λ=ê_λ, φ=ê_φ)`, `spherical_to_cartesian` / `cartesian_to_spherical`
↔ `(; x,y,z)` / `(; λ,φ,r)`.

## Storage

A structured grid's cell measure is stored as its per-axis factors, not the `∏ Nᵈ` products — it is a
real `AbstractArray`, so indexing and broadcasting are unchanged, but 2000² costs 0.046 MiB instead
of 61 MiB and `sum` is `O(∑ Nᵈ)`. An all-active mask (`AllActive`) stores only its size.

```julia
FG.measure_factors(grid)     # the factors, or `nothing`
FG.measure_array(grid)       # materialize densely, if you truly need it
```

## Defaults

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

## Extensions

| Load | Unlocks |
|------|---------|
| `NearestNeighbors` | k-d-tree neighbours for `UnstructuredGrid` |
| `Quickhull` | spherical Voronoi areas for arbitrary point sets |
| `DelaunayTriangulation` | planar Voronoi areas |
| `SparseArrays` | `sparse_adjacency_matrix` → CSC |
| `StaticArrays` | `SVector` / `MVector` points end to end |
| `AbstractFFTs` | `O(n log n)` equiangular quadrature weights |
| `Adapt` | move a grid to another storage backend |
| `ComputationalBackends` | opt-in threading, bit-identical to serial |

```julia
using ComputationalBackends: ThreadedBackend
FG.build_connectivity(grid; backend = ThreadedBackend())   # 3.7–4.3× on 8 threads
```

## Documentation

Full docs: <https://jbphyswx.github.io/FlowGeometries.jl>

```julia
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'   # build locally
julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl      # regenerate figures
```
