```@meta
CurrentModule = FlowGeometries
```

# FlowGeometries.jl

Coordinate metrics, spherical samplings, and grid types for flow-field analysis on the plane and the
sphere.

The package has no dependencies. Everything optional — spatial search, tessellation, sparse output,
static vectors, FFT-based quadrature, device transfer, threading — arrives through package
extensions, so you pay for exactly what you load.

## Three orthogonal choices

![Six spherical samplings](assets/samplings.png)


The central idea is that three questions people usually conflate are kept separate:

| question | answered by | examples |
|---|---|---|
| What is the **metric**? | [`Geometry`](@ref geometry-page) | `CartesianGeometry`, `SphericalGeometry` |
| Where are the **points**? | [`SphericalSampling`](@ref sampling-page) | Gauss–Legendre, HEALPix, cubed sphere, icosahedral, Yin–Yang |
| How is the data **stored and connected**? | [`Grids`](@ref grids-page), [`Connectivity`](@ref connectivity-page) | structured, curvilinear, unstructured |

A Gauss–Legendre *sampling* can live in a structured *grid* under a spherical *geometry*; so can a
lat–lon sampling. Changing one does not force a change in the others.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/jbphyswx/FlowGeometries.jl")
```

## Quick start

The package exports nothing. Import it qualified:

```julia
using FlowGeometries: FlowGeometries as FG
```

A spherical grid from a spectral sampling:

```julia
geo  = FG.SphericalGeometry()                                   # unit-agnostic; default Earth radius
grid = FG.structured_grid(FG.GaussLegendreSampling(), 64)       # 127 × 64 lon/lat

FG.coords(grid, 3, 5)          # (λ = …, φ = …) — named for the geometry, not (x, y)
FG.measure(grid, 3, 5)         # that cell's area
sum(FG.measure(grid)) / (4π * geo.R^2)   # ≈ 1: the cells tile the sphere
```

Nodes and quadrature weights together, from a single solve:

```julia
q = FG.spherical_quadrature(FG.GaussLegendreSampling(), 1024)
q.λ, q.φ, q.w                  # longitudes, latitudes, latitude weights
sum(q.w)                       # 2, i.e. ∫₋₁¹ dμ
```

An unstructured grid on a quasi-uniform sampling, with exact dual-cell areas:

```julia
g = FG.unstructured_grid(FG.IcosahedralSampling(16))   # 2562 nodes
sum(FG.measure(g)) / (4π * FG.SphericalGeometry().R^2) # 1.0 to machine precision
FG.neighbors(g, 1)                                     # this node's neighbours
```

Neighbours on a structured grid, without materializing anything:

```julia
out = Vector{Int}(undef, 8)
n = FG.neighbors!(out, grid, 3, 5; stencil = :vertex)  # writes n indices into `out`
```

## Conventions

- **Nothing is exported.** Call through `FG.` or reach into the submodules
  (`FG.Geometry`, `FG.SphericalSampling`, `FG.Grids`, `FG.Connectivity`, `FG.Execution`).
- **Coordinates are named by the geometry.** A spherical grid has `grid.λ`, `grid.φ`; a Cartesian one
  has `grid.x`, `grid.y`. Asking for the wrong one is an error, never a silent alias for the other.
- **Spherical coordinates are the polar frame** `(λ, φ, r)` — longitude, geographic latitude, radius —
  not east/north.
- **Cells are cell-centred.** A sampling's points are cell centres, and the connectivity treats each
  point as one cell. The two never disagree.
- **`f!` means it writes into your buffers** and allocates nothing beyond its return value.
- **Nonstandard element types work.** `Float32` and `BigFloat` are supported throughout; algorithms
  that cannot reach a given precision defer to ones that can.

## Contents

```@contents
Pages = ["geometry.md", "sampling.md", "grids.md", "connectivity.md", "extensions.md", "performance.md", "api.md"]
Depth = 2
```
