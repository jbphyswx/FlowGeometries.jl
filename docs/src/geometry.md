```@meta
CurrentModule = FlowGeometries.Geometry
```

```@setup geometry
using FlowGeometries: FlowGeometries as FG
sph  = FG.Geometry.SphericalGeometry()
cart = FG.Geometry.CartesianGeometry()
grid = FG.Connectivity.structured_grid(FG.SphericalSampling.GaussLegendreSampling(), 16)
λ, φ, Δλ, Δφ = 0.3, 0.4, 0.01, 0.02
r, Δr = 6.3e6, 1.0e4
```

# [Geometry](@id geometry-page)

A geometry answers one question: **what is the metric?** It carries no points and no topology — only
how to measure distance, area and volume, and what the coordinate directions are called.

```@example geometry
using FlowGeometries: FlowGeometries as FG

cart = FG.Geometry.CartesianGeometry()          # flat metric; carries no spacing
sph  = FG.Geometry.SphericalGeometry()          # default Earth radius
unit = FG.Geometry.SphericalGeometry(1.0)       # unit sphere
wgs  = FG.Geometry.SpheroidGeometry()           # WGS 84 oblate spheroid
```

## Element type

A geometry is parameterized by the float type it computes in, and that type fixes the width of
everything measured against it — coordinates, distances, metric factors. [`float_type`](@ref) reads
it and [`similar_geometry`](@ref) carries a geometry to another width, keeping its shape:

```@example geometry
FG.Geometry.float_type(sph), FG.Geometry.similar_geometry(Float32, wgs)
```

Anything that builds at a chosen width takes the type first, positionally, as `zeros` and `rand` do, so
it takes part in dispatch and the result's element type is known from the signature. Where a
constructor takes both a width and a geometry, the width wins and the geometry is carried to it; where
only a geometry is given, its own width is the one meant:

```@example geometry
g32 = FG.Connectivity.structured_grid(Float32, FG.SphericalSampling.GaussLegendreSampling(), 16)
eltype(FG.Grids.axis(g32, 1)), FG.Geometry.float_type(FG.Grids.grid_geometry(g32))
```

## Coordinate names

The geometry decides what a point's components are called, and the grid types inherit that.
Cartesian is `(x, y[, z])`; spherical is `(λ, φ[, r])` — longitude, **geographic** latitude, radius.

The names are enforced: `grid.x` on a spherical grid raises a `FieldError`. Reading `x` and getting
longitude is the kind of bug that survives code review and shows up as a wrong answer months later.

```@example geometry
FG.Grids.coords(grid, 2, 3)          # (λ = …, φ = …) on a spherical grid
FG.Grids.coordinate_names(grid)      # (:λ, :φ)
```

## Distance

`distance` is great-circle on a sphere and Euclidean on a plane, and takes points in whatever
representation you have — tuples, `NamedTuple`s, or (with the StaticArrays extension) static vectors.

```@example geometry
FG.Geometry.distance(sph, (0.0, 0.0), (π/2, 0.0))     # quarter of the equator
```

The spherical version uses the haversine form, which stays accurate for nearby points where the
law-of-cosines form loses precision to cancellation.

## Cell measures

```@example geometry
FG.Geometry.area_element(sph, φ, Δλ, Δφ)              # R²·cosφ·Δλ·Δφ
FG.Geometry.volume_element(sph, r, φ, Δλ, Δφ, Δr)     # r²·cosφ·Δλ·Δφ·Δr
FG.Geometry.area_element(cart, 2.0, 3.0)              # dx·dy, from the cell's own extents
```

These are the pointwise elements. A grid's per-cell measure is built from them once at construction —
see [Grids](@ref grids-page).

## Cartesian ↔ spherical

```@example geometry
p = FG.Geometry.spherical_to_cartesian(sph, (λ, φ))   # (x = …, y = …, z = …)
FG.Geometry.cartesian_to_spherical(sph, (p.x, p.y, p.z))   # (λ = …, φ = …, r = …)
```

Vectors transform differently from points, because a vector at a point is expressed in that point's
local frame. Those have their own pair:

```@example geometry
# components first, then the position they are expressed at
v = FG.Geometry.vector_to_cartesian(sph, 1.0, -0.5, 0.0, λ, φ)   # (u_λ, u_φ, u_r) → (x, y, z)
FG.Geometry.vector_from_cartesian(sph, v.x, v.y, v.z, λ, φ)      # and back
```

## Tangent-plane geometry

```@example geometry
ê = FG.Geometry.local_tangent_basis(sph, (λ, φ))     # (; λ = ê_λ, φ = ê_φ)
FG.Geometry.project_to_tangent_plane(sph, (λ, φ), (λ + 1e-6, φ))
```

`project_to_tangent_plane` gives a neighbour's offset in the tangent plane at `centre`: the quantity a
finite-difference or structure-function calculation on a sphere differences against.

## Nonuniform derivatives

```@example geometry
FG.Geometry.nonuniform_first_derivative(1.0, 0.0, 4.0, 1.0, 2.0)
```

Second-order accurate on an unequally spaced stencil — the usual three-point formula degrades to
first order when `h₋ ≠ h₊`, which is the common case on a stretched grid.

## A spheroid drives the whole stack

`SpheroidGeometry` overrides `distance` (Vincenty), `area_element`, `volume_element` and
`scale_factors`; the grid, sampling and connectivity layers are inherited.

```@example geometry
FG.Geometry.distance(wgs, (0.0, 0.0), (0.0, π/2))    # quarter meridian, WGS 84
```

```@example geometry
FG.Geometry.prime_vertical_radius(wgs, 0.0), FG.Geometry.meridional_radius(wgs, π/2)
```

Grid directions are `(λ, φ, h)`, with `h` the height **above the ellipsoid** — not the absolute radius
that a spherical grid's third direction carries.

```@example geometry
λ = range(0, 2π; length = 9)[1:8]
φ = range(-π/2, π/2; length = 7)
gs = FG.Grids.StructuredGrid(wgs, λ, φ)
FG.Grids.coordinate_names(gs), FG.Grids.isperiodic(gs, 1), size(gs)
```

The surface element `M(φ)·N(φ)cosφ·Δλ·Δφ` factors, so the measure stays separable and an interior cell
matches the geometry's own element exactly:

```@example geometry
Δλ, Δφ = FG.Grids.spacing(gs, 1), FG.Grids.spacing(gs, 2)
FG.Grids.measure(gs, 3, 4) ≈ FG.Geometry.area_element(wgs, φ[4], Δλ, Δφ)
```

Adding a height direction changes that. The geodetic volume element offsets *both* curvature radii by
`h`, coupling `φ` and `h`, so no product of per-axis factors reproduces it and the measure is stored
dense — which `measure_factors` reports by returning `nothing`:

```@example geometry
g3 = FG.Grids.StructuredGrid(wgs, λ, φ, range(0.0, 2000.0; length = 3))
FG.Grids.measure_factors(gs) !== nothing, FG.Grids.measure_factors(g3) === nothing
```

```@example geometry
Δh = FG.Grids.spacing(g3, 3)
FG.Grids.measure(g3, 3, 4, 2) ≈ FG.Geometry.volume_element(wgs, φ[4], 1000.0, Δλ, Δφ, Δh)
```

## Adding a geometry

Subtype `AbstractCartesianGeometry`, `AbstractSphericalGeometry` or `AbstractEllipsoidalGeometry` and
you inherit the whole grid, sampling and connectivity stack. Each hierarchy asks for its shape
parameters through accessors, so that is all a new geometry has to supply — a sphere defines
[`radius`](@ref), an ellipsoid [`semimajor_axis`](@ref) and [`flattening`](@ref), and no method reads a
field, so store them however you like:

```@example geometry
struct UnitSphere{T} <: FG.Geometry.AbstractSphericalGeometry{T} end
FG.Geometry.radius(::UnitSphere{T}) where {T} = one(T)

u = UnitSphere{Float64}()
gu = FG.Grids.StructuredGrid(u, λ, φ)
sum(FG.Grids.measure(gu)), FG.Geometry.distance(u, (0.0, 0.0), (0.0, π/2))
```

That total is the unit sphere's area to the discretization's accuracy, and the distance is a quarter
great circle — both from the one method above.

## Rotated frames

```@example geometry
rot = FG.Geometry.PoleRotation(0.7, 0.3)     # the frame whose north pole is at (0.7, 0.3)
FG.Geometry.rotate(rot, 0.7, 0.3)            # that pole maps to φ = π/2
```

```@example geometry
FG.Geometry.unrotate(rot, FG.Geometry.rotate(rot, 1.2, -0.4)...)   # round-trips
```

A whole point set rotates in place — the shape a sampling's `spherical_points` output has — and the
array forms allocate nothing:

```@example geometry
λs = [0.1, 1.2, 3.0, 5.5]
φs = [0.0, -0.4, 0.9, 0.2]
FG.Geometry.rotate!(λs, φs, rot)
λs
```

Rotating a rectilinear spherical grid **warps** it, and only its own frame's axes stay separable. The
warping is a formula, so the result keeps the mesh and the rotation and evaluates a cell's position
where it is asked for. `unrotate` is the usual direction, taking a rotated-pole grid's `(λ′, φ′)` axes
to the geographic coordinates of each cell.

```@example geometry
sph = FG.Geometry.SphericalGeometry()
λr = range(0, 2π; length = 25)[1:24]
φr = range(-1.2, 1.2; length = 13)
grot = FG.Grids.unrotate(FG.Grids.StructuredGrid(sph, λr, φr), rot)
typeof(grot).name.name, size(grot), FG.Grids.coordinate_names(grot)
```

The cell measure carries over *exactly*, a rotation being an isometry of the sphere, so the array is
shared; recomputing it from the rotated corners adds roundoff. The index topology carries over too:
same mesh, same neighbours, so a direction that wrapped still wraps.

```@example geometry
gplain = FG.Grids.StructuredGrid(sph, λr, φr)
FG.Grids.measure(grot, 3, 4) == FG.Grids.measure(gplain, 3, 4),
FG.Grids.isperiodic(grot, 1)
```

So rotating costs one `PoleRotation`, where materializing the warp cost two centre arrays, two corner
arrays and a dense copy of that measure:

```@example geometry
Base.summarysize(grot) - Base.summarysize(gplain), sizeof(rot)
```
