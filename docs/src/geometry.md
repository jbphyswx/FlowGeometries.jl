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

## Coordinate names

The geometry decides what a point's components are called, and the grid types inherit that.
Cartesian is `(x, y[, z])`; spherical is `(λ, φ[, r])` — longitude, **geographic** latitude, radius.

This is enforced rather than conventional: `grid.x` on a spherical grid is a `FieldError`, not a
silent alias for longitude. Reading `x` and getting longitude is the kind of bug that survives code
review and shows up as a wrong answer months later.

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

`project_to_tangent_plane` gives a neighbour's offset in the tangent plane at `centre`, which is what
finite-difference and structure-function calculations on a sphere actually need.

## Nonuniform derivatives

```@example geometry
FG.Geometry.nonuniform_first_derivative(1.0, 0.0, 4.0, 1.0, 2.0)
```

Second-order accurate on an unequally spaced stencil — the usual three-point formula degrades to
first order when `h₋ ≠ h₊`, which is the common case on a stretched grid.

## Adding a geometry

Subtype `AbstractCartesianGeometry`, `AbstractSphericalGeometry` or `AbstractEllipsoidalGeometry` and
you inherit the whole grid, sampling and connectivity stack. Only override what genuinely differs —
which `SpheroidGeometry` demonstrates: it supplies its own `distance` (Vincenty), `area_element` and
`scale_factors`, and nothing else changes.

```@example geometry
FG.Geometry.distance(wgs, (0.0, 0.0), (0.0, π/2))    # quarter meridian, WGS 84
```

```@example geometry
FG.Geometry.prime_vertical_radius(wgs, 0.0), FG.Geometry.meridional_radius(wgs, π/2)
```

## Rotated frames

```@example geometry
rot = FG.Geometry.PoleRotation(0.7, 0.3)     # the frame whose north pole is at (0.7, 0.3)
FG.Geometry.rotate(rot, 0.7, 0.3)            # that pole maps to φ = π/2
```

```@example geometry
FG.Geometry.unrotate(rot, FG.Geometry.rotate(rot, 1.2, -0.4)...)   # round-trips
```
