```@meta
CurrentModule = FlowGeometries.Geometry
```

# [Geometry](@id geometry-page)

A geometry answers one question: **what is the metric?** It carries no points and no topology — only
how to measure distance, area and volume, and what the coordinate directions are called.

```julia
using FlowGeometries: FlowGeometries as FG

cart = FG.CartesianGeometry(1.0, 1.0)        # dx, dy  (dz optional, for 3-D)
sph  = FG.SphericalGeometry()                 # default Earth radius
unit = FG.SphericalGeometry(1.0)              # unit sphere
```

## Coordinate names

The geometry decides what a point's components are called, and the grid types inherit that.
Cartesian is `(x, y[, z])`; spherical is `(λ, φ[, r])` — longitude, **geographic** latitude, radius.

This is enforced rather than conventional: `grid.x` on a spherical grid is a `FieldError`, not a
silent alias for longitude. Reading `x` and getting longitude is the kind of bug that survives code
review and shows up as a wrong answer months later.

```julia
FG.coords(grid, 2, 3)          # (λ = …, φ = …) on a spherical grid
FG.coordinate_names(grid)      # (:λ, :φ)
```

## Distance

`distance` is great-circle on a sphere and Euclidean on a plane, and takes points in whatever
representation you have — tuples, `NamedTuple`s, or (with the StaticArrays extension) static vectors.

```julia
FG.distance(sph, (0.0, 0.0), (π/2, 0.0))     # quarter of the equator
```

The spherical version uses the haversine form, which stays accurate for nearby points where the
law-of-cosines form loses precision to cancellation.

## Cell measures

```julia
FG.area_element(sph, λ, φ, Δλ, Δφ)           # R²·cosφ·Δλ·Δφ
FG.volume_element(sph, λ, φ, r, Δλ, Δφ, Δr)  # r²·cosφ·Δλ·Δφ·Δr
```

These are the pointwise elements. A grid's per-cell measure is built from them once at construction —
see [Grids](@ref grids-page).

## Cartesian ↔ spherical

```julia
FG.spherical_to_cartesian(sph, (λ, φ))       # (x = …, y = …, z = …)
FG.cartesian_to_spherical(sph, (x, y, z))    # (λ = …, φ = …, r = …)
```

Vectors transform differently from points, because a vector at a point is expressed in that point's
local frame. Those have their own pair:

```julia
FG.vector_to_cartesian(sph, point, vec)
FG.vector_from_cartesian(sph, point, vec)
```

## Tangent-plane geometry

```julia
ê = FG.local_tangent_basis(sph, (λ, φ))      # (; λ = ê_λ, φ = ê_φ) — unit vectors in the polar frame
FG.project_to_tangent_plane(sph, centre, neighbour)
```

`project_to_tangent_plane` gives a neighbour's offset in the tangent plane at `centre`, which is what
finite-difference and structure-function calculations on a sphere actually need.

## Nonuniform derivatives

```julia
FG.nonuniform_first_derivative(y₋, y₀, y₊, h₋, h₊)
```

Second-order accurate on an unequally spaced stencil — the usual three-point formula degrades to
first order when `h₋ ≠ h₊`, which is the common case on a stretched grid.

## Adding a geometry

Subtype `AbstractCartesianGeometry` or `AbstractSphericalGeometry` and you inherit the whole grid,
sampling and connectivity stack. Only override what genuinely differs — an oblate spheroid, for
instance, needs its own `distance` and `area_element` but nothing else.
