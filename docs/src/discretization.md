```@meta
CurrentModule = FlowGeometries.Discretization
```

# [Discretization](@id discretization-page)

The geometric inputs a numerical method needs from a grid: where a point falls, the weights that
interpolate to it, where a cell's faces are, the metric factors of the coordinate system, and the
finite-difference weights of any stencil.

All of it is a function of coordinates alone. **Applying** weights to a field is not here — that needs
a result location, a boundary-condition policy and a halo convention, which are the caller's to
choose, not this package's to impose.

## Centres and faces

```@example disc
using FlowGeometries: FlowGeometries as FG
D = FG.Discretization
A = FG.Axes

x = [0.0, 1.0, 3.0, 6.0]
D.faces(x)
```

`N` centres give `N+1` faces, at the midpoints, with the outermost two extrapolated a half-cell
beyond the end centres. Note that centres do **not** determine faces — the system is underdetermined
by one — so that midpoint rule is a stated convention, the same one the curvilinear corner
reconstruction uses.

A uniform axis keeps its uniformity, so nothing is lost by asking for its faces:

```@example disc
u = A.UniformAxis(0.0, 0.5, 6)
A.isuniform(D.faces(u)), A.spacing(D.faces(u)), length(D.faces(u))
```

```@example disc
D.nodes(u, D.Center()) === u, length(D.nodes(u, D.Face()))
```

## Point location

```@example disc
D.locate(u, 0.2), D.locate(u, 0.4), D.locate(u, -5.0)
```

`locate` returns the cell containing a coordinate, or `0` outside it. Cells are the intervals between
faces, so cell 1 of `u` spans `[-0.25, 0.25)`. It is `O(1)` on a uniform axis — a direct payoff of the
spacing living in the type — and `O(log n)` by bisection on a stretched one. Both storage orders work.

```@example disc
xs = cumsum([0.0, 1.0, 0.3, 2.5, 0.7, 4.0])
D.locate(xs, 2.0), D.nearest_index(xs, 2.0)
```

`nearest_index` always returns a valid index, clamping rather than reporting "outside"; exact ties go
to the lower index, and the uniform closed form and the general scan agree on that.

## Interpolation weights

```@example disc
i, w = D.interpolation_weights(xs, 2.0)
i, w, sum(w)
```

Weights only — applying them is `w[1]*f[i] + w[2]*f[i+1]` at the call site. For higher order on an
arbitrarily spaced axis, `lagrange_weights` is exact for polynomials up to `nodes-1`:

```@example disc
idx, wl = D.lagrange_weights(xs, 2.0, 4)
idx, sum(wl)
```

## Finite-difference weights

`fd_weights` is the recursion of Fornberg (1988), *Math. Comp.* **51**, 699–706. One recursion covers
every case: any derivative order, any node count (hence any order of accuracy), any evaluation point,
and arbitrarily spaced nodes.

```@example disc
D.fd_weights([-1.0, 0.0, 1.0], 0.0, 1)      # centred first difference
```

```@example disc
D.fd_weights([-1.0, 0.0, 1.0], 0.0, 2)      # centred second difference
```

```@example disc
D.fd_weights([-2.0, -1.0, 0.0, 1.0, 2.0], 0.0, 1)   # fourth-order first derivative
```

With `m` nodes the weights are exact for polynomials of degree `m-1`, so the accuracy order is
`m - order`. Nothing about that requires equal spacing:

```@example disc
D.fd_weights([-1.7, -0.4, 0.9], 0.0, 1)     # same order, unequal nodes
```

The axis form centres the stencil on a sample and shifts it inward at a boundary, so the node count —
and therefore the accuracy order — is the same everywhere, rather than degrading at the two ends:

```@example disc
xg = collect(range(0.0, 1.0; length = 11))
idx1, w1 = D.fd_weights(xg, 1, 1, 5)        # at the first sample
idxm, wm = D.fd_weights(xg, 6, 1, 5)        # in the interior
idx1, idxm
```

`FG.Geometry.nonuniform_first_derivative` is the three-node, first-derivative case of the same thing.

## Metric factors

```@example disc
G = FG.Geometry
sph = G.SphericalGeometry(2.0)
G.scale_factors(sph, (0.0, π/3)), G.scale_factors(sph, (0.0, π/3, 5.0))
```

`scale_factors` gives the physical length of a unit coordinate step in each direction — `(R cosφ, R)`
on a sphere's surface, `(r cosφ, r, 1)` with a radius direction, and `1` throughout for a Cartesian
metric. That is what turns a coordinate derivative into a physical one, `∂/∂sᵈ = (1/hᵈ)·∂/∂ξᵈ`, so a
divergence or a curl is assembled from these plus `fd_weights` with its conventions stated at the call
site.

```@example disc
G.jacobian(sph, (0.0, 0.5)), 4cos(0.5)
```

`jacobian` is their product: the volume element per unit coordinate volume, and the quantity the
grid's own cell measure is built from.

An ellipsoidal metric supplies its own curvature radii, and nothing else in the stack changes:

```@example disc
wgs = G.SpheroidGeometry()
G.scale_factors(wgs, (0.0, 0.0)), G.meridional_radius(wgs, π/2) > wgs.a
```
