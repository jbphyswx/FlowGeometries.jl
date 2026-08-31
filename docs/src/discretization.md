```@meta
CurrentModule = FlowGeometries.Discretization
```

# [Discretization](@id discretization-page)

The geometric inputs a numerical method needs from a grid: where a point falls, the weights that
interpolate to it, where a cell's faces are, the metric factors of the coordinate system, and the
finite-difference weights of any stencil.

All of it is a function of coordinates alone. **Applying** weights to a field is not here: that needs a
result location, a boundary-condition policy and a halo convention, all of them the caller's to choose.

## Centres and faces

```@example disc
using FlowGeometries: FlowGeometries as FG
D = FG.Discretization
O = FG.Operators
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

## Gaps and cell widths

[`local_spacing`](@ref) is the gap either side of one sample, and [`cell_width`](@ref) the width of one
cell. Both are a scalar subtraction of two stored numbers — no array is built — so they are the forms
to call per grid point, where `faces` would materialize the whole axis to answer about one cell.

```@example disc
xs = cumsum([0.0, 1.0, 0.3, 2.5, 0.7, 4.0])
D.local_spacing(xs, 3), D.cell_width(xs, 3)
```

`cell_width` is the distance between the cell's two faces:

```@example disc
f = D.faces(xs)
[D.cell_width(xs, i) for i in eachindex(xs)] ≈ abs.(diff(f))
```

The gaps are **signed** and the width is not. A derivative needs the sign, since it distinguishes an
axis stored increasing from one stored decreasing; a width is a length and cannot be negative. Reverse
the axis and cell `i` moves to `n+1-i`, where the gaps come back negated and swapped — the same two
neighbours, now on the other side — while the width is unchanged:

```@example disc
xr = reverse(xs)                              # cell 3 of xs is cell 4 of xr
D.local_spacing(xr, 4), D.cell_width(xr, 4)
```

At a bounded end the outward gap is `0` and the caller falls back to a one-sided stencil. Given a
`period` it wraps, so a seam behaves like the interior:

```@example disc
λ = collect(range(0, 2π * (1 - 1/8); length = 8))
D.local_spacing(λ, 8), D.local_spacing(λ, 8, 2π)
```

[`cell_widths`](@ref) is the whole axis at once. A uniform axis returns an
[`Axes.ConstantVector`](@ref) — one number and a length — so nothing is materialized for it either:

```@example disc
D.cell_widths(u), D.cell_widths(xs)
```

On a grid, `Grids.local_spacing`, `Grids.cell_width` and `Grids.cell_widths` take the direction's axis
and its wrap period from the grid, so a periodic seam is right without being asked for — see the
[Grids](@ref grids-page) page.

## Point location

```@example disc
D.locate(u, 0.2), D.locate(u, 0.4), D.locate(u, -5.0)
```

`locate` returns the cell containing a coordinate, or `0` outside it. Cells are the intervals between
faces, so cell 1 of `u` spans `[-0.25, 0.25)`. It is `O(1)` on a uniform axis — a direct payoff of the
spacing living in the type — and `O(log n)` by bisection on a stretched one. Both storage orders work.

```@example disc
D.locate(xs, 2.0), D.nearest_index(xs, 2.0)
```

`nearest_index` always returns a valid index, clamping a coordinate outside the axis to the nearer end;
exact ties go to the lower index. It brackets and compares, so it is `O(log n)` on a stretched axis and
`O(1)` on a uniform one, and both paths compare the same two samples: a uniform axis and its `collect`
hold the same numbers and agree about which is nearest.

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
and therefore the accuracy order — holds at the two ends as well as the interior:

```@example disc
xg = collect(range(0.0, 1.0; length = 11))
idx1, w1 = D.fd_weights(xg, 1, 1, 5)        # at the first sample
idxm, wm = D.fd_weights(xg, 6, 1, 5)        # in the interior
idx1, idxm
```

`FG.Geometry.nonuniform_first_derivative` is the three-node, first-derivative case of the same thing.
Fed the gaps above it is a complete derivative at a point, which is the shape an operator assembled at
the call site takes — a divergence, a curl, a staggered difference are each this plus the metric
factors and whatever boundary policy the caller chose:

```@example disc
g(x) = 3x^2 - 2x + 5                        # exact for a quadratic, on any spacing
maximum(abs(FG.Geometry.nonuniform_first_derivative(
                g(xs[i-1]), g(xs[i]), g(xs[i+1]), D.local_spacing(xs, i)...) - (6xs[i] - 2))
        for i in 2:length(xs)-1)
```

Signed gaps are what carry that through: the same expression on a descending axis differentiates with
respect to the coordinate, with no correction at the call site:

```@example disc
maximum(abs(FG.Geometry.nonuniform_first_derivative(
                g(xr[i-1]), g(xr[i]), g(xr[i+1]), D.local_spacing(xr, i)...) - (6xr[i] - 2))
        for i in 2:length(xr)-1)
```

## Applying a weight set

[`apply_stencil!`](@ref FlowGeometries.Operators.apply_stencil!) is the one function here that touches a field, and only along a single
direction with the result left where the input was. That case needs no convention the package has not
already fixed: nothing to stagger, `fd_weights`' inward shift at a bounded end, wrapping on a periodic
one — so no halo either.

```@example disc
x = collect(range(0.0, 2.0; length = 11))
f = @. 3x^2 - 2x + 5
out = similar(f)
O.apply_stencil!(out, f, x, 1; order = 1, nodes = 3)
maximum(abs, out .- (6x .- 2))          # exact for a quadratic, ends included
```

A stretched axis is equally exact, the weights being built per sample:

```@example disc
xs = [0.0, 0.11, 0.37, 0.9, 1.05, 1.6, 1.62, 2.0]
outs = similar(xs)
O.apply_stencil!(outs, (@. 3xs^2 - 2xs + 5), xs, 1; order = 1, nodes = 3)
maximum(abs, outs .- (6xs .- 2))
```

Given a `period` the stencil stays centred and wraps, carrying the wrapped samples' coordinates across
the seam so the spacing there is the true one — the seam is then no worse than the interior:

```@example disc
λ = collect(range(0, 2π; length = 65)[1:64])
o = similar(λ)
O.apply_stencil!(o, sin.(λ), λ, 1; order = 1, nodes = 5, period = 2π)
maximum(abs, o .- cos.(λ)), abs(o[1] - cos(λ[1]))
```

The grid form takes the axis, the wrap period and the mask from the grid, so a periodic direction wraps
without being told. Where a mask bites, a cell whose stencil reads an inactive sample is written as
`masked`:

```@example disc
geo = FG.Geometry.CartesianGeometry()
X = collect(range(0.0, 1.0; length = 9)); Y = collect(range(0.0, 2.0; length = 7))
F = [xi^2 + 3yi for xi in X, yi in Y]
mk = trues(9, 7); mk[5, 3] = false
gm = FG.Grids.StructuredGrid(geo, X, Y, mk)
Om = similar(F)
O.apply_stencil!(Om, F, gm, 1; order = 1, nodes = 3, masked = NaN)
Om[3:7, 3]                               # the masked cell and the two that read it
```

### What a mask edge does to the stencil

Blanking is the default, and it is not free: a cell is blanked when *any* sample its window reads is
inactive, so a single masked cell takes out up to `nodes - 1` cells either side of it. A five-point
derivative on a short axis can be annihilated outright by one hole.

The window already shifts inward at the end of an axis, precisely so the node count — and the accuracy
order — survives a boundary. The end of an active *run* is the same situation, and
[`ShiftWithinRun`](@ref FlowGeometries.Operators.ShiftWithinRun) treats it that way:

```@example disc
xr = collect(0.0:1.0:6.0)
mr = trues(7, 1); mr[4, 1] = false
gr = FG.Grids.StructuredGrid(geo, xr, [0.0], mr)
fr = reshape(collect(0.0:6.0), 7, 1)         # f = x, so df/dx is exactly 1

blanked  = zeros(7, 1); shifted = zeros(7, 1)
O.apply_stencil!(blanked, fr, gr, 1; order = 1, nodes = 5, masked = NaN)
O.apply_stencil!(shifted, fr, gr, 1; order = 1, nodes = 5, masked = NaN,
                 policy = O.ShiftWithinRun())
vec(blanked), vec(shifted)
```

Every cell is blanked in the first, and every active cell is exact in the second. The policies are:

| policy | at a run edge |
|---|---|
| [`BlankMasked`](@ref FlowGeometries.Operators.BlankMasked) | the default — write `masked` wherever the window reads an inactive sample |
| [`ShiftWithinRun`](@ref FlowGeometries.Operators.ShiftWithinRun) | shift the window to fit inside the run, keeping `nodes` and the accuracy order; `masked` only where the run is shorter than `nodes` |
| [`ReduceInRun`](@ref FlowGeometries.Operators.ReduceInRun) | as above, and where the run cannot hold `nodes`, use the largest window it can, down to `order + 1` |

`ReduceInRun` is the one policy that lowers the accuracy order without saying so, hence its own name: a
narrow strait wants an answer at reduced order, and elsewhere `ShiftWithinRun` reports a run too short
by writing `masked`.

A cell whose window already lies inside its run reuses the precomputed row, so a run's interior is
bit-for-bit what `BlankMasked` gives, and only the cells within `nodes - 1` of an edge cost anything
extra. A run that wraps a periodic seam is a single run.

Build the weights once with [`axis_stencils`](@ref) to reuse them across many fields; applying a
precomputed set allocates nothing.

```@example disc
idx, w = D.axis_stencils(X, 1, 3)
size(idx), size(w)
```

## The physical derivative

`apply_stencil!` differentiates with respect to a **coordinate**. On any curved geometry that is not
the derivative a physical law is written in — that one is per unit *distance*, `∂f/∂sᵈ = (1/hᵈ)·∂f/∂ξᵈ`.
[`derivative!`](@ref FlowGeometries.Operators.derivative!) is the two together:

```@example disc
R   = 6.371e6
sph = FG.Geometry.SphericalGeometry(R)
lon = collect(range(0, 2π * (1 - 1/48); length = 48))
lat = collect(range(-π/2, π/2; length = 25))          # both poles are rows of this grid
gs  = FG.Grids.StructuredGrid(sph, lon, lat)
fs  = [sin(φ) for _ in lon, φ in lat]                 # ∂/∂north should be cos(φ)/R
dn  = zeros(48, 25)
O.derivative!(dn, fs, gs, 2; order = 1, nodes = 5, masked = NaN)
dn[1, 12], cos(lat[12]) / R
```

Where the metric degenerates the derivative does not exist, and `masked` is written. Longitude at a pole
is that case — `h_λ = R cos φ → 0`:

```@example disc
de = zeros(48, 25)
O.derivative!(de, [sin(λ) * cos(φ) for λ in lon, φ in lat], gs, 1;
              order = 1, nodes = 5, masked = NaN)
all(isnan, de[:, 1]), all(isnan, de[:, 25]), any(isnan, de[:, 2:24])
```

The threshold is [`metric_floor`](@ref), `L·√eps(T)` — relative to the geometry's size *and* to the
element type. An absolute constant cannot serve both: `1e-12` is below `eps(Float32)`, and in `Float32`
`cos(Float32(π/2)) ≈ -4.4e-8`, so `h_λ` at the pole is around `0.28` metres and a fixed small threshold
never fires.

A divergence or a curl remains the caller's to assemble, needing a result location and a
boundary-condition policy this does not choose. When you do, note the **flux form** — on a sphere

```math
\nabla\cdot\mathbf{u} = \frac{1}{R\cos\varphi}\left[\frac{\partial u_\lambda}{\partial\lambda}
                        + \frac{\partial (u_\varphi\cos\varphi)}{\partial\varphi}\right]
```

so the second term differentiates `u_φ·cos φ`, not `u_φ`. Adding two physical derivatives is a
different expression, and a wrong one.

## Off a rectilinear grid: the least-squares gradient

`apply_stencil!` needs a separable axis to difference along. A `CurvilinearGrid` has none, and a node
set's neighbours come from connectivity, so neither has a stencil.
`Operators.gradient_plan` builds a least-squares gradient for them, and
[`gradient!`](@ref FlowGeometries.Operators.gradient!) applies it:

```@example disc
cart = FG.Geometry.CartesianGeometry{Float64}()
nn   = 14
xg   = [t + 0.35u for t in range(0, 10; length = nn), u in range(0, 6; length = nn)]   # sheared
yg   = [u - 0.2t  for t in range(0, 10; length = nn), u in range(0, 6; length = nn)]
cgrid = FG.Grids.CurvilinearGrid(cart, xg, yg, trues(nn, nn); measure = fill(1.0, nn, nn))
plan  = FG.Operators.gradient_plan(cgrid)

fld = 2.0 .* xg .- 3.0 .* yg .+ 7        # ∇ = (2, -3) everywhere
g1, g2 = zeros(nn, nn), zeros(nn, nn)
O.gradient!(g1, g2, fld, plan)
maximum(abs, g1 .- 2), maximum(abs, g2 .+ 3)
```

Exact for a linear field on **any** stencil, however skewed — the least-squares combination cancels the
leading truncation term, which inverting a 2×2 index-space Jacobian does not. It is second order on
locally symmetric stencils and degrades toward first on strongly skewed cells, and where the stencil is
separable and orthogonal it *is* the centred difference, so it agrees with `apply_stencil!` where both
apply.

`A` depends only on the geometry, so the plan holds the per-neighbour coefficients and each apply is
one dot product per cell, allocating nothing. Where `A` is rank deficient — every neighbour on one
line, at a boundary or beside a mask — that component is **zeroed**, under the rule `apply_stencil!`
states for a mask.

## Evaluating a field at a coordinate

Observational data arrives with a coordinate. [`interpolate`](@ref FlowGeometries.Operators.interpolate) evaluates a field there:

```@example disc
xa = collect(range(0, 2; length = 11)); ya = collect(range(-1, 3; length = 9))
ga = FG.Grids.StructuredGrid(cart, xa, ya)
fa = [2xi - 3yj + 0.5xi * yj + 7 for xi in xa, yj in ya]
O.interpolate(fa, ga, (0.7, 1.1)), 2*0.7 - 3*1.1 + 0.5*0.7*1.1 + 7
```

Multilinear on a rectilinear grid; a weighted least-squares plane on a curvilinear grid or a node set,
which is exact for a linear field and reproduces a cell's own value at its centre. A **periodic
direction interpolates across its seam**, so a coordinate past the last sample wraps to the first.

The mask policies mean here what they mean for a stencil: [`BlankMasked`](@ref FlowGeometries.Operators.BlankMasked) returns `masked` when a
contributor is inactive, [`ReduceInRun`](@ref FlowGeometries.Operators.ReduceInRun) renormalizes over the active ones, and
[`ShiftWithinRun`](@ref FlowGeometries.Operators.ShiftWithinRun) is refused, there being no window to shift.

Anything that *does* need a convention the package has not chosen — a staggered difference, or a
multi-direction operator like a divergence or a curl, which also need a result location and a
boundary-condition policy — is assembled at the call site from these weights and the metric factors
below.

## A batch of fields on one grid

A field may carry trailing axes beyond the grid's own — many tracers, an ensemble, a time window —
sharing one geometry. Those axes are **batch**: every entry point takes them, differencing only along
`dim`, which indexes the grid's directions.

```@example disc
gb = FG.Grids.StructuredGrid(cart, xa, ya)
fb = cat((fa .+ 100k for k in 1:4)...; dims = 3)      # (11, 9, 4) against a 2-D grid
ob = similar(fb)
O.derivative!(ob, fb, gb, 1; order = 1, nodes = 3)
size(ob), ob[5, 4, 1] ≈ ob[5, 4, 4]                   # same derivative, offset field
```

Nothing about the grid depends on the batch, so the grid-only work — the stencil table, the metric
factors, the interpolation weights, a gradient plan's coefficients — is computed once and reused across
it. One pass over the batch is therefore *less* work than a pass per slice.

It also matters for a device backend. `apply_stencil!` launches over the whole output, batch included,
so a batched call is one launch over `prod(spatial) · prod(batch)` work items where a slice loop is one
launch each over `prod(spatial)`. A 64×64 slice is 4096 work items, well short of saturating a GPU; the
batch is the axis with the parallelism to fill it.

Evaluating a batch at a coordinate gives one value per element, through the `!` form or its allocating
wrapper as elsewhere in the package:

```@example disc
out = Vector{Float64}(undef, 4)
O.interpolate!(out, fb, gb, (0.7, 1.1))
out ≈ O.interpolate(fb, gb, (0.7, 1.1))               # the allocating form agrees
```

An unbatched field still returns a scalar — the rank decides, so nothing is inferred from a length at
run time. A mask stays the grid's own shape and applies to every element; a field whose *spatial* extent
disagrees with the grid is still an error, and so is asking to difference along a batch axis.

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
