```@meta
CurrentModule = FlowGeometries.Axes
```

# [Axes](@id axes-page)

An axis answers: **where along this direction are the samples, and is the spacing constant?**

That second half is a property of the axis's *type*, not of its values, so the fast paths that depend
on constant spacing can be selected at compile time.

## Uniform or not

```@example axes
using FlowGeometries: FlowGeometries as FG
A = FG.Axes

u = A.UniformAxis(0.0, 0.25, 5)     # origin, Δ, n
v = collect(u)                       # the same numbers, as a plain Vector

A.isuniform(u), A.isuniform(v)
```

`isuniform` is the compile-time question, and it is the only one this package asks. A `Vector` holding
an arithmetic sequence answers `false` because nothing in its type says otherwise, and **no code path
inspects coordinate values to decide otherwise** — no constructor sniffs your data, and there is no
tolerance anywhere.

Where the data question genuinely matters, the spacing accessors answer it exactly: an axis is equally
spaced precisely when its smallest gap equals its largest.

```@example axes
geo = FG.Geometry.CartesianGeometry()
gv = FG.Grids.StructuredGrid(geo, v, v)
FG.Grids.minimum_spacing(gv, 1) == FG.Grids.maximum_spacing(gv, 1)
```

If that holds and you want the fast path, build the axis and say so — the conversion replaces your
coordinates with the exact arithmetic sequence, which is your call to make, not this package's:

```@example axes
Δ = FG.Grids.minimum_spacing(gv, 1)            # equals maximum_spacing, so it IS the spacing
A.spacing(A.UniformAxis(first(v), Δ, length(v)))
```

A [`UniformAxis`](@ref) stores three numbers and computes `origin + (i-1)·Δ` in its own element type.
That matters against `range`: `range(0f0; step = 0.25f0, length = 5)` is a
`StepRangeLen{Float32, Float64, Float64, Int}` — `Float32` elements over a `Float64` offset and step —
so a `Float32` grid built from one would do `Float64` index arithmetic.

```@example axes
typeof(range(0.0f0; step = 0.25f0, length = 5)), typeof(A.uniform_axis(Float32, range(0.0f0; step = 0.25f0, length = 5)))
```

`Δ` may be negative, for an axis stored descending — which is routine, and which nothing in the
package assumes away:

```@example axes
d = A.UniformAxis(1.0, -0.25, 5)
A.spacing(d), extrema(d), collect(d)
```

A slice, a reversal, or any affine map of a uniform axis is still uniform, so the guarantee is not
lost by manipulating it:

```@example axes
A.isuniform(u[2:4]), A.isuniform(reverse(u)), A.isuniform(2.0 .* u .+ 1.0)
```

Anything that is not affine becomes a plain array, as it must:

```@example axes
A.isuniform(cos.(u))
```

## It is an `AbstractRange`

```@example axes
u isa AbstractRange, step(u)
```

Two things follow. Base solves `searchsorted` on a range in closed form rather than by bisection, so
lookup cost does not grow with the axis length — measured flat at 42 ns over `n = 10 … 10⁷`, against
35→69 ns for the equivalent `Vector`. And any package that already tests `isa AbstractRange` to pick a
uniform-grid fast path accepts this type without knowing it exists.

```@example axes
big = A.UniformAxis(0.0, 1e-7, 10^7)
searchsortedfirst(big, 0.5), searchsortedfirst(big, 0.5) == searchsortedfirst(collect(0.0:1e-7:0.9999999), 0.5)
```

## What it buys

`minimum`, `maximum`, `extrema` and `sum` are closed forms rather than scans, and the axis is
`isbits`, so moving it to another storage backend costs nothing.

```@example axes
sum(u), extrema(u), isbits(u), sizeof(u)
```

On a grid the payoff is the cell measure. A uniform axis's per-cell width is one number repeated, so
it is stored as a [`ConstantVector`](@ref) rather than `N` copies of the same value:

```@example axes
geo = FG.Geometry.CartesianGeometry()
N = 2000
g = FG.Grids.StructuredGrid(geo, range(0.0; step = 0.5, length = N),
                            range(0.0; step = 0.25, length = N))
typeof.(FG.Grids.measure_factors(g)), Base.summarysize(g)
```

That whole 2000×2000 grid is a few hundred bytes. A stretched axis is fully supported and simply
stores what it must:

```@example axes
xs = cumsum([0.0, 1.0, 0.3, 2.5, 0.7])
gs = FG.Grids.StructuredGrid(geo, xs, xs)
FG.Grids.isuniform(gs), typeof(FG.Grids.measure_factors(gs)[1])
```

## Grid accessors

```@example axes
FG.Grids.isuniform(g, 1), FG.Grids.spacing(g, 1), FG.Grids.origin(g, 1),
FG.Grids.bounds(g, 1), FG.Grids.extent(g, 1),
FG.Grids.minimum_spacing(gs, 1), FG.Grids.maximum_spacing(gs, 1)
```

`minimum_spacing` and `maximum_spacing` are `O(1)` on a uniform direction and `O(N)` otherwise; they
bound how far an index window must reach to cover a given physical distance, which is what a
neighbourhood-by-distance query needs on a stretched axis.
