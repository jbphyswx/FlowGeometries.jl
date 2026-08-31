```@meta
CurrentModule = FlowGeometries.Axes
```

# [Axes](@id axes-page)

An axis answers: **where along this direction are the samples, and is the spacing constant?**

The second half is answered by the axis's *type*, so the fast paths that depend on constant spacing are
selected at compile time.

## Uniform or not

```@example axes
using FlowGeometries: FlowGeometries as FG
A = FG.Axes

u = A.UniformAxis(0.0, 0.25, 5)     # origin, Δ, n
v = collect(u)                       # the same numbers, as a plain Vector

A.isuniform(u), A.isuniform(v)
```

`isuniform` reads the type, and that is the only question this package asks. A `Vector` holding an
arithmetic sequence answers `false`, and **no code path inspects coordinate values to decide it** — no
constructor sniffs your data, and there is no tolerance anywhere.

Where the data question genuinely matters, the spacing accessors answer it exactly: an axis is equally
spaced precisely when its smallest gap equals its largest.

```@example axes
geo = FG.Geometry.CartesianGeometry()
gv = FG.Grids.StructuredGrid(geo, v, v)
FG.Grids.minimum_spacing(gv, 1) == FG.Grids.maximum_spacing(gv, 1)
```

If that holds and you want the fast path, build the axis yourself. The conversion replaces your
coordinates with the exact arithmetic sequence, so it is yours to ask for:

```@example axes
Δ = FG.Grids.minimum_spacing(gv, 1)            # equals maximum_spacing, so it IS the spacing
A.spacing(A.UniformAxis(first(v), Δ, length(v)))
```

## Your axis type is kept

A grid stores the axis you hand it. Any `AbstractRange` already of the geometry's element type passes
through untouched — your own range subtype, a `StepRangeLen` whose `TwicePrecision` internals you want,
a `BigFloat`-backed range:

```@example axes
r = range(0.0; step = 0.5, length = 9)
g = FG.Grids.StructuredGrid(geo, r, r)
FG.Grids.coordinates(g, 1) === r, typeof(FG.Grids.coordinates(g, 1))
```

Nothing needs converting to earn the fast paths — they dispatch on `spacing_trait`, which is
`UniformSpacing()` for *every* range, so whatever you brought gets them:

```@example axes
FG.Grids.isuniform(g), FG.Grids.spacing(g, 1), typeof.(FG.Grids.measure_factors(g))
```

Conversion happens only where the element type must change, since an arbitrary range subtype cannot be
rebuilt at a new eltype generically. That case becomes a `UniformAxis{T}`:

```@example axes
gi = FG.Grids.StructuredGrid(geo, 0:8, 0:8)      # Int range, Float64 grid
typeof(FG.Grids.coordinates(gi, 1))
```

A [`UniformAxis`](@ref) stores three numbers and computes `origin + (i-1)·Δ` in its own element type;
`uniform_axis` builds one. `range(0f0; step = 0.25f0, length = 5)` is a
`StepRangeLen{Float32, Float64, Float64, Int}` — `Float32` elements over a `Float64` offset and step —
so a `Float32` grid built from one does `Float64` index arithmetic. The grid keeps what you passed, and
converting is opt-in:

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

## Your own uniform axis type

If you want a uniform axis carrying something extra — a name, a unit tag, a provenance field — subtype
[`AbstractUniformAxis`](@ref) and implement the three methods `AbstractRange` requires anyway:

```@example axes
struct NamedAxis{T} <: A.AbstractUniformAxis{T}
    start::T
    h::T
    count::Int
    name::Symbol
end
Base.first(a::NamedAxis) = a.start
Base.step(a::NamedAxis) = a.h
Base.length(a::NamedAxis) = a.count

z = NamedAxis(0.0, 0.25, 5, :depth)
collect(z), sum(z), extrema(z), z[3], collect(z[2:4]), collect(reverse(z))
```

That is the whole contract. Indexing, the `O(1)` reductions, slicing, reversal and the affine
arithmetic all follow, and no generic method reads a field — the names above are nothing like
`UniformAxis`'s and it makes no difference:

```@example axes
collect(2.0 .* z .+ 1.0), A.isuniform(z), A.spacing(z)
```

A derived axis is a plain `UniformAxis`, since the extra field has no generic meaning under a slice.
Define [`similar_axis`](@ref) to say what it should be instead:

```@example axes
A.similar_axis(a::NamedAxis{T}, origin, Δ, n::Integer) where {T} =
    NamedAxis{T}(convert(T, origin), convert(T, Δ), Int(n), a.name)

zz = reverse(z)[2:3] .+ 1.0
typeof(zz), zz.name, collect(zz)
```

Such an axis is stored as itself on a grid and takes every uniform fast path:

```@example axes
gz = FG.Grids.StructuredGrid(geo, z, z)
FG.Grids.coordinates(gz, 1) === z, FG.Grids.isuniform(gz),
typeof.(FG.Grids.measure_factors(gz)), Base.summarysize(gz)
```

## It is an `AbstractRange`

```@example axes
u isa AbstractRange, step(u)
```

Two things follow. Base solves `searchsorted` on a range in closed form, so lookup cost is flat in the
axis length — 42 ns over `n = 10 … 10⁷`, against 35→69 ns for the equivalent `Vector`. And any package
that already tests `isa AbstractRange` to pick a uniform-grid fast path accepts this type without
knowing it exists.

```@example axes
big = A.UniformAxis(0.0, 1e-7, 10^7)
searchsortedfirst(big, 0.5), searchsortedfirst(big, 0.5) == searchsortedfirst(collect(0.0:1e-7:0.9999999), 0.5)
```

## What it buys

`minimum`, `maximum`, `extrema` and `sum` are closed forms, and the axis is `isbits`, so moving it to
another storage backend costs nothing.

```@example axes
sum(u), extrema(u), isbits(u), sizeof(u)
```

On a grid the payoff is the cell measure. A uniform axis's per-cell width is one number repeated, held
as a [`ConstantVector`](@ref):

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

`minimum_spacing` and `maximum_spacing` are `O(1)` on a uniform direction and `O(N)` otherwise. They
bound how far an index window must reach to cover a given physical distance, as a
neighbourhood-by-distance query on a stretched axis needs.
