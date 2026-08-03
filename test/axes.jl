Test.@testset "Axis eltype conversion preserves the container type" begin
    geom = FG.Geometry.CartesianGeometry()
    # A Float32 Vector into a Float64 geometry must copy — but into the same kind of container,
    # not unconditionally into a plain `Vector`.
    x32 = Float32[0, 1, 2, 3]
    grid = FG.Grids.StructuredGrid(geom, x32, x32, trues(4, 4))
    Test.@test FG.Grids.coordinates(grid, 1) isa Vector{Float64}
    # A uniform axis keeps its uniformity across an eltype conversion — that property, not the
    # container, is what later fast paths dispatch on.
    r32 = Float32(0):Float32(1):Float32(3)
    gridr = FG.Grids.StructuredGrid(geom, r32, r32, trues(4, 4))
    Test.@test FG.Grids.isuniform(gridr, 1)
    Test.@test eltype(FG.Grids.coordinates(gridr, 1)) === Float64
    Test.@test FG.Grids.spacing(gridr, 1) === 1.0
    Test.@test collect(FG.Grids.coordinates(gridr, 1)) == Float64[0, 1, 2, 3]

    # An axis already of the geometry's element type is kept EXACTLY as given, whatever type it is.
    # `range(0f0; step=0.25f0, …)` is a `StepRangeLen{Float32, Float64, Float64, Int}` — Float32
    # elements over a Float64 offset and step — and that is the caller's choice to make, not this
    # package's to override. `Axes.uniform_axis` is the opt-in for Float32-throughout arithmetic.
    geo32 = FG.Geometry.CartesianGeometry(Float32)
    r32f = range(0.0f0; step = 0.25f0, length = 5)
    g32 = FG.Grids.StructuredGrid(geo32, r32f, r32f, trues(5, 5))
    ax32 = FG.Grids.coordinates(g32, 1)
    Test.@test ax32 === r32f                       # preserved, not replaced
    Test.@test eltype(ax32) === Float32
    Test.@test FG.Grids.spacing(g32, 1) === 0.25f0
    Test.@test FG.Grids.coords(g32, 2, 3) === (x = 0.25f0, y = 0.5f0)
    # …and converting deliberately gives Float32 throughout.
    u32 = FG.Axes.uniform_axis(Float32, r32f)
    Test.@test all(p -> !(p isa Type) || p === Float32, typeof(u32).parameters)
    Test.@test collect(u32) == collect(r32f)
end

Test.@testset "AbstractUniformAxis is a working extension point" begin
    A = FG.Axes
    # The contract is three methods, not a layout, so a subtype may name its fields anything.
    # These are named unlike `UniformAxis`'s so that a generic method reaching for a field is a
    # test failure instead of a coincidence.
    struct MinimalAxis{T} <: A.AbstractUniformAxis{T}
        lo::T
        h::T
        count::Int
    end
    Base.first(a::MinimalAxis) = a.lo
    Base.step(a::MinimalAxis) = a.h
    Base.length(a::MinimalAxis) = a.count

    m = MinimalAxis(0.0, 0.25, 5)
    ref = [0.0, 0.25, 0.5, 0.75, 1.0]
    # Those three methods are the whole contract; everything below is inherited.
    Test.@test collect(m) == ref
    Test.@test size(m) == (5,) && length(m) == 5
    Test.@test (first(m), last(m)) == (0.0, 1.0)
    Test.@test step(m) == 0.25 && A.spacing(m) == 0.25 && A.isuniform(m)
    Test.@test m isa AbstractRange
    Test.@test m[3] == 0.5
    Test.@test sum(m) == sum(ref)
    Test.@test (minimum(m), maximum(m)) == (0.0, 1.0) && extrema(m) == (0.0, 1.0)
    Test.@test collect(m[2:4]) == ref[2:4]
    Test.@test collect(m[1:2:5]) == ref[1:2:5]
    Test.@test collect(reverse(m)) == reverse(ref)
    Test.@test collect(m .+ 1.0) == ref .+ 1.0
    Test.@test collect(2.0 .* m) == 2.0 .* ref
    Test.@test collect(m ./ 2.0) == ref ./ 2.0
    Test.@test collect(-m) == -ref
    Test.@test collect(1.0 .- m) == 1.0 .- ref
    Test.@test collect(m .+ m) == ref .+ ref
    Test.@test collect(m .- m) == ref .- ref
    Test.@test cos.(m) ≈ cos.(ref) && !A.isuniform(cos.(m))
    Test.@test searchsortedfirst(m, 0.5) == searchsortedfirst(ref, 0.5)
    Test.@test copy(m) === m
    Test.@test m == MinimalAxis(0.0, 0.25, 5) && m == A.UniformAxis(0.0, 0.25, 5)
    Test.@test m != MinimalAxis(0.0, 0.25, 4) && m != MinimalAxis(0.0, 0.5, 5)
    Test.@test MinimalAxis(0.0, 1.0, 0) == A.UniformAxis(9.0, 3.0, 0)
    Test.@test MinimalAxis(2.0, 1.0, 1) == A.UniformAxis(2.0, 7.0, 1)
    Test.@test A.uniform_axis(Float32, m) === A.UniformAxis(0.0f0, 0.25f0, 5)
    # Messages name the actual type, not the supertype.
    Test.@test occursin("MinimalAxis", sprint(show, m))
    Test.@test_throws ArgumentError minimum(MinimalAxis(0.0, 1.0, 0))
    # Without the hook, a derived axis is correct but plain.
    Test.@test m[2:4] isa A.UniformAxis
    Test.@test A.isuniform(m[2:4]) && A.isuniform(reverse(m)) && A.isuniform(2.0 .* m)

    # `similar_axis` is the one method that makes derived axes keep the subtype.
    struct TaggedAxis2{T} <: A.AbstractUniformAxis{T}
        lo::T
        h::T
        count::Int
        tag::Symbol
    end
    Base.first(a::TaggedAxis2) = a.lo
    Base.step(a::TaggedAxis2) = a.h
    Base.length(a::TaggedAxis2) = a.count
    A.similar_axis(a::TaggedAxis2{T}, origin, Δ, n::Integer) where {T} =
        TaggedAxis2{T}(convert(T, origin), convert(T, Δ), Int(n), a.tag)

    t = TaggedAxis2(0.0, 0.25, 5, :mine)
    for got in (t[2:4], t[1:2:5], reverse(t), t .+ 1.0, 2.0 .* t, -t, t ./ 2.0, t .+ t, t .- t)
        Test.@test got isa TaggedAxis2 && got.tag === :mine
    end

    # And a subtype drives a grid, stored as itself, with every uniform fast path.
    geo = FG.Geometry.CartesianGeometry()
    g = FG.Grids.StructuredGrid(geo, t, t)
    ax = FG.Grids.coordinates(g, 1)
    Test.@test ax === t
    Test.@test FG.Grids.isuniform(g) && FG.Grids.spacing(g, 1) == 0.25
    Test.@test all(f -> f isa A.ConstantVector, FG.Grids.measure_factors(g))
    Test.@test FG.Grids.measure(g, 3, 4) == 0.25^2
    Test.@test Base.summarysize(g) < 512
    Test.@test length(Set(FG.Discretization.local_spacing(ax, i)[2] for i in 1:4)) == 1
    Test.@test FG.Discretization.locate(ax, 0.6) ==
               FG.Discretization.locate(collect(ax), 0.6)
    Test.@test A.isuniform(FG.Discretization.faces(ax))
    # Mixing two uniform axis types lands on the canonical concrete one.
    Test.@test promote_type(typeof(t), typeof(m)) === A.UniformAxis{Float64}
end

Test.@testset "A caller's own axis type is preserved, and still gets every fast path" begin
    # An arbitrary `AbstractRange` subtype must survive untouched: nothing about the uniform fast
    # paths requires converting it, because they dispatch on `spacing_trait`, which is
    # `UniformSpacing()` for every range.
    struct TaggedAxis{T} <: AbstractRange{T}
        o::T
        d::T
        n::Int
        tag::Symbol
    end
    Base.size(a::TaggedAxis) = (a.n,)
    Base.length(a::TaggedAxis) = a.n
    Base.IndexStyle(::Type{<:TaggedAxis}) = IndexLinear()
    Base.getindex(a::TaggedAxis{T}, i::Int) where {T} =
        (Base.@boundscheck checkbounds(a, i); a.o + T(i - 1) * a.d)
    Base.step(a::TaggedAxis) = a.d
    Base.first(a::TaggedAxis) = a.o
    Base.last(a::TaggedAxis{T}) where {T} = a.o + T(a.n - 1) * a.d

    geo = FG.Geometry.CartesianGeometry()
    mine = TaggedAxis(0.0, 0.5, 9, :mine)
    g = FG.Grids.StructuredGrid(geo, mine, mine)
    ax = FG.Grids.coordinates(g, 1)
    Test.@test ax === mine                          # the same object, not a copy
    Test.@test ax isa TaggedAxis && ax.tag === :mine # and the extra field survives

    # Every uniform fast path applies to it.
    Test.@test FG.Grids.isuniform(g) && FG.Grids.spacing(g, 1) === 0.5
    Test.@test all(f -> f isa FG.Axes.ConstantVector, FG.Grids.measure_factors(g))
    Test.@test Base.summarysize(g) < 512
    Test.@test FG.Grids.minimum_spacing(g, 1) == FG.Grids.maximum_spacing(g, 1) == 0.5
    Test.@test FG.Grids.measure(g, 3, 4) === 0.25
    # `local_spacing` returns `step` rather than differencing, so the gap is exactly constant.
    Test.@test length(Set(FG.Discretization.local_spacing(ax, i)[2] for i in 1:8)) == 1
    Test.@test FG.Discretization.locate(ax, 2.0) == FG.Discretization.locate(collect(ax), 2.0)
    Test.@test FG.Discretization.nearest_index(ax, 2.0) == 5
    Test.@test FG.Axes.isuniform(FG.Discretization.faces(ax))

    # Base ranges are likewise handed back as given, including the ones whose internals a caller
    # might specifically want.
    for r in (range(0.0; step = 0.5, length = 9), LinRange(0.0, 4.0, 9),
              FG.Axes.UniformAxis(0.0, 0.5, 9))
        gg = FG.Grids.StructuredGrid(geo, r, r)
        Test.@test FG.Grids.coordinates(gg, 1) === r
        Test.@test FG.Grids.isuniform(gg) && FG.Grids.spacing(gg, 1) == 0.5
    end
    # An integer range cannot stay one in a Float64 grid, so that is the case that rebuilds.
    gi = FG.Grids.StructuredGrid(geo, 0:8, 0:8)
    Test.@test FG.Grids.coordinates(gi, 1) isa FG.Axes.UniformAxis{Float64}
    Test.@test collect(FG.Grids.coordinates(gi, 1)) == collect(0.0:1.0:8.0)

    # BigFloat survives end to end at full precision.
    gb = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(BigFloat),
                                 range(BigFloat(0); step = BigFloat(1) / 3, length = 7),
                                 range(BigFloat(0); step = BigFloat(1) / 3, length = 7))
    Test.@test eltype(FG.Grids.coordinates(gb, 1)) === BigFloat
    Test.@test FG.Grids.spacing(gb, 1) * 3 == 1
end

Test.@testset "A uniform axis is an AbstractRange, and behaves like one" begin
    A = FG.Axes
    a = A.UniformAxis(0.0, 0.25, 5)
    v = collect(a)
    Test.@test a isa AbstractRange
    Test.@test step(a) === 0.25 && first(a) === 0.0 && last(a) === 1.0
    Test.@test length(a) == 5 && size(a) == (5,)

    # Base's range machinery rebuilds a range from its endpoints and step. Without the protocol
    # methods these recurse until the stack runs out, so each is asserted.
    Test.@test collect(a .+ a) == v .+ v
    Test.@test collect(a .- a) == v .- v
    Test.@test collect(a .+ 1.0) == v .+ 1.0
    Test.@test collect(1.0 .- a) == 1.0 .- v
    Test.@test collect(2.0 .* a) == 2.0 .* v
    Test.@test collect(a ./ 2.0) == v ./ 2.0
    Test.@test collect(-a) == -v
    Test.@test collect(a + 1.0) == v .+ 1.0
    Test.@test collect(a * 2.0) == v .* 2.0
    Test.@test copy(a) === a                      # immutable
    Test.@test occursin("UniformAxis", sprint(show, a))
    Test.@test a == A.UniformAxis(0.0, 0.25, 5)
    Test.@test a != A.UniformAxis(0.0, 0.5, 5)
    Test.@test a != A.UniformAxis(0.0, 0.25, 6)
    Test.@test promote_type(A.UniformAxis{Float32}, A.UniformAxis{Float64}) ===
               A.UniformAxis{Float64}

    # An affine map of an affine sequence is affine, so the spacing guarantee survives arithmetic
    # instead of collapsing to a Vector.
    Test.@test A.isuniform(a .+ 1.0) && A.spacing(a .+ 1.0) === 0.25
    Test.@test A.isuniform(2.0 .* a) && A.spacing(2.0 .* a) === 0.5
    Test.@test A.isuniform(a .+ a) && A.spacing(a .+ a) === 0.5
    Test.@test A.isuniform(-a) && A.spacing(-a) === -0.25
    # Anything not affine must become a plain array rather than claim to be uniform.
    Test.@test !A.isuniform(cos.(a))
    Test.@test cos.(a) ≈ cos.(v)
    Test.@test_throws DimensionMismatch A.UniformAxis(0.0, 1.0, 3) .+ A.UniformAxis(0.0, 1.0, 4)

    # The reason for the subtyping: being an `AbstractRange` is what gets Base's closed-form
    # `searchsorted` instead of a bisection. Asserted as the dispatch itself — the property that
    # causes the speed — rather than as a duration; the duration is in `benchmark/`.
    Test.@test A.UniformAxis(0.0, 1e-1, 10) isa AbstractRange
    Test.@test which(searchsortedfirst, Tuple{A.UniformAxis{Float64},Float64}) ===
               which(searchsortedfirst, Tuple{StepRangeLen{Float64},Float64})
    # …and it still gives the right index.
    for n in (10, 1000)
        u = A.UniformAxis(0.0, 1 / n, n)
        for t in (0.0, 0.25, 0.5, 1.0)
            Test.@test searchsortedfirst(u, t) == searchsortedfirst(collect(u), t)
            Test.@test searchsortedlast(u, t) == searchsortedlast(collect(u), t)
        end
    end
end

Test.@testset "Uniform axes and constant factors are O(1), and exact" begin
    A = FG.Axes
    a = A.UniformAxis(0.0, 0.25, 5)
    Test.@test collect(a) == [0.0, 0.25, 0.5, 0.75, 1.0]
    Test.@test A.isuniform(a) && A.spacing(a) === 0.25
    Test.@test isbits(a)                          # nothing to move to another storage backend
    Test.@test (first(a), last(a)) == (0.0, 1.0)
    # Closed forms, not scans.
    Test.@test sum(a) == sum(collect(a))
    Test.@test extrema(a) == extrema(collect(a))
    # A descending axis is a first-class axis: spacing is SIGNED, extrema are still ordered.
    d = A.UniformAxis(1.0, -0.25, 5)
    Test.@test A.spacing(d) === -0.25
    Test.@test extrema(d) == (0.0, 1.0)
    Test.@test collect(d) == reverse([0.0, 0.25, 0.5, 0.75, 1.0])
    # A slice and a reversal of a uniform axis are uniform, not collapsed to a Vector.
    Test.@test A.isuniform(a[2:4]) && collect(a[2:4]) == [0.25, 0.5, 0.75]
    Test.@test A.isuniform(reverse(a)) && collect(reverse(a)) == reverse(collect(a))
    Test.@test_throws BoundsError a[6]
    Test.@test_throws BoundsError a[0]

    # `isuniform` is the TYPE question, and nothing here answers it from the values: a Vector
    # holding an arithmetic sequence is still not `isuniform`, and no constructor inspects data to
    # decide otherwise. Where the data question genuinely matters, the existing spacing accessors
    # answer it exactly — identical gaps iff the smallest equals the largest — with no tolerance.
    v = collect(a)
    Test.@test !A.isuniform(v)
    geo0 = FG.Geometry.CartesianGeometry()
    gv = FG.Grids.StructuredGrid(geo0, v, v)
    Test.@test FG.Grids.minimum_spacing(gv, 1) == FG.Grids.maximum_spacing(gv, 1)
    gs2 = FG.Grids.StructuredGrid(geo0, [0.0, 1.0, 3.0, 6.0], [0.0, 1.0, 3.0, 6.0])
    Test.@test FG.Grids.minimum_spacing(gs2, 1) != FG.Grids.maximum_spacing(gs2, 1)
    Test.@test_throws ArgumentError A.spacing([0.0, 1.0, 3.0])
    Test.@test_throws ArgumentError A.uniform_axis(Float64, [0.0, 1.0, 3.0])

    c = A.ConstantVector(2.5, 4)
    Test.@test collect(c) == fill(2.5, 4)
    Test.@test sum(c) == 10.0 && prod(c) == 2.5^4 && extrema(c) == (2.5, 2.5)
    Test.@test sum(abs2, c) == 4 * abs2(2.5)
    Test.@test count(>(2), c) == 4
    Test.@test A.isuniform(A.ConstantVector(1.0, 3)) == false   # a factor list, not an axis
    Test.@test_throws BoundsError c[5]

    # The whole point: a large uniform grid carries no per-cell and no per-axis storage.
    geo = FG.Geometry.CartesianGeometry()
    N = 2000
    g = FG.Grids.StructuredGrid(geo, range(0.0; step = 0.5, length = N),
                          range(0.0; step = 0.25, length = N), FG.Grids.AllActive((N, N)))
    Test.@test all(f -> f isa A.ConstantVector, FG.Grids.measure_factors(g))
    Test.@test Base.summarysize(g) < 512          # three numbers per axis, not N per axis
    Test.@test FG.Grids.measure(g, 3, 4) === 0.5 * 0.25
    # Constant factors and dense factors must agree, and the uniform answer is the exact one:
    # `Δ` is stored, so every cell width IS `Δ`, where recovering it by differencing reconstructed
    # coordinates leaves an ulp of noise.
    dense = FG.Discretization._cell_widths_dense(collect(FG.Grids.coordinates(g, 1)))
    Test.@test collect(FG.Grids.measure_factors(g)[1]) ≈ dense rtol = 1e-15
    Test.@test all(==(0.5), FG.Grids.measure_factors(g)[1])
end

Test.@testset "Axis widths use bulk operations, not per-element scalar reads" begin
    # An array type that counts scalar `getindex` calls. Device arrays make scalar indexing an
    # error precisely because it is a per-element round trip; the property that matters is that
    # the O(n) work is done in bulk, so the scalar count must not grow with n.
    mutable struct CountingVector{T} <: AbstractVector{T}
        data::Vector{T}
        scalar_reads::Int
    end
    CountingVector(v::Vector{T}) where {T} = CountingVector{T}(v, 0)
    Base.size(c::CountingVector) = size(c.data)
    Base.IndexStyle(::Type{<:CountingVector}) = IndexLinear()
    function Base.getindex(c::CountingVector, i::Int)
        c.scalar_reads += 1
        return c.data[i]
    end
    Base.setindex!(c::CountingVector, v, i::Int) = (c.data[i] = v)
    Base.similar(c::CountingVector, ::Type{S}, dims::Dims) where {S} =
        CountingVector{S}(Vector{S}(undef, dims...), 0)
    # Views of a CountingVector go through the parent's getindex, so bulk broadcasts over
    # `@view` still register — count them via the parent after the fact.

    for n in (50, 500, 5000)
        c = CountingVector(collect(range(0.0; step = 1.5, length = n)))
        w = FG.Discretization.cell_widths(c)
        Test.@test collect(w) ≈ [FG.Discretization.cell_width(c.data, i) for i in 1:n]
        # Bulk broadcasts do touch elements, but the count must be O(n) with a small constant
        # and never O(n) per output element.
        Test.@test c.scalar_reads <= 4n + 16
    end

    # The endpoint handling is the only genuinely scalar part, and it is O(1).
    n = 4000
    c1 = CountingVector(collect(range(0.0; step = 1.0, length = n)))
    c2 = CountingVector(collect(range(0.0; step = 1.0, length = 2n)))
    FG.Discretization.cell_widths(c1)
    FG.Discretization.cell_widths(c2)
    Test.@test c2.scalar_reads - c1.scalar_reads <= 4n + 16   # grows linearly, not quadratically
end
