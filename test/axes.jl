Test.@testset "Axis eltype conversion preserves the container type" begin
    geom = FG.Geometry.CartesianGeometry()
    # A uniform axis keeps its uniformity across an eltype conversion.
    r32 = Float32(0):Float32(1):Float32(3)
    gridr = FG.Grids.StructuredGrid(geom, r32, r32, trues(4, 4))
    Test.@test FG.Grids.isuniform(gridr, 1)
    Test.@test FG.Grids.spacing(gridr, 1) === 1.0
    Test.@test collect(FG.Grids.coordinates(gridr, 1)) == Float64[0, 1, 2, 3]

    geo32 = FG.Geometry.CartesianGeometry(Float32)
    r32f = range(0.0f0; step = 0.25f0, length = 5)
    g32 = FG.Grids.StructuredGrid(geo32, r32f, r32f, trues(5, 5))
    Test.@test FG.Grids.spacing(g32, 1) === 0.25f0
    Test.@test FG.Grids.coords(g32, 2, 3) === (x = 0.25f0, y = 0.5f0)
    Test.@test collect(FG.Axes.uniform_axis(Float32, r32f)) == collect(r32f)
end

Test.@testset "AbstractUniformAxis is a working extension point" begin
    A = FG.Axes
    # The contract is three methods, so a subtype may name its fields anything. These are named
    # unlike `UniformAxis`'s, so a generic method reaching for a field fails here.
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
    Test.@test (first(m), last(m)) == (0.0, 1.0)
    Test.@test A.spacing(m) == 0.25 && A.isuniform(m)
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
    Test.@test m == MinimalAxis(0.0, 0.25, 5) && m == A.UniformAxis(0.0, 0.25, 5)
    Test.@test m != MinimalAxis(0.0, 0.25, 4) && m != MinimalAxis(0.0, 0.5, 5)
    Test.@test MinimalAxis(0.0, 1.0, 0) == A.UniformAxis(9.0, 3.0, 0)
    Test.@test MinimalAxis(2.0, 1.0, 1) == A.UniformAxis(2.0, 7.0, 1)
    Test.@test A.uniform_axis(Float32, m) === A.UniformAxis(0.0f0, 0.25f0, 5)
    Test.@test_throws ArgumentError minimum(MinimalAxis(0.0, 1.0, 0))
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

    # A subtype drives a grid.
    geo = FG.Geometry.CartesianGeometry()
    g = FG.Grids.StructuredGrid(geo, t, t)
    ax = FG.Grids.coordinates(g, 1)
    Test.@test FG.Grids.isuniform(g) && FG.Grids.spacing(g, 1) == 0.25
    Test.@test FG.Grids.measure(g, 3, 4) == 0.25^2
    Test.@test length(Set(FG.Discretization.local_spacing(ax, i)[2] for i in 1:4)) == 1
    Test.@test FG.Discretization.locate(ax, 0.6) ==
               FG.Discretization.locate(collect(ax), 0.6)
    Test.@test A.isuniform(FG.Discretization.faces(ax))
end

Test.@testset "A caller's own axis type is preserved, and still gets every fast path" begin
    # Any `AbstractRange` has `UniformSpacing()`, so a caller's own range gets the uniform paths.
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
    Test.@test FG.Grids.isuniform(g) && FG.Grids.spacing(g, 1) === 0.5
    Test.@test FG.Grids.minimum_spacing(g, 1) == FG.Grids.maximum_spacing(g, 1) == 0.5
    Test.@test FG.Grids.measure(g, 3, 4) === 0.25
    # `local_spacing` returns `step` itself, so the gap is exactly constant.
    Test.@test length(Set(FG.Discretization.local_spacing(ax, i)[2] for i in 1:8)) == 1
    Test.@test FG.Discretization.locate(ax, 2.0) == FG.Discretization.locate(collect(ax), 2.0)
    Test.@test FG.Discretization.nearest_index(ax, 2.0) == 5
    Test.@test FG.Axes.isuniform(FG.Discretization.faces(ax))

    for r in (range(0.0; step = 0.5, length = 9), LinRange(0.0, 4.0, 9),
              FG.Axes.UniformAxis(0.0, 0.5, 9))
        gg = FG.Grids.StructuredGrid(geo, r, r)
        Test.@test FG.Grids.isuniform(gg) && FG.Grids.spacing(gg, 1) == 0.5
    end
    gi = FG.Grids.StructuredGrid(geo, 0:8, 0:8)
    Test.@test collect(FG.Grids.coordinates(gi, 1)) == collect(0.0:1.0:8.0)

    # BigFloat survives end to end at full precision.
    gb = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(BigFloat),
                                 range(BigFloat(0); step = BigFloat(1) / 3, length = 7),
                                 range(BigFloat(0); step = BigFloat(1) / 3, length = 7))
    Test.@test FG.Grids.spacing(gb, 1) == BigFloat(1) / 3
end

Test.@testset "A uniform axis is an AbstractRange, and behaves like one" begin
    A = FG.Axes
    a = A.UniformAxis(0.0, 0.25, 5)
    v = collect(a)
    Test.@test step(a) === 0.25 && first(a) === 0.0 && last(a) === 1.0
    Test.@test collect(a + 1.0) == v .+ 1.0
    Test.@test collect(a * 2.0) == v .* 2.0

    # An affine map of an affine sequence is affine, so the spacing guarantee survives arithmetic.
    Test.@test A.isuniform(a .+ 1.0) && A.spacing(a .+ 1.0) === 0.25
    Test.@test A.isuniform(2.0 .* a) && A.spacing(2.0 .* a) === 0.5
    Test.@test A.isuniform(a .+ a) && A.spacing(a .+ a) === 0.5
    Test.@test A.isuniform(-a) && A.spacing(-a) === -0.25
    Test.@test_throws DimensionMismatch A.UniformAxis(0.0, 1.0, 3) .+ A.UniformAxis(0.0, 1.0, 4)
end

Test.@testset "Uniform axes and constant factors are O(1), and exact" begin
    A = FG.Axes
    a = A.UniformAxis(0.0, 0.25, 5)
    Test.@test collect(a) == [0.0, 0.25, 0.5, 0.75, 1.0]
    Test.@test A.isuniform(a) && A.spacing(a) === 0.25
    # A descending axis: spacing keeps its sign, extrema are still ordered.
    d = A.UniformAxis(1.0, -0.25, 5)
    Test.@test A.spacing(d) === -0.25
    Test.@test extrema(d) == (0.0, 1.0)
    Test.@test collect(d) == reverse([0.0, 0.25, 0.5, 0.75, 1.0])
    Test.@test_throws BoundsError a[6]

    # The spacing accessors answer whether the gaps are identical exactly: the smallest equals the
    # largest.
    v = collect(a)
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
    Test.@test_throws BoundsError c[5]

    # A large uniform grid carries no per-cell and no per-axis storage.
    geo = FG.Geometry.CartesianGeometry()
    N = 2000
    g = FG.Grids.StructuredGrid(geo, range(0.0; step = 0.5, length = N),
                          range(0.0; step = 0.25, length = N), FG.Grids.AllActive((N, N)))
    Test.@test Base.summarysize(g) < 512
    Test.@test FG.Grids.measure(g, 3, 4) === 0.5 * 0.25
    # Every cell width of a uniform axis is `Δ` to the bit.
    Test.@test all(==(0.5), FG.Grids.measure_factors(g)[1])
end

Test.@testset "A periodic axis is n equal cells around one period" begin
    A = FG.Axes
    r = A.PeriodicUniformAxis(0.0, 2π, 12)
    Test.@test A.isuniform(r) && step(r) == 2π / 12 && A.period(r) == 2π
    Test.@test collect(r) == [(i - 1) * (2π / 12) for i in 1:12]
    # Descending: a negative period, the same magnitude.
    d = A.PeriodicUniformAxis(2π, -2π, 12)
    Test.@test step(d) == -step(r) && A.period(d) == 2π && first(d) == 2π
    Test.@test collect(r[2:5]) == collect(r)[2:5]
    Test.@test A.period(A.PeriodicUniformAxis{Float32}(0, 2π, 8)) == Float32(2π)
    Test.@test_throws ArgumentError A.PeriodicUniformAxis(0.0, 2π, 0)
    Test.@test_throws ArgumentError A.PeriodicUniformAxis(0.0, 0.0, 4)
    Test.@test_throws ArgumentError A.PeriodicUniformAxis(0.0, Inf, 4)
end

Test.@testset "Axis widths use bulk operations, not per-element scalar reads" begin
    # An array type that counts scalar `getindex` calls.
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

    for n in (50, 500, 5000)
        c = CountingVector(collect(range(0.0; step = 1.5, length = n)))
        w = FG.Discretization.cell_widths(c)
        Test.@test collect(w) ≈ [FG.Discretization.cell_width(c.data, i) for i in 1:n]
        # O(n) reads with a small constant.
        Test.@test c.scalar_reads <= 4n + 16
    end
end

Test.@testset "An analytic axis is a formula, and inverts instead of searching" begin
    A = FG.Axes
    D = FG.Discretization
    GD = FG.Grids

    axes_under_test = (
        ("GeometricAxis r=1.15", A.GeometricAxis(0.0, 10.0, 1.15, 40)),
        ("GeometricAxis r=0.85", A.GeometricAxis(0.0, 100.0, 0.85, 40)),
        ("GeometricAxis descending", A.GeometricAxis(1000.0, -20.0, 1.1, 30)),
        ("GeometricAxis r=2.5", A.GeometricAxis(1.0, 0.5, 2.5, 12)),
        ("PowerAxis p=2", A.PowerAxis(0.0, 5000.0, 2.0, 40)),
        ("PowerAxis p=0.5", A.PowerAxis(0.0, 5000.0, 0.5, 40)),
        ("PowerAxis descending", A.PowerAxis(0.0, -6000.0, 1.7, 25)),
    )

    for (_, a) in axes_under_test
        n = length(a)
        # `index_at` inverts the indexing.
        Test.@test all(abs(A.index_at(a, a[i]) - i) < 1e-9 for i in 1:n)
        # Strictly monotone, so that inverse is single-valued.
        Test.@test all(sign(a[i + 1] - a[i]) == sign(a[2] - a[1]) for i in 1:(n - 1))
        Test.@test_throws ArgumentError A.spacing(a)
        # The endpoint reductions are closed form, and agree with the samples.
        Test.@test extrema(a) == extrema(collect(a))
        Test.@test first(a) === a[1] && last(a) === a[n]
    end

    # A geometric axis's gaps form a geometric progression, the property it is named for.
    let a = A.GeometricAxis(0.0, 10.0, 1.15, 40)
        g = [a[i + 1] - a[i] for i in 1:(length(a) - 1)]
        Test.@test g[1] ≈ a.Δ
        Test.@test all(isapprox(g[i + 1], g[i] * a.ratio; rtol = 1e-12) for i in 1:(length(g) - 1))
    end

    # Every query answers exactly what the search over the same samples answers — at the samples,
    # inside every cell, on every face, and outside both ends. The closed form keeps the same
    # convention: faces are the coordinate midpoints.
    for (_, a) in axes_under_test
        n = length(a)
        dense = collect(a)
        probes = Float64[]
        for i in 1:n
            push!(probes, a[i])
            i < n && append!(probes,
                             [a[i] + t * (a[i + 1] - a[i]) for t in (0.001, 0.25, 0.5, 0.75, 0.999)])
        end
        append!(probes, [D._face_at(a, j, n) for j in 1:(n + 1)])
        w = abs(a[n] - a[1])
        append!(probes, [a[1] - 0.3w, a[1] - 1e-9w, a[n] + 1e-9w, a[n] + 0.3w])
        Test.@test all(D.locate(a, v) == D.locate(dense, v) for v in probes)
        Test.@test all(D.nearest_index(a, v) == D.nearest_index(dense, v) for v in probes)
        Test.@test all(D.interpolation_weights(a, v) == D.interpolation_weights(dense, v)
                       for v in probes)
    end

    # As a grid's last direction it measures what its samples do.
    let sph = FG.Geometry.SphericalGeometry(6.371e6),
        λ = collect(range(0, 2π; length = 13)[1:12]),
        φ = collect(range(-1.2, 1.2; length = 9)),
        z = A.PowerAxis(6.371e6, 8.0e4, 1.6, 20)
        g = GD.StructuredGrid(sph, λ, φ, z)
        gd = GD.StructuredGrid(sph, λ, φ, collect(z))
        Test.@test GD.coords(g, 2, 3, 4) == (λ = λ[2], φ = φ[3], r = z[4])
        Test.@test all(GD.measure(g, i, j, k) == GD.measure(gd, i, j, k)
                       for i in 1:12, j in 1:9, k in 1:20)
        Test.@test GD.locate(g, (0.3, 0.1, z[7] + 1.0)) == GD.locate(gd, (0.3, 0.1, z[7] + 1.0))
    end

    # What cannot be inverted single-valuedly is refused at construction.
    Test.@test_throws ArgumentError A.GeometricAxis(0.0, 1.0, 1.0, 10)    # constant gaps: a UniformAxis
    Test.@test_throws ArgumentError A.GeometricAxis(0.0, 1.0, -2.0, 10)   # not monotone
    Test.@test_throws ArgumentError A.GeometricAxis(0.0, 0.0, 1.2, 10)    # no first gap
    Test.@test_throws ArgumentError A.GeometricAxis(0.0, 1.0, 1.2, -1)
    Test.@test_throws ArgumentError A.PowerAxis(0.0, 1.0, 0.0, 10)        # a zero exponent is flat
    Test.@test_throws ArgumentError A.PowerAxis(0.0, 1.0, -1.0, 10)
    Test.@test_throws ArgumentError A.PowerAxis(0.0, 0.0, 2.0, 10)
    Test.@test_throws ArgumentError A.PowerAxis(0.0, 1.0, 2.0, 1)
end
