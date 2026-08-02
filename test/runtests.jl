using FlowGeometries: FlowGeometries
using Test: Test

# Every weak dependency is loaded up front so the extensions are exercised by the whole suite.
using Adapt: Adapt
using DelaunayTriangulation: DelaunayTriangulation
using KernelAbstractions: KernelAbstractions
using NearestNeighbors: NearestNeighbors
using Quickhull: Quickhull
using SparseArrays: SparseArrays
using StaticArrays: StaticArrays

const FG = FlowGeometries

# A caller's own stencil shape: forward-only in every direction, radius in the type. `offsets` is the
# whole contract — everything else in Stencils is written in terms of it.
struct Upwind{R} <: FG.Stencils.AbstractStencil end
Upwind(r::Integer) = Upwind{Int(r)}()
FG.Stencils.offsets(::Upwind{R}, ::Val{N}) where {R,N} =
    ntuple(i -> ntuple(d -> d == cld(i, R) ? mod1(i, R) : 0, Val(N)), Val(N * R))

# Fixed arity, and the cell index travels as a tuple that is splatted inside the callee: forwarding
# through `args...` allocates 240 bytes on its own, so a vararg harness would measure itself.
for n in 1:6
    args = [Symbol(:a, i) for i in 1:n]
    @eval function _alloc(f::F, $(args...)) where {F}
        a = typemax(Int)
        for _ in 1:4                       # first calls carry compilation
            a = min(a, @allocated f($(args...)))
        end
        return a
    end
end

q_coords(g, I)           = FG.Grids.coords(g, I...)
q_measure(g, I)          = FG.Grids.measure(g, I...)
q_active(g, I)           = FG.Grids.isactive(g, I...)
q_dist(g, I, J)          = FG.Geometry.distance(g, I, J)
q_disp(g, I, J)          = FG.Grids.displacement(g, I, J)
q_nn(g, I)               = FG.Connectivity.nneighbors(g, I...)
q_nbrs!(out, g, I)       = FG.Connectivity.neighbors!(out, g, I...)
q_iter(g, I)             = (t = 0; for j in FG.Grids.neighbors(g, I...); t += j; end; t)
q_top(g)                 = FG.Connectivity.MetricTopology(g)
q_win(g, I, r, mt)       = FG.Connectivity.metric_window(g, I, r, mt)
q_nwithin(g, r, I)       = FG.Connectivity.nneighbors_within(g, I...; ball = r)
q_within!(b, g, r, I)    = FG.Connectivity.neighbors_within!(b, g, I...; ball = r)
q_fold(g, r, I)          = FG.Connectivity.fold_within((a, _, d) -> a + d, 0.0, g, I...; ball = r)
q_knn!(ix, dt, g, k, I)  = FG.Connectivity.k_nearest!(ix, dt, g, I...; k = k)
q_within_top(g, r, I, t) = FG.Connectivity.nneighbors_within(g, I...; ball = r, topology = t)
q_fold_at(g, p, r, t, s) = FG.Connectivity.fold_at((a, _, _) -> a + 1, 0, g, p;
                                                   ball = r, topology = t, scratch = s)
q_locate(g, p)           = FG.Grids.locate(g, p)
q_locate_top(g, p, t, s) = FG.Grids.locate(g, p; topology = t, scratch = s)
q_stencil!(o, f, ix, w, d) = FG.Discretization.apply_stencil!(o, f, ix, w, d)
q_stencil_m!(o, f, ix, w, d, msk) = FG.Discretization.apply_stencil!(o, f, ix, w, d; mask = msk)
q_area(g, I)             = FG.Grids.area(g, I...)
q_coords!(o, g, I)       = FG.Grids.coords!(o, g, I...)
q_axis(g, d)             = length(FG.Grids.axis(g, d))
q_spacing(g, d)          = FG.Grids.spacing(g, d)
q_origin(g, d)           = FG.Grids.origin(g, d)
q_extent(g, d)           = FG.Grids.extent(g, d)
q_bounds(g, d)           = FG.Grids.bounds(g, d)
q_isper(g, d)            = FG.Grids.isperiodic(g, d)
q_isuni(g, d)            = FG.Grids.isuniform(g, d)
q_period(g, d)           = FG.Grids.period(g, d)
q_minsp(g, d)            = FG.Grids.minimum_spacing(g, d)
q_maxsp(g, d)            = FG.Grids.maximum_spacing(g, d)
q_size(g)                = FG.Grids.size_tuple(g)
q_mask(g)                = FG.Grids.mask(g)
q_names(g)               = FG.Grids.coordinate_names(g)
q_perflags(g)            = FG.Grids.periodic_flags(g)
q_topo(g)                = FG.Grids.topology(g)

function check_shape(label, g, I)
    r = FG.Grids.grid_geometry(g) isa FG.Geometry.AbstractSphericalGeometry ? 6.371e5 : 2.5
    out = Vector{Int}(undef, 64)
    buf = Vector{Int}(undef, 4096)
    pt = Vector{Float64}(undef, length(I))
    dts = Vector{Float64}(undef, 64)
    mt = FG.Connectivity.MetricTopology(g)
    J = ntuple(d -> I[d] + 1, length(I))
    checks = (
        ("coords",             _alloc(q_coords, g, I)),
        ("measure",            _alloc(q_measure, g, I)),
        ("isactive",           _alloc(q_active, g, I)),
        ("distance(g,I,J)",    _alloc(q_dist, g, I, J)),
        ("displacement",       _alloc(q_disp, g, I, J)),
        ("nneighbors",         _alloc(q_nn, g, I)),
        ("neighbors!",         _alloc(q_nbrs!, out, g, I)),
        ("neighbors iterator", _alloc(q_iter, g, I)),
        ("MetricTopology",     _alloc(q_top, g)),
        ("nneighbors_within",  _alloc(q_nwithin, g, r, I)),
        ("neighbors_within!",  _alloc(q_within!, buf, g, r, I)),
        ("fold_within",        _alloc(q_fold, g, r, I)),
        ("area",               _alloc(q_area, g, I)),
        ("k_nearest!",         _alloc(q_knn!, out, dts, g, 6, I)),
        ("coords!",            _alloc(q_coords!, pt, g, I)),
        ("size_tuple",         _alloc(q_size, g)),
        ("mask",               _alloc(q_mask, g)),
        ("coordinate_names",   _alloc(q_names, g)),
        ("periodic_flags",     _alloc(q_perflags, g)),
        ("topology",           _alloc(q_topo, g)),
    )
    for (name, a) in checks
        a == 0 || println("    ", label, " / ", name, " -> ", a, " B")
        Test.@test a == 0
    end
    if g isa FG.Grids.StructuredGrid
        a = _alloc(q_win, g, I, r, mt)
        a == 0 || println("    ", label, " / metric_window -> ", a, " B")
        Test.@test a == 0
    end
    for d in 1:length(I)
        # `spacing` is the constant-spacing accessor and errors on an irregular axis, by design.
        if FG.Grids.isuniform(g, d)
            a = _alloc(q_spacing, g, d)
            a == 0 || println("    ", label, " / spacing(d=", d, ") -> ", a, " B")
            Test.@test a == 0
        end
        for (name, a) in (("origin", _alloc(q_origin, g, d)), ("extent", _alloc(q_extent, g, d)),
                          ("bounds", _alloc(q_bounds, g, d)), ("isperiodic", _alloc(q_isper, g, d)),
                          ("isuniform", _alloc(q_isuni, g, d)), ("period", _alloc(q_period, g, d)))
            a == 0 || println("    ", label, " / ", name, "(d=", d, ") -> ", a, " B")
            Test.@test a == 0
        end
        # spacing bounds are a rectilinear notion; a curvilinear grid has no axes to gap
        if g isa FG.Grids.StructuredGrid
            for (name, a) in (("minimum_spacing", _alloc(q_minsp, g, d)),
                              ("maximum_spacing", _alloc(q_maxsp, g, d)))
                a == 0 || println("    ", label, " / ", name, "(d=", d, ") -> ", a, " B")
                Test.@test a == 0
            end
        end
    end
    # A cell-list query enumerates through a fold, so it holds no candidate buffer at all — the property
    # the device path depends on.
    let cl = FG.Connectivity.MetricTopology(g; index = FG.Grids.cell_list(g; ball = r))
        a = _alloc(q_within_top, g, r, I, cl)
        a == 0 || println("    ", label, " / cell-list query -> ", a, " B")
        Test.@test a == 0
        Test.@test q_within_top(g, r, I, cl) == q_nwithin(g, r, I)
    end

    Test.@test (Test.@inferred q_coords(g, I)) isa NamedTuple
    Test.@test (Test.@inferred q_dist(g, I, J)) isa Float64
    Test.@test (Test.@inferred q_nwithin(g, r, I)) isa Int
    Test.@test (Test.@inferred q_top(g)) isa FG.Connectivity.MetricTopology
    return nothing
end

Test.@testset "FlowGeometries.jl" begin

    Test.@testset "Abstract hierarchy" begin
        Test.@test FG.Geometry.AbstractCartesianGeometry <: FG.Geometry.AbstractGeometry
        Test.@test FG.Geometry.AbstractSphericalGeometry <: FG.Geometry.AbstractGeometry
        Test.@test FG.Geometry.CartesianGeometry <: FG.Geometry.AbstractCartesianGeometry
        Test.@test FG.Geometry.SphericalGeometry <: FG.Geometry.AbstractSphericalGeometry
        Test.@test FG.Grids.AbstractStructuredGrid <: FG.Grids.AbstractGrid
        Test.@test FG.Grids.AbstractCurvilinearGrid <: FG.Grids.AbstractGrid
        Test.@test FG.Grids.AbstractUnstructuredGrid <: FG.Grids.AbstractGrid
        Test.@test FG.Grids.StructuredGrid <: FG.Grids.AbstractStructuredGrid
        Test.@test FG.Grids.CurvilinearGrid <: FG.Grids.AbstractCurvilinearGrid
        Test.@test FG.Grids.UnstructuredGrid <: FG.Grids.AbstractUnstructuredGrid
    end

    Test.@testset "User geometry subtype participates" begin
        struct StretchedCartesian{T} <: FG.Geometry.AbstractCartesianGeometry{T} end
        g = StretchedCartesian{Float64}()
        Test.@test g isa FG.Geometry.AbstractGeometry
        Test.@test FG.Geometry.area_element(g, 2.0, 3.0) ≈ 6.0
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        Test.@test FG.Geometry.distance(g, p1, p2) ≈ 5.0

        # A spherical geometry supplies `radius` and nothing else. The field is named unlike
        # `SphericalGeometry`'s so that any method reaching for `.R` fails here.
        struct ScaledSphere{T} <: FG.Geometry.AbstractSphericalGeometry{T}
            r0::T
        end
        FG.Geometry.radius(g::ScaledSphere) = g.r0

        s = ScaledSphere(6.371e6)
        ref = FG.Geometry.SphericalGeometry(6.371e6)
        Test.@test FG.Geometry.radius(s) == FG.Geometry.radius(ref)
        for (a, b) in ((( 0.0, 0.0), (0.0, π / 2)), ((0.3, -0.4), (2.9, 1.1)))
            Test.@test FG.Geometry.distance(s, a, b) == FG.Geometry.distance(ref, a, b)
        end
        Test.@test FG.Geometry.spherical_to_cartesian(s, (0.3, 0.4)) ==
                   FG.Geometry.spherical_to_cartesian(ref, (0.3, 0.4))
        Test.@test FG.Geometry.area_element(s, 0.4, 0.01, 0.02) ==
                   FG.Geometry.area_element(ref, 0.4, 0.01, 0.02)
        Test.@test FG.Geometry.scale_factors(s, (0.3, 0.4)) ==
                   FG.Geometry.scale_factors(ref, (0.3, 0.4))
        # …and the whole grid stack downstream of it, including the cell measure.
        λ = range(0.0; step = 2π / 8, length = 8)
        φ = range(-1.2; step = 0.3, length = 9)
        gs, gr = FG.Grids.StructuredGrid(s, λ, φ), FG.Grids.StructuredGrid(ref, λ, φ)
        Test.@test FG.Grids.measure_array(gs) == FG.Grids.measure_array(gr)
        Test.@test sum(FG.Grids.measure(gs)) == sum(FG.Grids.measure(gr))
        Test.@test FG.Grids.area(gs, 3, 4) == FG.Grids.area(gr, 3, 4)
        Test.@test FG.Grids.coords(gs, 3, 4) == FG.Grids.coords(gr, 3, 4)

        # An ellipsoid supplies the two shape parameters; everything else is derived.
        struct MySpheroid{T} <: FG.Geometry.AbstractEllipsoidalGeometry{T}
            eq::T
            flat::T
        end
        FG.Geometry.semimajor_axis(g::MySpheroid) = g.eq
        FG.Geometry.flattening(g::MySpheroid) = g.flat

        e = MySpheroid(6378137.0, inv(298.257223563))
        eref = FG.Geometry.SpheroidGeometry()
        for f in (FG.Geometry.semimajor_axis, FG.Geometry.flattening, FG.Geometry.semiminor_axis,
                  FG.Geometry.eccentricity²)
            Test.@test f(e) == f(eref)
        end
        Test.@test FG.Geometry.prime_vertical_radius(e, 0.7) ==
                   FG.Geometry.prime_vertical_radius(eref, 0.7)
        Test.@test FG.Geometry.meridional_radius(e, 0.7) ==
                   FG.Geometry.meridional_radius(eref, 0.7)
        Test.@test FG.Geometry.distance(e, (0.0, 0.0), (0.5, 0.6)) ==
                   FG.Geometry.distance(eref, (0.0, 0.0), (0.5, 0.6))
        Test.@test FG.Geometry.area_element(e, 0.4, 0.01, 0.02) ==
                   FG.Geometry.area_element(eref, 0.4, 0.01, 0.02)
        Test.@test FG.Geometry.scale_factors(e, (0.3, 0.4)) ==
                   FG.Geometry.scale_factors(eref, (0.3, 0.4))
    end

    Test.@testset "Cartesian geometry" begin
        geom = FG.Geometry.CartesianGeometry()
        p1 = (0.0, 0.0)
        p2 = (3000.0, 4000.0)
        Test.@test FG.Geometry.distance(geom, p1, p2) ≈ 5000.0
        # The geometry carries no spacing: a cell's extents are the caller's (in practice, the grid's).
        Test.@test FG.Geometry.area_element(geom, 1000.0, 1000.0) ≈ 1.0e6
        Test.@test FG.Geometry.volume_element(geom, 1.0, 2.0, 3.0) ≈ 6.0
        Test.@test FG.Geometry.CartesianGeometry(Float32) === FG.Geometry.CartesianGeometry{Float32}()
        Test.@test FG.Geometry.CartesianGeometry{Float32}() isa FG.Geometry.AbstractCartesianGeometry{Float32}
    end

    Test.@testset "Spherical geometry + Cartesian" begin
        geom = FG.Geometry.SphericalGeometry(6.371e6)
        london = (deg2rad(-0.1276), deg2rad(51.5074))
        paris = (deg2rad(2.3522), deg2rad(48.8566))
        d_km = FG.Geometry.distance(geom, london, paris) / 1000.0
        Test.@test 300 < d_km < 400

        λ, φ = 0.3, 0.4
        uλ, uφ = 1.0, -0.5
        p = FG.Geometry.vector_to_cartesian(geom, uλ, uφ, λ, φ)
        back = FG.Geometry.vector_from_cartesian(geom, p, λ, φ)
        Test.@test back.λ ≈ uλ
        Test.@test back.φ ≈ uφ
        Test.@test abs(back.r) < 1e-12
    end

    Test.@testset "nonuniform_first_derivative" begin
        # Uniform: recovers (f_p - f_m)/(2h)
        Test.@test FG.Geometry.nonuniform_first_derivative(1.0, 2.0, 3.0, 1.0, 1.0) ≈ 1.0
        # Exact for quadratic on nonuniform stencil
        # f(x)=x^2 at x=0 with neighbors -1 and 2: f'=2x=0 at center... use x=-1,0,2
        f_m, f_0, f_p = 1.0, 0.0, 4.0  # (-1)^2, 0, 2^2
        Test.@test FG.Geometry.nonuniform_first_derivative(f_m, f_0, f_p, 1.0, 2.0) ≈ 0.0 atol = 1e-12
    end

    Test.@testset "StructuredGrid preserves uniform spacing" begin
        geom = FG.Geometry.CartesianGeometry()
        x = 0.0:2000.0:20_000.0
        y = 0.0:2000.0:10_000.0
        mask = trues(length(x), length(y))
        grid = FG.Grids.StructuredGrid(geom, x, y, mask)
        # What a range axis is FOR is the guarantee of constant spacing, so that is what is asserted:
        # the property survives, the spacing is exactly the one asked for, and the samples are equal
        # to the input. The concrete container is an implementation detail (`Axes.UniformAxis`, which
        # unlike `StepRangeLen` keeps its arithmetic in the grid's own element type).
        Test.@test FG.Grids.isuniform(grid)
        Test.@test FG.Grids.isuniform(grid, 1) && FG.Grids.isuniform(grid, 2)
        Test.@test FG.Grids.spacing(grid, 1) === 2000.0
        Test.@test FG.Grids.spacing(grid, 2) === 2000.0
        Test.@test collect(grid.x) == collect(x)
        Test.@test collect(grid.y) == collect(y)
        Test.@test FG.Grids.bounds(grid, 1) == (0.0, 20_000.0)
        Test.@test FG.Grids.extent(grid, 1) == 20_000.0
        Test.@test FG.Grids.origin(grid, 1) == 0.0
        # A uniform axis has one gap, so both ends of its range of gaps are that gap.
        Test.@test FG.Grids.minimum_spacing(grid, 1) == FG.Grids.maximum_spacing(grid, 1) == 2000.0
        Test.@test FG.Grids.size_tuple(grid) == (length(x), length(y))
        Test.@test FG.Grids.area(grid, 2, 2) ≈ 2000.0 * 2000.0
        Test.@test FG.Grids.coords(grid, 2, 3) == (x = 2000.0, y = 4000.0)
        Test.@test FG.Grids.coords(NTuple{2,Float64}, grid, 2, 3) == (2000.0, 4000.0)
        out = zeros(2)
        FG.Grids.coords!(out, grid, 2, 3)
        Test.@test out == [2000.0, 4000.0]
        Test.@test FG.Grids.isactive(grid, 1, 1)
        Test.@test FG.Grids.grid_geometry(grid) === geom
        Test.@test !FG.Grids.isperiodic(grid, 1)  # Cartesian not auto-periodic
    end

    Test.@testset "StructuredGrid with Vector axes is still structured" begin
        geom = FG.Geometry.CartesianGeometry()
        x = collect(0.0:1.0:4.0)
        y = collect(0.0:1.0:3.0)
        grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
        Test.@test grid isa FG.Grids.StructuredGrid
        Test.@test grid.x isa Vector
        Test.@test grid.y isa Vector
        Test.@test FG.Grids.size_tuple(grid) == (5, 4)
    end

    Test.@testset "Spherical StructuredGrid auto-periodic longitude" begin
        R = 6.371e6
        geom = FG.Geometry.SphericalGeometry(R)
        # Full-circle longitude (closed to within one cell)
        nλ, nφ = 8, 5
        dλ = 2π / nλ
        lon = range(0.0; step = dλ, length = nλ)
        lat = range(-π / 4, π / 4; length = nφ)
        grid = FG.Grids.StructuredGrid(geom, lon, lat, trues(nλ, nφ))
        Test.@test FG.Grids.isperiodic(grid, 1)
        Test.@test !FG.Grids.isperiodic(grid, 2)
        Test.@test FG.Grids.isuniform(grid, 1)
        Test.@test FG.Grids.spacing(grid, 1) ≈ dλ
        Test.@test all(FG.Grids.area(grid, i, j) > 0 for i in 1:nλ, j in 1:nφ)
        p = FG.Grids.coords(grid, 1, 1)
        Test.@test keys(p) == (:λ, :φ)
        Test.@test p.λ ≈ 0.0
        Test.@test p.φ ≈ -π / 4
    end

    Test.@testset "Coordinate names follow the geometry, never stand in for each other" begin
        cgeom = FG.Geometry.CartesianGeometry()
        sgeom = FG.Geometry.SphericalGeometry(6.371e6)
        xs = collect(0.0:1.0:4.0)
        cgrid = FG.Grids.StructuredGrid(cgeom, xs, xs, trues(5, 5))
        sgrid = FG.Grids.StructuredGrid(sgeom, deg2rad.(xs), deg2rad.(xs), trues(5, 5))

        # A spherical grid holds longitude/latitude and says so; asking it for `x` is an error
        # rather than silently handing back λ.
        Test.@test FG.Grids.coordinate_names(cgrid) == (:x, :y)
        Test.@test FG.Grids.coordinate_names(sgrid) == (:λ, :φ)
        Test.@test cgrid.x === FG.Grids.coordinates(cgrid, 1)
        Test.@test sgrid.λ === FG.Grids.coordinates(sgrid, 1)
        Test.@test sgrid.φ === FG.Grids.coordinates(sgrid, 2)
        Test.@test_throws FieldError sgrid.x
        Test.@test_throws FieldError cgrid.λ
        Test.@test :λ in propertynames(sgrid)
        Test.@test :x in propertynames(cgrid)

        # `axis` is the rectilinear spelling of the same thing; `coordinates` works on every
        # architecture, including ones with no axes at all.
        Test.@test FG.Grids.axis(sgrid, 2) === FG.Grids.coordinates(sgrid, 2)
        Test.@test FG.Grids.coordinates(cgrid) === (FG.Grids.coordinates(cgrid, 1), FG.Grids.coordinates(cgrid, 2))

        nx, ny = 4, 3
        xm = [Float64(i) for i in 1:nx, j in 1:ny]
        ym = [Float64(j) for i in 1:nx, j in 1:ny]
        cv = FG.Grids.CurvilinearGrid(sgeom, deg2rad.(xm), deg2rad.(ym), trues(nx, ny))
        Test.@test FG.Grids.coordinate_names(cv) == (:λ, :φ)
        Test.@test cv.λ === FG.Grids.coordinates(cv, 1)
        Test.@test_throws FieldError cv.x
        Test.@test size(FG.Grids.corners(cv, 1)) == (nx + 1, ny + 1)
        Test.@test keys(FG.Grids.corner_coords(cv, 1, 1)) == (:λ, :φ)

        un = FG.Grids.UnstructuredGrid(sgeom, [0.0, 0.1, 0.2], [0.0, 0.1, 0.0], [1.0, 1.0, 1.0], trues(3))
        Test.@test FG.Grids.coordinate_names(un) == (:λ, :φ)
        Test.@test un.λ == [0.0, 0.1, 0.2]
        Test.@test_throws FieldError un.y
    end

    Test.@testset "Grids implement the Base collection surface" begin
        geom = FG.Geometry.CartesianGeometry()
        xs = collect(0.0:1.0:4.0)
        ys = collect(0.0:1.0:3.0)
        grid = FG.Grids.StructuredGrid(geom, xs, ys, trues(5, 4))
        Test.@test size(grid) == (5, 4)
        Test.@test size(grid, 2) == 4
        Test.@test length(grid) == 20
        Test.@test ndims(grid) == 2
        Test.@test eltype(grid) === Float64
        Test.@test axes(grid) == (Base.OneTo(5), Base.OneTo(4))
        Test.@test FG.Grids.size_tuple(grid) == size(grid)
        # `show` summarizes rather than dumping every coordinate array.
        s = sprint(show, MIME"text/plain"(), grid)
        Test.@test occursin("StructuredGrid{Float64} 5×4", s)
        Test.@test occursin("20 active", s)
        Test.@test count(==('\n'), s) < 10
        Test.@test sprint(show, grid) == "StructuredGrid{Float64}(5×4)"
    end

    Test.@testset "Cell measure is a separable outer product matching the metric formulas" begin
        cw = FG.Grids._cell_width
        sgeo = FG.Geometry.SphericalGeometry(6.371e6)
        cgeo = FG.Geometry.CartesianGeometry()
        cgeo3 = FG.Geometry.CartesianGeometry()

        # Spherical area must equal R²cosφ·Δλ·Δφ cell by cell, on a nonuniform grid.
        λ = collect(range(0.0; step = 2π / 12, length = 12))
        φ = cumsum([-1.0, 0.2, 0.5, 0.15, 0.4, 0.3])
        g = FG.Grids.StructuredGrid(sgeo, λ, φ, trues(length(λ), length(φ)))
        λper = FG.Grids.isperiodic(g, 1) ? 2π : nothing
        ref = [FG.Geometry.area_element(sgeo, φ[j], cw(λ, i, λper), cw(φ, j))
               for i in eachindex(λ), j in eachindex(φ)]
        Test.@test FG.Grids.measure(g) ≈ ref rtol = 1e-14

        # Spherical volume must equal r²cosφ·Δλ·Δφ·Δr.
        r = collect(6.30e6:1.0e4:6.34e6)
        g3 = FG.Grids.StructuredGrid(sgeo, λ, φ, r, trues(length(λ), length(φ), length(r)))
        ref3 = [FG.Geometry.volume_element(sgeo, r[k], φ[j], cw(λ, i, λper), cw(φ, j), cw(r, k))
                for i in eachindex(λ), j in eachindex(φ), k in eachindex(r)]
        Test.@test FG.Grids.measure(g3) ≈ ref3 rtol = 1e-14

        # A degenerate angular axis drops the differential it no longer has, rather than substituting
        # a placeholder into the 2-D area formula.
        zonal = FG.Grids.StructuredGrid(sgeo, λ, [0.4], trues(length(λ), 1))
        Test.@test FG.Grids.measure(zonal) ≈ [sgeo.R * cos(0.4) * cw(λ, i, 2π) for i in eachindex(λ), _ in 1:1]
        merid = FG.Grids.StructuredGrid(sgeo, [0.3], φ, trues(1, length(φ)))
        Test.@test FG.Grids.measure(merid) ≈ [sgeo.R * cw(φ, j) for _ in 1:1, j in eachindex(φ)]

        # A whole sphere's cells sum to 4πR².
        n = 200
        λf = collect(range(0.0; step = 2π / n, length = n))
        φf = collect(range(-π / 2 + π / (2n), π / 2 - π / (2n); length = n))
        gf = FG.Grids.StructuredGrid(sgeo, λf, φf, trues(n, n))
        Test.@test sum(FG.Grids.measure(gf)) ≈ 4π * sgeo.R^2 rtol = 1e-4

        # A periodic axis wraps its boundary cell in 3-D exactly as it already did in 2-D.
        xnu = cumsum([0.0, 10.0, 40.0, 15.0, 60.0, 25.0])
        yy = collect(0.0:50.0:100.0)
        zz = collect(0.0:10.0:20.0)
        # A NONUNIFORM periodic Cartesian direction has no period to infer — its samples do not
        # determine the seam gap — so it must be stated rather than taken from whichever gap is first.
        xper = FG.Grids._cartesian_period(xnu)
        Test.@test_throws ArgumentError FG.Grids.StructuredGrid(
            cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz)); periodic = true)
        gp = FG.Grids.StructuredGrid(cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz));
                               periodic = true, period = xper)
        gn = FG.Grids.StructuredGrid(cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz)); periodic = false)
        Test.@test FG.Grids.measure(gp) ≈ [cw(xnu, i, xper) * cw(yy, j) * cw(zz, k)
                                     for i in eachindex(xnu), j in eachindex(yy), k in eachindex(zz)]
        Test.@test FG.Grids.measure(gp) != FG.Grids.measure(gn)
        # …and 2-D and 3-D agree on that wrapped width.
        g2p = FG.Grids.StructuredGrid(cgeo, xnu, yy, trues(length(xnu), length(yy));
                                periodic = true, period = xper)
        Test.@test FG.Grids.measure(g2p)[end, 1] / cw(yy, 1) ≈ FG.Grids.measure(gp)[end, 1, 1] / (cw(yy, 1) * cw(zz, 1))
    end

    Test.@testset "measure is the dimension-agnostic name for area/volume" begin
        geom3 = FG.Geometry.CartesianGeometry()
        x = 0.0:1.0:3.0
        z = 0.0:1.0:2.0
        g3 = FG.Grids.StructuredGrid(geom3, x, x, z, trues(4, 4, 3))
        Test.@test FG.Grids.measure(g3, 2, 2, 2) ≈ 1.0
        Test.@test FG.Grids.measure(g3, 2, 2, 2) == FG.Grids.area(g3, 2, 2, 2)
        Test.@test size(FG.Grids.measure(g3)) == (4, 4, 3)
    end

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
        Test.@test length(Set(FG.Grids._local_spacing(ax, i)[2] for i in 1:4)) == 1
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
        # `_local_spacing` returns `step` rather than differencing, so the gap is exactly constant.
        Test.@test length(Set(FG.Grids._local_spacing(ax, i)[2] for i in 1:8)) == 1
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

        # The reason for the subtyping: Base solves `searchsorted` on a range in closed form, so the
        # cost does not grow with the axis length the way bisection does.
        best(f) = (f(); minimum(@elapsed(f()) for _ in 1:2000))
        t_small = best(() -> searchsortedfirst(A.UniformAxis(0.0, 1e-1, 10), 0.5))
        t_large = best(() -> searchsortedfirst(A.UniformAxis(0.0, 1e-7, 10^7), 0.5))
        Test.@test t_large < 2 * t_small
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
        dense = FG.Grids._axis_widths_dense(collect(FG.Grids.coordinates(g, 1)))
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
            w = FG.Grids._axis_widths(c)
            Test.@test collect(w) ≈ [FG.Grids._cell_width(c.data, i) for i in 1:n]
            # Bulk broadcasts do touch elements, but the count must be O(n) with a small constant
            # and never O(n) per output element.
            Test.@test c.scalar_reads <= 4n + 16
        end

        # The endpoint handling is the only genuinely scalar part, and it is O(1).
        n = 4000
        c1 = CountingVector(collect(range(0.0; step = 1.0, length = n)))
        c2 = CountingVector(collect(range(0.0; step = 1.0, length = 2n)))
        FG.Grids._axis_widths(c1)
        FG.Grids._axis_widths(c2)
        Test.@test c2.scalar_reads - c1.scalar_reads <= 4n + 16   # grows linearly, not quadratically
    end

    Test.@testset "UnstructuredGrid carries a caller-chosen index type" begin
        geom = FG.Geometry.CartesianGeometry()
        x = [0.0, 1.0, 0.0]
        y = [0.0, 0.0, 1.0]
        areas = [0.5, 0.5, 0.5]
        # Int32 CSR indices: half the memory of Int64 on a large mesh, and the width GPU kernels use.
        nbrs = Int32[2, 3, 1, 1]
        ptr = Int32[1, 3, 4, 5]
        grid = FG.Grids.UnstructuredGrid(geom, x, y, areas, trues(3), nbrs, ptr)
        Test.@test eltype(grid.neighbor_nbrs) === Int32
        Test.@test collect(FG.Grids.neighbors(grid, 1)) == Int32[2, 3]
        Test.@test FG.Grids.measure(grid, 2) ≈ 0.5
        # Mismatched adjacency length is caught at construction.
        Test.@test_throws ArgumentError FG.Grids.UnstructuredGrid(geom, x, y, areas, trues(3), nbrs, Int32[1, 3])
    end

    Test.@testset "1D and 3D StructuredGrid" begin
        geom = FG.Geometry.CartesianGeometry()
        x = 0.0:1.0:9.0
        g1 = FG.Grids.StructuredGrid(geom, x, trues(length(x)))
        Test.@test FG.Grids.size_tuple(g1) == (10,)
        Test.@test FG.Grids.area(g1, 2) ≈ 1.0

        geom3 = FG.Geometry.CartesianGeometry()
        z = 0.0:1.0:4.0
        g3 = FG.Grids.StructuredGrid(geom3, x, x, z, trues(length(x), length(x), length(z)))
        Test.@test FG.Grids.size_tuple(g3) == (10, 10, 5)
        Test.@test FG.Grids.area(g3, 2, 2, 2) ≈ 1.0
    end

    Test.@testset "CurvilinearGrid" begin
        geom = FG.Geometry.CartesianGeometry()
        nx, ny = 4, 3
        x = [Float64(i) for i in 1:nx, j in 1:ny]
        y = [Float64(j) for i in 1:nx, j in 1:ny]
        grid = FG.Grids.CurvilinearGrid(geom, x, y, trues(nx, ny))
        Test.@test FG.Grids.size_tuple(grid) == (nx, ny)
        Test.@test FG.Grids.coords(grid, 2, 2) == (x = 2.0, y = 2.0)
        Test.@test FG.Grids.area(grid, 2, 2) > 0
    end

    Test.@testset "UnstructuredGrid explicit adjacency" begin
        geom = FG.Geometry.CartesianGeometry()
        x = [0.0, 1.0, 0.0]
        y = [0.0, 0.0, 1.0]
        areas = [0.5, 0.5, 0.5]
        mask = trues(3)
        # CSR: node 1 → [2,3], node 2 → [1], node 3 → [1]
        nbrs = [2, 3, 1, 1]
        ptr = [1, 3, 4, 5]
        grid = FG.Grids.UnstructuredGrid(geom, x, y, areas, mask, nbrs, ptr)
        Test.@test FG.Grids.size_tuple(grid) == (3,)
        Test.@test collect(FG.Grids.neighbors(grid, 1)) == [2, 3]
        Test.@test FG.Grids.coords(grid, 2) == (x = 1.0, y = 0.0)

        # Convenience no-neighbor constructor
        g0 = FG.Grids.UnstructuredGrid(geom, x, y, areas, mask)
        Test.@test isempty(FG.Grids.neighbors(g0, 1))
    end

    Test.@testset "UnstructuredGrid auto-build works with the extensions loaded" begin
        # With the extensions loaded, the auto-build must actually work, not merely throw without them.
        geom = FG.Geometry.CartesianGeometry()
        n = 24
        x = [0.5 + 0.4cos(2π * k / n) for k in 1:n]
        y = [0.5 + 0.4sin(2π * k / n) for k in 1:n]
        g = FG.Grids.UnstructuredGrid(geom, x, y, trues(n); k = 4)
        Test.@test FG.Grids.size_tuple(g) == (n,)
        Test.@test all(1 ≤ length(FG.Grids.neighbors(g, i)) ≤ 4 for i in 1:n)
        Test.@test all(>(0), FG.Grids.measure(g))          # Voronoi areas, via DelaunayTriangulation
        Test.@test all(isfinite, FG.Grids.measure(g))
    end

    Test.@testset "local_tangent_basis / project_to_tangent_plane" begin
        cgeom = FG.Geometry.CartesianGeometry()
        c = (0.0, 0.0)
        n = (1.0, 2.0)
        Test.@test FG.Geometry.project_to_tangent_plane(cgeom, c, n) == (; x = 1.0, y = 2.0)
        Test.@test FG.Geometry.distance(cgeom, (x = 0.0, y = 0.0), (x = 3.0, y = 4.0)) ≈ 5.0

        sgeom = FG.Geometry.SphericalGeometry(1.0)
        ê = FG.Geometry.local_tangent_basis(sgeom, (0.0, 0.0))
        Test.@test abs(sqrt(sum(abs2, ê.λ)) - 1) < 1e-14
        Test.@test abs(sqrt(sum(abs2, ê.φ)) - 1) < 1e-14
        Test.@test FG.Geometry.distance(sgeom, (λ = 0.0, φ = 0.0), (λ = 0.1, φ = 0.0)) ≈ sgeom.R * 0.1 atol = 1e-10
    end

    Test.@testset "Spherical sampling hierarchy" begin
        Test.@test FG.SphericalSampling.GaussLegendreSampling <: FG.SphericalSampling.AbstractGaussLegendreSampling
        Test.@test FG.SphericalSampling.DriscollHealySampling <: FG.SphericalSampling.AbstractDriscollHealySampling
        Test.@test FG.SphericalSampling.ClenshawCurtisSampling <: FG.SphericalSampling.AbstractClenshawCurtisSampling
        Test.@test FG.SphericalSampling.McEwenWiauxSampling <: FG.SphericalSampling.AbstractMcEwenWiauxSampling
        Test.@test FG.SphericalSampling.LatLonSampling <: FG.SphericalSampling.AbstractLatLonSampling
        Test.@test FG.SphericalSampling.HEALPixSampling <: FG.SphericalSampling.AbstractHEALPixSampling
        Test.@test FG.SphericalSampling.CubedSphereSampling <: FG.SphericalSampling.AbstractCubedSphereSampling
        Test.@test FG.SphericalSampling.IcosahedralSampling <: FG.SphericalSampling.AbstractIcosahedralSampling
        Test.@test FG.SphericalSampling.YinYangSampling <: FG.SphericalSampling.AbstractYinYangSampling
        Test.@test FG.SphericalSampling.ScatteredSphericalSampling <: FG.SphericalSampling.AbstractScatteredSphericalSampling

        Test.@test FG.SphericalSampling.is_tensor_product(FG.SphericalSampling.ClenshawCurtisSampling())
        Test.@test FG.SphericalSampling.is_iso_latitude(FG.SphericalSampling.HEALPixSampling(2))
        Test.@test FG.SphericalSampling.is_equal_area(FG.SphericalSampling.HEALPixSampling(2))
        Test.@test FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.GaussLegendreSampling())
        Test.@test !FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.LatLonSampling())
        Test.@test !FG.SphericalSampling.is_tensor_product(FG.SphericalSampling.HEALPixSampling(1))
    end

    Test.@testset "Clenshaw–Curtis = FastSphericalHarmonics sph_points" begin
        nθ = 8
        (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.ClenshawCurtisSampling(), nθ)
        Test.@test length(φ) == nθ
        Test.@test length(λ) == 2nθ - 1
        Test.@test FG.SphericalSampling.bandlimit(FG.SphericalSampling.ClenshawCurtisSampling(), nθ) == nθ - 1
        # θ = π(i−1/2)/N ; φ_geo = π/2 − θ
        θ = [π * (i - 0.5) / nθ for i in 1:nθ]
        Test.@test φ ≈ (π / 2 .- θ)
        Test.@test λ[1] ≈ 0
        Test.@test λ[2] - λ[1] ≈ 2π / (2nθ - 1)
        # Open: no poles
        Test.@test all(abs.(φ) .< π / 2 - 1e-12)
    end

    Test.@testset "Gauss–Legendre nodes / weights" begin
        nθ = 16
        (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.GaussLegendreSampling(), nθ)
        w = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.GaussLegendreSampling(), nθ)
        Test.@test length(λ) == 2nθ - 1
        Test.@test length(φ) == nθ
        Test.@test sum(w) ≈ 2 atol = 1e-12          # ∫_{-1}^{1} dμ
        Test.@test FG.SphericalSampling.bandlimit(FG.SphericalSampling.GaussLegendreSampling(), nθ) == nθ - 1
        # μ = sin(φ) should be symmetric about 0
        μ = sin.(φ)
        Test.@test μ ≈ -reverse(μ) atol = 1e-12
    end

    Test.@testset "Gauss–Legendre solves once for axes and weights, and is exact to 2n-1" begin
        for n in (1, 2, 3, 7, 8, 64, 257)
            r = FG.SphericalSampling._gauss_legendre_μ(n)
            Test.@test issorted(r.μ)                       # ascending, as the eigen route was
            Test.@test r.μ ≈ -reverse(r.μ) atol = 1e-15    # roots come in ± pairs
            Test.@test all(>(0), r.w)
            Test.@test sum(r.w) ≈ 2 rtol = 1e-14
            # An n-point rule is exact for every polynomial up to degree 2n-1.
            for d in (0, 1, 2n - 2, 2n - 1)
                exact = isodd(d) ? 0.0 : 2 / (d + 1)
                Test.@test sum(r.w .* r.μ .^ d) ≈ exact atol = 1e-11
            end
        end
        # Odd n has its central node exactly at zero (and not at -0.0).
        Test.@test FG.SphericalSampling._gauss_legendre_μ(9).μ[5] === 0.0

        # Two regimes with a crossover at n = 60: the asymptotic expansion is a fixed-order Float64
        # coefficient set, poor for small n and unable to exceed Float64 precision; Newton is exact
        # arithmetic but O(n²). Each is used only where it is the better answer.
        SS = FG.SphericalSampling
        for n in (60, 61, 62)
            r = SS._gauss_legendre_μ(n)
            Test.@test issorted(r.μ)
            Test.@test sum(r.w) ≈ 2 rtol = 1e-14
            d = 2n - 2
            Test.@test sum(r.w .* r.μ .^ d) ≈ 2 / (d + 1) rtol = 1e-11
        end
        # Both regimes solve the same problem, so above the crossover they must still agree — to the
        # Newton's accuracy, which is the looser of the two there.
        for n in (64, 200, 1024)
            m = (n + 1) ÷ 2
            asy = SS._gauss_legendre_μ(n)                      # takes the asymptotic path
            mμ = Vector{Float64}(undef, n); mw = Vector{Float64}(undef, n)
            SS._gauss_legendre_newton!(Float64, Float64, mμ, mw, n, m)
            Test.@test maximum(abs.(asy.μ .- mμ)) < 1e-13
            Test.@test maximum(abs.(asy.w .- mw) ./ mw) < 1e-9
            Test.@test maximum(abs.(asy.μ .+ reverse(asy.μ))) < 1e-15   # ± symmetric
            Test.@test sum(mw) ≈ 2 rtol = 1e-13
        end
        # Each output stays independently optional on the asymptotic path too.
        Test.@test SS._gauss_legendre_asy!(Float64, nothing, nothing, 128, 64) === nothing
        # A wider element type must NOT take the Float64 coefficient path — it would cap precision.
        μb, wb = setprecision(BigFloat, 192) do
            μ = Vector{BigFloat}(undef, 128); w = Vector{BigFloat}(undef, 128)
            SS._gauss_legendre_μ!(μ, w); (μ, w)
        end
        Test.@test abs(sum(wb) - 2) < 1e-40        # far past what Float64 coefficients could give
        Test.@test eltype(wb) === BigFloat
        # …and the Float64 result agrees with that reference to Float64 precision.
        r64 = SS._gauss_legendre_μ(128)
        Test.@test maximum(abs.(r64.w .- Float64.(wb)) ./ Float64.(wb)) < 1e-14
        Test.@test maximum(abs.(r64.μ .- Float64.(μb))) < 1e-15

        # The Bonnet recurrence sums n terms, so Float32 cannot resolve the roots on its own; the
        # solve runs at Float64 and rounds once, keeping the result correctly rounded.
        r32 = FG.SphericalSampling._gauss_legendre_μ(64; T = Float32)
        r64 = FG.SphericalSampling._gauss_legendre_μ(64)
        Test.@test eltype(r32.μ) === Float32
        Test.@test maximum(abs.(Float64.(r32.w) .- r64.w) ./ r64.w) < 4 * eps(Float32)
        Test.@test maximum(abs.(Float64.(r32.μ) .- r64.μ)) < 4 * eps(Float32)

        # One solve serves both, and agrees with the two separate entry points.
        for n in (5, 32)
            q = FG.SphericalSampling.spherical_quadrature(FG.SphericalSampling.GaussLegendreSampling(), n)
            a = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.GaussLegendreSampling(), n)
            Test.@test q.λ == a.λ
            Test.@test q.φ == a.φ
            Test.@test q.w == FG.SphericalSampling.latitude_weights(FG.SphericalSampling.GaussLegendreSampling(), n)
        end
        # Non-GL samplings have independent closed forms and take the generic method.
        for s in (FG.SphericalSampling.DriscollHealySampling(), FG.SphericalSampling.DriscollHealyEqualSampling(), FG.SphericalSampling.ClenshawCurtisSampling())
            q = FG.SphericalSampling.spherical_quadrature(s, 12)
            a = FG.SphericalSampling.spherical_axes(s, 12)
            Test.@test q.φ == a.φ && q.λ == a.λ
            Test.@test q.w == FG.SphericalSampling.latitude_weights(s, 12)
        end
        # McEwen–Wiaux has nodes but deliberately no weights, so the combined form must refuse too
        # rather than inventing a rule that is not exact even at l = 0.
        Test.@test_throws ArgumentError FG.SphericalSampling.spherical_quadrature(FG.SphericalSampling.McEwenWiauxSampling(), 12)
        # The in-place form writes into caller buffers and allocates no scratch of its own.
        n = 64
        sz = FG.SphericalSampling.axes_lengths(FG.SphericalSampling.GaussLegendreSampling(), n)
        λb = Vector{Float64}(undef, sz.nlon)
        φb = Vector{Float64}(undef, sz.nlat)
        wb = Vector{Float64}(undef, sz.nlat)
        FG.SphericalSampling.spherical_quadrature!(λb, φb, wb, FG.SphericalSampling.GaussLegendreSampling(), n)
        Test.@test φb == FG.SphericalSampling.spherical_quadrature(FG.SphericalSampling.GaussLegendreSampling(), n).φ
        # Only the returned NamedTuple; the solve itself needs O(1) scratch, not the O(n²)
        # eigenvector matrix a Golub–Welsch decomposition would.
        nalloc(f) = (f(); Base.gc_alloc_count((@timed f()).gcstats))
        Test.@test nalloc(() -> FG.SphericalSampling.spherical_quadrature!(λb, φb, wb, FG.SphericalSampling.GaussLegendreSampling(), n)) <= 1
        Test.@test nalloc(() -> FG.SphericalSampling._gauss_legendre_μ!(φb, wb)) <= 1
        # O(1) scratch means the count cannot grow with n.
        big = Vector{Float64}(undef, 4n)
        bigw = similar(big)
        Test.@test nalloc(() -> FG.SphericalSampling._gauss_legendre_μ!(big, bigw)) <= 1
        Test.@test_throws DimensionMismatch FG.SphericalSampling.spherical_quadrature!(
            λb, φb, Vector{Float64}(undef, n + 1), FG.SphericalSampling.GaussLegendreSampling(), n)
    end

    Test.@testset "Driscoll–Healy DH1 / DH2" begin
        lmax = 5
        nlat = FG.SphericalSampling.nlat_for_bandlimit(FG.SphericalSampling.DriscollHealySampling(), lmax)
        Test.@test nlat == 2 * (lmax + 1)
        ax = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.DriscollHealySampling(), nlat)
        λ2 = ax.λ
        φ = ax.φ
        ax = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.DriscollHealyEqualSampling(), nlat)
        λ1 = ax.λ
        φ1 = ax.φ
        Test.@test length(λ2) == 2nlat
        Test.@test length(λ1) == nlat
        Test.@test φ ≈ φ1
        Test.@test φ[1] ≈ π / 2 atol = 1e-14       # north pole
        Test.@test φ[end] > -π / 2 + 1e-10         # south excluded
        w = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), nlat)
        Test.@test w[1] ≈ 0 atol = 1e-14           # north pole weight vanishes
        Test.@test FG.SphericalSampling.bandlimit(FG.SphericalSampling.DriscollHealySampling(), nlat) == lmax
    end

    Test.@testset "McEwen–Wiaux axes" begin
        lmax = 7
        nlat = FG.SphericalSampling.nlat_for_bandlimit(FG.SphericalSampling.McEwenWiauxSampling(), lmax)
        (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.McEwenWiauxSampling(), nlat)
        L = lmax + 1
        Test.@test length(φ) == L
        Test.@test length(λ) == 2L - 1
        θ = FG.SphericalSampling.colatitude.(φ)
        Test.@test θ[1] ≈ π / (2L - 1)
        Test.@test θ[end] ≈ π
    end

    Test.@testset "LatLonSampling" begin
        (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.LatLonSampling(), 5; nlon = 8)
        Test.@test length(λ) == 8
        Test.@test length(φ) == 5
        Test.@test φ[1] ≈ -π / 2
        Test.@test φ[end] ≈ π / 2
    end

    Test.@testset "HEALPix" begin
        s = FG.SphericalSampling.HEALPixSampling(1)
        Test.@test FG.SphericalSampling.healpix_npix(s) == 12
        Test.@test FG.SphericalSampling.healpix_nring(s) == 3
        Test.@test FG.SphericalSampling.healpix_pixel_area(s) ≈ 4π / 12
        (; λ, φ) = FG.SphericalSampling.spherical_points(s)
        Test.@test length(λ) == 12
        Test.@test length(φ) == 12
        s4 = FG.SphericalSampling.HEALPixSampling(4)
        Test.@test FG.SphericalSampling.healpix_npix(s4) == 12 * 16
        ax = FG.SphericalSampling.spherical_points(s4)
        λ4 = ax.λ
        φ4 = ax.φ
        Test.@test length(λ4) == 192
        # Equal-area: all pixels same area by construction; centers on |φ| < π/2
        Test.@test all(abs.(φ4) .≤ π / 2 + 1e-12)
        # nside=32 first ring: 4 pixels at same latitude, φ = 45° + 90°k
        ax = FG.SphericalSampling.spherical_points(FG.SphericalSampling.HEALPixSampling(32))
        λ32 = ax.λ
        φ32 = ax.φ
        Test.@test length(unique(round.(φ32[1:4]; digits = 12))) == 1
        Test.@test rad2deg.(λ32[1:4]) ≈ [45.0, 135.0, 225.0, 315.0] atol = 1e-8
        Test.@test rad2deg(FG.SphericalSampling.colatitude(φ32[1])) ≈ 1.46197116 atol = 1e-5
    end

    Test.@testset "Cubed sphere / Yin–Yang / icosahedral" begin
        (; λ, φ, panel) = FG.SphericalSampling.cubed_sphere_points(4)
        Test.@test length(λ) == 6 * 16
        Test.@test extrema(panel) == (1, 6)
        Test.@test all(abs.(φ) .≤ π / 2 + 1e-10)

        yy = FG.SphericalSampling.yin_yang_panels(6, 4)
        Test.@test length(yy.yin.λ) == 6
        Test.@test length(yy.yin.φ) == 4
        Test.@test size(yy.yang.λ) == (6, 4)
        Test.@test size(yy.yang.φ) == (6, 4)

        ax = FG.SphericalSampling.icosahedral_vertices(1)

        λi = ax.λ

        φi = ax.φ
        Test.@test length(λi) == 12
        ax = FG.SphericalSampling.icosahedral_vertices(2)
        λ2 = ax.λ
        φ2 = ax.φ
        # ν=2 geodesic: 10ν²+2 = 42 vertices
        Test.@test length(λ2) == 42
    end

    Test.@testset "Sampling → StructuredGrid" begin
        geom = FG.Geometry.SphericalGeometry(1.0)
        (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.ClenshawCurtisSampling(), 6)
        grid = FG.Grids.StructuredGrid(geom, λ, φ, trues(length(λ), length(φ)))
        Test.@test FG.Grids.isperiodic(grid, 1)
        Test.@test FG.Grids.size_tuple(grid) == (length(λ), length(φ))
    end


    Test.@testset "bang spherical_axes! / points!" begin
        nθ = 8
        sz = FG.SphericalSampling.axes_lengths(FG.SphericalSampling.ClenshawCurtisSampling(), nθ)
        λ = Vector{Float64}(undef, sz.nlon)
        φ = Vector{Float64}(undef, sz.nlat)
        FG.SphericalSampling.spherical_axes!(λ, φ, FG.SphericalSampling.ClenshawCurtisSampling(), nθ)
        ax = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.ClenshawCurtisSampling(), nθ)
        λ2 = ax.λ
        φ2 = ax.φ
        Test.@test λ == λ2
        Test.@test φ == φ2

        n = FG.SphericalSampling.npoints(FG.SphericalSampling.HEALPixSampling(2))
        Λ = Vector{Float64}(undef, n)
        Φ = Vector{Float64}(undef, n)
        FG.SphericalSampling.spherical_points!(Λ, Φ, FG.SphericalSampling.HEALPixSampling(2))
        ax = FG.SphericalSampling.spherical_points(FG.SphericalSampling.HEALPixSampling(2))
        Λ2 = ax.λ
        Φ2 = ax.φ
        Test.@test Λ == Λ2
        Test.@test Φ == Φ2

        w = Vector{Float64}(undef, 12)
        FG.SphericalSampling.latitude_weights!(w, FG.SphericalSampling.GaussLegendreSampling(), 12)
        Test.@test sum(w) ≈ 2 atol = 1e-12
    end

    Test.@testset "Connectivity CSR / structured neighbors" begin
        geom = FG.Geometry.CartesianGeometry()
        grid = FG.Grids.StructuredGrid(geom, 0.0:1.0:2.0, 0.0:1.0:2.0, trues(3, 3); periodic = (false, false))
        nbr = FG.Grids.neighbors(grid, 2, 2)
        Test.@test Set(nbr) == Set([
            FG.Connectivity.linear_index(grid, 1, 2),
            FG.Connectivity.linear_index(grid, 3, 2),
            FG.Connectivity.linear_index(grid, 2, 1),
            FG.Connectivity.linear_index(grid, 2, 3),
        ])
        Test.@test FG.Connectivity.nneighbors(grid, 2, 2) == 4
        Test.@test FG.Connectivity.nneighbors(grid, 1, 1) == 2
        out = Vector{Int}(undef, 4)
        Test.@test FG.Connectivity.neighbors!(out, grid, 2, 2) == 4
        Test.@test Set(out) == Set(nbr)

        conn = FG.Connectivity.build_connectivity(grid)
        Test.@test conn isa FG.Connectivity.CSRConnectivity
        Test.@test FG.Connectivity.nnodes(conn) == 9
        Test.@test Set(FG.Grids.neighbors(conn, FG.Connectivity.linear_index(grid, 2, 2))) == Set(nbr)
        Test.@test FG.Connectivity.nedges(conn) == sum(FG.Connectivity.nneighbors(grid, Tuple(ci)...) for ci in CartesianIndices((3, 3)))

        mask = trues(3, 3)
        mask[2, 2] = false
        g2 = FG.Grids.StructuredGrid(geom, 0.0:1.0:2.0, 0.0:1.0:2.0, mask)
        Test.@test FG.Connectivity.nneighbors(g2, 2, 2) == 0
        Test.@test FG.Connectivity.nneighbors(g2, 1, 2) == 2

        ug = FG.Grids.UnstructuredGrid(geom, [0.0, 1.0, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], trues(3),
            [2, 3, 1, 3, 1, 2], [1, 3, 5, 7])
        uc = FG.Connectivity.build_connectivity(ug)
        Test.@test Set(FG.Grids.neighbors(ug, 1)) == Set([2, 3])
        Test.@test Set(FG.Grids.neighbors(uc, 1)) == Set([2, 3])

        A = FG.Connectivity.adjacency_matrix(grid)
        Test.@test A isa Matrix{Bool}
        Test.@test size(A) == (9, 9)
        Test.@test A[FG.Connectivity.linear_index(grid, 2, 2), FG.Connectivity.linear_index(grid, 2, 3)]
        Abuf = falses(9, 9)
        FG.Connectivity.adjacency_matrix!(Abuf, grid)
        Test.@test Abuf == A
        FG.Connectivity.adjacency_matrix!(Abuf, conn)
        Test.@test Abuf == A
    end

    Test.@testset "Curvilinear periodicity" begin
        geom = FG.Geometry.CartesianGeometry()
        x = [Float64(i) for i in 1:3, j in 1:2]
        y = [Float64(j) for i in 1:3, j in 1:2]
        g = FG.Grids.CurvilinearGrid(geom, x, y, trues(3, 2); periodic = (true, false))
        Test.@test FG.Grids.isperiodic(g, 1)
        Test.@test !FG.Grids.isperiodic(g, 2)
        Test.@test FG.Connectivity.nneighbors(g, 1, 1) == 3  # wrap in x, open in y
        nbr = FG.Grids.neighbors(g, 1, 1)
        Test.@test FG.Connectivity.linear_index(g, 3, 1) in nbr  # periodic wrap
        Test.@test FG.Connectivity.linear_index(g, 1, 2) in nbr
    end

    Test.@testset "Spherical sampling connectivity" begin
        function _symmetric(conn)
            A = FG.Connectivity.adjacency_matrix(conn)
            return A == A'
        end
        function _min_degree(conn, dmin)
            return all(i -> FG.Connectivity.nneighbors(conn, i) ≥ dmin, 1:FG.Connectivity.nnodes(conn))
        end

        # Tensor-product → structured lon-periodic
        sg = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 8)
        Test.@test FG.Grids.isperiodic(sg, 1)
        Test.@test !FG.Grids.isperiodic(sg, 2)
        cc = FG.Connectivity.build_connectivity(FG.SphericalSampling.ClenshawCurtisSampling(), 8)
        Test.@test FG.Connectivity.nnodes(cc) == length(sg.mask)
        Test.@test _symmetric(cc)

        # Cubed sphere: seams + symmetry
        n = 4
        csc = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), n)
        Test.@test FG.Connectivity.nnodes(csc) == 6 * n * n
        Test.@test _symmetric(csc)
        Test.@test _min_degree(csc, 2)
        # Interior panel cell (away from edges) has 4 face neighbors
        # face 1, i=j=2 → lin = (2-1)*n + 2 = n+2 when j=2,i=2 → (f-1)*n²+(j-1)*n+i
        lin_int = (1 - 1) * n * n + (2 - 1) * n + 2
        Test.@test FG.Connectivity.nneighbors(csc, lin_int) == 4
        # Edge (not corner): still 4 after seam fold
        lin_edge = (1 - 1) * n * n + (1 - 1) * n + 2  # j=1, i=2 on face 1
        Test.@test FG.Connectivity.nneighbors(csc, lin_edge) == 4

        # Yin–Yang: two disconnected panels
        nlon, nlat = 5, 4
        yyc = FG.Connectivity.build_connectivity(FG.SphericalSampling.YinYangSampling(), nlon, nlat)
        Test.@test FG.Connectivity.nnodes(yyc) == 2 * nlon * nlat
        Test.@test _symmetric(yyc)
        yin_nodes = 1:(nlon * nlat)
        yang_nodes = (nlon * nlat + 1):(2 * nlon * nlat)
        for i in yin_nodes
            Test.@test all(j -> j in yin_nodes, FG.Grids.neighbors(yyc, i))
        end
        for i in yang_nodes
            Test.@test all(j -> j in yang_nodes, FG.Grids.neighbors(yyc, i))
        end

        # HEALPix: documented RING neighbors for nside=4, pix=1 (0-based)
        nbr0 = FG.Connectivity.healpix_neighbors(4, 1)
        Test.@test Set(nbr0) == Set([16, 6, 5, 0, 3, 2, 8, 7])
        hpc = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(2))
        Test.@test FG.Connectivity.nnodes(hpc) == 12 * 2 * 2
        Test.@test _symmetric(hpc)
        # nside=1 → every pixel has 6 neighbors
        hp1 = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(1))
        Test.@test all(i -> FG.Connectivity.nneighbors(hp1, i) == 6, 1:12)
        Test.@test _symmetric(hp1)

        # Icosahedral ν=1: 12 vertices, each degree 5; 30 undirected edges → 60 directed
        ic1 = FG.Connectivity.build_connectivity(FG.SphericalSampling.IcosahedralSampling(1))
        Test.@test FG.Connectivity.nnodes(ic1) == 12
        Test.@test all(i -> FG.Connectivity.nneighbors(ic1, i) == 5, 1:12)
        Test.@test FG.Connectivity.nedges(ic1) == 60
        Test.@test _symmetric(ic1)
        mesh2 = FG.SphericalSampling.icosahedral_mesh(2)
        Test.@test length(mesh2.λ) == FG.SphericalSampling.icosahedral_nvertices(2)
        ic2 = FG.Connectivity.build_connectivity(FG.SphericalSampling.IcosahedralSampling(2))
        Test.@test FG.Connectivity.nnodes(ic2) == length(mesh2.λ)
        Test.@test _symmetric(ic2)
        Test.@test _min_degree(ic2, 5)

        ug = FG.Connectivity.unstructured_grid(FG.SphericalSampling.CubedSphereSampling(), 3)
        Test.@test ug isa FG.Grids.UnstructuredGrid
        Test.@test length(FG.Grids.neighbors(ug, 1)) == FG.Connectivity.nneighbors(FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), 3), 1)
    end

    Test.@testset "SparseArrays sparse_adjacency_matrix (optional)" begin
        using SparseArrays: SparseArrays as Sp
        geom = FG.Geometry.CartesianGeometry()
        grid = FG.Grids.StructuredGrid(geom, 0.0:1.0:1.0, 0.0:1.0:1.0, trues(2, 2))
        conn = FG.Connectivity.build_connectivity(grid)
        ne = FG.Connectivity.nedges(conn)
        I = Vector{Int}(undef, ne)
        J = Vector{Int}(undef, ne)
        Test.@test FG.Connectivity.sparse_adjacency_coo!(I, J, conn) == ne
        S = FG.Connectivity.sparse_adjacency_matrix(conn)
        Test.@test S isa Sp.SparseMatrixCSC
        Test.@test size(S) == (4, 4)
        Test.@test Sp.nnz(S) == ne
        Test.@test S[1, 2] && S[2, 1]
        Test.@test Matrix(S) == FG.Connectivity.adjacency_matrix(conn)
    end

    Test.@testset "StaticArrays extension" begin
        using StaticArrays: StaticArrays as SA
        geom = FG.Geometry.SphericalGeometry(1.0)
        p1 = SA.SVector{2,Float64}(0.0, 0.0)
        p2 = SA.SVector{2,Float64}(0.1, 0.2)
        d = FG.Geometry.distance(geom, p1, p2)
        Test.@test d ≈ FG.Geometry.distance(geom, Tuple(p1), Tuple(p2))
        ê = FG.Geometry.local_tangent_basis(geom, p1)
        Test.@test ê.λ isa NTuple
        Test.@test Test.@inferred(FG.Geometry.as_ntuple(p2)) === (0.1, 0.2)
    end

    Test.@testset "Connectivity is built into contiguous CSR, not per-node vectors" begin
        # The allocation count must not scale with the node count: everything lands in one neighbor
        # block plus one offset array, however many nodes there are.
        allocs(f) = (f(); minimum(Base.gc_alloc_count((@timed f()).gcstats) for _ in 1:5))
        # nside 8 → 32 is a 16× jump in node count (768 → 12288). Both sizes are past the point
        # where the count settles: the very smallest grids take one or two fewer allocations, so
        # anchoring on nside = 4 would measure that step rather than any scaling with n.
        small = allocs(() -> FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(8)))
        large = allocs(() -> FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(32)))
        Test.@test large <= small
        Test.@test large < 16
        Test.@test allocs(() -> FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), 16)) < 16

        # Degrees and reciprocity are unaffected by the storage change.
        conn = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(4))
        n = FG.Connectivity.nnodes(conn)
        Test.@test n == FG.SphericalSampling.healpix_npix(4)
        Test.@test all(6 ≤ FG.Connectivity.nneighbors(conn, i) ≤ 8 for i in 1:n)
        Test.@test all(!in(i, FG.Grids.neighbors(conn, i)) for i in 1:n)          # no self-loops
        Test.@test all(issorted(FG.Grids.neighbors(conn, i)) for i in 1:n)        # sorted, deduped rows
        Test.@test all(allunique(FG.Grids.neighbors(conn, i)) for i in 1:n)
        Test.@test all(i in FG.Grids.neighbors(conn, j) for i in 1:n for j in FG.Grids.neighbors(conn, i))

        # A duplicated edge collapses to one.
        mesh = FG.SphericalSampling.icosahedral_mesh(1)
        c1 = FG.Connectivity.build_connectivity(FG.SphericalSampling.IcosahedralSampling(1))
        Test.@test FG.Connectivity.nedges(c1) == 2 * length(unique(map(e -> minmax(e[1], e[2]), mesh.edges)))
    end

    Test.@testset "Spectral samplings integrate band-limited fields exactly" begin
        # The property that DEFINES a spectral quadrature sampling: Σ_j w_j P_l(sin φ_j) vanishes
        # for every 1 ≤ l ≤ lmax. A wrong node set or weight set fails this even when the point
        # count is right.
        function legendre(l, x)
            l == 0 && return one(x)
            p0, p1 = one(x), x
            for k in 1:(l - 1)
                p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
            end
            return p1
        end
        for s in (FG.SphericalSampling.GaussLegendreSampling(), FG.SphericalSampling.DriscollHealySampling(),
                  FG.SphericalSampling.DriscollHealyEqualSampling(), FG.SphericalSampling.ClenshawCurtisSampling())
            for nlat in (8, 16, 24)
                ax = FG.SphericalSampling.spherical_axes(s, nlat)
                w = FG.SphericalSampling.latitude_weights(s, nlat)
                # One normalization for every sampling: Σ w = ∫₀^π sinθ dθ = 2, carrying the sinθ
                # Jacobian and nothing else, so the longitude factor is always the caller's.
                Test.@test sum(w) ≈ 2 rtol = 1e-13
                Test.@test all(>(0), w) || s isa FG.SphericalSampling.AbstractDriscollHealySampling  # DH's polar node has zero weight
                # Exact for every single P_l the node count can resolve.
                for l in 1:(nlat - 1)
                    Test.@test abs(sum(w[j] * legendre(l, sin(ax.φ[j])) for j in eachindex(ax.φ))) < 1e-11
                end
                # Gauss–Legendre goes further: exact to 2N-1, which is what makes it the sampling
                # whose quadrature is exact at its own stated band limit.
                if s isa FG.SphericalSampling.AbstractGaussLegendreSampling
                    for l in nlat:(2nlat - 1)
                        Test.@test abs(sum(w[j] * legendre(l, sin(ax.φ[j])) for j in eachindex(ax.φ))) < 1e-11
                    end
                end
            end
        end

        # Float32 all the way through.
        w32 = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.ClenshawCurtisSampling(), 12; T = Float32)
        Test.@test eltype(w32) === Float32
        Test.@test sum(w32) ≈ 2 rtol = 1e-5

        # Samplings without a quadrature say so, without pointing at some other package.
        for s in (FG.SphericalSampling.McEwenWiauxSampling(), FG.SphericalSampling.LatLonSampling())
            err = try
                FG.SphericalSampling.latitude_weights(s, 8)
                nothing
            catch e
                sprint(showerror, e)
            end
            Test.@test err !== nothing
            Test.@test !occursin("FastTransforms", err)
            Test.@test !occursin("SSHT /", err)
        end
    end

    Test.@testset "Sampling axes are well formed" begin
        for (s, nlat, kw) in (
            (FG.SphericalSampling.GaussLegendreSampling(), 33, (;)),
            (FG.SphericalSampling.DriscollHealySampling(), 32, (;)),
            (FG.SphericalSampling.ClenshawCurtisSampling(), 33, (;)),
            (FG.SphericalSampling.McEwenWiauxSampling(), 33, (;)),
            (FG.SphericalSampling.LatLonSampling(), 33, (; nlon = 66)),
        )
            ax = FG.SphericalSampling.spherical_axes(s, nlat; kw...)
            sz = FG.SphericalSampling.axes_lengths(s, nlat; kw...)
            Test.@test length(ax.λ) == sz.nlon
            Test.@test length(ax.φ) == sz.nlat
            Test.@test issorted(ax.φ) || issorted(ax.φ; rev = true)
            Test.@test all(φ -> -π / 2 - 1e-12 ≤ φ ≤ π / 2 + 1e-12, ax.φ)
            Test.@test all(λ -> -1e-12 ≤ λ < 2π + 1e-12, ax.λ)
            Test.@test allunique(round.(ax.φ; digits = 12))
            p = FG.SphericalSampling.spherical_points(s, nlat; kw...)
            Test.@test length(p.λ) == sz.nlon * sz.nlat == FG.SphericalSampling.npoints(s, nlat; kw...)
        end
    end

    Test.@testset "Cubed-sphere points are cell centres, so panels do not share nodes" begin
        # Endpoint-inclusive panel coordinates put nodes ON the seams, so adjacent panels emit
        # coincident points — 12(n-2)+16 of them — while `_cubed_neighbor` treats those same edges
        # as folding onto a *different* panel's cells. Cell centres keep points and connectivity
        # consistent and give 6n² genuinely distinct nodes.
        for n in (1, 2, 4, 8, 16)
            p = FG.SphericalSampling.cubed_sphere_points(n)
            Test.@test length(p.λ) == 6n^2
            Test.@test length(unique(collect(zip(p.λ, p.φ)))) == 6n^2
            Test.@test length(unique(p.panel)) == 6
            Test.@test all(v -> abs(hypot(cos(v[2]) * cos(v[1]), cos(v[2]) * sin(v[1]), sin(v[2])) - 1) < 1e-12,
                           zip(p.λ, p.φ))
            # No point may sit exactly on a panel boundary (|ξ| = π/4 maps to the cube edges).
            Test.@test all(abs(abs(ξ) - π / 4) > 1e-12 for ξ in
                           range(-π / 4 + (π / 2 / n) / 2; step = π / 2 / n, length = n))
        end
    end

    Test.@testset "Default node areas follow the sampling's equal-area trait" begin
        using Quickhull: Quickhull
        R = 6.371e6
        geo = FG.Geometry.SphericalGeometry(R)
        tot = 4π * R^2

        # Equal-area by construction: uniform is exact.
        gh = FG.Connectivity.unstructured_grid(FG.SphericalSampling.HEALPixSampling(4); geometry = geo)
        ah = FG.Grids.measure(gh)
        Test.@test all(≈(tot / length(ah)), ah)
        Test.@test sum(ah) ≈ tot rtol = 1e-12

        # NOT equal-area: a uniform default would be silently wrong, so real dual areas are used.
        for g in (FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(4); geometry = geo),
                  FG.Connectivity.unstructured_grid(FG.SphericalSampling.CubedSphereSampling(), 8; geometry = geo))
            a = FG.Grids.measure(g)
            Test.@test sum(a) ≈ tot rtol = 1e-8      # still tiles the sphere exactly
            Test.@test all(>(0), a)                   # no degenerate zero-area cells
            Test.@test minimum(a) / maximum(a) < 0.95 # genuinely non-uniform
        end

        # An explicit `areas` always wins over any default.
        gx = FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(2); geometry = geo, areas = fill(1.0, 42))
        Test.@test all(==(1.0), FG.Grids.measure(gx))
    end

    Test.@testset "Yin–Yang cells tile each panel exactly; the overlap is resolution-independent" begin
        geo = FG.Geometry.SphericalGeometry()
        R2 = geo.R^2
        box = sqrt(2.0) * (3π / 2) * R2   # one panel's exact [-3π/4,3π/4] × [-π/4,π/4] area
        for (nlon, nlat) in ((8, 6), (16, 12), (48, 32))
            g = FG.Connectivity.unstructured_grid(FG.SphericalSampling.YinYangSampling(), nlon, nlat; geometry = geo)
            a = FG.Grids.measure(g)
            np = nlon * nlat
            Test.@test length(a) == 2np
            # Cell centres, not panel edges: the nlon×nlat cells tile the panel box exactly. Sampling
            # the endpoints instead inflates this by nlon/(nlon-1) × nlat/(nlat-1).
            Test.@test sum(@view a[1:np]) ≈ box rtol = 1e-14
            # Yang is a rigid rotation of yin, so the two blocks are elementwise identical.
            Test.@test @view(a[1:np]) == @view(a[(np + 1):(2np)])
            # The panels overlap by construction: the excess over the sphere is exactly 3√2π/4π at
            # every resolution. A resolution-*dependent* excess would mean a discretisation bug.
            Test.@test sum(a) / (4π * R2) ≈ 3 * sqrt(2) / 4 rtol = 1e-14
            # Areas vary as cos φ across the panel, and nowhere degenerate.
            Test.@test all(>(0), a)
            pts = collect(zip(round.(g.λ; digits = 10), round.(g.φ; digits = 10)))
            Test.@test length(unique(pts)) == 2np
        end
        Test.@test minimum(FG.Grids.measure(FG.Connectivity.unstructured_grid(FG.SphericalSampling.YinYangSampling(), 192, 128))) /
                   maximum(FG.Grids.measure(FG.Connectivity.unstructured_grid(FG.SphericalSampling.YinYangSampling(), 192, 128))) ≈
                   cos(π / 4) rtol = 1e-2
    end

    Test.@testset "Icosahedral dual areas are exact and need no tessellation dependency" begin
        geo = FG.Geometry.SphericalGeometry()
        tot = 4π * geo.R^2
        for ν in (1, 2, 4, 8, 16)
            mesh = FG.SphericalSampling.icosahedral_mesh(ν)
            Test.@test length(mesh.triangles) == 20ν^2
            Test.@test length(mesh.verts) == length(mesh.λ) == 10ν^2 + 2
            # Euler: V - E + F = 2.
            Test.@test (10ν^2 + 2) - length(mesh.edges) + length(mesh.triangles) == 2
            # Every triangle references three distinct in-range vertices.
            Test.@test all(t -> length(unique(t)) == 3 && all(1 .<= t .<= 10ν^2 + 2), mesh.triangles)

            g = FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(ν); geometry = geo)
            a = FG.Grids.measure(g)
            # The per-triangle Voronoi shares tile each triangle, so the cells tile the sphere.
            Test.@test sum(a) ≈ tot rtol = 1e-14
            Test.@test all(>(0), a)
            # It is the true spherical Voronoi dual — cross-checked against the convex-hull route.
            vor = FG.Grids._voronoi_areas(geo, g.λ, g.φ)
            Test.@test maximum(abs.(a .- vor) ./ vor) < 1e-12
        end
        # A geodesic sphere has exactly 12 pentagons (the icosahedron's corners); they are the
        # smallest cells, and all the rest are hexagons.
        a8 = FG.Grids.measure(FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(8); geometry = geo))
        Test.@test count(x -> x < minimum(a8) * (1 + 1e-9), a8) == 12
        # A uniform 4πR²/N would be ~±25% wrong here, so it is not the default.
        Test.@test minimum(a8) / maximum(a8) < 0.6
    end

    Test.@testset "Coarsest cubed sphere (one node per face) is constructible" begin
        # `range(a, b; length = 1)` is an error, so n = 1 needs its own handling — and
        # `build_connectivity` accepts n ≥ 1, so this size is reachable.
        p = FG.SphericalSampling.cubed_sphere_points(1)
        Test.@test length(p.λ) == 6
        Test.@test length(unique(p.panel)) == 6
        # Each face center is a distinct point on the sphere.
        pts = [(cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ)) for (λ, φ) in zip(p.λ, p.φ)]
        Test.@test length(unique(x -> round.(x; digits = 9), pts)) == 6
        Test.@test all(v -> abs(hypot(v...) - 1) < 1e-12, pts)
        conn = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), 1)
        Test.@test FG.Connectivity.nnodes(conn) == 6
        Test.@test_throws ArgumentError FG.SphericalSampling.cubed_sphere_points(0)
        for n in (1, 2, 5)
            Test.@test length(FG.SphericalSampling.cubed_sphere_points(n).λ) == 6n^2
        end
    end

    Test.@testset "Icosahedral mesh is indexed topologically, not by hashing coordinates" begin
        for ν in (1, 2, 3, 5, 8)
            mesh = FG.SphericalSampling.icosahedral_mesh(ν)
            nv = length(mesh.λ)
            Test.@test nv == FG.SphericalSampling.icosahedral_nvertices(ν) == 10ν^2 + 2
            # A geodesic sphere has exactly 30ν² edges, no duplicates and no self-loops.
            Test.@test length(mesh.edges) == 30ν^2
            Test.@test allunique(mesh.edges)
            Test.@test all(e -> e[1] < e[2], mesh.edges)
            # …and exactly 12 degree-5 vertices (the icosahedron corners); every other is degree 6.
            deg = zeros(Int, nv)
            for (a, b) in mesh.edges
                deg[a] += 1
                deg[b] += 1
            end
            Test.@test count(==(5), deg) == 12
            Test.@test count(==(6), deg) == nv - 12
            # Vertices are distinct points on the unit sphere.
            pts = [(cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ)) for (λ, φ) in zip(mesh.λ, mesh.φ)]
            Test.@test all(p -> abs(hypot(p...) - 1) < 1e-12, pts)
            Test.@test length(unique(p -> round.(p; digits = 9), pts)) == nv
            # Edge lengths are quasi-uniform, as a geodesic subdivision must be.
            elen = [hypot((pts[a] .- pts[b])...) for (a, b) in mesh.edges]
            Test.@test maximum(elen) / minimum(elen) < 1.5
        end

        # Construction cost must not scale with a per-vertex hash table.
        function allocs(f)
            f()
            r = @timed f()
            return Base.gc_alloc_count(r.gcstats)
        end
        Test.@test allocs(() -> FG.SphericalSampling.icosahedral_mesh(4)) < 200
        Test.@test allocs(() -> FG.SphericalSampling.icosahedral_mesh(32)) < 200
    end

    Test.@testset "Neighbor traversal allocates nothing" begin
        geom = FG.Geometry.CartesianGeometry()
        n = 40
        xs = collect(0.0:1.0:(n - 1))
        grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(n, n))

        function sweep(g, n)
            c = 0
            for j in 1:n, i in 1:n
                for v in FG.Grids.neighbors(g, i, j)
                    c += v
                end
            end
            return c
        end
        sweep(grid, 3)
        Test.@test @allocated(sweep(grid, n)) == 0

        # The lazy sequence agrees with the buffer-filling form, element for element.
        buf = Vector{Int}(undef, 8)
        for (i, j) in ((1, 1), (1, 5), (n, n), (7, 9))
            k = FG.Connectivity.neighbors!(buf, grid, i, j)
            Test.@test collect(FG.Grids.neighbors(grid, i, j)) == buf[1:k]
            Test.@test length(FG.Grids.neighbors(grid, i, j)) == k
        end

        # A masked-out cell reports no neighbors, matching nneighbors.
        m = trues(n, n)
        m[5, 5] = false
        masked = FG.Grids.StructuredGrid(geom, xs, xs, m)
        Test.@test isempty(collect(FG.Grids.neighbors(masked, 5, 5)))
        Test.@test length(FG.Grids.neighbors(masked, 5, 5)) == 0
        Test.@test !(FG.Connectivity.linear_index(masked, 5, 5) in collect(FG.Grids.neighbors(masked, 5, 6)))
        # …but is still reachable when the caller does not filter on the mask.
        Test.@test FG.Connectivity.linear_index(masked, 5, 5) in collect(FG.Grids.neighbors(masked, 5, 6; active_only = false))

        # Curvilinear grids take the same lazy path.
        nx, ny = 6, 5
        xm = [Float64(i) for i in 1:nx, j in 1:ny]
        ym = [Float64(j) for i in 1:nx, j in 1:ny]
        cv = FG.Grids.CurvilinearGrid(geom, xm, ym, trues(nx, ny))
        Test.@test length(FG.Grids.neighbors(cv, 3, 3)) == 4
        Test.@test collect(FG.Grids.neighbors(cv, 3, 3)) == let b = Vector{Int}(undef, 8)
            k = FG.Connectivity.neighbors!(b, cv, 3, 3); b[1:k]
        end
    end

    Test.@testset "k-d-tree adjacency, open and wrapping" begin
        using NearestNeighbors: NearestNeighbors
        cgeo = FG.Geometry.CartesianGeometry()
        n, L = 4, 4.0
        xs = Float64[i for i in 0:(n - 1), j in 0:(n - 1)][:]
        ys = Float64[j for i in 0:(n - 1), j in 0:(n - 1)][:]
        N = length(xs)
        areas = ones(N)

        # The non-periodic build must work for every input, not only the wrapping one.
        g_open = FG.Grids.UnstructuredGrid(cgeo, xs, ys, trues(N); k = 4, areas = areas)
        Test.@test FG.Connectivity.nnodes(FG.Connectivity.build_connectivity(g_open)) == N
        Test.@test all(1 ≤ length(FG.Grids.neighbors(g_open, i)) ≤ 4 for i in 1:N)
        Test.@test !FG.Grids.isperiodic(g_open, 1)

        g_per = FG.Grids.UnstructuredGrid(cgeo, xs, ys, trues(N); k = 4, areas = areas,
                                    periodic = (true, true), period = (L, L))
        Test.@test FG.Grids.isperiodic(g_per, 1) && FG.Grids.isperiodic(g_per, 2)
        Test.@test FG.Grids.period(g_per, 1) == L
        # On a wrapped lattice every node is interior: exactly four neighbours, each one cell away.
        Test.@test all(length(FG.Grids.neighbors(g_per, i)) == 4 for i in 1:N)
        Test.@test sort(collect(FG.Grids.neighbors(g_per, 1))) == [2, 4, 5, 13]
        minsep(a, b, L) = min(abs(a - b), L - abs(a - b))
        Test.@test all(minsep(xs[i], xs[j], L)^2 + minsep(ys[i], ys[j], L)^2 ≈ 1.0
                       for i in 1:N for j in FG.Grids.neighbors(g_per, i))
        Test.@test all(i in FG.Grids.neighbors(g_per, j) for i in 1:N for j in FG.Grids.neighbors(g_per, i))
        # Wrapping must change the graph, not merely be recorded.
        Test.@test any(sort(collect(FG.Grids.neighbors(g_open, i))) != sort(collect(FG.Grids.neighbors(g_per, i)))
                       for i in 1:N)
        # Radius queries honor it too.
        g_rad = FG.Grids.UnstructuredGrid(cgeo, xs, ys, trues(N); radius = 1.01, areas = areas,
                                    periodic = (true, true), period = (L, L))
        Test.@test all(length(FG.Grids.neighbors(g_rad, i)) == 4 for i in 1:N)

        # A Cartesian box has no period to infer, so wrapping without one is an error.
        Test.@test_throws ArgumentError FG.Grids.UnstructuredGrid(cgeo, xs, ys, trues(N);
                                                            k = 4, areas = areas, periodic = true)

        # Spherical longitude wraps with no ghosting: the embedding identifies λ with λ+2π.
        sgeo = FG.Geometry.SphericalGeometry(1.0)
        gs = FG.Grids.UnstructuredGrid(sgeo, [0.01, 6.27, 3.14], [0.0, 0.0, 0.0], trues(3);
                                 k = 1, areas = ones(3))
        Test.@test FG.Grids.isperiodic(gs, 1)
        Test.@test only(FG.Grids.neighbors(gs, 1)) == 2      # across the seam, not the far-away node 3
    end

    Test.@testset "HEALPix RING neighbours are emitted in ascending order" begin
        C = FG.Connectivity
        # The dedup pass is an insertion sort, so its cost is the inversion count. Walking the compass
        # offsets in RING order leaves the emission almost sorted. The sort still runs and still
        # guarantees the order; this measures only that its input is nearly ordered.
        out = Vector{Int}(undef, 8)
        for nside in (1, 2, 4, 16, 64)
            npix = 12 * nside^2
            inversions = 0
            presorted = 0
            valid = true
            for p in 0:(npix - 1)
                m = C.healpix_neighbors!(out, nside, p)
                v = view(out, 1:m)
                k = 0
                for a in 1:m, b in (a + 1):m
                    v[a] > v[b] && (k += 1)
                end
                inversions += k
                k == 0 && (presorted += 1)
                # Aggregate rather than assert per pixel: the emitted set must be distinct in-range
                # pixel ids that exclude the node itself, whatever the order.
                valid &= m ≥ 6 && all(0 .≤ v .< npix) && length(unique(v)) == m && p ∉ v
            end
            Test.@test valid
            # Only the ring seam is left out of order, and the seam is ~4·nside of 12·nside² pixels,
            # so the residual disorder falls like 1/nside rather than to a constant.
            Test.@test inversions / npix < 6 / nside
            Test.@test presorted / npix > 1 - 8 / nside
        end
        # The CSR is unchanged by the reordering: rows sorted, adjacency reciprocal, degrees 7 or 8.
        for nside in (1, 2, 8, 32)
            c = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(nside))
            n = FG.Connectivity.nnodes(c)
            Test.@test all(issorted(FG.Grids.neighbors(c, i)) for i in 1:n)
            Test.@test FG.Connectivity.is_symmetric_adjacency(c)
            Test.@test sort(unique(length(FG.Grids.neighbors(c, i)) for i in 1:n)) ⊆ [6, 7, 8]
        end
    end

    Test.@testset "Threading is opt-in and changes no result" begin
        using ComputationalBackends: ComputationalBackends as CB
        Test.@test Base.get_extension(FG, :FlowGeometriesComputationalBackendsExt) !== nothing

        # Chunking must partition exactly — no gaps, no overlap, contiguous, whatever the remainder.
        for (n, k) in ((10, 3), (10, 1), (3, 10), (1, 1), (100, 7))
            rs = FG.Execution.chunk_ranges(n, k)
            Test.@test reduce(vcat, collect.(rs)) == collect(1:n)
            Test.@test all(!isempty, rs) && length(rs) ≤ max(1, min(k, n))
        end
        # `nothing` hands the body the whole range in one call: the serial path adds no partitioning.
        seen = UnitRange{Int}[]
        FG.Execution.run_chunks(17, nothing) do r; push!(seen, r); end
        Test.@test seen == [1:17]

        # Every threaded kernel must be bit-identical to serial, not merely close.
        geo = FG.Geometry.SphericalGeometry()
        thr = CB.ThreadedBackend()
        for n in (7, 64)
            a = FG.SphericalSampling.cubed_sphere_points(n)
            b = FG.SphericalSampling.cubed_sphere_points(n; backend = thr)
            c = FG.SphericalSampling.cubed_sphere_points(n; backend = CB.SerialBackend())
            Test.@test a.λ == b.λ == c.λ
            Test.@test a.φ == b.φ == c.φ
            Test.@test a.panel == b.panel == c.panel
            sp = FG.SphericalSampling.spherical_points(FG.SphericalSampling.CubedSphereSampling(), n; backend = thr)
            Test.@test sp.λ == a.λ && sp.φ == a.φ
        end
        for (g, lbl) in ((geo, "spherical"), (FG.Geometry.CartesianGeometry(), "cartesian"))
            for n in (3, 40)   # n = 3 gives fewer rows than threads, exercising the short case
                λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
                φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
                m = fill(true, n, n)
                Test.@test FG.Grids.measure(FG.Grids.CurvilinearGrid(g, λ, φ, m)) ==
                           FG.Grids.measure(FG.Grids.CurvilinearGrid(g, λ, φ, m; backend = thr))
            end
        end

        # Connectivity: both cell passes write only slots their own cell owns. Masked and periodic
        # cases matter most — they make the per-cell degree vary, so a chunk boundary landing
        # mid-row would show up as a wrong offset rather than a wrong count.
        for topo in (FG.Connectivity.IndexTopology((37, 21), (true, false), nothing),
                     FG.Connectivity.IndexTopology((37, 21), (true, true), nothing),
                     FG.Connectivity.IndexTopology((5, 4), (false, false), nothing),
                     FG.Connectivity.IndexTopology((31, 29), (true, false),
                                      [isodd(i * 7 + j * 3) for i in 1:31, j in 1:29]))
            for st in (FG.Stencils.Axial(1), FG.Stencils.Moore(1)), ao in (true, false)
                a = FG.Connectivity.build_connectivity(topo; stencil = st, active_only = ao)
                b = FG.Connectivity.build_connectivity(topo; stencil = st, active_only = ao, backend = thr)
                Test.@test a.ptr == b.ptr
                Test.@test a.nbrs == b.nbrs
            end
        end
        g = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 33)
        Test.@test FG.Connectivity.build_connectivity(g).nbrs ==
                   FG.Connectivity.build_connectivity(g; backend = thr).nbrs

        # The metric-ball builder chunks the same two owned-slot passes, and per-cell degrees vary
        # even more (polar rows reach every longitude), so the same chunk-boundary argument applies.
        gball = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6),
                                        range(0, 2π; length = 25)[1:24],
                                        range(-π / 2, π / 2; length = 13))
        a = FG.Connectivity.build_connectivity_within(gball; ball = 2.0e6)
        b = FG.Connectivity.build_connectivity_within(gball; ball = 2.0e6, backend = thr)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs

        # The candidate builder emits and dedups concurrently, so each sampling's `emit!` has to be
        # free of state shared between nodes. nside = 1 and 2 cover the singular pixels, where a node
        # has 7 neighbours rather than 8.
        for s in (FG.SphericalSampling.HEALPixSampling(1), FG.SphericalSampling.HEALPixSampling(2), FG.SphericalSampling.HEALPixSampling(8))
            a = FG.Connectivity.build_connectivity(s)
            b = FG.Connectivity.build_connectivity(s; backend = thr)
            Test.@test a.ptr == b.ptr
            Test.@test a.nbrs == b.nbrs
        end
        for n in (1, 2, 9), st in (FG.Stencils.Axial(1), FG.Stencils.Moore(1))
            a = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), n; stencil = st)
            b = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), n; stencil = st, backend = thr)
            Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
        end
        for (nlon, nlat) in ((1, 1), (7, 5)), st in (FG.Stencils.Axial(1), FG.Stencils.Moore(1))
            a = FG.Connectivity.build_connectivity(FG.SphericalSampling.YinYangSampling(), nlon, nlat; stencil = st)
            b = FG.Connectivity.build_connectivity(FG.SphericalSampling.YinYangSampling(), nlon, nlat; stencil = st, backend = thr)
            Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
        end
    end

    Test.@testset "Grids can be moved to another storage backend" begin
        using Adapt: Adapt, adapt
        # A stand-in device array: a distinct type, so a successful adapt shows in the grid's type.
        struct FakeDev end
        struct DevArr{T,N} <: AbstractArray{T,N}
            a::Array{T,N}
        end
        Base.size(d::DevArr) = size(d.a)
        Base.getindex(d::DevArr, I...) = getindex(d.a, I...)
        Adapt.adapt_storage(::FakeDev, x::Array) = DevArr(x)

        geo = FG.Geometry.SphericalGeometry()
        g = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 9)
        d = adapt(FakeDev(), g)
        Test.@test FG.Grids.coordinates(d, 1) isa DevArr && FG.Grids.coordinates(d, 2) isa DevArr
        # A separable measure must adapt its FACTORS — materializing the outer product onto a device
        # is exactly what the factored form exists to avoid.
        Test.@test FG.Grids.measure(d) isa FG.Grids.SeparableMeasure
        Test.@test all(f -> f isa DevArr, FG.Grids.measure_factors(d))
        Test.@test all(FG.Grids.measure(d)[i, j] == FG.Grids.measure(g)[i, j]
                       for i in 1:size(g, 1), j in 1:size(g, 2))
        Test.@test FG.Grids.mask(d) isa FG.Grids.AllActive           # size only; nothing to move
        Test.@test size(d) == size(g) && FG.Grids.grid_geometry(d) === FG.Grids.grid_geometry(g)
        Test.@test FG.Grids.isperiodic(d, 1) == FG.Grids.isperiodic(g, 1)

        n = 12
        λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
        φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
        cg0 = FG.Grids.CurvilinearGrid(geo, λ, φ, fill(true, n, n))
        cg = adapt(FakeDev(), cg0)
        Test.@test FG.Grids.coordinates(cg, 1) isa DevArr && FG.Grids.measure(cg) isa DevArr
        Test.@test all(FG.Grids.measure(cg)[i, j] == FG.Grids.measure(cg0)[i, j] for i in 1:n, j in 1:n)

        ug0 = FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(3))
        ug = adapt(FakeDev(), ug0)
        Test.@test FG.Grids.coordinates(ug, 1) isa DevArr
        Test.@test getfield(ug, :neighbor_nbrs) isa DevArr
        Test.@test getfield(ug, :neighbor_ptr) isa DevArr
        Test.@test all(FG.Grids.measure(ug)[i] == FG.Grids.measure(ug0)[i] for i in eachindex(FG.Grids.measure(ug0)))

        c = adapt(FakeDev(), FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(2)))
        Test.@test c.nbrs isa DevArr && c.ptr isa DevArr

        t = FG.Connectivity.IndexTopology((4, 3), (true, false), nothing)
        Test.@test adapt(FakeDev(), t) === t              # no mask, nothing to move
        t2 = FG.Connectivity.IndexTopology((4, 3), (true, false), fill(true, 4, 3))
        Test.@test adapt(FakeDev(), t2).mask isa DevArr
        Test.@test adapt(FakeDev(), FG.Grids.AllActive((5, 5))) isa FG.Grids.AllActive
    end

    Test.@testset "A symmetric adjacency is read as CSC without transposing a second copy" begin
        using SparseArrays: SparseArrays
        for (g, lbl) in ((FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 17), "structured"),
                         (FG.Grids.CurvilinearGrid(FG.Geometry.SphericalGeometry(),
                                             [2π * (i - 1) / 12 for i in 1:12, j in 1:9],
                                             [asin(2 * (j - 0.5) / 9 - 1) for i in 1:12, j in 1:9],
                                             trues(12, 9)), "curvilinear"))
            conn = FG.Connectivity.build_connectivity(g)
            n = FG.Connectivity.nnodes(conn)
            Test.@test FG.Connectivity.is_symmetric_adjacency(conn)          # what licenses the shortcut
            A = FG.Connectivity.sparse_adjacency_matrix(g)                   # shortcut route
            B = FG.Connectivity.sparse_adjacency_matrix(FG.Connectivity.build_connectivity(g))  # transpose route
            Test.@test A == B                                   # identical matrix, not merely similar
            Test.@test Matrix(A) == FG.Connectivity.adjacency_matrix(conn)
            Test.@test SparseArrays.nnz(A) == FG.Connectivity.nedges(conn)
            Test.@test all(issorted(@view SparseArrays.rowvals(A)[SparseArrays.nzrange(A, j)])
                           for j in 1:n)
            # A non-default index type cannot alias Int buffers, so it falls back and must still match.
            Test.@test FG.Connectivity.sparse_adjacency_matrix(g; Ti = Int32) == A
            Test.@test eltype(FG.Connectivity.sparse_adjacency_matrix(g; Ti = Int32).colptr) === Int32
        end
        # `sort_neighbors!` orders each block in place and changes nothing else.
        c = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(4))
        before = [sort(collect(FG.Grids.neighbors(c, i))) for i in 1:FG.Connectivity.nnodes(c)]
        FG.Connectivity.sort_neighbors!(c)
        Test.@test all(issorted(FG.Grids.neighbors(c, i)) for i in 1:FG.Connectivity.nnodes(c))
        Test.@test all(collect(FG.Grids.neighbors(c, i)) == before[i] for i in 1:FG.Connectivity.nnodes(c))
        # k-nearest adjacency is NOT symmetric in general, so it must not take the shortcut.
        λ = [2π * ((i * 0.6180339887498949) % 1) for i in 1:150]
        φ = [asin(2 * (i / 151) - 1) for i in 1:150]
        ug = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(), λ, φ, trues(150); k = 4, areas = ones(150))
        uc = FG.Connectivity.build_connectivity(ug)
        Test.@test !FG.Connectivity.is_symmetric_adjacency(uc)
        Test.@test Matrix(FG.Connectivity.sparse_adjacency_matrix(uc)) == FG.Connectivity.adjacency_matrix(uc)
    end

    Test.@testset "Equiangular weights: FFT path and recurrence fallback agree with the definition" begin
        using FFTW: FFTW          # loads AbstractFFTs, which fires the extension
        Test.@test Base.get_extension(FG, :FlowGeometriesAbstractFFTsExt) !== nothing

        # Literal evaluation of the defining sum — no recurrence, no transform.
        function direct(family, nlat)
            nterm = (nlat + 1) ÷ 2
            θ(i) = family === :open ? π * (i - 0.5) / nlat : π * (i - 1) / nlat
            return [(4 / nlat) * sin(θ(i)) * sum(sin((2k + 1) * θ(i)) / (2k + 1) for k in 0:(nterm - 1))
                    for i in 1:nlat]
        end
        for nlat in (8, 64, 512)
            dh = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), nlat)
            cc = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.ClenshawCurtisSampling(), nlat)
            Test.@test maximum(abs.(dh .- direct(:closed, nlat))) < 1e-13
            Test.@test maximum(abs.(cc .- direct(:open, nlat))) < 1e-13
            Test.@test sum(dh) ≈ 2 atol = 1e-13
            Test.@test sum(cc) ≈ 2 atol = 1e-13
        end
        # The transform is O(n log n), so the cost ratio over a 4× size step must be far below the
        # 16× a quadratic rule would show.
        best(f) = (f(); minimum(@elapsed f() for _ in 1:3))
        t1 = best(() -> FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), 1024))
        t2 = best(() -> FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), 4096))
        Test.@test t2 / t1 < 8

        # Without an FFT implementation loaded the recurrence must still produce the same weights.
        script = """
        using FlowGeometries
        const SS = FlowGeometries.SphericalSampling
        w = SS.latitude_weights(SS.ClenshawCurtisSampling(), 64)
        v = SS.latitude_weights(SS.DriscollHealySampling(), 64)
        print(Base.get_extension(FlowGeometries, :FlowGeometriesAbstractFFTsExt) === nothing,
              " ", sum(w), " ", sum(v), " ", w[7], " ", v[7])
        """
        out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`, String)
        parts = split(out)
        Test.@test parts[1] == "true"                       # extension genuinely absent
        Test.@test parse(Float64, parts[2]) ≈ 2 atol = 1e-13
        Test.@test parse(Float64, parts[3]) ≈ 2 atol = 1e-13
        Test.@test parse(Float64, parts[4]) ≈ direct(:open, 64)[7] atol = 1e-13
        Test.@test parse(Float64, parts[5]) ≈ direct(:closed, 64)[7] atol = 1e-13
    end

    Test.@testset "Curvilinear areas hold two corner rows, not the whole field" begin
        geo = FG.Geometry.SphericalGeometry()
        for n in (5, 32)
            λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
            φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
            g = FG.Grids.CurvilinearGrid(geo, λ, φ, trues(n, n))
            a = FG.Grids.measure(g)
            Test.@test size(a) == (n, n)
            Test.@test all(>(0), a)
            # Reference: the same two triangles per cell, from a full materialized corner field.
            xc, yc = FG.Grids._centers_to_corners(λ), FG.Grids._centers_to_corners(φ)
            dirs = [(cos(yc[i, j]) * cos(xc[i, j]), cos(yc[i, j]) * sin(xc[i, j]), sin(yc[i, j]))
                    for i in 1:(n + 1), j in 1:(n + 1)]
            ref = [geo.R^2 * (FG.Geometry.spherical_excess(dirs[i, j], dirs[i + 1, j], dirs[i + 1, j + 1]) +
                              FG.Geometry.spherical_excess(dirs[i, j], dirs[i + 1, j + 1], dirs[i, j + 1]))
                   for i in 1:n, j in 1:n]
            Test.@test a == ref     # same arithmetic, only the buffering differs
        end
        # Construction memory must scale with the grid's own stored content (corners + areas), not
        # carry an extra full-size unit-vector field on top of it.
        function mib(n)
            λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
            φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
            m = trues(n, n)
            f = () -> FG.Grids.CurvilinearGrid(geo, λ, φ, m)
            f()
            return (@allocated f()) / 2^20
        end
        n = 200
        stored = 3 * 8 * (n + 1)^2 / 2^20      # xc, yc, areas
        Test.@test mib(n) < 1.6 * stored
    end

    Test.@testset "The ! forms allocate nothing beyond their return value" begin
        # Minimum of several samples: a single `@timed` can catch a collection mid-call. The true count
        # is a floor, so the minimum converges to it.
        nalloc(f) = (f(); minimum(Base.gc_alloc_count((@timed f()).gcstats) for _ in 1:5))

        # Gauss–Legendre: the node and weight outputs are independently optional, so asking for one
        # does not force a scratch vector for the other.
        n = 128
        gl = FG.SphericalSampling.GaussLegendreSampling()
        w = Vector{Float64}(undef, n)
        φ = Vector{Float64}(undef, n)
        λ = Vector{Float64}(undef, FG.SphericalSampling.axes_lengths(gl, n).nlon)
        FG.SphericalSampling.latitude_weights!(w, gl, n)
        FG.SphericalSampling.spherical_axes!(λ, φ, gl, n)
        Test.@test nalloc(() -> FG.SphericalSampling.latitude_weights!(w, gl, n)) == 0
        Test.@test nalloc(() -> FG.SphericalSampling.spherical_axes!(λ, φ, gl, n)) <= 1
        # …and both still agree with the combined solve.
        q = FG.SphericalSampling.spherical_quadrature(gl, n)
        Test.@test w == q.w && φ == q.φ && λ == q.λ

        # Cubed sphere: the panel id is not part of `spherical_points!`'s result, so it is not built.
        m = 32
        N = 6m^2
        cλ = Vector{Float64}(undef, N); cφ = Vector{Float64}(undef, N)
        FG.SphericalSampling.spherical_points!(cλ, cφ, FG.SphericalSampling.CubedSphereSampling(), m)
        Test.@test nalloc(() -> FG.SphericalSampling.spherical_points!(cλ, cφ, FG.SphericalSampling.CubedSphereSampling(), m)) <= 1
        p = FG.SphericalSampling.cubed_sphere_points(m)
        Test.@test cλ == p.λ && cφ == p.φ          # identical to the panel-carrying form
        Test.@test length(unique(p.panel)) == 6     # which still reports panels

        # Icosahedral vertices need no edge or triangle list, and must not build one.
        for ν in (8, 16)
            full = FG.SphericalSampling.icosahedral_mesh(ν)
            vo = FG.SphericalSampling.icosahedral_mesh(ν; topology = false)
            Test.@test vo.λ == full.λ && vo.φ == full.φ && vo.verts == full.verts
            Test.@test isempty(vo.edges) && isempty(vo.triangles)
            iλ = Vector{Float64}(undef, FG.SphericalSampling.icosahedral_nvertices(ν))
            iφ = similar(iλ)
            FG.SphericalSampling.icosahedral_vertices!(iλ, iφ, ν)
            Test.@test iλ == full.λ && iφ == full.φ
            # Allocation count must not grow with ν once the topology is skipped.
        end
        a8 = nalloc(() -> FG.SphericalSampling.icosahedral_vertices(8))
        a32 = nalloc(() -> FG.SphericalSampling.icosahedral_vertices(32))
        Test.@test a32 <= a8 + 2
    end

    Test.@testset "The icosahedron's faces are a written-down constant, and the right one" begin
        # The table replaces a runtime rediscovery of a fixed combinatorial fact. Re-derive it here
        # from the vertex geometry so the table cannot drift from the vertex ordering.
        faces = FG.SphericalSampling._ICOSAHEDRON_FACES
        Test.@test length(faces) == 20
        φg = (1.0 + sqrt(5.0)) / 2
        raw = [(0.0, 1.0, φg), (0.0, -1.0, φg), (0.0, 1.0, -φg), (0.0, -1.0, -φg),
               (1.0, φg, 0.0), (-1.0, φg, 0.0), (1.0, -φg, 0.0), (-1.0, -φg, 0.0),
               (φg, 0.0, 1.0), (φg, 0.0, -1.0), (-φg, 0.0, 1.0), (-φg, 0.0, -1.0)]
        v = [p ./ sqrt(sum(abs2, p)) for p in raw]
        d(i, j) = sqrt(sum(abs2, v[i] .- v[j]))
        edge = minimum(d(i, j) for i in 1:12 for j in (i + 1):12)
        # Every listed face is an equilateral triangle of three mutually adjacent vertices.
        for f in faces
            Test.@test length(unique(f)) == 3 && all(1 .<= f .<= 12)
            Test.@test all(d(a, b) ≈ edge for (a, b) in ((f[1], f[2]), (f[2], f[3]), (f[1], f[3])))
        end
        # Exactly the 20 such triangles exist, each listed once.
        derived = Set(Tuple(sort(collect(f))) for f in faces)
        expected = Set(Tuple(sort([a, b, c])) for a in 1:12 for b in (a + 1):12 for c in (b + 1):12
                       if d(a, b) ≈ edge && d(b, c) ≈ edge && d(a, c) ≈ edge)
        Test.@test derived == expected && length(derived) == 20
        # Every vertex is in 5 faces, and V - E + F = 2.
        Test.@test all(count(f -> v in f, faces) == 5 for v in 1:12)
        Test.@test length(unique(Tuple(sort([f[i], f[j]])) for f in faces
                                 for (i, j) in ((1, 2), (2, 3), (1, 3)))) == 30
        Test.@test 12 - 30 + 20 == 2
    end

    Test.@testset "Cell measure is stored factored, not materialized" begin
        geo = FG.Geometry.SphericalGeometry()
        for (nx, ny) in ((16, 9), (64, 40))
            x = collect(range(0, 2π; length = nx))
            y = collect(range(-1.3, 1.3; length = ny))
            g = FG.Grids.StructuredGrid(geo, x, y, FG.Grids.AllActive((nx, ny)))
            m = FG.Grids.measure(g)
            Test.@test m isa FG.Grids.SeparableMeasure
            Test.@test size(m) == (nx, ny) == size(g)
            # Indexes exactly like the dense outer product it replaces — bit-identical, not close.
            wx, wy = FG.Grids.measure_factors(g)
            dense = wx .* transpose(wy)
            Test.@test all(m[i, j] === dense[i, j] for i in 1:nx, j in 1:ny)
            Test.@test FG.Grids.measure_array(g) == dense
            Test.@test collect(m) == dense
            Test.@test FG.Grids.measure(g, 3, 4) === dense[3, 4]
            # sum is ∏ᵈ ∑ᵢ, i.e. O(Nx+Ny); it must still agree with the dense sum to roundoff.
            Test.@test sum(m) ≈ sum(dense) rtol = 1e-14
            Test.@test_throws BoundsError m[nx + 1, 1]
            # Storage is the factors, not the cells.
            Test.@test sizeof(wx) + sizeof(wy) < sizeof(dense)
        end
        # A 1-D grid is separable too: its single factor IS the measure.
        g1 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), collect(0.0:0.5:5.0), FG.Grids.AllActive((11,)))
        Test.@test FG.Grids.measure(g1) isa FG.Grids.SeparableMeasure
        f1 = FG.Grids.measure_factors(g1)
        Test.@test f1 !== nothing && length(f1) == 1
        Test.@test collect(only(f1)) == collect(FG.Grids.measure(g1))
        Test.@test FG.Grids.measure(g1, 3) ≈ 0.5

        # `show` must not sum every cell to print one line.
        big = FG.Grids.StructuredGrid(geo, collect(range(0, 2π; length = 2000)),
                                collect(range(-1.4, 1.4; length = 1000)), FG.Grids.AllActive((2000, 1000)))
        Test.@test occursin("2000×1000", sprint(show, MIME"text/plain"(), big))
    end

    Test.@testset "An all-active mask carries no per-cell storage" begin
        m = FG.Grids.AllActive((7, 5))
        Test.@test size(m) == (7, 5) && length(m) == 35
        Test.@test all(m[i, j] for i in 1:7, j in 1:5)
        Test.@test count(m) == 35 && all(m) && any(m)
        Test.@test collect(m) == trues(7, 5)
        Test.@test_throws BoundsError m[8, 1]
        # It is what the sampling constructors reach for, and it is smaller than a BitArray.
        g = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 33)
        Test.@test FG.Grids.mask(g) isa FG.Grids.AllActive
        Test.@test all(FG.Grids.isactive(g, i, j) for i in 1:size(g, 1), j in 1:size(g, 2))
        Test.@test Base.summarysize(FG.Grids.AllActive((4000, 2000))) < Base.summarysize(trues(4000, 2000))
        # An explicit mask still overrides it and still masks.
        mm = trues(FG.SphericalSampling.axes_lengths(FG.SphericalSampling.ClenshawCurtisSampling(), 9).nlon, 9)
        mm[2, 3] = false
        gm = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 9; mask = mm)
        Test.@test !FG.Grids.isactive(gm, 2, 3)
    end

    Test.@testset "Sampling connectivity is built from index topology, not a discarded grid" begin
        # Identical CSR to routing through `structured_grid`, without evaluating the axes (for
        # Gauss–Legendre that is the O(n²) root solve), the dense measure, or a `trues` mask.
        for (s, nlat, kw) in (
            (FG.SphericalSampling.GaussLegendreSampling(), 12, (;)),
            (FG.SphericalSampling.ClenshawCurtisSampling(), 9, (;)),
            (FG.SphericalSampling.DriscollHealySampling(), 8, (;)),
            (FG.SphericalSampling.LatLonSampling(), 7, (; nlon = 10)),
        )
            for st in (FG.Stencils.Axial(1), FG.Stencils.Moore(1))
                direct = FG.Connectivity.build_connectivity(s, nlat; stencil = st, kw...)
                viagrid = FG.Connectivity.build_connectivity(FG.Connectivity.structured_grid(s, nlat; kw...); stencil = st)
                Test.@test FG.Connectivity.nnodes(direct) == FG.Connectivity.nnodes(viagrid)
                Test.@test direct.nbrs == viagrid.nbrs
                Test.@test direct.ptr == viagrid.ptr
            end
        end
        # A mask still applies, and `nothing` means "all active" without materializing one.
        sz = FG.SphericalSampling.axes_lengths(FG.SphericalSampling.ClenshawCurtisSampling(), 9)
        m = trues(sz.nlon, sz.nlat)
        m[2, 2] = false
        masked = FG.Connectivity.build_connectivity(FG.SphericalSampling.ClenshawCurtisSampling(), 9; mask = m)
        Test.@test FG.Connectivity.nnodes(masked) == sz.nlon * sz.nlat
        Test.@test length(FG.Grids.neighbors(masked, sz.nlon + 2)) == 0   # (i,j) = (2,2), i fastest

        # The grid-free path must not scale its allocations with nlat the way building a grid does.
        nalloc(f) = (f(); Base.gc_alloc_count((@timed f()).gcstats))
        small = nalloc(() -> FG.Connectivity.build_connectivity(FG.SphericalSampling.GaussLegendreSampling(), 16))
        large = nalloc(() -> FG.Connectivity.build_connectivity(FG.SphericalSampling.GaussLegendreSampling(), 128))
        Test.@test large <= small + 2
    end

    Test.@testset "k-d-tree knn allocations do not scale with the node count" begin
        using NearestNeighbors: NearestNeighbors
        geo = FG.Geometry.SphericalGeometry(1.0)
        nalloc(f) = (f(); Base.gc_alloc_count((@timed f()).gcstats))
        build(n) = begin
            λ = [2π * (i * 0.6180339887498949 % 1) for i in 1:n]
            φ = [asin(2 * (i / (n + 1)) - 1) for i in 1:n]
            () -> FG.Grids._build_kdtree_neighbors(geo, (λ, φ); k = 6)
        end
        # Batch `knn` returns a Vector{Vector{Int}} plus a Vector{Vector{Float64}} — ~4 heap
        # allocations per query point (160,064 at N = 40k). Querying through `knn!` into reused
        # buffers, behind a function barrier so the abstractly-inferred tree does not force a
        # dynamic dispatch per call, makes the count flat in N.
        #
        # Measured on the QUERY loop with the tree passed in: `KDTree` construction itself spawns a
        # task per subtree when threads are available, so its own allocation count scales with N
        # (32 at one thread, 527 at four, N = 20k). That is upstream and not what this fix is about.
        E = Base.get_extension(FG, :FlowGeometriesNearestNeighborsExt)
        function query_allocs(n)
            λ = [2π * (i * 0.6180339887498949 % 1) for i in 1:n]
            φ = [asin(2 * (i / (n + 1)) - 1) for i in 1:n]
            pts = Matrix{Float64}(undef, 3, n)
            @. pts[1, :] = cos(φ) * cos(λ); @. pts[2, :] = cos(φ) * sin(λ); @. pts[3, :] = sin(φ)
            tree = NearestNeighbors.KDTree(pts)
            nbrs = Vector{Int}(undef, n * 6); ptr = Vector{Int}(undef, n + 1)
            return nalloc(() -> E._knn_loop!(nbrs, ptr, tree, pts, n, 6, 7))
        end
        small = query_allocs(2_000)
        large = query_allocs(20_000)
        Test.@test large <= small + 2
        Test.@test large < 20
        # The whole path still must not carry a per-point allocation.
        Test.@test nalloc(build(20_000)) < 0.05 * 20_000

        # Correctness is unchanged: the k nearest by great-circle distance, nearest-first.
        n = 200
        λ = [2π * (i * 0.6180339887498949 % 1) for i in 1:n]
        φ = [asin(2 * (i / (n + 1)) - 1) for i in 1:n]
        nbrs, ptr = FG.Grids._build_kdtree_neighbors(geo, (λ, φ); k = 5)
        v = [(cos(φ[i]) * cos(λ[i]), cos(φ[i]) * sin(λ[i]), sin(φ[i])) for i in 1:n]
        for i in 1:n
            d = [j == i ? Inf : sum(abs2, v[i] .- v[j]) for j in 1:n]
            Test.@test sort(nbrs[ptr[i]:(ptr[i + 1] - 1)]) == sort(partialsortperm(d, 1:5))
            # nearest-first within the node's block
            Test.@test issorted([d[j] for j in nbrs[ptr[i]:(ptr[i + 1] - 1)]])
        end
    end

    Test.@testset "Sparse adjacency assembles straight into CSC" begin
        using SparseArrays: SparseArrays
        conn = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(8))
        n, ne = FG.Connectivity.nnodes(conn), FG.Connectivity.nedges(conn)
        A = FG.Connectivity.sparse_adjacency_matrix(conn)
        Test.@test size(A) == (n, n)
        Test.@test SparseArrays.nnz(A) == ne
        # Same matrix the dense builder produces.
        Test.@test Matrix(A) == FG.Connectivity.adjacency_matrix(conn)
        # Row indices ascending within each column, as CSC requires — obtained without a sort.
        Test.@test all(issorted(@view SparseArrays.rowvals(A)[SparseArrays.nzrange(A, j)]) for j in 1:n)

        # A caller-supplied index type, and caller-owned buffers reused with no allocation of the
        # matrix's own storage.
        Test.@test eltype(FG.Connectivity.sparse_adjacency_matrix(conn; Ti = Int32).colptr) === Int32
        colptr = Vector{Int}(undef, n + 1)
        rowval = Vector{Int}(undef, ne)
        nzval = Vector{Bool}(undef, ne)
        B = FG.Connectivity.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)
        Test.@test B == A
        Test.@test B.colptr === colptr && B.rowval === rowval && B.nzval === nzval
        FG.Connectivity.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)
        Test.@test @allocated(FG.Connectivity.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)) < 200

        # The COO route still agrees with the direct one.
        I = Vector{Int}(undef, ne); J = Vector{Int}(undef, ne)
        Test.@test FG.Connectivity.sparse_adjacency_coo!(I, J, conn) == ne
        Test.@test SparseArrays.sparse(I, J, trues(ne), n, n) == A
    end

    Test.@testset "HEALPix pixel centers are distinct and correctly ringed" begin
        # These tile the sphere, so no two may coincide, and the ring structure is fully determined:
        # 4nside-1 rings holding 4, 8, … 4(nside-1), then 4nside for 2nside+1 rings, then back down.
        # The ring structure is fully determined, so it is asserted ring by ring, not just by count.
        for nside in (1, 2, 4, 8, 16)
            p = FG.SphericalSampling.spherical_points(FG.SphericalSampling.HEALPixSampling(nside))
            npix = FG.SphericalSampling.healpix_npix(nside)
            Test.@test length(p.λ) == npix
            Test.@test length(unique(collect(zip(p.λ, p.φ)))) == npix

            zs = sort(unique(round.(sin.(p.φ); digits = 11)); rev = true)
            Test.@test length(zs) == FG.SphericalSampling.healpix_nring(nside) == 4nside - 1
            counts = [count(φ -> isapprox(sin(φ), z; atol = 1e-10), p.φ) for z in zs]
            expected = vcat([4r for r in 1:(nside - 1)], fill(4nside, 2nside + 1),
                            [4r for r in (nside - 1):-1:1])
            Test.@test counts == expected
            Test.@test sum(counts) == npix

            # Nearest-neighbour separation scales as ~1/nside and is never zero.
            Test.@test all(-π / 2 ≤ φ ≤ π / 2 for φ in p.φ)
            Test.@test all(0 ≤ λ < 2π + 1e-12 for λ in p.λ)
        end
    end

    Test.@testset "Spherical Voronoi areas tile the sphere" begin
        using Quickhull: Quickhull
        R = 6.371e6
        geo = FG.Geometry.SphericalGeometry(R)
        for nside in (2, 4)
            p = FG.SphericalSampling.spherical_points(FG.SphericalSampling.HEALPixSampling(nside))
            a = FG.Grids._voronoi_areas(geo, p.λ, p.φ)
            Test.@test length(a) == length(p.λ)
            Test.@test all(>(0), a)
            Test.@test sum(a) ≈ 4π * R^2 rtol = 1e-10   # a tessellation covers the sphere exactly
        end
        # Non-equal-area sampling: still tiles exactly, but the cells genuinely differ.
        m = FG.SphericalSampling.icosahedral_mesh(4)
        ai = FG.Grids._voronoi_areas(geo, m.λ, m.φ)
        Test.@test sum(ai) ≈ 4π * R^2 rtol = 1e-10
        Test.@test minimum(ai) / maximum(ai) < 0.8

        # Float32 all the way through, and a clear error rather than a degenerate hull.
        p32 = FG.SphericalSampling.spherical_points(FG.SphericalSampling.HEALPixSampling(2); T = Float32)
        a32 = FG.Grids._voronoi_areas(FG.Geometry.SphericalGeometry(Float32(R)), p32.λ, p32.φ)
        Test.@test eltype(a32) === Float32
        Test.@test sum(a32) ≈ 4Float32(π) * Float32(R)^2 rtol = 1e-4
        Test.@test_throws ArgumentError FG.Grids._voronoi_areas(geo, [0.0, 1.0], [0.0, 0.5])
    end

    Test.@testset "Planar Voronoi areas are complete and degeneracy-safe" begin
        using DelaunayTriangulation: DelaunayTriangulation
        cgeo = FG.Geometry.CartesianGeometry()
        n = 40
        xs = [0.5 + 0.4cos(2π * k / n) + 0.03sin(7k) for k in 1:n]
        ys = [0.5 + 0.4sin(2π * k / n) + 0.03cos(5k) for k in 1:n]
        a = FG.Grids._voronoi_areas(cgeo, xs, ys)
        Test.@test length(a) == n
        Test.@test all(isfinite, a)
        Test.@test all(≥(0), a)
        Test.@test 0 < sum(a) < 1.0          # clipped to the hull, inside the unit square

        # A duplicate point is silently dropped by the triangulation, so its slot is never assigned.
        # It must read as zero, not as whatever happened to be in an `undef` buffer.
        xd = [0.1, 0.9, 0.5, 0.5, 0.7]
        yd = [0.1, 0.2, 0.9, 0.9, 0.6]
        ad = FG.Grids._voronoi_areas(cgeo, xd, yd)
        Test.@test all(isfinite, ad)
        Test.@test all(≥(0), ad)
        Test.@test count(iszero, ad) ≥ 1

        # Degenerate input is rejected up front, by our own precondition, not by an opaque
        # internal error escaping the triangulator.
        Test.@test_throws ArgumentError FG.Grids._voronoi_areas(cgeo, [0.0, 1.0], [0.0, 1.0])
        Test.@test_throws ArgumentError FG.Grids._voronoi_areas(cgeo, [0.1, 0.2, 0.3, 0.4],
                                                                      [0.1, 0.2, 0.3, 0.4])
        Test.@test_throws ArgumentError FG.Grids._voronoi_areas(cgeo, fill(0.5, 4), fill(0.5, 4))

        # Float32 in, Float32 out (the tessellation itself runs in Float64 for predicate robustness).
        a32 = FG.Grids._voronoi_areas(FG.Geometry.CartesianGeometry(Float32), Float32.(xs), Float32.(ys))
        Test.@test eltype(a32) === Float32
        Test.@test sum(a32) ≈ Float32(sum(a)) rtol = 1e-5
    end

    Test.@testset "Dense and sparse adjacency agree; sparse is the scalable one" begin
        using SparseArrays: SparseArrays
        cgeo = FG.Geometry.CartesianGeometry()
        small = FG.Grids.StructuredGrid(cgeo, range(0.0, 1.0; length = 20), range(0.0, 1.0; length = 20),
                                  trues(20, 20))
        A = FG.Connectivity.adjacency_matrix(small)
        Test.@test size(A) == (400, 400)
        Test.@test A == A'
        Test.@test Matrix(FG.Connectivity.sparse_adjacency_matrix(small)) == A

        # Dense is n² bytes — quadratic in nodes, quartic in grid side — so a 1000² grid is ~10¹²
        # bytes and simply is not the right tool. The sparse route on that same grid is routine.
        big = FG.Grids.StructuredGrid(cgeo, range(0.0, 1.0; length = 1000), range(0.0, 1.0; length = 1000),
                                trues(1000, 1000))
        S = FG.Connectivity.sparse_adjacency_matrix(big)
        Test.@test size(S) == (10^6, 10^6)
        Test.@test SparseArrays.nnz(S) == FG.Connectivity.nedges(FG.Connectivity.build_connectivity(big))
    end

    Test.@testset "Quadrature-exactness trait matches measured exactness" begin
        # The trait is about integrating PRODUCTS of two degree-lmax functions, which is what
        # spectral analysis forms. Clenshaw–Curtis's grid represents to N-1 but its quadrature only
        # integrates a single P_l to N-1, so it cannot claim exactness at its own band limit.
        Test.@test FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.GaussLegendreSampling())
        Test.@test FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.DriscollHealySampling())
        Test.@test !FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.ClenshawCurtisSampling())
        Test.@test !FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.McEwenWiauxSampling())
        Test.@test !FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.LatLonSampling())

        # Back the claim numerically: GL integrates degree 2·lmax exactly; CC does not.
        function legendre(l, x)
            l == 0 && return one(x)
            p0, p1 = one(x), x
            for k in 1:(l - 1)
                p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
            end
            p1
        end
        n = 16
        for s in (FG.SphericalSampling.GaussLegendreSampling(), FG.SphericalSampling.ClenshawCurtisSampling())
            ax = FG.SphericalSampling.spherical_axes(s, n)
            w = FG.SphericalSampling.latitude_weights(s, n)
            lmax = FG.SphericalSampling.bandlimit(s, n)
            err = abs(sum(w[j] * legendre(2lmax, sin(ax.φ[j])) for j in eachindex(ax.φ)))
            if FG.SphericalSampling.admits_exact_bandlimited_quadrature(s)
                Test.@test err < 1e-10
            else
                Test.@test err > 1e-6      # genuinely inexact at 2·lmax, as the trait now says
            end
        end
    end

    Test.@testset "No method ambiguities or unbound static parameters" begin
        Test.@test isempty(Test.detect_ambiguities(FG; recursive = true))
        Test.@test isempty(Test.detect_unbound_args(FG; recursive = true))
    end

    Test.@testset "Point functions accept every representation and element type" begin
        using StaticArrays: StaticArrays as SA
        cgeo = FG.Geometry.CartesianGeometry()     # Float64 geometry …
        sgeo = FG.Geometry.SphericalGeometry(6.371e6)

        R = 6.371e6

        # … fed points whose element type is NOT the geometry's: they are converted on entry, and
        # every representation reaches the same kernel rather than recursing through normalization.
        Test.@test FG.Geometry.distance(cgeo, (0, 0), (3, 4)) ≈ 5.0
        Test.@test FG.Geometry.distance(cgeo, (0.0f0, 0.0f0), (3.0f0, 4.0f0)) ≈ 5.0
        Test.@test FG.Geometry.distance(cgeo, (x = 0.0, y = 0.0), SA.SVector(3.0, 4.0)) ≈ 5.0
        Test.@test FG.Geometry.distance(cgeo, [0.0, 0.0], (3.0, 4.0)) ≈ 5.0
        Test.@test FG.Geometry.distance(sgeo, (0.1f0, 0.2f0), (0.3f0, 0.4f0)) ≈
                   FG.Geometry.distance(sgeo, (Float64(0.1f0), Float64(0.2f0)), (Float64(0.3f0), Float64(0.4f0)))

        # 3-component spherical points carry an absolute radius; on the reference sphere the chord
        # distance agrees with the 2-component great-circle distance to within chord-vs-arc.
        Test.@test FG.Geometry.distance(sgeo, (0.1, 0.2, R), (0.1, 0.2, R)) ≈ 0.0 atol = 1e-9
        Test.@test FG.Geometry.distance(sgeo, (0.0, 0.0, R), (0.0, 0.0, 2R)) ≈ R

        # An unsupported point length is an error, not unbounded recursion.
        Test.@test_throws MethodError FG.Geometry.distance(sgeo, (0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4))
        Test.@test_throws ArgumentError FG.Geometry.distance(cgeo, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])

        # `spherical_to_cartesian` — (λ, φ) = (0, 0) is the +x axis at radius R.
        P0 = FG.Geometry.spherical_to_cartesian(sgeo, (0.0, 0.0))
        Test.@test all(((P0.x, P0.y, P0.z) .- (R, 0.0, 0.0)) .< 1e-6)
        Pref = FG.Geometry.spherical_to_cartesian(sgeo, (0.1, 0.2))
        for pt in ((0.1, 0.2), (λ = 0.1, φ = 0.2), SA.SVector(0.1, 0.2), [0.1, 0.2])
            Test.@test FG.Geometry.spherical_to_cartesian(sgeo, pt) === Pref
        end

        # The spherical tangent-plane projection is built on it, and was unreachable before.
        Δ0 = FG.Geometry.project_to_tangent_plane(sgeo, (λ = 0.1, φ = 0.2), (λ = 0.1, φ = 0.2))
        Test.@test Δ0.λ ≈ 0.0 atol = 1e-9
        Test.@test Δ0.φ ≈ 0.0 atol = 1e-9
        # A small eastward step lies along ê_λ only, with arc length R·cos(φ)·Δλ.
        λ0, φ0, Δλ = 0.7, 0.2, 1e-6
        Δe = FG.Geometry.project_to_tangent_plane(sgeo, (λ0, φ0), (λ0 + Δλ, φ0))
        Test.@test Δe.λ ≈ R * cos(φ0) * Δλ rtol = 1e-6
        Test.@test abs(Δe.φ) < 1e-3 * abs(Δe.λ)
    end

    Test.@testset "Requested point representation is honored exactly" begin
        using StaticArrays: StaticArrays as SA
        sgeo = FG.Geometry.SphericalGeometry(6.371e6)
        cgeo = FG.Geometry.CartesianGeometry()
        xs = collect(0.0:1.0:4.0)
        cgrid = FG.Grids.StructuredGrid(cgeo, xs, xs, trues(5, 5))
        sgrid = FG.Grids.StructuredGrid(sgeo, deg2rad.(xs), deg2rad.(xs), trues(5, 5))

        Test.@test Test.@inferred(FG.Grids.coords(cgrid, 2, 3)) === (x = 1.0, y = 2.0)
        Test.@test Test.@inferred(FG.Grids.coords(NamedTuple, sgrid, 2, 3)) === FG.Grids.coords(sgrid, 2, 3)
        Test.@test Test.@inferred(FG.Grids.coords(Tuple, cgrid, 2, 3)) === (1.0, 2.0)
        Test.@test Test.@inferred(FG.Grids.coords(NTuple{2,Float32}, cgrid, 2, 3)) === (1.0f0, 2.0f0)
        Test.@test Test.@inferred(FG.Grids.coords(SA.SVector{2,Float64}, cgrid, 2, 3)) === SA.SVector(1.0, 2.0)
        Test.@test Test.@inferred(FG.Grids.coords(SA.SVector, cgrid, 2, 3)) === SA.SVector(1.0, 2.0)
        Test.@test Test.@inferred(FG.Grids.coords(SA.MVector{2,Float64}, cgrid, 2, 3)) == SA.MVector(1.0, 2.0)
        Test.@test Test.@inferred(FG.Grids.coords(Vector{Float64}, cgrid, 2, 3)) == [1.0, 2.0]

        # The vector-returning geometry functions take the same leading-type escape.
        p, q = SA.SVector(0.1, 0.2), SA.SVector(0.11, 0.21)
        Test.@test Test.@inferred(FG.Geometry.project_to_tangent_plane(SA.SVector{2,Float64}, sgeo, p, q)) ==
                   SA.SVector(Tuple(FG.Geometry.project_to_tangent_plane(sgeo, p, q)))
        Test.@test Test.@inferred(FG.Geometry.spherical_to_cartesian(SA.SVector{3,Float64}, sgeo, p)) ==
                   SA.SVector(Tuple(FG.Geometry.spherical_to_cartesian(sgeo, p)))
        Test.@test Test.@inferred(FG.Geometry.vector_to_cartesian(SA.SVector{3,Float64}, sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)) ==
                   SA.SVector(Tuple(FG.Geometry.vector_to_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)))
        Test.@test Test.@inferred(FG.Geometry.vector_from_cartesian(SA.MVector{3,Float64}, sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)) ==
                   SA.MVector(Tuple(FG.Geometry.vector_from_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)))
        Test.@test Test.@inferred(FG.Geometry.local_tangent_basis(SA.SVector{3,Float64}, sgeo, p)).λ isa SA.SVector{3,Float64}
        # A velocity may itself be given in any representation.
        Test.@test FG.Geometry.vector_from_cartesian(sgeo, SA.SVector(1.0, 2.0, 3.0), 0.1, 0.2) ===
                   FG.Geometry.vector_from_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)
    end

    Test.@testset "Point handling allocates nothing" begin
        using StaticArrays: StaticArrays as SA
        sgeo = FG.Geometry.SphericalGeometry(6.371e6)
        xs = deg2rad.(collect(0.0:1.0:9.0))
        sgrid = FG.Grids.StructuredGrid(sgeo, xs, xs, trues(10, 10))

        function svec_loop(grid, geo, n)
            s = 0.0
            c0 = FG.Grids.coords(SA.SVector{2,Float64}, grid, 1, 1)
            for j in 1:n, i in 1:n
                nb = FG.Grids.coords(SA.SVector{2,Float64}, grid, i, j)
                s += FG.Geometry.distance(geo, c0, nb + SA.SVector{2,Float64}(1e-3, 1e-3))
            end
            return s
        end
        function namedtuple_loop(grid, geo, n)
            s = 0.0
            c0 = FG.Grids.coords(grid, 1, 1)
            for j in 1:n, i in 1:n
                Δ = FG.Geometry.project_to_tangent_plane(geo, c0, FG.Grids.coords(grid, i, j))
                s += Δ[1]^2 + Δ[2]^2
            end
            return s
        end
        svec_loop(sgrid, sgeo, 2)
        namedtuple_loop(sgrid, sgeo, 2)
        Test.@test @allocated(svec_loop(sgrid, sgeo, 10)) == 0
        Test.@test @allocated(namedtuple_loop(sgrid, sgeo, 10)) == 0
    end

    Test.@testset "An out-of-range index errors, and @inbounds still opts out" begin
        # An out-of-range index must error rather than read past the end of the array. `@boundscheck`
        # is elided at an `@inbounds` call site, so a hot loop still pays nothing.
        geom = FG.Geometry.CartesianGeometry()
        g = FG.Grids.StructuredGrid(geom, 0.0:1.0:4.0, 0.0:1.0:3.0, trues(5, 4))
        Test.@test_throws BoundsError FG.Grids.measure(g, 99, 99)
        Test.@test_throws BoundsError FG.Grids.area(g, 99, 99)
        Test.@test_throws BoundsError FG.Grids.isactive(g, 99, 99)
        Test.@test_throws BoundsError FG.Grids.coords(g, 99, 99)
        Test.@test_throws BoundsError FG.Grids.coords(NTuple{2,Float64}, g, 6, 1)
        Test.@test_throws BoundsError FG.Grids.coords!(zeros(2), g, 1, 5)
        Test.@test_throws BoundsError FG.Grids.measure(g, 0, 1)

        nx, ny = 4, 3
        xm = [Float64(i) for i in 1:nx, j in 1:ny]
        ym = [Float64(j) for i in 1:nx, j in 1:ny]
        cv = FG.Grids.CurvilinearGrid(geom, xm, ym, trues(nx, ny))
        Test.@test_throws BoundsError FG.Grids.coords(cv, nx + 1, 1)
        Test.@test_throws BoundsError FG.Grids.corner_coords(cv, nx + 3, 1)
        Test.@test_throws BoundsError FG.Grids.measure(cv, 1, ny + 1)

        un = FG.Grids.UnstructuredGrid(geom, [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], trues(2))
        Test.@test_throws BoundsError FG.Grids.coords(un, 3)
        Test.@test_throws BoundsError FG.Grids.measure(un, 3)

        # In-range access is unaffected, and a caller that opts out still pays nothing.
        Test.@test FG.Grids.measure(g, 2, 2) ≈ 1.0
        Test.@test FG.Grids.coords(g, 2, 3) == (x = 1.0, y = 2.0)
        function sweep(grid, n)
            acc = 0.0
            @inbounds for j in 1:n, i in 1:n
                acc += FG.Grids.measure(grid, i, j) + (FG.Grids.isactive(grid, i, j) ? 1.0 : 0.0)
            end
            return acc
        end
        sweep(g, 2)
        Test.@test @allocated(sweep(g, 4)) == 0
    end

    Test.@testset "Auto-periodicity does not depend on which way an axis is stored" begin
        # A circle is a circle whichever way it is traversed, and a descending axis is routine.
        sg = FG.Geometry.SphericalGeometry()
        n = 8
        asc = collect(range(0.0; step = 2π / n, length = n))
        for x in (asc, reverse(asc))
            Test.@test FG.Grids.isperiodic(FG.Grids.StructuredGrid(sg, x, [0.0, 0.1], trues(n, 2)), 1)
        end
        # A regional span is not periodic in either orientation.
        reg = collect(range(0.0; step = 0.05, length = n))
        for x in (reg, reverse(reg))
            Test.@test !FG.Grids.isperiodic(FG.Grids.StructuredGrid(sg, x, [0.0, 0.1], trues(n, 2)), 1)
        end
        # The measure follows the periodicity, so a descending full circle must tile the same as an
        # ascending one rather than losing its seam cell.
        φ = collect(range(-1.0, 1.0; length = 5))
        ma = FG.Grids.measure(FG.Grids.StructuredGrid(sg, asc, φ, FG.Grids.AllActive((n, 5))))
        md = FG.Grids.measure(FG.Grids.StructuredGrid(sg, reverse(asc), φ, FG.Grids.AllActive((n, 5))))
        Test.@test sum(ma) ≈ sum(md) rtol = 1e-14
    end

    Test.@testset "A scattered point set is reachable through the documented entry points" begin
        using NearestNeighbors: NearestNeighbors
        using Quickhull: Quickhull
        # The documented entry point for an arbitrary point set, through to Voronoi cell areas.
        geo = FG.Geometry.SphericalGeometry()
        nn = 60
        λ = [2π * ((i * 0.6180339887498949) % 1) for i in 1:nn]
        φ = [asin(2 * (i / (nn + 1)) - 1) for i in 1:nn]
        g = FG.Connectivity.unstructured_grid(FG.SphericalSampling.ScatteredSphericalSampling(), λ, φ; geometry = geo)
        Test.@test g isa FG.Grids.UnstructuredGrid
        Test.@test FG.Grids.size_tuple(g) == (nn,)
        Test.@test all(>(0), FG.Grids.measure(g))
        # A tessellation covers the sphere exactly, whatever the point set.
        Test.@test sum(FG.Grids.measure(g)) ≈ 4π * geo.R^2 rtol = 1e-10
        Test.@test all(1 ≤ length(FG.Grids.neighbors(g, i)) ≤ 6 for i in 1:nn)
        # Caller-supplied areas and a radius query both still apply.
        g2 = FG.Connectivity.unstructured_grid(FG.SphericalSampling.ScatteredSphericalSampling(), λ, φ;
                                  geometry = geo, areas = ones(nn), k = 3)
        Test.@test all(==(1.0), FG.Grids.measure(g2))

        # The bang form must actually write into the buffers it is given.
        lo = zeros(nn); po = zeros(nn)
        r = FG.SphericalSampling.spherical_points!(lo, po, FG.SphericalSampling.ScatteredSphericalSampling(), λ, φ)
        Test.@test r.λ === lo && r.φ === po
        Test.@test lo == λ && po == φ
        Test.@test FG.SphericalSampling.npoints(FG.SphericalSampling.ScatteredSphericalSampling(), λ, φ) == nn
        Test.@test_throws DimensionMismatch FG.SphericalSampling.spherical_points!(
            zeros(nn - 1), zeros(nn - 1), FG.SphericalSampling.ScatteredSphericalSampling(), λ, φ)
        # …and the non-allocating identity form still hands the caller's own arrays back.
        p = FG.SphericalSampling.spherical_points(FG.SphericalSampling.ScatteredSphericalSampling(), λ, φ)
        Test.@test p.λ === λ && p.φ === φ
    end

    Test.@testset "Every ! form allocates nothing, at every size" begin
        nalloc(f) = (f(); minimum(Base.gc_alloc_count((@timed f()).gcstats) for _ in 1:5))

        # `!` means no allocation beyond the return value, at any size.
        nlat = 256
        s = FG.SphericalSampling.GaussLegendreSampling()
        n = FG.SphericalSampling.npoints(s, nlat)
        L = Vector{Float64}(undef, n); P = Vector{Float64}(undef, n)
        FG.SphericalSampling.spherical_points!(L, P, s, nlat)
        Test.@test nalloc(() -> FG.SphericalSampling.spherical_points!(L, P, s, nlat)) <= 1
        Test.@test (L, P) == (FG.SphericalSampling.spherical_points(s, nlat).λ, FG.SphericalSampling.spherical_points(s, nlat).φ)

        nlon, nlt = 64, 32
        n2 = FG.SphericalSampling.npoints(FG.SphericalSampling.YinYangSampling(), nlon, nlt)
        L2 = Vector{Float64}(undef, n2); P2 = Vector{Float64}(undef, n2)
        FG.SphericalSampling.spherical_points!(L2, P2, FG.SphericalSampling.YinYangSampling(), nlon, nlt)
        Test.@test nalloc(() -> FG.SphericalSampling.spherical_points!(L2, P2, FG.SphericalSampling.YinYangSampling(), nlon, nlt)) <= 1
        yy = FG.SphericalSampling.spherical_points(FG.SphericalSampling.YinYangSampling(), nlon, nlt)
        Test.@test L2 == yy.λ && P2 == yy.φ

        # Allocation count must be flat in the problem size, not merely small at one size.
        function ico_allocs(ν)
            nv = FG.SphericalSampling.icosahedral_nvertices(ν)
            a = Vector{Float64}(undef, nv); b = Vector{Float64}(undef, nv)
            FG.SphericalSampling.icosahedral_vertices!(a, b, ν)
            return nalloc(() -> FG.SphericalSampling.icosahedral_vertices!(a, b, ν))
        end
        Test.@test ico_allocs(8) <= 1
        Test.@test ico_allocs(32) <= 1
        Test.@test ico_allocs(64) <= 1
        # …and the direct write must agree exactly with the mesh route it replaced.
        for ν in (1, 2, 5, 16)
            nv = FG.SphericalSampling.icosahedral_nvertices(ν)
            a = Vector{Float64}(undef, nv); b = Vector{Float64}(undef, nv)
            FG.SphericalSampling.icosahedral_vertices!(a, b, ν)
            m = FG.SphericalSampling.icosahedral_mesh(ν)
            Test.@test a == m.λ && b == m.φ
        end
        Test.@test_throws DimensionMismatch FG.SphericalSampling.icosahedral_vertices!(zeros(3), zeros(3), 4)
        Test.@test_throws ArgumentError FG.SphericalSampling.icosahedral_vertices!(zeros(12), zeros(12), 0)
    end

    Test.@testset "Separable reductions factor, and only where the algebra allows" begin
        A = FG.Axes
        geo = FG.Geometry.SphericalGeometry()
        N = 240
        g = FG.Grids.StructuredGrid(geo, collect(range(0, 2π; length = N)),
                              collect(range(-1.4, 1.4; length = N)), FG.Grids.AllActive((N, N)))
        m = FG.Grids.measure(g)
        dense = FG.Grids.measure_array(g)
        # Every reduction that factors must agree with the dense evaluation.
        Test.@test sum(m) ≈ sum(dense) rtol = 1e-12
        Test.@test maximum(m) ≈ maximum(dense) rtol = 1e-12
        Test.@test minimum(m) ≈ minimum(dense) rtol = 1e-12
        Test.@test all(isapprox.(extrema(m), extrema(dense); rtol = 1e-12))
        Test.@test sum(abs2, m) ≈ sum(abs2, dense) rtol = 1e-12
        Test.@test sum(sqrt, m) ≈ sum(sqrt, dense) rtol = 1e-12
        Test.@test sum(abs, m) ≈ sum(abs, dense) rtol = 1e-12
        Test.@test maximum(abs2, m) ≈ maximum(abs2, dense) rtol = 1e-12
        Test.@test findmax(m)[2] == findmax(dense)[2]
        Test.@test findmin(m)[2] == findmin(dense)[2]
        Test.@test argmax(m) == argmax(dense)

        # A NON-multiplicative `f` does not factor, so it must fall through to the dense path and stay
        # correct rather than take a shortcut that does not hold.
        for f in (exp, log, sin)
            Test.@test sum(f, m) ≈ sum(f, dense) rtol = 1e-9
        end

        # Reducing a direction away leaves a still-separable object of the reduced shape.
        for dims in (1, 2, (1, 2))
            r = sum(m; dims = dims)
            Test.@test r isa FG.Grids.SeparableMeasure
            Test.@test size(r) == size(sum(dense; dims = dims))
            Test.@test collect(r) ≈ sum(dense; dims = dims) rtol = 1e-12
        end

        # max/min of a product are taken from the per-axis endpoints, which is exact for factors of any
        # sign — `∏ maximum` alone would be wrong as soon as a factor could go negative.
        ms = FG.Grids.SeparableMeasure(([-2.0, 1.0, 3.0], [-1.0, 4.0]))
        d2 = [ms[i, j] for i in 1:3, j in 1:2]
        Test.@test extrema(ms) == extrema(d2)
        Test.@test maximum(ms) == maximum(d2)
        Test.@test minimum(ms) == minimum(d2)
        Test.@test prod(ms) ≈ prod(d2) rtol = 1e-12

        # 3-D, so the factoring is exercised beyond the two-factor case.
        n3 = 24
        g3 = FG.Grids.StructuredGrid(geo, collect(range(0, 2π; length = n3)),
                               collect(range(-1.0, 1.0; length = n3)),
                               collect(range(6.3e6, 6.4e6; length = n3)),
                               FG.Grids.AllActive((n3, n3, n3)))
        m3 = FG.Grids.measure(g3); d3 = FG.Grids.measure_array(g3)
        Test.@test sum(m3) ≈ sum(d3) rtol = 1e-12
        Test.@test maximum(m3) ≈ maximum(d3) rtol = 1e-12
        Test.@test minimum(m3) ≈ minimum(d3) rtol = 1e-12
        Test.@test findmax(m3)[2] == findmax(d3)[2]

        # An all-active mask answers every reduction from its size.
        aa = FG.Grids.AllActive((7, 5))
        Test.@test sum(aa) == 35 && count(aa) == 35 && prod(aa) == true
        Test.@test extrema(aa) == (true, true)
        Test.@test all(identity, aa) && !any(!, aa)
        Test.@test count(identity, aa) == 35
        Test.@test findfirst(aa) == CartesianIndex(1, 1)
        Test.@test length(findall(aa)) == 35
        Test.@test collect(aa) == trues(7, 5)
    end

    Test.@testset "Stencils are any shape, any radius, any dimension" begin
        S = FG.Stencils
        C = FG.Connectivity
        # The radius-1 shapes must reproduce the conventional offset sets exactly, in order.
        Test.@test S.offsets(S.Axial(1), Val(2)) == ((-1, 0), (1, 0), (0, -1), (0, 1))
        Test.@test S.offsets(S.Moore(1), Val(2)) ==
            ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
        Test.@test S.offsets(S.Axial(1), Val(1)) == ((-1,), (1,))
        Test.@test S.nstencil(S.Axial(1), Val(3)) == 6
        Test.@test S.nstencil(S.Moore(1), Val(3)) == 26
        # Counts follow the defining formulas at any radius and dimension.
        for N in 1:5, r in 1:3
            Test.@test S.nstencil(S.Axial(r), Val(N)) == 2 * N * r
            Test.@test S.nstencil(S.Moore(r), Val(N)) == (2r + 1)^N - 1
            Test.@test S.nstencil(S.Diagonal(r), Val(N)) == 2^N * r
            # von Neumann is the L1 ball; Axial and Moore bracket it.
            Test.@test S.nstencil(S.Axial(r), Val(N)) ≤ S.nstencil(S.VonNeumann(r), Val(N)) ≤
                       S.nstencil(S.Moore(r), Val(N))
        end
        Test.@test Set(S.offsets(S.VonNeumann(1), Val(2))) == Set(S.offsets(S.Axial(1), Val(2)))
        # Every offset satisfies its shape's defining inequality, and none is the origin.
        for N in 2:4, r in 1:3
            Test.@test all(o -> 0 < sum(abs, o) ≤ r, S.offsets(S.VonNeumann(r), Val(N)))
            Test.@test all(o -> 0 < maximum(abs, o) ≤ r, S.offsets(S.Moore(r), Val(N)))
            Test.@test all(o -> count(!iszero, o) == 1, S.offsets(S.Axial(r), Val(N)))
            Test.@test all(o -> all(==(abs(o[1])), abs.(o)), S.offsets(S.Diagonal(r), Val(N)))
            Test.@test allunique(S.offsets(S.Moore(r), Val(N)))
        end
        Test.@test S.offsets(S.Anisotropic((2, 0)), Val(2)) == ((-2, 0), (-1, 0), (1, 0), (2, 0))
        Test.@test S.offsets(S.Custom(((1, 0), (0, 1))), Val(2)) == ((1, 0), (0, 1))

        Test.@testset "A caller's own shape needs only _offset_list" begin
            Test.@test S.offsets(Upwind(1), Val(2)) == ((1, 0), (0, 1))
            Test.@test S.offsets(Upwind(2), Val(2)) == ((1, 0), (2, 0), (0, 1), (0, 2))
            Test.@test S.nstencil(Upwind(3), Val(4)) == 12
            Test.@test S.reach(Upwind(3), Val(2)) == (3, 3)
            Test.@test S.fold_offsets((a, o) -> a + sum(o), 0, Upwind(2), Val(3)) == 9
            acc = Int[]
            S.foreach_offset(o -> push!(acc, sum(o)), Upwind(2), Val(2))
            Test.@test acc == [1, 2, 1, 2]

            # And it drives the connectivity stack, allocation-free like the built-ins.
            geo = FG.Geometry.CartesianGeometry()
            g = FG.Grids.StructuredGrid(geo, range(0.0; step = 1.0, length = 6),
                                        range(0.0; step = 1.0, length = 5))
            Test.@test FG.Connectivity.nneighbors(g, 2, 2; stencil = Upwind(1)) == 2
            Test.@test FG.Connectivity.nneighbors(g, 6, 5; stencil = Upwind(1)) == 0
            buf = Vector{Int}(undef, 8)
            n = FG.Connectivity.neighbors!(buf, g, 2, 2; stencil = Upwind(1))
            # Linear indices into the 6×5 grid: (3,2) and (2,3).
            Test.@test sort(buf[1:n]) == sort([3 + 1 * 6, 2 + 2 * 6])
            sweep(gr) = begin
                t = 0
                for j in 1:size(gr, 2), i in 1:size(gr, 1)
                    t += FG.Connectivity.nneighbors(gr, i, j; stencil = Upwind(2))
                end
                t
            end
            sweep(g)
            Test.@test @allocated(sweep(g)) == 0
        end
        Test.@test S.reach(S.Moore(3), Val(2)) == (3, 3)
        Test.@test S.reach(S.Anisotropic((3, 1)), Val(2)) == (3, 1)
        Test.@test S.Vertex === S.Moore
        # A stencil is named by its type; there is no symbol-to-stencil conversion to go wrong.
        Test.@test !hasmethod(C._stencil_val, Tuple{Symbol})
        Test.@test_throws ArgumentError S.Axial(0)
        Test.@test_throws ArgumentError S.Custom(((0, 0),))
        Test.@test_throws ArgumentError S.Anisotropic((0, 0))
        Test.@test_throws DimensionMismatch S.offsets(S.Anisotropic((1, 1)), Val(3))

        # Connectivity honours every shape, and the CSR agrees with per-cell counting.
        geo = FG.Geometry.CartesianGeometry()
        g = FG.Grids.StructuredGrid(geo, 0.0:1.0:9.0, 0.0:1.0:9.0, FG.Grids.AllActive((10, 10)))
        for st in (S.Axial(1), S.Axial(2), S.VonNeumann(2), S.Moore(1), S.Moore(2),
                   S.Anisotropic((3, 1)), S.Diagonal(1))
            conn = FG.Connectivity.build_connectivity(g; stencil = st)
            Test.@test FG.Connectivity.nedges(conn) ==
                sum(C.nneighbors(g, Tuple(ci)...; stencil = st) for ci in CartesianIndices((10, 10)))
            buf = Vector{Int}(undef, 256)
            k = C.neighbors!(buf, g, 5, 5; stencil = st)
            Test.@test sort(buf[1:k]) == sort(collect(FG.Grids.neighbors(g, 5, 5; stencil = st)))
            Test.@test k == C.nneighbors(g, 5, 5; stencil = st)
        end
        # A wide stencil at an interior cell sees its full offset set; at a corner it sees fewer.
        Test.@test C.nneighbors(g, 5, 5; stencil = S.Moore(2)) == 24
        Test.@test C.nneighbors(g, 1, 1; stencil = S.Moore(2)) < 24

        # The lazy iterator allocates nothing with a concrete stencil, including the default.
        function sweep(grid, n, st)
            c = 0
            for j in 1:n, i in 1:n
                for v in FG.Grids.neighbors(grid, i, j; stencil = st)
                    c += v
                end
            end
            return c
        end
        function sweep_default(grid, n)
            c = 0
            for j in 1:n, i in 1:n
                for v in FG.Grids.neighbors(grid, i, j)
                    c += v
                end
            end
            return c
        end
        # Every traversal allocates nothing at all — any shape, any radius, and with the stencil
        # written as an inline literal at the call site, which is the form that has to work.
        function sweep_def(grid, n)
            c = 0
            for j in 1:n, i in 1:n
                for v in FG.Grids.neighbors(grid, i, j)
                    c += v
                end
            end
            return c
        end
        sweep_def(g, 2)
        Test.@test @allocated(sweep_def(g, 10)) == 0

        # One sweeper per shape, each with the stencil written inline. The stencil is spelled through
        # the const `FG`, not the local alias `S`: a non-const binding cannot be constant-folded, so the
        # stencil type would not reach the call site and the measurement would be of that, not of the
        # package.
        sweep_ax1(grid, n) = (c = 0; for j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j; stencil = FG.Stencils.Axial(1)); c += v; end; end; c)
        sweep_ax3(grid, n) = (c = 0; for j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j; stencil = FG.Stencils.Axial(3)); c += v; end; end; c)
        sweep_vn2(grid, n) = (c = 0; for j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j; stencil = FG.Stencils.VonNeumann(2)); c += v; end; end; c)
        sweep_mo1(grid, n) = (c = 0; for j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j; stencil = FG.Stencils.Moore(1)); c += v; end; end; c)
        sweep_mo3(grid, n) = (c = 0; for j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j; stencil = FG.Stencils.Moore(3)); c += v; end; end; c)
        sweep_dia(grid, n) = (c = 0; for j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j; stencil = FG.Stencils.Diagonal(2)); c += v; end; end; c)
        sweep_ani(grid, n) = (c = 0; for j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j; stencil = FG.Stencils.Anisotropic((4, 1))); c += v; end; end; c)
        sweep_cus(grid, n) = (c = 0; for j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j; stencil = FG.Stencils.Custom(((1, 0), (0, 1)))); c += v; end; end; c)

        # A masked, wrapping grid exercises both extra branches of the neighbour kernel.
        mm = trues(10, 10); mm[4:6, 4:6] .= false
        gmask = FG.Grids.StructuredGrid(geo, 0.0:1.0:9.0, 0.0:1.0:9.0, mm;
                                        periodic = true, period = 10.0)
        # Called by name, not through a collection: iterating over a tuple of functions would make the
        # CALL dynamically dispatched and so allocate in the harness rather than in the package.
        sweep_ax1(g, 2); sweep_ax3(g, 2); sweep_vn2(g, 2); sweep_mo1(g, 2)
        sweep_mo3(g, 2); sweep_dia(g, 2); sweep_ani(g, 2); sweep_cus(g, 2)
        Test.@test @allocated(sweep_ax1(g, 10)) == 0
        Test.@test @allocated(sweep_ax3(g, 10)) == 0
        Test.@test @allocated(sweep_vn2(g, 10)) == 0
        Test.@test @allocated(sweep_mo1(g, 10)) == 0
        Test.@test @allocated(sweep_mo3(g, 10)) == 0
        Test.@test @allocated(sweep_dia(g, 10)) == 0
        Test.@test @allocated(sweep_ani(g, 10)) == 0
        Test.@test @allocated(sweep_cus(g, 10)) == 0

        sweep_ax1(gmask, 2); sweep_ax3(gmask, 2); sweep_vn2(gmask, 2); sweep_mo1(gmask, 2)
        sweep_mo3(gmask, 2); sweep_dia(gmask, 2); sweep_ani(gmask, 2); sweep_cus(gmask, 2)
        Test.@test @allocated(sweep_ax1(gmask, 10)) == 0
        Test.@test @allocated(sweep_ax3(gmask, 10)) == 0
        Test.@test @allocated(sweep_vn2(gmask, 10)) == 0
        Test.@test @allocated(sweep_mo1(gmask, 10)) == 0
        Test.@test @allocated(sweep_mo3(gmask, 10)) == 0
        Test.@test @allocated(sweep_dia(gmask, 10)) == 0
        Test.@test @allocated(sweep_ani(gmask, 10)) == 0
        Test.@test @allocated(sweep_cus(gmask, 10)) == 0

        # The buffer-filling and counting forms likewise, at any stencil width.
        buf2 = Vector{Int}(undef, 512)
        fill_ax(grid, n, b) = (k = 0; for j in 1:n, i in 1:n
            k += FG.Connectivity.neighbors!(b, grid, i, j; stencil = FG.Stencils.Axial(1)); end; k)
        fill_mo(grid, n, b) = (k = 0; for j in 1:n, i in 1:n
            k += FG.Connectivity.neighbors!(b, grid, i, j; stencil = FG.Stencils.Moore(3)); end; k)
        cnt_mo(grid, n) = (k = 0; for j in 1:n, i in 1:n
            k += FG.Connectivity.nneighbors(grid, i, j; stencil = FG.Stencils.Moore(3)); end; k)
        fill_ax(g, 2, buf2); fill_mo(g, 2, buf2); cnt_mo(g, 2)
        Test.@test @allocated(fill_ax(g, 10, buf2)) == 0
        Test.@test @allocated(fill_mo(g, 10, buf2)) == 0
        Test.@test @allocated(cnt_mo(g, 10)) == 0

        # And in more than three dimensions.
        g4a = FG.Grids.StructuredGrid(geo, ntuple(_ -> 0.0:1.0:4.0, 4)...)
        sweep4(grid, n) = (c = 0; for l in 1:n, k in 1:n, j in 1:n, i in 1:n
            for v in FG.Grids.neighbors(grid, i, j, k, l; stencil = FG.Stencils.Moore(1)); c += v; end; end; c)
        sweep4(g4a, 2)
        Test.@test @allocated(sweep4(g4a, 4)) == 0

        # The bulk builder's allocation count is flat in BOTH grid size and stencil width: the only
        # allocations are the CSR output arrays.
        nalloc(f) = (f(); minimum(Base.gc_alloc_count((@timed f()).gcstats) for _ in 1:3))
        counts = [nalloc(() -> FG.Connectivity.build_connectivity(
                      FG.Grids.StructuredGrid(geo, range(0.0, 1.0; length = m),
                                              range(0.0, 1.0; length = m)); stencil = st))
                  for m in (30, 60, 120), st in (S.Axial(1), S.Moore(2), S.Moore(3))]
        Test.@test all(==(first(counts)), counts)
        Test.@test first(counts) < 20
    end

    Test.@testset "Public names the suite had not been calling" begin
        GE = FG.Geometry
        C = FG.Connectivity
        S = FG.SphericalSampling

        # cartesian_to_spherical: the documented inverse of spherical_to_cartesian.
        sph = GE.SphericalGeometry(6.371e6)
        for p in ((0.0, 0.0), (1.2, -0.4), (5.9, 1.1))
            xyz = GE.spherical_to_cartesian(sph, p)
            back = GE.cartesian_to_spherical(sph, xyz)
            Test.@test mod(back.λ, 2π) ≈ mod(p[1], 2π) atol = 1e-12
            Test.@test back.φ ≈ p[2] atol = 1e-12
        end
        # A 3-D point carries its own radius through the round trip.
        r3 = GE.cartesian_to_spherical(sph, GE.spherical_to_cartesian(sph, (0.3, 0.4, 7.0e6)))
        Test.@test r3.r ≈ 7.0e6

        # cartesian_index is linear_index's inverse on every architecture that has one.
        geo = GE.CartesianGeometry()
        g = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:2.0, 0.0:1.0:1.0)
        Test.@test all(C.linear_index(g, Tuple(C.cartesian_index(g, k))...) == k
                       for k in 1:length(FG.Grids.mask(g)))
        Test.@test C.cartesian_index(g, 1) == CartesianIndex(1, 1, 1)

        # empty_csr / csr_connectivity: the storage type's own constructors.
        e = C.empty_csr(5)
        Test.@test C.nnodes(e) == 5 && C.nedges(e) == 0
        Test.@test all(isempty(e.nbrs[e.ptr[i]:(e.ptr[i + 1] - 1)]) for i in 1:5)
        Test.@test eltype(C.empty_csr(4, Int32).ptr) === Int32
        csr = C.csr_connectivity([2, 1, 3, 2], [1, 2, 4, 5])
        Test.@test C.nnodes(csr) == 3 && C.nedges(csr) == 4
        Test.@test collect(csr.nbrs[csr.ptr[2]:(csr.ptr[3] - 1)]) == [1, 3]
        Test.@test C.is_symmetric_adjacency(csr)
        # …and it validates, rather than trusting the buffers.
        Test.@test_throws ArgumentError C.csr_connectivity([1], [2, 2])        # ptr[1] != 1
        Test.@test_throws ArgumentError C.csr_connectivity([1, 2], [1, 2])     # length mismatch

        # StencilNeighbors is the lazy per-cell sequence; iterating it must equal neighbors!.
        gm = FG.Grids.StructuredGrid(geo, 0.0:1.0:4.0, 0.0:1.0:3.0)
        it = FG.Grids.neighbors(gm, 2, 2)
        Test.@test it isa C.StencilNeighbors
        buf = Vector{Int}(undef, 8)
        n = C.neighbors!(buf, gm, 2, 2)
        Test.@test sort(collect(it)) == sort(buf[1:n])
        Test.@test length(collect(FG.Grids.neighbors(gm, 1, 1))) == C.nneighbors(gm, 1, 1)

        # geographic_latitude / colatitude are each other's inverse.
        Test.@test S.geographic_latitude(0.0) ≈ π / 2
        Test.@test S.geographic_latitude(π) ≈ -π / 2
        Test.@test S.geographic_latitude(S.colatitude(0.37)) ≈ 0.37
        Test.@test S.colatitude(S.geographic_latitude(1.1)) ≈ 1.1

        # ring_info: rings tile the map contiguously, and each one's latitude is pix2ang's.
        for nside in (1, 2, 4, 8)
            nrings = 4 * nside - 1
            infos = [S.ring_info(nside, r) for r in 1:nrings]
            Test.@test sum(i -> i.ringpix, infos) == 12 * nside^2
            Test.@test infos[1].startpix == 0
            Test.@test all(infos[r].startpix + infos[r].ringpix == infos[r + 1].startpix
                           for r in 1:(nrings - 1))
            Test.@test all(S.pix2ang(nside, infos[r].startpix)[1] ≈ infos[r].colatitude
                           for r in 1:nrings)
            Test.@test all(i -> i.latitude ≈ π / 2 - i.colatitude, infos)
            Test.@test all(infos[r].ringpix == 4r for r in 1:(nside - 1))          # polar cap
            Test.@test all(infos[r].ringpix == 4nside for r in nside:(3nside))     # equatorial belt
            Test.@test all(infos[r].ringpix == infos[nrings + 1 - r].ringpix for r in 1:nrings)
            Test.@test all(infos[r].latitude ≈ -infos[nrings + 1 - r].latitude for r in 1:nrings)
        end
        Test.@test abs(S.ring_info(8, 16).latitude) < 1e-15      # ring 2·nside is the equator
        Test.@test S.ring_info(4, 3; T = Float32).colatitude isa Float32
        Test.@test_throws ArgumentError S.ring_info(4, 0)
        Test.@test_throws ArgumentError S.ring_info(4, 16)
        Test.@test_throws ArgumentError S.ring_info(0, 1)
    end

    Test.@testset "A separable measure stays separable under the operations that preserve it" begin
        GR = FG.Grids
        geo = FG.Geometry.CartesianGeometry()
        n = 400
        g = GR.StructuredGrid(geo, range(0.0; step = 0.5, length = n),
                              range(0.0; step = 0.25, length = n))
        m = GR.measure(g)
        dense = collect(m)
        probes = ((1, 1), (7, 13), (n, n))

        # Scaling, a multiplicative map, and a factor-wise product all keep the product form — so a
        # unit conversion costs bytes rather than the ∏Nᵈ values a dense result would.
        for got in (2.0 .* m, m .* 2.0, m ./ 4.0, abs.(m), abs2.(m), sqrt.(m), inv.(m),
                    m .* m, m ./ m)
            Test.@test got isa GR.SeparableMeasure
            Test.@test Base.summarysize(got) < 200
        end
        Test.@test all(≈((2.0 .* m)[I...], 2 * dense[I...]) for I in probes)
        Test.@test all(≈(abs2.(m)[I...], abs2(dense[I...])) for I in probes)
        Test.@test all(≈((m .* m)[I...], dense[I...]^2) for I in probes)
        Test.@test Base.summarysize(dense) > 1_000_000     # what the lazy form avoids

        # The closed-form reductions still apply to the result, so the saving compounds.
        s = 3.0 .* m
        Test.@test sum(s) ≈ 3 * sum(m)
        Test.@test maximum(s) ≈ 3 * maximum(m)
        Test.@test all(extrema(s) .≈ 3 .* extrema(m))
        Test.@test findmax(s)[2] == findmax(m)[2]
        Test.@test all(f -> f isa FG.Axes.ConstantVector, GR.measure_factors(s))

        # Everything else materializes — correct, just dense. A negative scale is deliberately in this
        # group: `findmax`'s per-axis argmax is only valid for non-negative factors.
        for got in (exp.(m), log.(m), m .+ m, m .+ 1.0, -1.0 .* m, -2.0 .* m)
            Test.@test !(got isa GR.SeparableMeasure)
        end
        Test.@test exp.(m)[3, 4] ≈ exp(dense[3, 4])
        Test.@test (-2.0 .* m)[3, 4] ≈ -2 * dense[3, 4]
        Test.@test findmax(-2.0 .* m)[1] ≈ maximum(-2 .* dense)

        Test.@test typeof(similar(m)) == Matrix{Float64}
        d = Array{Float64}(undef, size(m))
        copyto!(d, m)
        Test.@test all(d[I...] == dense[I...] for I in probes)
        Test.@test GR.measure_factors(collect(m)) === nothing
        Test.@test_throws DimensionMismatch m .* GR.measure(
            GR.StructuredGrid(geo, range(0.0; step = 1.0, length = 3),
                              range(0.0; step = 1.0, length = 3)))
    end

    Test.@testset "A pole rotation applies to a point set and to a grid" begin
        GE = FG.Geometry
        GR = FG.Grids
        rot = GE.PoleRotation(0.7, 0.3)
        Test.@test GE.rotate(rot, 0.7, 0.3)[2] ≈ π / 2      # the rotation's own pole

        λ = [0.1, 1.2, 3.0, 5.5]
        φ = [0.0, -0.4, 0.9, 0.2]
        Λ, Φ = GE.rotate(rot, λ, φ)
        Test.@test all((Λ[i], Φ[i]) == GE.rotate(rot, λ[i], φ[i]) for i in eachindex(λ))
        λ2, φ2 = copy(λ), copy(φ)
        GE.rotate!(λ2, φ2, rot)
        Test.@test λ2 == Λ && φ2 == Φ
        GE.unrotate!(λ2, φ2, rot)
        Test.@test all(isapprox.(λ2, λ; atol = 1e-12)) && all(isapprox.(φ2, φ; atol = 1e-12))
        rr() = FG.Geometry.rotate!(λ2, φ2, rot)
        rr()
        Test.@test @allocated(rr()) == 0
        # 2-D fields (a grid's coordinate arrays) go through the same method.
        Λm = [0.1i for i in 1:3, _ in 1:4]
        Test.@test size(GE.rotate!(Λm, [0.1j for _ in 1:3, j in 1:4], rot)[1]) == (3, 4)
        Test.@test_throws DimensionMismatch GE.rotate!(zeros(3), zeros(4), rot)

        # Rotating a rectilinear spherical grid gives a curvilinear one — which is what it is — and
        # the cell measure carries over EXACTLY, because a rotation is an isometry of the sphere.
        sph = FG.Geometry.SphericalGeometry(6.371e6)
        λa = range(0, 2π; length = 25)[1:24]
        φa = range(-1.2, 1.2; length = 13)
        gs = GR.StructuredGrid(sph, λa, φa)
        gr = GR.unrotate(gs, rot)
        Test.@test gr isa GR.CurvilinearGrid && size(gr) == (24, 13)
        Test.@test all(GR.measure(gr, i, j) == GR.measure(gs, i, j) for i in 1:24, j in 1:13)
        Test.@test sum(GR.measure(gr)) ≈ sum(GR.measure(gs))
        Test.@test GR._raw_coords(gr, 3, 4) == GE.unrotate(rot, λa[3], φa[4])
        Test.@test GR._raw_coords(GR.rotate(gs, rot), 3, 4) == GE.rotate(rot, λa[3], φa[4])
        # The mesh is unchanged, so its index topology and wrap length are too (λ is an angle in
        # either frame).
        Test.@test GR.isperiodic(gr, 1) && !GR.isperiodic(gr, 2)
        Test.@test GR.period(gr, 1) ≈ 2π
        Test.@test size(GR.corners(gr, 1)) == (25, 14)
        Test.@test GR.coordinate_names(gr) == (:λ, :φ)
        Test.@test FG.Connectivity.nneighbors(gr, 1, 5) == 4     # wraps, like the original
    end

    Test.@testset "apply_stencil! differentiates a field along one direction" begin
        D = FG.Discretization
        geo = FG.Geometry.CartesianGeometry()

        # Exact for any polynomial the node count spans, at EVERY sample — the ends included, because
        # the stencil shifts inward rather than clipping to a lower order.
        x = collect(range(0.0, 2.0; length = 11))
        f = @. 3x^2 - 2x + 5
        out = similar(f)
        D.apply_stencil!(out, f, x, 1; order = 1, nodes = 3)
        Test.@test maximum(abs, out .- (6 .* x .- 2)) < 1e-12
        D.apply_stencil!(out, f, x, 1; order = 2, nodes = 3)
        Test.@test maximum(abs, out .- 6.0) < 1e-11

        # A stretched axis is equally exact: the weights are per-sample, not one set reused.
        xs = [0.0, 0.11, 0.37, 0.9, 1.05, 1.6, 1.62, 2.0]
        outs = similar(xs)
        D.apply_stencil!(outs, (@. 3xs^2 - 2xs + 5), xs, 1; order = 1, nodes = 3)
        Test.@test maximum(abs, outs .- (6 .* xs .- 2)) < 1e-11
        # 3 nodes cannot span a cubic; 4 can. Both statements matter — the first shows the test bites.
        D.apply_stencil!(outs, xs .^ 3, xs, 1; order = 1, nodes = 3)
        Test.@test maximum(abs, outs .- 3 .* xs .^ 2) > 1e-3
        D.apply_stencil!(outs, xs .^ 3, xs, 1; order = 1, nodes = 4)
        Test.@test maximum(abs, outs .- 3 .* xs .^ 2) < 1e-11

        # Periodic: the stencil stays centred and wraps, so the seam is no worse than the interior,
        # and a 5-node stencil converges at 4th order rather than to machine precision.
        perr(m) = begin
            lm = collect(range(0, 2π; length = m + 1)[1:m])
            o = similar(lm)
            D.apply_stencil!(o, sin.(lm), lm, 1; order = 1, nodes = 5, period = 2π)
            (maximum(abs, o .- cos.(lm)),
             max(abs(o[1] - cos(lm[1])), abs(o[m] - cos(lm[m]))),
             maximum(abs, o[(m ÷ 4):(m ÷ 2)] .- cos.(lm[(m ÷ 4):(m ÷ 2)])))
        end
        e32, _, _ = perr(32)
        e64, seam64, mid64 = perr(64)
        Test.@test 12 < e32 / e64 < 20
        Test.@test seam64 ≤ 3 * mid64
        # A descending axis wraps the other way, and getting the sign wrong shows up here.
        λd = collect(range(2π, 0; length = 65)[1:64])
        od = similar(λd)
        D.apply_stencil!(od, sin.(λd), λd, 1; order = 1, nodes = 5, period = 2π)
        Test.@test maximum(abs, od .- cos.(λd)) < 1e-5
        Test.@test FG.Axes.wrap_sign(λd) == -1.0 && FG.Axes.wrap_sign(-λd) == 1.0

        # Only the named direction is differenced.
        X = collect(range(0.0, 1.0; length = 9))
        Y = collect(range(0.0, 2.0; length = 7))
        F = [xi^2 + 3yi for xi in X, yi in Y]
        O = similar(F)
        D.apply_stencil!(O, F, X, 1; order = 1, nodes = 3)
        Test.@test maximum(abs, O .- [2xi for xi in X, _ in Y]) < 1e-11
        D.apply_stencil!(O, F, Y, 2; order = 1, nodes = 3)
        Test.@test maximum(abs, O .- 3.0) < 1e-11
        F3 = [xi^2 + 3yi + 2zi for xi in X, yi in Y, zi in 0.0:0.5:1.0]
        O3 = similar(F3)
        D.apply_stencil!(O3, F3, collect(0.0:0.5:1.0), 3; order = 1, nodes = 3)
        Test.@test maximum(abs, O3 .- 2.0) < 1e-11

        # The grid form supplies axis, wrap period and mask, so none of it is restated.
        gx = FG.Grids.StructuredGrid(geo, X, Y)
        Og = similar(F)
        D.apply_stencil!(Og, F, gx, 1; order = 1, nodes = 3)
        Test.@test Og ≈ [2xi for xi in X, _ in Y]
        λ = collect(range(0, 2π; length = 65)[1:64])
        gp = FG.Grids.StructuredGrid(geo, λ, [0.0, 1.0]; periodic = true, period = 2π)
        Op = similar([sin(l) for l in λ, _ in 1:2])
        D.apply_stencil!(Op, [sin(l) for l in λ, _ in 1:2], gp, 1; order = 1, nodes = 5)
        Test.@test maximum(abs, Op[:, 1] .- cos.(λ)) < 1e-5

        # A derivative that would read an inactive cell is not invented.
        mk = trues(9, 7)
        mk[5, 3] = false
        gm = FG.Grids.StructuredGrid(geo, X, Y, mk)
        Om = fill(NaN, 9, 7)
        D.apply_stencil!(Om, F, gm, 1; order = 1, nodes = 3, masked = -1.0)
        Test.@test Om[5, 3] == -1.0                       # the inactive cell itself
        Test.@test Om[4, 3] == -1.0 && Om[6, 3] == -1.0   # its stencil neighbours
        Test.@test Om[2, 3] ≈ 2X[2] && Om[8, 3] ≈ 2X[8]   # cells that never read it
        Test.@test Om[5, 4] ≈ 2X[5]                       # a different row is unaffected
        D.apply_stencil!(Om, F, gm, 1; order = 1, nodes = 3, active_only = false)
        Test.@test Om[5, 3] ≈ 2X[5]

        # A precomputed weight set gives the same answer and applies allocation-free.
        idx, w = D.axis_stencils(X, 1, 3)
        Test.@test size(idx) == (9, 3) && size(w) == (9, 3)
        O2 = similar(F)
        D.apply_stencil!(O2, F, idx, w, 1)
        Test.@test O2 ≈ [2xi for xi in X, _ in Y]
        ap() = FG.Discretization.apply_stencil!(O2, F, idx, w, 1)
        ap()
        Test.@test @allocated(ap()) == 0

        Test.@test_throws ArgumentError D.axis_stencils(X, 2, 2)          # too few nodes for order 2
        Test.@test_throws ArgumentError D.axis_stencils([0.0, 1.0], 1, 5) # more nodes than samples
        Test.@test_throws ArgumentError D.apply_stencil!(O, F, X, 3)      # no direction 3 in a matrix
        Test.@test_throws DimensionMismatch D.apply_stencil!(O, F, X[1:5], 1)
        Test.@test_throws DimensionMismatch D.apply_stencil!(similar(F, 3, 3), F, idx, w, 1)
    end

    Test.@testset "Curvilinear and node grids work in any number of dimensions" begin
        geo = FG.Geometry.CartesianGeometry()
        C = FG.Connectivity
        GR = FG.Grids

        # 2-D still derives its corner areas: a uniform 1.0 × 0.5 mesh has exactly that cell area.
        X2 = [x for x in 0.0:1.0:4.0, _ in 0.0:0.5:2.0]
        Y2 = [y for _ in 0.0:1.0:4.0, y in 0.0:0.5:2.0]
        g2 = GR.CurvilinearGrid(geo, X2, Y2, trues(5, 5))
        Test.@test all(≈(0.5), GR.measure(g2))
        Test.@test GR.coords(g2, 2, 3) == (x = 1.0, y = 1.0)
        # The reconstructed corners sit exactly a half-cell outside the outermost centres.
        Test.@test GR.corner_coords(g2, 1, 1) == (x = -0.5, y = -0.25)

        X3 = [x for x in 0.0:1.0:3.0, _ in 1:3, _ in 1:2]
        Y3 = [y for _ in 1:4, y in 0.0:2.0:4.0, _ in 1:2]
        Z3 = [z for _ in 1:4, _ in 1:3, z in 0.0:0.5:0.5]
        # Past 2-D the corner-area kernel does not apply, and the error has to say so rather than
        # quietly producing a number from a 2-D formula.
        Test.@test_throws ArgumentError GR.CurvilinearGrid(geo, X3, Y3, Z3, trues(4, 3, 2))
        vol = fill(1.0 * 2.0 * 0.5, 4, 3, 2)
        g3 = GR.CurvilinearGrid(geo, X3, Y3, Z3, vol, trues(4, 3, 2))
        Test.@test size(g3) == (4, 3, 2) && ndims(g3) == 3
        Test.@test GR.coordinate_names(g3) == (:x, :y, :z)
        Test.@test GR.coords(g3, 2, 3, 2) == (x = 1.0, y = 4.0, z = 0.5)
        Test.@test GR.measure(g3, 2, 2, 1) == 1.0
        Test.@test size(GR.corners(g3, 1)) == (5, 4, 3)
        Test.@test GR.corner_coords(g3, 1, 1, 1) == (x = -0.5, y = -1.0, z = -0.25)

        W = ntuple(d -> [Tuple(ci)[d] * 1.0 for ci in CartesianIndices((3, 3, 2, 2))], 4)
        g4 = GR.CurvilinearGrid(geo, W..., ones(3, 3, 2, 2), trues(3, 3, 2, 2))
        Test.@test size(g4) == (3, 3, 2, 2)
        Test.@test GR.coords(g4, 2, 3, 1, 2) == (x1 = 2.0, x2 = 3.0, x3 = 1.0, x4 = 2.0)

        # The ghost ring is exact for a field linear in each direction, corners included — the
        # property the reconstruction exists to have, in 2-D and beyond.
        L3 = [1.5i - 2.0j + 0.5k for i in 1:4, j in 1:5, k in 1:3]
        Test.@test all(FG.Grids._ghosted(L3, (i, j, k)) ≈ 1.5i - 2.0j + 0.5k
                       for i in 0:5, j in 0:6, k in 0:4)
        # …and reconstruction costs only its own output. Minimum of several samples: a single
        # `@allocated` can land on a collection or on the tail of compilation, and the true cost is a
        # floor, so the minimum converges to it.
        cc(A) = (FG.Grids._centers_to_corners(A);
                 minimum(@allocated(FG.Grids._centers_to_corners(A)) for _ in 1:3))
        big = [1.0i + 2.0j for i in 1:120, j in 1:120]
        Test.@test cc(big) < 1.2 * 8 * 121^2

        # Connectivity follows into N-D, and the dense adjacency agrees with the CSR.
        Test.@test C.nneighbors(g3, 2, 2, 1) == 5          # 6 face neighbours minus one wall
        Test.@test C.linear_index(g3, 2, 3, 2) == 2 + 2 * 4 + 1 * 12
        Test.@test length(collect(GR.neighbors(g3, 2, 2, 1))) == 5
        Test.@test C.neighbors!(Vector{Int}(undef, 8), g3, 2, 2, 1) == 5
        csr3 = C.build_connectivity(g3)
        Test.@test C.is_symmetric_adjacency(csr3)
        Test.@test C.adjacency_matrix!(Matrix{Bool}(undef, 24, 24), g3) == C.adjacency_matrix(csr3)
        Test.@test C.adjacency_matrix!(Matrix{Bool}(undef, 25, 25), g2) ==
                   C.adjacency_matrix(C.build_connectivity(g2))
        ball3 = C.build_connectivity_within(g3; ball = 1.01)
        k3 = C.linear_index(g3, 2, 2, 1)
        Test.@test sort(ball3.nbrs[ball3.ptr[k3]:(ball3.ptr[k3 + 1] - 1)]) ==
                   sort(C.neighbors_within(g3, 2, 2, 1; ball = 1.01))

        # Periodicity is per-direction in N-D too.
        gp = GR.CurvilinearGrid(geo, X3, Y3, Z3, vol, trues(4, 3, 2);
                                periodic = (true, false, false), period = (4.0, 0.0, 0.0))
        Test.@test GR.isperiodic(gp, 1) && !GR.isperiodic(gp, 2)
        Test.@test GR.period(gp, 1) == 4.0
        Test.@test C.nneighbors(gp, 1, 2, 1) == 5

        # Node grids: coordinates as a tuple, since a run of vectors cannot say how many are
        # coordinates (a CurvilinearGrid counts them from `ndims(mask)`; a node set has no such handle).
        x, y, z = rand(6), rand(6), rand(6)
        gu3 = GR.UnstructuredGrid(geo, (x, y, z), ones(6), trues(6))
        Test.@test GR.coordinate_names(gu3) == (:x, :y, :z)
        Test.@test GR.coords(gu3, 4) == (x = x[4], y = y[4], z = z[4])
        Test.@test isempty(GR.neighbors(gu3, 4))          # no CSR given ⇒ no adjacency
        gu3a = GR.UnstructuredGrid(geo, (x, y, z), ones(6), trues(6), [2, 1], [1, 2, 3, 3, 3, 3, 3])
        Test.@test collect(GR.neighbors(gu3a, 1)) == [2]
        gu1 = GR.UnstructuredGrid(geo, ([0.0, 1.0, 5.0],), ones(3), trues(3))
        Test.@test GR.coordinate_names(gu1) == (:x,)
        Test.@test sort(C.neighbors_within(gu1, 1; ball = 1.5)) == [2]
        # The two-direction positional form is unchanged.
        Test.@test GR.coordinate_names(
            GR.UnstructuredGrid(geo, rand(8), rand(8), ones(8), trues(8))) == (:x, :y)

        # A k-d tree is dimension-agnostic, so the neighbour search generalizes even though the
        # Voronoi control volumes behind the 2-D measure do not.
        nn = 40
        px, py, pz, pw = rand(nn), rand(nn), rand(nn), rand(nn)
        bruteknn(pts, i, kk) = sort([t[2] for t in sort(
            [(sqrt(sum((pts[q][i] - pts[q][j])^2 for q in eachindex(pts))), j)
             for j in 1:nn if j != i])[1:kk]])
        nbrs3, ptr3 = GR._build_kdtree_neighbors(geo, (px, py, pz); k = 4)
        Test.@test all(sort(nbrs3[ptr3[i]:(ptr3[i + 1] - 1)]) == bruteknn((px, py, pz), i, 4)
                       for i in 1:nn)
        nbrs4, ptr4 = GR._build_kdtree_neighbors(geo, (px, py, pz, pw); k = 3)
        Test.@test all(sort(nbrs4[ptr4[i]:(ptr4[i + 1] - 1)]) == bruteknn((px, py, pz, pw), i, 3)
                       for i in 1:nn)
        # Periodic ghost images are placed per direction, so a face node finds the opposite face.
        fx, fy, fz = [0.02, 0.98, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        nbw, ptw = GR._build_kdtree_neighbors(geo, (fx, fy, fz); k = 1,
                                              periodic = (true, false, false),
                                              period = (1.0, 0.0, 0.0))
        Test.@test nbw[ptw[1]] == 2
        nbb, ptb = GR._build_kdtree_neighbors(geo, (fx, fy, fz); k = 1)
        Test.@test nbb[ptb[1]] == 3
        # `(λ, φ, r)` embeds at its own radius, so a radius query there is a true chord.
        sph1 = FG.Geometry.SphericalGeometry(1.0)
        sλ, sφ, sr = [0.0, 0.0], [0.0, 0.0], [1.0, 1.5]
        Test.@test GR._build_kdtree_neighbors(sph1, (sλ, sφ, sr); k = 1, radius = 0.6)[2][2] == 2
        Test.@test GR._build_kdtree_neighbors(sph1, (sλ, sφ, sr); k = 1, radius = 0.4)[2][2] == 1
        Test.@test_throws ArgumentError GR._build_kdtree_neighbors(sph1, (sλ, sφ, sr, sr); k = 1)
        g3kd = GR.UnstructuredGrid(geo, (px, py, pz), trues(nn); k = 4, areas = ones(nn))
        Test.@test length(collect(GR.neighbors(g3kd, 1))) == 4
        Test.@test_throws ArgumentError GR.UnstructuredGrid(geo, (px, py, pz), trues(nn); k = 4)
    end

    Test.@testset "Distance and displacement between cells honour the topology" begin
        GE = FG.Geometry
        GR = FG.Grids
        geo = GE.CartesianGeometry()
        n, Δ = 10, 1.0
        L = n * Δ
        gp = GR.StructuredGrid(geo, range(0.0; step = Δ, length = n),
                               range(0.0; step = Δ, length = 6); periodic = true, period = L)
        gb = GR.StructuredGrid(geo, range(0.0; step = Δ, length = n),
                               range(0.0; step = Δ, length = 6))

        # The point form on the raw coordinates would give the full extent; across a seam the cells are
        # one spacing apart.
        Test.@test GE.distance(gp, (1, 1), (n, 1)) ≈ Δ
        Test.@test GE.distance(gb, (1, 1), (n, 1)) ≈ (n - 1) * Δ
        Test.@test GE.distance(gp, (3, 2), (5, 4)) ≈ sqrt(2 * (2Δ)^2)   # interior is unaffected
        Test.@test GE.distance(gp, (1, 1), (n, 3)) ≈ GE.distance(gp, (n, 3), (1, 1))
        Test.@test GE.distance(gp, (4, 4), (4, 4)) == 0
        Test.@test GE.distance(gp, CartesianIndex(1, 1), CartesianIndex(n, 1)) ≈ Δ

        d = GR.displacement(gp, (1, 1), (n, 1))
        Test.@test d[1] ≈ -Δ && d[2] == 0                                # the short way round
        Test.@test sqrt(sum(abs2, GR.displacement(gp, (2, 5), (n, 2)))) ≈
                   GE.distance(gp, (2, 5), (n, 2))
        Test.@test all(GR.displacement(gp, (3, 1), (7, 4)) .≈ .-GR.displacement(gp, (7, 4), (3, 1)))
        Test.@test GR.displacement(gb, (1, 1), (n, 1))[1] ≈ (n - 1) * Δ

        # Longitude is intrinsically 2π-periodic, so the cell form must agree with the point form on the
        # raw coordinates — the minimum image changes nothing there.
        sph = FG.Geometry.SphericalGeometry(6.371e6)
        λ = collect(range(0, 2π; length = 25)[1:24])
        φ = collect(range(-1.0, 1.0; length = 9))
        gs = GR.StructuredGrid(sph, λ, φ)
        Test.@test GR.isperiodic(gs, 1)
        Test.@test all(GE.distance(gs, (i, j), (k, l)) ≈ GE.distance(sph, (λ[i], φ[j]), (λ[k], φ[l]))
                       for i in (1, 5, 24), j in (1, 5, 9), k in (1, 12, 24), l in (1, 5, 9))

        Λ = [λi for λi in range(0, π; length = 7), _ in 1:5]
        Φ = [φj for _ in 1:7, φj in range(-0.8, 0.8; length = 5)]
        gc = GR.CurvilinearGrid(sph, Λ, Φ, trues(7, 5))
        Test.@test GE.distance(gc, (2, 2), (4, 4)) ≈
                   GE.distance(sph, (Λ[2, 2], Φ[2, 2]), (Λ[4, 4], Φ[4, 4]))
        gu = GR.UnstructuredGrid(geo, ([0.5, 2.0, 9.5],), ones(3), trues(3);
                                 periodic = (true,), period = (10.0,))
        Test.@test GE.distance(gu, 1, 3) ≈ 1.0
        Test.@test GR.displacement(gu, 1, 3)[1] ≈ -1.0

        dd() = FG.Geometry.distance(gp, (3, 2), (9, 5))
        pp() = FG.Grids.displacement(gp, (3, 2), (9, 5))
        dd(); pp()
        Test.@test @allocated(dd()) == 0
        Test.@test @allocated(pp()) == 0
    end

    Test.@testset "A ball query can sum periodic images, which a convolution needs" begin
        C = FG.Connectivity
        GR = FG.Grids
        Nx, Δx = 32, 62.5
        L = Nx * Δx
        ax = range(0.0; step = Δx, length = Nx)
        g = GR.StructuredGrid(FG.Geometry.CartesianGeometry(), ax, ax;
                              periodic = (true, true), period = (L, L))
        # ℓ = L, so the kernel still carries 0.22 of its peak at L/2 — the regime where the images the
        # nearest-image convention discards actually matter.
        ℓ, α = 2000.0, 6.0
        kern(d) = exp(-α * d^2 / ℓ^2)
        fold(I, rad, conv) = C.fold_within((0.0, 0.0, 0), g, I...;
                                           ball = rad, images = conv, self = true) do a, J, d
            w = kern(d)
            (a[1] + w * cos(2π * ax[J[1]] / L), a[2] + w, a[3] + 1)
        end
        # Brute force: every cell at every image within ±K periods, gated on the radius. The definition
        # of the periodized sum, with no window and no minimum image.
        function brute(I, rad; K = 3)
            p0 = (ax[I[1]], ax[I[2]])
            num = 0.0; den = 0.0; cnt = 0
            for jy in 1:Nx, jx in 1:Nx, ky in -K:K, kx in -K:K
                dd = hypot(ax[jx] + kx * L - p0[1], ax[jy] + ky * L - p0[2])
                dd ≤ rad || continue
                w = kern(dd)
                num += w * cos(2π * ax[jx] / L); den += w; cnt += 1
            end
            return (num, den, cnt)
        end

        # `rad = 3500 > L` reaches images several turns out, which is what the uncapped window is for:
        # `metric_window`'s cap at the axis length would drop every one of them.
        for rad in (1469.0, 3500.0), I in ((1, 1), (5, 9), (17, 32))
            v = fold(I, rad, C.AllImages())
            b = brute(I, rad)
            Test.@test v[3] == b[3]
            Test.@test v[1] ≈ b[1] rtol = 1e-14
            Test.@test v[2] ≈ b[2] rtol = 1e-14
        end
        # Below L/2 the conventions are the same walk, so nothing existing changes.
        for rad in (200.0, 600.0, 999.0)
            a = fold((7, 11), rad, C.AllImages())
            b = fold((7, 11), rad, C.NearestImage())
            Test.@test a == b
        end

        # The physics: filtering one Fourier mode must reproduce the analytic Gaussian transfer.
        # Projected over the whole field rather than cell-by-cell, since `cos` vanishes at some cells.
        want = exp(-(2π / L)^2 * ℓ^2 / (4α))
        function transfer(rad, conv)
            num = 0.0; den = 0.0
            for j in 1:Nx, i in 1:Nx
                a = fold((i, j), rad, conv)
                f0 = cos(2π * ax[i] / L)
                num += (a[1] / a[2]) * f0; den += f0 * f0
            end
            return num / den
        end
        errs = [abs(transfer(rad, C.AllImages()) - want) / want for rad in (1469.0, 2600.0, 4000.0)]
        Test.@test errs[1] > errs[2] > errs[3]        # only truncation is left, and it shrinks
        Test.@test errs[3] < 1e-10
        # The nearest-image error is irreducible: widening the support cannot recover images the
        # convention discards.
        Test.@test abs(transfer(4000.0, C.NearestImage()) - want) / want > 0.3
        Test.@test fold((1, 1), 2600.0, C.NearestImage())[2] <
                   0.95 * fold((1, 1), 2600.0, C.AllImages())[2]

        # Summing images is a statement about a translation, so an angular periodic direction refuses it.
        gs = GR.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6),
                               collect(range(0, 2π; length = 25)[1:24]),
                               collect(range(-1.0, 1.0; length = 9)))
        Test.@test GR.isperiodic(gs, 1)
        Test.@test_throws ArgumentError C.fold_within((a, J, d) -> a, 0, gs, 3, 4;
                                                      ball = 2.0e6, images = C.AllImages())
        Test.@test C.fold_within((a, J, d) -> a + 1, 0, gs, 3, 4; ball = 2.0e6) ==
                   C.nneighbors_within(gs, 3, 4; ball = 2.0e6)
        # Nothing periodic ⇒ there are no images, so the request is a no-op rather than an error.
        gnp = GR.StructuredGrid(FG.Geometry.CartesianGeometry(), ax, ax)
        Test.@test C.fold_within((a, J, d) -> a + 1, 0, gnp, 5, 5;
                                 ball = 300.0, images = C.AllImages()) ==
                   C.fold_within((a, J, d) -> a + 1, 0, gnp, 5, 5; ball = 300.0)

        # The existing entry points are this fold, and both stay allocation-free over a whole sweep.
        Test.@test C.nneighbors_within(g, 7, 11; ball = 600.0) ==
                   C.fold_within((a, J, d) -> a + 1, 0, g, 7, 11; ball = 600.0)
        cnt(gr, m) = (t = 0; for j in 1:m, i in 1:m
            t += FG.Connectivity.fold_within((a, J, d) -> a + 1, 0, gr, i, j; ball = 600.0) end; t)
        nnw(gr, m) = (t = 0; for j in 1:m, i in 1:m
            t += FG.Connectivity.nneighbors_within(gr, i, j; ball = 600.0) end; t)
        cnt(g, Nx); nnw(g, Nx)
        Test.@test @allocated(cnt(g, Nx)) == 0
        Test.@test @allocated(nnw(g, Nx)) == 0
    end

    Test.@testset "MetricBall queries match a brute-force scan of the same metric" begin
        C = FG.Connectivity
        GE = FG.Geometry
        # Every cell, same minimum-image + geometry-distance filter, no window: equality proves the
        # window never under-covers and the traversal never duplicates a cell.
        function brute(grid, I, r; active_only = true)
            N = ndims(FG.Grids.mask(grid))
            sz = size(FG.Grids.mask(grid))
            prd = ntuple(d -> FG.Grids.isperiodic(grid, d) ? FG.Grids.period(grid, d) : 0.0, N)
            geom = FG.Grids.grid_geometry(grid)
            p0 = FG.Grids._raw_coords(grid, I...)
            out = Int[]
            for ci in CartesianIndices(sz)
                J = Tuple(ci)
                J == I && continue
                active_only && !FG.Grids.isactive(grid, J...) && continue
                pt = FG.Grids._raw_coords(grid, J...)
                q = ntuple(N) do d
                    p = prd[d]
                    p > 0 ? p0[d] + (pt[d] - p0[d] - p * round((pt[d] - p0[d]) / p)) : pt[d]
                end
                GE.distance(geom, p0, q) ≤ r || continue
                push!(out, C._linidx(sz, J...))
            end
            return sort(out)
        end
        agrees(grid, I, r; kw...) = begin
            got = sort(C.neighbors_within(grid, I...; ball = r, kw...))
            got == brute(grid, I, r; kw...) &&
                C.nneighbors_within(grid, I...; ball = r, kw...) == length(got) && allunique(got)
        end

        geo = FG.Geometry.CartesianGeometry()
        gu = FG.Grids.StructuredGrid(geo, range(0.0; step = 1.0, length = 12),
                                     range(0.0; step = 0.7, length = 9))
        for (I, r) in (((6, 5), 2.5), ((1, 1), 3.0), ((12, 9), 0.7), ((6, 5), 0.0), ((6, 5), 100.0))
            Test.@test agrees(gu, I, r)
        end
        xs = cumsum([0.0, 1.0, 0.3, 2.5, 0.7, 0.2, 1.4])
        gst = FG.Grids.StructuredGrid(geo, xs, cumsum([0.0, 0.5, 1.7, 0.2, 0.9]))
        for (I, r) in (((3, 2), 1.5), ((1, 5), 2.0), ((7, 3), 3.1))
            Test.@test agrees(gst, I, r)
        end
        # Periodic: the seam wraps by minimum image, and a ball reaching all the way around must not
        # count any cell twice.
        gp = FG.Grids.StructuredGrid(geo, range(0.0; step = 1.0, length = 10),
                                     range(0.0; step = 1.0, length = 6);
                                     periodic = true, period = 10.0)
        for (I, r) in (((1, 3), 2.2), ((10, 1), 1.5), ((5, 3), 4.9), ((5, 3), 50.0))
            Test.@test agrees(gp, I, r)
        end
        g4 = FG.Grids.StructuredGrid(geo, ntuple(_ -> range(0.0; step = 1.0, length = 5), 4)...)
        Test.@test agrees(g4, (3, 3, 3, 3), 1.8)
        Test.@test agrees(g4, (1, 2, 5, 4), 2.6)

        R = 6.371e6
        sph = FG.Geometry.SphericalGeometry(R)
        λ = range(0, 2π; length = 25)[1:24]
        φ = range(-π / 2, π / 2; length = 13)
        g2 = FG.Grids.StructuredGrid(sph, λ, φ)
        # Pole cells (full longitude ring in range), the seam, and a radius past the antipode.
        for (I, r) in (((5, 7), 2.0e6), ((1, 13), 1.5e6), ((3, 1), 1.0e6), ((24, 6), 2.0e6),
                       ((5, 7), 2.5e7))
            Test.@test agrees(g2, I, r)
        end
        g2s = FG.Grids.StructuredGrid(sph, collect(range(2π; step = -2π / 16, length = 16)),
                                      asin.(range(-1, 1; length = 9)))
        Test.@test agrees(g2s, (4, 8), 2.0e6)
        Test.@test agrees(g2s, (16, 2), 3.0e6)
        # 3-D is the CHORD metric, shorter than the arc — a window derived for arcs under-covers it,
        # which is what the near-antipodal radii on the unit shell probe.
        g3 = FG.Grids.StructuredGrid(sph, λ, φ, range(R, R + 3e5; length = 4))
        for (I, r) in (((5, 7, 2), 2.0e6), ((1, 13, 1), 1.0e6), ((3, 7, 4), 1.5e5))
            Test.@test agrees(g3, I, r)
        end
        gunit = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(1.0), λ, φ,
                                        range(1.0, 1.1; length = 3))
        for (I, r) in (((5, 7, 2), 1.95), ((5, 7, 2), 1.2), ((1, 1, 1), 2.05))
            Test.@test agrees(gunit, I, r)
        end

        # A 1-D transect's metric is the shorter arc of its circle — the same convention its measure
        # uses — so a ball can wrap the seam.
        g1 = FG.Grids.StructuredGrid(sph, λ)
        Test.@test GE.distance(sph, (0.1,), (0.1 + 2π - 0.3,)) ≈ R * 0.3
        for (I, r) in (((1,), 2.0e6), ((24,), 1.7e6), ((12,), 2.5e7))
            Test.@test agrees(g1, I, r)
        end

        wgs = FG.Geometry.SpheroidGeometry()
        ge2 = FG.Grids.StructuredGrid(wgs, λ, φ)
        for (I, r) in (((5, 7), 2.0e6), ((1, 13), 1.0e6), ((24, 6), 8.0e5))
            Test.@test agrees(ge2, I, r)
        end
        ge1 = FG.Grids.StructuredGrid(wgs, λ)
        Test.@test GE.distance(wgs, (0.0,), (π,)) ≈ π * GE.semimajor_axis(wgs)
        Test.@test agrees(ge1, (1,), 2.0e6)
        ge3 = FG.Grids.StructuredGrid(wgs, λ, φ, range(0.0, 3.0e5; length = 4))
        for (I, r) in (((5, 7, 2), 2.0e6), ((1, 13, 1), 1.0e6), ((3, 7, 4), 1.5e5), ((5, 7, 2), 1.4e7))
            Test.@test agrees(ge3, I, r)
        end
        # The geodetic 3-D metric is the ECEF chord: at f = 0 it must equal the sphere's, and the
        # closed forms pin the conversion.
        Test.@test GE.distance(FG.Geometry.SpheroidGeometry(R, 0.0), (0.3, 0.4, 1e4), (1.1, -0.2, 3e4)) ≈
                   GE.distance(sph, (0.3, 0.4, R + 1e4), (1.1, -0.2, R + 3e4)) rtol = 1e-12
        a, b = GE.semimajor_axis(wgs), GE.semiminor_axis(wgs)
        Test.@test GE.distance(wgs, (0.0, 0.0, 0.0), (0.0, π / 2, 0.0)) ≈ sqrt(a^2 + b^2)
        Test.@test GE.distance(wgs, (0.7, 0.5, 0.0), (0.7, 0.5, 250.0)) ≈ 250.0
        ecef = GE.geodetic_to_cartesian(wgs, (0.0, 0.0, 0.0))
        Test.@test (ecef.x, ecef.y, ecef.z) == (a, 0.0, 0.0)
        Test.@test GE.geodetic_to_cartesian(wgs, (0.0, π / 2, 100.0)).z ≈ b + 100.0

        # Mask, MetricBall-vs-bare-radius, argument errors, and the two-call buffer pattern.
        mk = trues(24, 13)
        mk[4, 7] = false
        gm = FG.Grids.StructuredGrid(sph, λ, φ, mk)
        Test.@test agrees(gm, (5, 7), 2.0e6)
        Test.@test C.nneighbors_within(gm, 4, 7; ball = 2.0e6) == 0
        Test.@test sort(C.neighbors_within(g2, 5, 7; ball = FG.Stencils.MetricBall(2.0e6))) ==
                   brute(g2, (5, 7), 2.0e6)
        Test.@test_throws ArgumentError C.nneighbors_within(g2, 5, 7; ball = -1.0)
        Test.@test_throws ArgumentError FG.Stencils.MetricBall(-2.0)
        Test.@test_throws BoundsError C.nneighbors_within(g2, 99, 7; ball = 1.0)
        nb = C.nneighbors_within(g2, 5, 7; ball = 2.0e6)
        buf = Vector{Int}(undef, nb)
        Test.@test C.neighbors_within!(buf, g2, 5, 7; ball = 2.0e6) == nb
        Test.@test sort(buf) == brute(g2, (5, 7), 2.0e6)
        Test.@test_throws ArgumentError C.neighbors_within!(Vector{Int}(undef, nb - 1), g2, 5, 7;
                                                            ball = 2.0e6)

        # The bulk builder: row k of the CSR is exactly the per-cell query for cell k, the graph is
        # symmetric because the metric is, and the stencil builder's own default is untouched by it.
        csrb = C.build_connectivity_within(g2; ball = 2.0e6)
        Test.@test all(1:prod(size(g2))) do k
            Ik = Tuple(CartesianIndices(size(g2))[k])
            sort(csrb.nbrs[csrb.ptr[k]:(csrb.ptr[k + 1] - 1)]) ==
                sort(C.neighbors_within(g2, Ik...; ball = 2.0e6))
        end
        Test.@test C.is_symmetric_adjacency(csrb)
        Test.@test C.build_connectivity_within(g2; ball = FG.Stencils.MetricBall(2.0e6)).nbrs == csrb.nbrs
        Test.@test C.nedges(C.build_connectivity(g2)) > 0
        csrbm = C.build_connectivity_within(gm; ball = 2.0e6)
        deadk = 4 + (7 - 1) * 24
        Test.@test csrbm.ptr[deadk] == csrbm.ptr[deadk + 1] && deadk ∉ csrbm.nbrs

        # Curvilinear and unstructured scan every cell against the same metric.
        Λm = [λi for λi in range(0, π; length = 9), _ in 1:7]
        Φm = [φj for _ in 1:9, φj in range(-1.0, 1.0; length = 7)]
        gc = FG.Grids.CurvilinearGrid(sph, Λm, Φm, trues(9, 7))
        rc = 3.0e6
        wantc = [ii + (jj - 1) * 9 for jj in 1:7, ii in 1:9 if !(ii == 4 && jj == 3) &&
                 GE.distance(sph, (Λm[4, 3], Φm[4, 3]), (Λm[ii, jj], Φm[ii, jj])) ≤ rc]
        Test.@test !isempty(wantc)
        Test.@test sort(C.neighbors_within(gc, 4, 3; ball = rc)) == sort(vec(wantc))
        Test.@test C.nneighbors_within(gc, 4, 3; ball = rc) == length(wantc)
        csrbc = C.build_connectivity_within(gc; ball = rc)
        k43 = 4 + (3 - 1) * 9
        Test.@test sort(csrbc.nbrs[csrbc.ptr[k43]:(csrbc.ptr[k43 + 1] - 1)]) == sort(vec(wantc))
        Test.@test C.is_symmetric_adjacency(csrbc)
        pts = [(0.1, 0.2), (0.3, 0.25), (2.0, -0.5), (0.11, 0.19), (1.0, 1.0)]
        gu2 = FG.Grids.UnstructuredGrid(sph, first.(pts), last.(pts),
                                        fill(1.0, 5), trues(5), Int[], ones(Int, 6))
        Test.@test sort(C.neighbors_within(gu2, 1; ball = 2.0e6)) ==
                   [k for k in 2:5 if GE.distance(sph, pts[1], pts[k]) ≤ 2.0e6]

        # Repeated queries allocate nothing on a rectilinear grid. Called through the const `FG` path:
        # a captured non-const module local would defeat const-folding and charge dispatch to the sweep.
        ballsweep(gr) = begin
            t = 0
            for j in 1:size(gr, 2), i in 1:size(gr, 1)
                t += FG.Connectivity.nneighbors_within(gr, i, j; ball = 2.5)
            end
            t
        end
        ballsweep(gu)
        Test.@test @allocated(ballsweep(gu)) == 0
    end

    Test.@testset "Grids and connectivity work in any number of dimensions" begin
        geo = FG.Geometry.CartesianGeometry()
        # A 4-D and a 5-D rectilinear grid, built through the one varargs constructor.
        g4 = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:3.0)
        Test.@test size(g4) == (4, 4, 4, 4)
        Test.@test FG.Grids.mask(g4) isa FG.Grids.AllActive          # no mask argument needed
        Test.@test ndims(g4) == 4
        Test.@test FG.Grids.measure(g4, 2, 2, 2, 2) ≈ 1.0
        Test.@test sum(FG.Grids.measure(g4)) ≈ 4.0^4 rtol = 1e-12
        Test.@test FG.Grids.coordinate_names(g4) == (:x1, :x2, :x3, :x4)
        Test.@test FG.Grids.coords(g4, 2, 3, 1, 4) == (x1 = 1.0, x2 = 2.0, x3 = 0.0, x4 = 3.0)
        Test.@test FG.Connectivity.nneighbors(g4, 2, 2, 2, 2) == 8      # 2N faces
        Test.@test FG.Connectivity.nneighbors(g4, 2, 2, 2, 2; stencil = FG.Stencils.Moore(1)) == 80  # 3^4-1
        Test.@test FG.Connectivity.nedges(FG.Connectivity.build_connectivity(g4)) > 0

        g5 = FG.Grids.StructuredGrid(geo, ntuple(_ -> 0.0:1.0:2.0, 5)...)
        Test.@test size(g5) == (3, 3, 3, 3, 3)
        Test.@test FG.Connectivity.nneighbors(g5, 2, 2, 2, 2, 2) == 10

        # A traversal must stay allocation-free past N = 3 too. Coordinate names are numbered rather
        # than lettered from N = 4 on, and a per-cell query reaches them; building those symbols at run
        # time would allocate on every call, so this covers more than the loop itself.
        sweep4(gr) = begin
            t = 0
            for l in 1:size(gr, 4), k in 1:size(gr, 3), j in 1:size(gr, 2), i in 1:size(gr, 1)
                t += FG.Connectivity.nneighbors(gr, i, j, k, l; stencil = FG.Stencils.Moore(1))
            end
            t
        end
        sweep5(gr) = begin
            t = 0
            for m in 1:size(gr, 5), l in 1:size(gr, 4), k in 1:size(gr, 3),
                j in 1:size(gr, 2), i in 1:size(gr, 1)
                t += FG.Connectivity.nneighbors(gr, i, j, k, l, m)
            end
            t
        end
        coordsweep4(gr) = begin
            t = 0.0
            for l in 1:size(gr, 4), k in 1:size(gr, 3), j in 1:size(gr, 2), i in 1:size(gr, 1)
                t += FG.Grids.coords(gr, i, j, k, l).x1 + FG.Grids.measure(gr, i, j, k, l)
            end
            t
        end
        sweep4(g4); sweep5(g5); coordsweep4(g4)
        Test.@test @allocated(sweep4(g4)) == 0
        Test.@test @allocated(sweep5(g5)) == 0
        Test.@test @allocated(coordsweep4(g4)) == 0
        # The names themselves must be a compile-time constant past the lettered directions.
        names4() = FG.Geometry.point_names(FG.Geometry.CartesianGeometry(), Val(4))
        names7() = FG.Geometry.point_names(FG.Geometry.SphericalGeometry(1.0), Val(7))
        names4(); names7()
        Test.@test @allocated(names4()) == 0
        Test.@test @allocated(names7()) == 0
        Test.@test names7() == (:λ, :φ, :r, :q4, :q5, :q6, :q7)

        # A spherical grid in 1-D measures arc length, and in 4-D keeps its metric directions.
        sgeo = FG.Geometry.SphericalGeometry(2.0)
        g1 = FG.Grids.StructuredGrid(sgeo, collect(range(0, 2π; length = 9)))
        Test.@test size(g1) == (9,)
        Test.@test FG.Grids.measure_factors(g1) !== nothing
        Test.@test sum(FG.Grids.measure(g1)) ≈ 2.0 * 2π rtol = 1e-2   # R·Δλ over the full circle
        g4s = FG.Grids.StructuredGrid(sgeo, collect(range(0, 2π; length = 8)),
                                collect(range(-1.0, 1.0; length = 6)),
                                collect(range(1.0, 2.0; length = 4)),
                                collect(range(0.0, 1.0; length = 3)))
        Test.@test FG.Grids.coordinate_names(g4s) == (:λ, :φ, :r, :q4)
        Test.@test all(>(0), FG.Grids.measure(g4s))

        # Topology lives in the type, and the mask defaults to all-active.
        Test.@test FG.Grids.topology(g4) == ntuple(_ -> FG.Grids.Bounded(), 4)
        gp = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0;
                               topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
        Test.@test FG.Grids.topology(gp, 1) === FG.Grids.Periodic()
        Test.@test FG.Grids.isperiodic(gp, 1) && !FG.Grids.isperiodic(gp, 2)
        Test.@test FG.Grids.periodic_flags(gp) == (true, false)
        Test.@test FG.Grids.period(gp, 1) ≈ 4.0            # uniform closure: n·|Δ|
        Test.@test FG.Grids.period(gp, 2) == 0.0           # bounded: no period
        # `periodic = true` still means "direction 1 wraps".
        gq = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0; periodic = true)
        Test.@test FG.Grids.isperiodic(gq, 1) && !FG.Grids.isperiodic(gq, 2)
        # A mask may be given positionally or by keyword, but not both.
        m = trues(4, 4); m[2, 2] = false
        Test.@test !FG.Grids.isactive(FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, m), 2, 2)
        Test.@test !FG.Grids.isactive(FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0; mask = m), 2, 2)
        Test.@test_throws ArgumentError FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, m; mask = m)
        Test.@test_throws DimensionMismatch FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, trues(3, 3))
    end

    Test.@testset "Mask topology: interior, boundary, components and holes" begin
        C = FG.Connectivity
        geo = FG.Geometry.CartesianGeometry()
        m = trues(9, 9)
        m[4:6, 4:6] .= false      # an enclosed block
        m[1, 1] = false            # touches the edge, so not enclosed
        g = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0, m)
        Test.@test C.count_holes(g) == 1
        labels, ncomp = C.connected_components(g)
        Test.@test ncomp == 1                       # the active region is one piece
        Test.@test all(labels[ci] == 0 for ci in CartesianIndices(m) if !m[ci])
        Test.@test all(labels[ci] == 1 for ci in CartesianIndices(m) if m[ci])
        # Interior and boundary partition the active cells.
        int = C.interior(g)
        bnd = C.boundary_cells(g)
        Test.@test count(int) + count(bnd) == count(m)
        Test.@test !any(int .& bnd)
        Test.@test all(!int[ci] for ci in CartesianIndices(m) if !m[ci])
        Test.@test !int[1, 5]                       # a domain edge is not interior
        Test.@test int[3, 3]

        # Wrapping removes the edges, so a region that only touched one becomes enclosed.
        gp = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0, m;
                               periodic = (true, true), period = (9.0, 9.0))
        Test.@test C.count_holes(gp) == 2
        # Two separated blocks are two holes.
        m2 = trues(11, 11); m2[3:4, 3:4] .= false; m2[8:9, 8:9] .= false
        Test.@test C.count_holes(FG.Grids.StructuredGrid(geo, 0.0:1.0:10.0, 0.0:1.0:10.0, m2)) == 2
        # A fully active grid has no holes and one component.
        gfull = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0)
        Test.@test C.count_holes(gfull) == 0
        Test.@test C.connected_components(gfull)[2] == 1
        # Two disconnected active regions are two components.
        m3 = falses(9, 3); m3[1:3, :] .= true; m3[7:9, :] .= true
        Test.@test C.connected_components(
            FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:2.0, m3))[2] == 2
        # A 3-D shell encloses its cavity.
        m4 = trues(7, 7, 7); m4[3:5, 3:5, 3:5] .= false
        Test.@test C.count_holes(
            FG.Grids.StructuredGrid(geo, 0.0:1.0:6.0, 0.0:1.0:6.0, 0.0:1.0:6.0, m4)) == 1
    end

    Test.@testset "Staggering, point location and interpolation weights" begin
        D = FG.Discretization
        A = FG.Axes
        # Faces bracket the centres, one more of them, at the midpoints.
        x = [0.0, 1.0, 3.0, 6.0]
        f = D.faces(x)
        Test.@test length(f) == length(x) + 1
        Test.@test f[2] ≈ 0.5 && f[3] ≈ 2.0 && f[4] ≈ 4.5
        Test.@test f[1] ≈ -0.5 && f[5] ≈ 7.5          # extrapolated a half-cell beyond the ends
        Test.@test issorted(f)
        # `centers` inverts `faces` exactly on a uniform axis. On a stretched one it cannot: `faces`
        # midpoints the centres and `centers` midpoints those back, which averages neighbouring widths.
        Test.@test length(D.centers(f)) == length(x)
        u = A.UniformAxis(0.0, 0.5, 6)
        Test.@test A.isuniform(D.faces(u))            # a uniform axis keeps its guarantee
        Test.@test collect(D.faces(u)) ≈ D.faces(collect(u))
        Test.@test collect(D.centers(D.faces(u))) ≈ collect(u)
        Test.@test D.nodes(u, D.Center()) === u
        Test.@test length(D.nodes(u, D.Face())) == length(u) + 1
        # The faces of a uniform axis are exactly the cell boundaries: their gaps are the spacing.
        Test.@test A.spacing(D.faces(u)) === A.spacing(u)

        # `locate` returns the cell containing a coordinate, and 0 outside.
        Test.@test D.locate(u, 0.0) == 1
        Test.@test D.locate(u, 0.2) == 1              # within the first cell's half-width
        Test.@test D.locate(u, 0.4) == 2              # past it, so the second cell
        Test.@test D.locate(u, 0.5) == 2
        Test.@test D.locate(u, -1.0) == 0
        Test.@test D.locate(u, 99.0) == 0
        # The uniform closed form and the general bisection must agree everywhere.
        v = collect(u)
        Test.@test all(D.locate(u, t) == D.locate(v, t) for t in range(-0.4, 2.9; length = 97))
        # …and on a genuinely stretched axis, against a direct scan of the faces.
        xs = cumsum([0.0, 1.0, 0.3, 2.5, 0.7, 4.0])
        fx = D.faces(xs)
        # The faces themselves are probed too: `f[i] ≤ v < f[i+1]` is a rule about exactly those
        # coordinates, so a reference that never lands on one cannot tell whether it is honoured.
        for t in vcat(collect(range(-1.0, 10.0; length = 121)), collect(fx))
            want = 0
            for i in eachindex(xs)
                last_cell = i == length(xs)
                fx[i] ≤ t && (t < fx[i+1] || (last_cell && t == fx[i+1])) && (want = i; break)
            end
            Test.@test D.locate(xs, t) == want
        end
        # A descending axis locates just as well.
        d = A.UniformAxis(3.0, -0.5, 7)
        Test.@test D.locate(d, 3.0) == 1 && D.locate(d, 0.0) == 7 && D.locate(d, 9.0) == 0

        # `nearest_index` always returns a valid index.
        Test.@test D.nearest_index(u, -5.0) == 1
        Test.@test D.nearest_index(u, 99.0) == length(u)
        Test.@test all(D.nearest_index(u, t) == D.nearest_index(v, t) for t in range(-1, 4; length = 61))
        # The two paths index the same numbers, so they must not part company at a midpoint, where
        # `(t - first)/Δ` can read as an exact tie while the two samples are not equidistant from `t`.
        let w = range(0.0, 1.0; length = 64), wv = collect(w)
            mids = [(wv[i] + wv[i+1]) / 2 for i in 1:(length(wv) - 1)]
            Test.@test all(D.nearest_index(w, t) == D.nearest_index(wv, t) for t in mids)
            Test.@test all(D.locate(w, t) == D.locate(wv, t) for t in mids)
            Test.@test all(D.nearest_index(wv, t) == argmin(abs.(wv .- t)) for t in mids)
        end

        # Both are searches, not sweeps: a stretched axis must not be walked, or materialize its
        # faces, on a query. Timed rather than only allocation-checked, since a scan need not allocate.
        let small = collect(range(0.0, 1.0; length = 64)),
            big = collect(range(0.0, 1.0; length = 1 << 18))
            for (nm, g) in (("locate", D.locate), ("nearest_index", D.nearest_index))
                g(small, 0.37); g(big, 0.37)
                Test.@test (@allocated g(big, 0.37)) == 0
                ts = minimum(@elapsed(g(small, 0.37 + 1e-9k)) for k in 1:200)
                tb = minimum(@elapsed(g(big, 0.37 + 1e-9k)) for k in 1:200)
                # 4096x the samples: a scan would be ~1000x slower, a bisection ~1.5x.
                tb < 20 * ts || println("    ", nm, " grew ", round(tb / ts; digits = 1), "x over 4096x n")
                Test.@test tb < 20 * ts
            end
        end

        # Linear weights sum to 1 and reproduce a linear function exactly.
        for t in range(0.0, 2.5; length = 37)
            i, w = D.interpolation_weights(v, t)
            Test.@test sum(w) ≈ 1.0
            Test.@test w[1] * v[i] + w[2] * v[i+1] ≈ t atol = 1e-12
        end
        # Outside the axis it clamps rather than extrapolating.
        i, w = D.interpolation_weights(v, -10.0)
        Test.@test w[1] * v[i] + w[2] * v[i+1] ≈ first(v)
        # Lagrange weights are exact for polynomials up to degree nodes-1, on a stretched axis.
        for k in 2:5
            for t in (0.7, 2.2, 5.5)
                idx, w = D.lagrange_weights(xs, t, k)
                Test.@test sum(w) ≈ 1.0 atol = 1e-10
                for p in 0:(k - 1)
                    Test.@test sum(w[a] * xs[idx[a]]^p for a in eachindex(w)) ≈ t^p rtol = 1e-8
                end
            end
        end
    end

    Test.@testset "Fornberg finite-difference weights are exact to their stated order" begin
        D = FG.Discretization
        # The classic equispaced formulas, reproduced from the general recursion.
        Test.@test D.fd_weights([-1.0, 0.0, 1.0], 0.0, 1) ≈ [-0.5, 0.0, 0.5]
        Test.@test D.fd_weights([-1.0, 0.0, 1.0], 0.0, 2) ≈ [1.0, -2.0, 1.0]
        Test.@test D.fd_weights([0.0, 1.0], 0.0, 1) ≈ [-1.0, 1.0]               # forward difference
        Test.@test D.fd_weights([-2.0, -1.0, 0.0], 0.0, 1) ≈ [0.5, -2.0, 1.5]   # backward, 2nd order
        Test.@test D.fd_weights([-2.0, -1.0, 0.0, 1.0, 2.0], 0.0, 1) ≈
                   [1/12, -2/3, 0.0, 2/3, -1/12]
        Test.@test D.fd_weights([-2.0, -1.0, 0.0, 1.0, 2.0], 0.0, 2) ≈
                   [-1/12, 4/3, -5/2, 4/3, -1/12]
        Test.@test D.fd_weights([-1.0, 0.0, 1.0], 0.0, 0) ≈ [0.0, 1.0, 0.0]     # order 0 interpolates

        # The defining property, on ARBITRARILY spaced nodes: with m nodes the weights differentiate
        # every polynomial of degree ≤ m-1 exactly.
        nodes = [-1.7, -0.4, 0.0, 0.9, 2.3, 4.1]
        for m in 1:length(nodes), order in 0:min(3, m - 1)
            w = D.fd_weights(nodes[1:m], 0.35, order)
            for p in 0:(m - 1)
                got = sum(w[a] * nodes[a]^p for a in 1:m)
                want = p < order ? 0.0 : prod(p - k for k in 0:(order - 1); init = 1.0) * 0.35^(p - order)
                Test.@test isapprox(got, want; atol = 1e-8, rtol = 1e-8)
            end
        end
        Test.@test_throws ArgumentError D.fd_weights([0.0, 1.0], 0.0, 2)   # too few nodes
        Test.@test_throws ArgumentError D.fd_weights([0.0, 0.0], 0.0, 1)   # repeated node
        Test.@test_throws ArgumentError D.fd_weights([0.0, 1.0], 0.0, -1)

        # The axis form keeps the node count at a boundary rather than clipping the stencil, so the
        # accuracy order is the same everywhere.
        xs = collect(range(0.0, 1.0; length = 11))
        for i in 1:11
            idx, w = D.fd_weights(xs, i, 1, 5)
            Test.@test length(idx) == 5
            # exact for a quartic, which 5 nodes must be
            fpoly(t) = 2t^4 - 3t^3 + t - 5
            dpoly(t) = 8t^3 - 9t^2 + 1
            Test.@test sum(w[a] * fpoly(xs[idx[a]]) for a in 1:5) ≈ dpoly(xs[i]) atol = 1e-9 rtol = 1e-8
        end
        # A stretched axis costs nothing extra and stays exact.
        xnu = cumsum([0.0, 0.05, 0.2, 0.07, 0.3, 0.12, 0.26])
        for i in eachindex(xnu)
            idx, w = D.fd_weights(xnu, i, 2, 4)
            fq(t) = 3t^3 - t^2 + 2t
            d2(t) = 18t - 2
            Test.@test sum(w[a] * fq(xnu[idx[a]]) for a in eachindex(w)) ≈ d2(xnu[i]) atol = 1e-7 rtol = 1e-6
        end
        # It generalizes the existing three-point first-derivative formula.
        h_m, h_p = 1.0, 2.0
        w = D.fd_weights([-h_m, 0.0, h_p], 0.0, 1)
        f_m, f_0, f_p = 1.3, -0.7, 2.9
        Test.@test w[1] * f_m + w[2] * f_0 + w[3] * f_p ≈
                   FG.Geometry.nonuniform_first_derivative(f_m, f_0, f_p, h_m, h_p)
    end

    Test.@testset "Metric scale factors and the Jacobian" begin
        G = FG.Geometry
        sg = G.SphericalGeometry(2.0)
        Test.@test G.scale_factors(sg, (0.3, 0.0)) == (2.0, 2.0)
        Test.@test all(isapprox.(G.scale_factors(sg, (0.0, π / 3)), (1.0, 2.0); atol = 1e-15))
        Test.@test all(isapprox.(G.scale_factors(sg, (0.0, π / 3, 5.0)), (2.5, 5.0, 1.0); atol = 1e-15))
        Test.@test G.scale_factors(FG.Geometry.CartesianGeometry(), (1.0, 2.0)) == (1.0, 1.0)
        Test.@test G.scale_factors(FG.Geometry.CartesianGeometry(), (1.0, 2.0, 3.0)) == (1.0, 1.0, 1.0)
        Test.@test G.jacobian(sg, (0.0, 0.5)) ≈ 4 * cos(0.5)
        Test.@test G.jacobian(FG.Geometry.CartesianGeometry(), (1.0, 2.0)) == 1.0
        # The Jacobian is the area element per unit coordinate area, which is what the grid's own
        # measure is built from.
        gg = FG.Grids.StructuredGrid(sg, collect(range(0, 2π; length = 40)),
                               collect(range(-1.2, 1.2; length = 30)))
        λ, φ = FG.Grids.coordinate_names(gg)
        i, j = 7, 11
        p = FG.Grids.coords(gg, i, j)
        Δλ = FG.Grids._cell_width(FG.Grids.coordinates(gg, 1), i, FG.Grids.period(gg, 1))
        Δφ = FG.Grids._cell_width(FG.Grids.coordinates(gg, 2), j)
        Test.@test FG.Grids.measure(gg, i, j) ≈ G.jacobian(sg, (p.λ, p.φ)) * Δλ * Δφ rtol = 1e-12
        # Any point representation is accepted.
        Test.@test G.scale_factors(sg, (λ = 0.0, φ = π / 3)) == G.scale_factors(sg, (0.0, π / 3))
        Test.@test G.scale_factors(sg, [0.0, π / 3]) == G.scale_factors(sg, (0.0, π / 3))
    end

    Test.@testset "Octahedral and reduced Gaussian grids" begin
        SS = FG.SphericalSampling
        # The defining rule: 20 longitudes on the ring nearest the pole, +4 per ring, 4N(N+9) in all.
        for N in (1, 2, 5, 16, 80, 320)
            s = SS.OctahedralGaussianSampling(N)
            counts = SS.nlon_per_ring(s)
            Test.@test length(counts) == SS.nrings(s) == 2N
            Test.@test counts[1] == 20
            Test.@test N == 1 || all(diff(counts[1:N]) .== 4)
            Test.@test sum(counts) == 4 * N * (N + 9) == SS.npoints(s)
            Test.@test counts == reverse(counts)          # the hemispheres mirror
        end
        # Latitudes are the Gaussian ones, north to south, and the weights are Gauss–Legendre's.
        for N in (4, 12)
            s = SS.OctahedralGaussianSampling(N)
            lats = SS.ring_latitudes(s)
            Test.@test issorted(lats; rev = true)
            Test.@test sort(lats) ≈ FG.SphericalSampling.spherical_axes(FG.SphericalSampling.GaussLegendreSampling(), 2N).φ atol = 1e-12
            w = SS.latitude_weights(s)
            Test.@test sum(w) ≈ 2 rtol = 1e-13
            Test.@test all(>(0), w)
            # Exact for every P_l the ring count resolves.
            function legendre(l, x)
                l == 0 && return one(x)
                p0, p1 = one(x), x
                for k in 1:(l - 1)
                    p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
                end
                return p1
            end
            for l in 1:(2N - 1)
                Test.@test abs(sum(w[j] * legendre(l, sin(lats[j])) for j in eachindex(lats))) < 1e-11
            end
            # The full-sphere integral, with the per-ring longitude factor the varying counts require.
            counts = SS.nlon_per_ring(s)
            Test.@test sum(w[j] * (2π / counts[j]) * counts[j] for j in eachindex(counts)) ≈ 4π rtol = 1e-13
        end
        # Points are distinct, on the sphere, and ring-ordered.
        for N in (2, 8)
            s = SS.OctahedralGaussianSampling(N)
            p = SS.spherical_points(s)
            n = SS.npoints(s)
            Test.@test length(p.λ) == length(p.φ) == n
            Test.@test all(φ -> -π/2 ≤ φ ≤ π/2, p.φ)
            Test.@test all(λ -> 0 ≤ λ < 2π + 1e-12, p.λ)
            Test.@test length(unique([(round(a; digits = 12), round(b; digits = 12))
                                      for (a, b) in zip(p.λ, p.φ)])) == n
        end
        # A ring sampling is not a tensor product, but it is iso-latitude and has an exact quadrature.
        oct = SS.OctahedralGaussianSampling(4)
        Test.@test !SS.is_tensor_product(oct)
        Test.@test SS.is_iso_latitude(oct)
        Test.@test SS.admits_exact_bandlimited_quadrature(oct)
        Test.@test_throws ArgumentError SS.OctahedralGaussianSampling(0)

        # An explicit table is the other way a reduced grid is specified.
        rg = SS.ReducedGaussianSampling([20, 24, 24, 20])
        Test.@test SS.nrings(rg) == 4 && SS.npoints(rg) == 88
        Test.@test length(SS.spherical_points(rg).λ) == 88
        Test.@test issorted(SS.ring_latitudes(rg); rev = true)
        Test.@test_throws ArgumentError SS.ReducedGaussianSampling(Int[])
        Test.@test_throws ArgumentError SS.ReducedGaussianSampling([4, 0])
    end

    Test.@testset "Fibonacci lattice" begin
        SS = FG.SphericalSampling
        for n in (1, 7, 100, 3000)
            s = SS.FibonacciSampling(n)
            p = SS.spherical_points(s)
            Test.@test SS.npoints(s) == n == length(p.λ) == length(p.φ)
            Test.@test all(φ -> -π/2 - 1e-12 ≤ φ ≤ π/2 + 1e-12, p.φ)
            Test.@test all(λ -> 0 ≤ λ < 2π + 1e-12, p.λ)
            Test.@test length(unique([(round(a; digits = 10), round(b; digits = 10))
                                      for (a, b) in zip(p.λ, p.φ)])) == n
            # `z` advances in exactly equal steps, which is what makes the bands equal-area.
            n < 2 || Test.@test maximum(abs.(diff(sin.(p.φ)) .- 2 / n)) < 1e-12
        end
        # Quasi-uniform: nearest-neighbour separation barely varies, unlike a lat–lon grid's.
        p = SS.spherical_points(SS.FibonacciSampling(1500))
        v = [(cos(p.φ[i]) * cos(p.λ[i]), cos(p.φ[i]) * sin(p.λ[i]), sin(p.φ[i])) for i in 1:1500]
        nn = [minimum(sqrt(sum(abs2, v[i] .- v[j])) for j in 1:1500 if j != i) for i in 1:150]
        Test.@test minimum(nn) / maximum(nn) > 0.5
        Test.@test SS.is_equal_area(SS.FibonacciSampling(10))
        Test.@test_throws ArgumentError SS.FibonacciSampling(0)
        # The bang form writes into the caller's buffers: its allocation count is flat in `n`, so
        # nothing is allocated per point. (The residual is the timing closure, not the function.)
        nalloc(f) = (f(); minimum(Base.gc_alloc_count((@timed f()).gcstats) for _ in 1:5))
        function fib_allocs(n)
            a = Vector{Float64}(undef, n); b = similar(a)
            fib = SS.FibonacciSampling(n)
            SS.spherical_points!(a, b, fib)
            return nalloc(() -> SS.spherical_points!(a, b, fib))
        end
        Test.@test fib_allocs(500) == fib_allocs(50_000)
        Test.@test fib_allocs(500) <= 2
        Test.@test_throws DimensionMismatch SS.spherical_points!(zeros(3), zeros(3),
                                                                 SS.FibonacciSampling(4))
    end

    Test.@testset "HEALPix pixel indexing round-trips in both schemes" begin
        SS = FG.SphericalSampling
        for nside in (1, 2, 3, 4, 8, 16)
            npix = SS.healpix_npix(nside)
            # Locating a pixel's own centre must return that pixel, for every pixel.
            for p in 0:(npix - 1)
                θ, ϕ = SS.pix2ang(nside, p)
                Test.@test SS.ang2pix(nside, θ, ϕ) == p
                Test.@test SS.vec2pix(nside, SS.pix2vec(nside, p)) == p
            end
            Test.@test_throws ArgumentError SS.pix2ang(nside, npix)
            Test.@test_throws ArgumentError SS.pix2ang(nside, -1)
        end
        # RING <-> NESTED is a bijection, and both name the same point.
        for nside in (1, 2, 4, 8)
            npix = SS.healpix_npix(nside)
            seen = falses(npix)
            for p in 0:(npix - 1)
                n = SS.ring2nest(nside, p)
                Test.@test 0 ≤ n < npix
                Test.@test !seen[n + 1]
                seen[n + 1] = true
                Test.@test SS.nest2ring(nside, n) == p
                a = SS.pix2ang(nside, n; scheme = SS.Nested())
                b = SS.pix2ang(nside, p)
                Test.@test a[1] ≈ b[1] && a[2] ≈ b[2]
                Test.@test SS.ang2pix(nside, b[1], b[2]; scheme = SS.Nested()) == n
            end
            Test.@test all(seen)
        end
        # NESTED needs a power-of-two nside; RING does not.
        Test.@test_throws ArgumentError SS.ring2nest(3, 0)
        Test.@test_throws ArgumentError SS.nest2ring(3, 0)
        Test.@test_throws ArgumentError SS.ang2pix(3, 1.0, 1.0; scheme = SS.Nested())
        Test.@test SS.ang2pix(3, 1.0, 1.0) isa Int
        # `pix2ang` must name the same points `spherical_points` lists.
        for nside in (2, 8)
            p = SS.spherical_points(FG.SphericalSampling.HEALPixSampling(nside))
            for i in 0:(SS.healpix_npix(nside) - 1)
                θ, ϕ = SS.pix2ang(nside, i)
                Test.@test SS.geographic_latitude(θ) ≈ p.φ[i + 1]
                Test.@test ϕ ≈ p.λ[i + 1]
            end
        end
    end

    Test.@testset "Oblate spheroid geometry" begin
        G = FG.Geometry
        g = G.SpheroidGeometry()                       # WGS 84
        Test.@test g.a == 6378137.0
        Test.@test G.semiminor_axis(g) ≈ 6356752.314245 atol = 1e-6
        Test.@test G.eccentricity²(g) ≈ 0.00669437999014 atol = 1e-12
        # The quarter meridian is a published WGS 84 constant, and it exercises the whole geodesic.
        Test.@test G.distance(g, (0.0, 0.0), (0.0, π / 2)) ≈ 10001965.729 atol = 1e-2
        # The equator is a circle of radius a, so a quarter of it is exactly π·a/2.
        Test.@test G.distance(g, (0.0, 0.0), (π / 2, 0.0)) ≈ π * g.a / 2 atol = 1e-3
        # A meridian geodesic IS the meridian arc, so it can be checked against ∫M(φ)dφ directly.
        q = FG.SphericalSampling.spherical_quadrature(FG.SphericalSampling.GaussLegendreSampling(), 400)
        function meridian_arc(geo, φa, φb)
            mid = (φa + φb) / 2
            half = (φb - φa) / 2
            return half * sum(q.w[j] * G.meridional_radius(geo, mid + half * sin(q.φ[j]))
                              for j in eachindex(q.φ))
        end
        for (φa, φb) in ((0.0, 0.3), (-0.9, 0.2), (0.5, 1.4), (-1.5, 1.5))
            Test.@test G.distance(g, (0.4, φa), (0.4, φb)) ≈ meridian_arc(g, φa, φb) atol = 1e-3
        end
        # Symmetry, the zero case, and a finite answer for a near-antipodal pair.
        p1, p2 = (0.3, -0.4), (1.9, 0.55)
        Test.@test G.distance(g, p1, p2) ≈ G.distance(g, p2, p1)
        Test.@test G.distance(g, p1, p1) == 0.0
        d = G.distance(g, (0.0, 0.0), (π - 1e-9, 1e-9))
        Test.@test isfinite(d) && 1.9e7 < d < 2.1e7
        # Zero flattening must reduce exactly to the sphere.
        Test.@test G.distance(G.SpheroidGeometry(6371000.0, 0.0), (0.1, 0.2), (0.3, 0.4)) ≈
                   G.distance(G.SphericalGeometry(6371000.0), (0.1, 0.2), (0.3, 0.4)) rtol = 1e-9
        # The area element integrates to the closed-form spheroid surface area.
        e = sqrt(G.eccentricity²(g))
        exact = 2π * g.a^2 * (1 + (1 - e^2) / e * atanh(e))
        total = 2π * sum(q.w[j] * G.meridional_radius(g, q.φ[j]) * G.prime_vertical_radius(g, q.φ[j])
                         for j in eachindex(q.φ))
        Test.@test total ≈ exact rtol = 1e-9
        # Curvature radii bracket a, and the scale factors are built from them.
        Test.@test G.prime_vertical_radius(g, 0.0) ≈ g.a
        Test.@test G.meridional_radius(g, 0.0) < g.a < G.meridional_radius(g, π / 2)
        Test.@test G.scale_factors(g, (0.0, 0.0)) == (g.a, G.meridional_radius(g, 0.0))
        Test.@test G.area_element(g, 0.0, 1.0, 1.0) ≈
                   G.meridional_radius(g, 0.0) * G.prime_vertical_radius(g, 0.0)
        # Coordinate names follow the geodetic convention.
        Test.@test G.point_names(g, Val(2)) == (:λ, :φ)
        Test.@test G.point_names(g, Val(3)) == (:λ, :φ, :h)
        Test.@test G.point_names(g, Val(5)) == (:λ, :φ, :h, :q4, :q5)
        Test.@test_throws ArgumentError G.SpheroidGeometry(-1.0, 0.1)
        Test.@test_throws ArgumentError G.SpheroidGeometry(1.0, 1.5)

        # The geodetic volume element offsets BOTH curvature radii by h, so it does not factor.
        Test.@test G.volume_element(g, 0.4, 0.0, 1.0, 1.0, 1.0) ≈
                   G.area_element(g, 0.4, 1.0, 1.0)
        Test.@test G.volume_element(g, 0.4, 100.0, 1e-3, 1e-3, 2.0) ≈
                   (G.prime_vertical_radius(g, 0.4) + 100.0) * cos(0.4) *
                   (G.meridional_radius(g, 0.4) + 100.0) * 1e-3 * 1e-3 * 2.0
    end

    Test.@testset "A spheroid drives the grid stack in every dimension" begin
        GE = FG.Geometry
        geo = GE.SpheroidGeometry()
        a, e² = GE.semimajor_axis(geo), GE.eccentricity²(geo)
        λ8 = collect(range(0, 2π; length = 9)[1:8])
        φ7 = collect(range(-π / 2, π / 2; length = 7))
        h4 = collect(range(0.0, 3000.0; length = 4))
        g1 = FG.Grids.StructuredGrid(geo, λ8)
        g2 = FG.Grids.StructuredGrid(geo, λ8, φ7)
        g3 = FG.Grids.StructuredGrid(geo, λ8, φ7, h4)
        g4 = FG.Grids.StructuredGrid(geo, λ8, φ7, h4, [0.0, 2.0])
        Test.@test size.((g1, g2, g3, g4)) == ((8,), (8, 7), (8, 7, 4), (8, 7, 4, 2))
        # Longitude closes the circle, so it is periodic without being asked; latitude is not.
        Test.@test FG.Grids.isperiodic(g2, 1) && !FG.Grids.isperiodic(g2, 2)
        Test.@test FG.Grids.coordinate_names(g3) == (:λ, :φ, :h)
        Test.@test FG.Grids.coords(g3, 2, 3, 2) == (λ = λ8[2], φ = φ7[3], h = h4[2])

        # 1-D: with no latitude the parallel is the equator, so the circumference is exactly 2πa.
        Test.@test sum(FG.Grids.measure(g1)) ≈ 2π * a rtol = 1e-14

        # 2-D factors; 3-D cannot, so it is stored dense.
        Test.@test FG.Grids.measure_factors(g2) !== nothing
        Test.@test FG.Grids.measure_factors(g3) === nothing

        # Every cell equals the geometry's own element, exactly — not to a tolerance.
        wλ = FG.Grids._axis_widths(FG.Grids.coordinates(g2, 1), 2π)
        wφ = FG.Grids._axis_widths(FG.Grids.coordinates(g2, 2), nothing)
        wh = FG.Grids._axis_widths(FG.Grids.coordinates(g3, 3), nothing)
        Test.@test all(
            FG.Grids.measure(g2, i, j) ≈ GE.area_element(geo, φ7[j], wλ[i], wφ[j])
            for j in eachindex(φ7), i in eachindex(λ8)
        )
        Test.@test all(
            FG.Grids.measure(g3, i, j, k) ≈
            GE.volume_element(geo, φ7[j], h4[k], wλ[i], wφ[j], wh[k])
            for k in eachindex(h4), j in eachindex(φ7), i in eachindex(λ8)
        )
        # A further direction enters as a plain width.
        Test.@test sum(FG.Grids.measure(g4)) ≈
                   sum(FG.Grids._axis_widths([0.0, 2.0], nothing)) * sum(FG.Grids.measure(g3)) rtol = 1e-13

        # Independent check: the closed-form ellipsoid area, approached at second order.
        e = sqrt(e²)
        A_exact = 2π * a^2 * (1 + (1 - e²) / e * atanh(e))
        errs = map((24, 96, 384)) do n
            gn = FG.Grids.StructuredGrid(geo, collect(range(0, 2π; length = n + 1)[1:n]),
                                         collect(range(-π / 2, π / 2; length = n ÷ 2 + 1)))
            abs(sum(FG.Grids.measure(gn)) - A_exact) / A_exact
        end
        Test.@test errs[3] < 3e-5
        Test.@test errs[1] / errs[2] > 8 && errs[2] / errs[3] > 8   # 4× refinement ⇒ ~16×
        # A sphere is the f = 0 spheroid, and must agree with SphericalGeometry cell for cell.
        gsph = FG.Grids.StructuredGrid(GE.SphericalGeometry(a), λ8, φ7)
        gflat = FG.Grids.StructuredGrid(GE.SpheroidGeometry(a, 0.0), λ8, φ7)
        Test.@test all(FG.Grids.measure(gflat, i, j) ≈ FG.Grids.measure(gsph, i, j)
                       for j in eachindex(φ7), i in eachindex(λ8))

        Test.@test FG.Connectivity.nneighbors(g2, 1, 3) == 4       # wraps in λ
        Test.@test FG.Connectivity.nedges(FG.Connectivity.build_connectivity(g2)) > 0

        # A degenerate angular direction drops the differential that no longer exists, so a transect
        # measures arc length rather than an area with a placeholder in it.
        # Zonal at the equator: the parallel radius is N(0)·cos0 = a, so the circle closes at 2πa.
        gzon = FG.Grids.StructuredGrid(geo, λ8, [0.0])
        Test.@test sum(FG.Grids.measure(gzon)) ≈ 2π * a rtol = 1e-14
        # Zonal at latitude φ: radius N(φ)·cosφ.
        gzon2 = FG.Grids.StructuredGrid(geo, λ8, [0.7])
        Test.@test sum(FG.Grids.measure(gzon2)) ≈
                   2π * GE.prime_vertical_radius(geo, 0.7) * cos(0.7) rtol = 1e-14
        # Meridional: each cell is exactly M(φ)·Δφ, the meridian arc element.
        φm = collect(range(-π / 2, π / 2; length = 33))
        gmer = FG.Grids.StructuredGrid(geo, [0.0], φm)
        wm = FG.Grids._axis_widths(φm, nothing)
        Test.@test all(FG.Grids.measure(gmer, 1, j) ≈ GE.meridional_radius(geo, φm[j]) * wm[j]
                       for j in eachindex(φm))
        # The total approaches ∫M dφ — twice the published quarter meridian — at FIRST order, because
        # `n` cell-centres own `n` widths and so span π(1 + 1/n). Unlike the area case there is no cosφ
        # factor to annihilate that end-cell excess, and the error is exactly the 1/n it predicts.
        mer = map((48, 192, 768)) do n
            gm = FG.Grids.StructuredGrid(geo, [0.0], collect(range(-π / 2, π / 2; length = n + 1)))
            abs(sum(FG.Grids.measure(gm)) - 2 * 10001965.729) / (2 * 10001965.729)
        end
        Test.@test all(isapprox(mer[i], 1 / (48 * 4^(i - 1)); rtol = 0.02) for i in 1:3)
        # A single point has no extent in either direction.
        Test.@test sum(FG.Grids.measure(FG.Grids.StructuredGrid(geo, [0.3], [0.4]))) == 1.0

        # The dense measure is a second storage form, so the accessors around it get their own checks.
        Test.@test size(FG.Grids.measure_array(g3)) == size(g3)
        Test.@test sum(FG.Grids.measure(g3)) == sum(FG.Grids.measure_array(g3))
        Test.@test_throws BoundsError FG.Grids.measure(g3, 99, 1, 1)
        Test.@test_throws BoundsError FG.Grids.coords(g3, 1, 99, 1)
        Test.@test occursin("h ", sprint(show, MIME"text/plain"(), g3))
        mk = trues(size(g3)...)
        mk[1, 1, 1] = false
        gm = FG.Grids.StructuredGrid(geo, λ8, φ7, h4, mk)
        Test.@test !FG.Grids.isactive(gm, 1, 1, 1) && FG.Grids.isactive(gm, 2, 2, 2)
        Test.@test FG.Connectivity.nneighbors(gm, 2, 2, 2) == 6
    end

    Test.@testset "Pole rotation" begin
        G = FG.Geometry
        for (λp, φp) in ((0.0, π / 2), (0.7, 0.3), (2.0, -0.9), (4.5, 0.0))
            rot = G.PoleRotation(λp, φp)
            # The rotation's own pole becomes the new north pole.
            Test.@test G.rotate(rot, λp, φp)[2] ≈ π / 2 atol = 1e-12
            # `unrotate` inverts `rotate` everywhere.
            for λ in range(0, 2π; length = 13), φ in range(-1.4, 1.4; length = 9)
                a, b = G.rotate(rot, λ, φ)
                c, d = G.unrotate(rot, a, b)
                Test.@test abs(mod(c - λ + π, 2π) - π) < 1e-12
                Test.@test abs(d - φ) < 1e-12
                Test.@test -π/2 - 1e-12 ≤ b ≤ π/2 + 1e-12
                Test.@test 0 ≤ a < 2π + 1e-12
            end
        end
        # A rotation is an isometry: it preserves angular distance.
        rot = G.PoleRotation(0.7, 0.3)
        u = G.SphericalGeometry(1.0)
        for (p, q) in (((0.2, 0.1), (1.1, -0.4)), ((3.0, 0.9), (5.5, -1.2)))
            Test.@test G.distance(u, p, q) ≈
                       G.distance(u, G.rotate(rot, p...), G.rotate(rot, q...)) atol = 1e-12
        end
        # The identity rotation leaves the frame alone.
        idr = G.PoleRotation(0.0, π / 2)
        Test.@test all(isapprox.(G.rotate(idr, 1.2, 0.4), (1.2, 0.4); atol = 1e-12))
    end

    Test.@testset "An arc longer than half the sphere still finds every node" begin
        # `2sin(σ/2)` is the chord of an arc σ, and it turns back down past σ = π: a query radius built
        # from it without a clamp SHRINKS as the requested arc grows, and vanishes at σ = 2π.
        R = 6.371e6
        geo = FG.Geometry.SphericalGeometry(R)
        λ = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        φ = [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9]
        for frac in (0.5, 1.0, 1.05, 2.0)      # a quarter turn, then at, past and twice the antipode
            r = frac * π * R
            g = FG.Grids.UnstructuredGrid(geo, (λ, φ), trues(7); radius = r)
            for k in 1:7
                got = sort(collect(FG.Grids.neighbors(g, k)))
                want = sort([j for j in 1:7 if j != k &&
                             FG.Geometry.distance(geo, (λ[k], φ[k]), (λ[j], φ[j])) ≤ r])
                Test.@test got == want
            end
            # `2sin(σ/2)` is not monotone past `σ = π`, so the query radius has to be clamped there: no
            # two points on a sphere are more than π R apart, so from the antipode up every node must see
            # all six others however much further the radius reaches.
            frac ≥ 1 && Test.@test all(k -> length(FG.Grids.neighbors(g, k)) == 6, 1:7)
        end
    end

    Test.@testset "A ball query recomputes no grid invariant" begin
        C = FG.Connectivity
        # `MetricTopology` carries the per-direction minimum steps, so bounding the candidate window is
        # O(1) per direction on a stretched axis rather than a scan of the axis per query.
        geo = FG.Geometry.CartesianGeometry{Float64}()
        # Spacing ~1 whatever `n` is, so a fixed radius spans a fixed number of cells and any growth in
        # per-query cost is overhead rather than more candidates.
        function stretched(n)
            x = cumsum(1.0 .+ 0.5 .* sin.(range(0, 3π; length = n)))
            return FG.Grids.StructuredGrid(geo, x, trues(n))
        end
        # The fully-qualified path, and a warm-up call: a local `C` is captured as a `Module` field, so
        # `C.nneighbors_within` is a dynamic lookup and boxes what it passes.
        function nalloc(g, mt)
            FG.Connectivity.nneighbors_within(g, 32; ball = 3.0, topology = mt)
            return @allocated FG.Connectivity.nneighbors_within(g, 32; ball = 3.0, topology = mt)
        end
        percall(g, mt, r, reps) = begin
            I = (length(FG.Grids.mask(g)) ÷ 2,)
            C.nneighbors_within(g, I...; ball = r, topology = mt)
            t = @elapsed for _ in 1:reps
                C.nneighbors_within(g, I...; ball = r, topology = mt)
            end
            return t / reps
        end
        ts = map((256, 4096)) do n
            g = stretched(n)
            mt = C.MetricTopology(g)
            Test.@test mt isa C.MetricTopology
            Test.@test isbits(mt)                              # nothing heap-allocated to carry
            Test.@test nalloc(g, mt) == 0
            (percall(g, mt, 3.0, 3000), g, mt)
        end
        # A 16× longer axis holding the window size fixed: cost must be flat, not 16× worse. The bound
        # is loose enough for a noisy machine and far tighter than the linear growth it replaces.
        Test.@test ts[2][1] < 4 * ts[1][1]
        # And the count is the truth, whichever topology computed it.
        for (_, g, mt) in ts
            I = (length(FG.Grids.mask(g)) ÷ 2,)
            Test.@test C.nneighbors_within(g, I...; ball = 3.0, topology = mt) ==
                       C.nneighbors_within(g, I...; ball = 3.0)
        end
    end

    Test.@testset "An index changes a ball query's cost, never its answer" begin
        C = FG.Connectivity
        GD = FG.Grids
        geo = FG.Geometry.CartesianGeometry{Float64}()
        sgeo = FG.Geometry.SphericalGeometry(6.371e6)
        function curv(n; periodic = false, mask = trues(n, n))
            x = [t for t in range(0.0, 10.0; length = n), _ in 1:n]
            y = [t for _ in 1:n, t in range(0.0, 10.0; length = n)]
            return GD.CurvilinearGrid(geo, x, y, mask; measure = fill(1.0, n, n),
                                      periodic = (periodic, false),
                                      period = (periodic ? 10.0 : 0.0, 0.0))
        end
        function curv_sph(nλ, nφ)
            λ = [l for l in range(0.0, 2π * (1 - 1 / nλ); length = nλ), _ in 1:nφ]
            φ = [f for _ in 1:nλ, f in range(-1.2, 1.2; length = nφ)]
            return GD.CurvilinearGrid(sgeo, λ, φ, trues(nλ, nφ))
        end
        Test.@test GD.has_spatial_index(curv(8))                # the extension is loaded

        R = 6.371e6
        wall = trues(16, 16); wall[:, 9] .= false
        # (λ, φ, r) on a sphere, where the metric is the 3-D chord of the same embedding the index uses.
        nλ, nφ, nr = 16, 9, 4
        sph3 = GD.CurvilinearGrid(
            sgeo,
            [l for l in range(0.0, 2π * (1 - 1 / nλ); length = nλ), _ in 1:nφ, _ in 1:nr],
            [f for _ in 1:nλ, f in range(-1.0, 1.0; length = nφ), _ in 1:nr],
            [q for _ in 1:nλ, _ in 1:nφ, q in range(6.371e6, 1.02 * 6.371e6; length = nr)],
            fill(1.0, nλ, nφ, nr), trues(nλ, nφ, nr),
        )
        # A spheroid, where the ECEF chord the index searches is a LOWER bound on the Vincenty geodesic
        # the gate applies: the index over-returns, which is what it is allowed to do, and the answer is
        # still exactly the scan's.
        spd = FG.Geometry.SpheroidGeometry(6.378137e6, 1 / 298.257223563)
        λ2 = [l for l in range(0.0, 2π * (1 - 1 / 20); length = 20), _ in 1:11]
        φ2 = [f for _ in 1:20, f in range(-1.2, 1.2; length = 11)]
        spd2 = GD.CurvilinearGrid(spd, λ2, φ2, fill(1.0, 20, 11), trues(20, 11))
        nh = 3
        spd3 = GD.CurvilinearGrid(
            spd,
            [l for l in range(0.0, 2π * (1 - 1 / 20); length = 20), _ in 1:11, _ in 1:nh],
            [f for _ in 1:20, f in range(-1.2, 1.2; length = 11), _ in 1:nh],
            [h for _ in 1:20, _ in 1:11, h in range(0.0, 2.0e5; length = nh)],
            fill(1.0, 20, 11, nh), trues(20, 11, nh),
        )
        # A box wrapping in two of three directions, so the tree needs ghost images in both.
        nb = 8
        axb = collect(range(0.0, 7.0; length = nb))
        box3 = GD.CurvilinearGrid(
            geo,
            [x for x in axb, _ in 1:nb, _ in 1:nb],
            [y for _ in 1:nb, y in axb, _ in 1:nb],
            [z for _ in 1:nb, _ in 1:nb, z in axb],
            fill(1.0, nb, nb, nb), trues(nb, nb, nb);
            periodic = (true, true, false), period = (8.0, 8.0, 0.0),
        )
        cases = (
            (curv(16), ((1, 1), (8, 8), (16, 16), (1, 16)), 2.0),
            (curv(16; periodic = true), ((1, 1), (8, 8), (16, 8), (1, 16)), 2.0),
            (curv(16; mask = wall), ((8, 5), (1, 1), (14, 14)), 3.0),
            (curv_sph(24, 12), ((1, 1), (12, 6), (24, 12)), 0.2R),
            (curv_sph(24, 12), ((12, 6),), 3.0R),               # reaches past the antipode
            (sph3, ((1, 1, 1), (8, 5, 2), (16, 9, 4)), 0.3R),
            (sph3, ((8, 5, 2),), 3.0R),
            (spd2, ((1, 1), (10, 6), (20, 11)), 2.0e6),
            (spd2, ((10, 6),), 1.2e7),
            (spd3, ((1, 1, 1), (10, 6, 2), (20, 11, 3)), 2.0e6),
            (box3, ((1, 1, 1), (4, 4, 4), (8, 8, 8)), 2.5),
        )
        for (g, seeds, r) in cases
            ix = C.indexed(g)
            s = C.ball_scratch()
            cl = C.MetricTopology(g; index = GD.cell_list(g; ball = r))
            for I in seeds
                # The same SET of cells. Order is whatever enumerated them — a window, a tree and a cell
                # list each walk their own way — and no entry point sorts.
                scan = sort(C.neighbors_within(g, I...; ball = r))
                for top in (ix, cl)
                    Test.@test sort(C.neighbors_within(g, I...; ball = r, topology = top)) == scan
                    Test.@test sort(C.neighbors_within(g, I...; ball = r, topology = top,
                                                       scratch = s)) == scan
                    Test.@test C.nneighbors_within(g, I...; ball = r, topology = top) == length(scan)
                    buf = Vector{Int}(undef, length(scan))
                    n = C.neighbors_within!(buf, g, I...; ball = r, topology = top, scratch = s)
                    Test.@test sort(buf[1:n]) == scan
                    Test.@test allunique(buf[1:n])          # no cell reported twice
                end
                # The fold visits the same cells with the same distances, whatever found them.
                visit(top; sc = nothing) = sort!(C.fold_within(
                    Tuple{Int,Float64}[], g, I...; ball = r, topology = top, scratch = sc,
                ) do acc, J, d
                    push!(acc, (C._linidx(size(GD.mask(g)), J...), d))
                    return acc
                end)
                Test.@test visit(ix; sc = s) == visit(C.MetricTopology(g))
                Test.@test visit(cl; sc = s) == visit(C.MetricTopology(g))
            end
        end

        # Node sets, including a radius past the antipode.
        gu = C.unstructured_grid(FG.SphericalSampling.HEALPixSampling(4))
        ixu = C.indexed(gu)
        su = C.ball_scratch()
        for (idx, r) in ((1, 0.3R), (100, 0.3R), (192, 0.3R), (1, 3.1R))
            scan = sort(C.neighbors_within(gu, idx; ball = r))
            Test.@test sort(C.neighbors_within(gu, idx; ball = r, topology = ixu, scratch = su)) == scan
            Test.@test C.nneighbors_within(gu, idx; ball = r, topology = ixu) == length(scan)
        end

        # Cost: fixed radius, growing n. The scan is linear per query; the index must not be.
        # Best of several blocks, not the mean of one: a single collection landing in the block would
        # otherwise decide the comparison, and by this point in the suite the heap is big enough for
        # that to happen. The query itself allocates nothing, so no block needs to collect.
        function percall(g, top, r, reps)
            I = (size(GD.mask(g), 1) ÷ 2, size(GD.mask(g), 2) ÷ 2)
            buf = Vector{Int}(undef, 4096)
            s = C.ball_scratch()
            C.neighbors_within!(buf, g, I...; ball = r, topology = top, scratch = s)
            best = Inf
            for _ in 1:5
                t = @elapsed for _ in 1:reps
                    C.neighbors_within!(buf, g, I...; ball = r, topology = top, scratch = s)
                end
                best = min(best, t / reps)
            end
            return best
        end
        small, big = curv(24), curv(96)                      # 576 vs 9216 cells, 16× more
        # `curv` spans the same extent at every `n`, so the radius has to shrink with the spacing to
        # hold the ball at a fixed cell count — otherwise the growth measured is more candidates rather
        # than more overhead, which is not the claim.
        cells = 2.5
        r_small, r_big = cells * 10.0 / 23, cells * 10.0 / 95
        Test.@test C.nneighbors_within(small, 12, 12; ball = r_small) ==
                   C.nneighbors_within(big, 48, 48; ball = r_big)
        ix_small = percall(small, C.indexed(small), r_small, 2000)
        ix_big = percall(big, C.indexed(big), r_big, 2000)
        sc_big = percall(big, C.MetricTopology(big), r_big, 100)
        Test.@test ix_big < 3 * ix_small                     # flat in `n`, loosely bounded
        Test.@test ix_big < sc_big                           # and cheaper than the scan it replaces
    end

    Test.@testset "Connected is the reachable part of the ball, not the ball" begin
        C = FG.Connectivity
        GD = FG.Grids
        St = FG.Stencils
        geo = FG.Geometry.CartesianGeometry{Float64}()
        function box(n; mask = trues(n, n))
            ax = collect(range(0.0, 1.0 * (n - 1); length = n))
            return GD.StructuredGrid(geo, ax, ax, mask)
        end

        # Maskless and convex: everything in the ball is reachable inside it, so the two agree.
        g = box(9)
        for I in ((5, 5), (1, 1), (9, 4))
            Test.@test sort(C.neighbors_within(g, I...; ball = 2.5, reach = C.Connected(St.Moore(1)))) ==
                       sort(C.neighbors_within(g, I...; ball = 2.5))
        end

        # A wall through the ball: strictly smaller, and a named cell is dropped.
        wall = trues(9, 9); wall[:, 5] .= false
        gm = box(9; mask = wall)
        I = (5, 3)
        u = sort(C.neighbors_within(gm, I...; ball = 3.5))
        c = sort(C.neighbors_within(gm, I...; ball = 3.5, reach = C.Connected(St.Moore(1))))
        lin = LinearIndices((9, 9))
        Test.@test lin[5, 6] in u              # inside the ball, on the far side of the wall
        Test.@test !(lin[5, 6] in c)
        Test.@test issubset(c, u) && length(c) < length(u)

        # The defining property: closed under adjacency through returned cells.
        function reaches_seed_within(grid, I, r, sten)
            got = C.neighbors_within(grid, I...; ball = r, reach = C.Connected(sten))
            sz = size(GD.mask(grid))
            li = LinearIndices(sz)
            set = Set(got); push!(set, li[I...])
            seen = Set([li[I...]]); stack = [li[I...]]
            while !isempty(stack)
                ci = Tuple(CartesianIndices(sz)[pop!(stack)])
                for δ in St.offsets(sten, Val(length(sz)))
                    J = ntuple(d -> ci[d] + δ[d], length(sz))
                    all(d -> 1 ≤ J[d] ≤ sz[d], 1:length(sz)) || continue
                    l = li[J...]
                    (l in set && !(l in seen)) || continue
                    push!(seen, l); push!(stack, l)
                end
            end
            return length(seen) == length(set)
        end
        Test.@test reaches_seed_within(gm, I, 3.5, St.Moore(1))
        Test.@test reaches_seed_within(g, (5, 5), 2.5, St.Axial(1))

        # P–Q–R: R is inside the ball but its only path there leaves it, so a pruned walk cannot find
        # it — which is exactly the difference between the two operators, not a bug in either.
        m = trues(5); m[2] = false
        chain = GD.StructuredGrid(geo, collect(0.0:4.0), m)
        Test.@test 3 in C.neighbors_within(chain, 1; ball = 2.0)
        Test.@test isempty(C.neighbors_within(chain, 1; ball = 2.0, reach = C.Connected()))

        # Every entry point agrees with every other, under Connected as under Unrestricted.
        for (grid, seed, r, sten) in ((gm, (5, 3), 3.5, St.Moore(1)), (g, (5, 5), 2.5, St.Axial(1)))
            rc = C.Connected(sten)
            n = C.nneighbors_within(grid, seed...; ball = r, reach = rc)
            l = C.neighbors_within(grid, seed...; ball = r, reach = rc)
            buf = Vector{Int}(undef, n)
            C.neighbors_within!(buf, grid, seed...; ball = r, reach = rc)
            Test.@test n == length(l)
            Test.@test buf == l
            # The distances a Connected fold sees are the ball's own.
            byfold = C.fold_within(Tuple{Int,Float64}[], grid, seed...; ball = r, reach = rc) do acc, J, d
                push!(acc, (C._linidx(size(GD.mask(grid)), J...), d))
                return acc
            end
            Test.@test [p[1] for p in byfold] == l
            ref = Dict(C.fold_within(Tuple{Int,Float64}[], grid, seed...; ball = r) do acc, J, d
                push!(acc, (C._linidx(size(GD.mask(grid)), J...), d))
                return acc
            end)
            Test.@test all(p -> ref[p[1]] == p[2], byfold)
        end

        # Curvilinear: the same, and the index does not change it.
        xs = [x for x in range(0.0, 8.0; length = 9), _ in 1:9]
        ys = [y for _ in 1:9, y in range(0.0, 8.0; length = 9)]
        cmask = trues(9, 9); cmask[:, 5] .= false
        gcv = GD.CurvilinearGrid(geo, xs, ys, cmask; measure = fill(1.0, 9, 9))
        rc = C.Connected(St.Moore(1))
        cc = sort(C.neighbors_within(gcv, 5, 3; ball = 3.5, reach = rc))
        Test.@test issubset(cc, sort(C.neighbors_within(gcv, 5, 3; ball = 3.5)))
        Test.@test length(cc) < length(C.neighbors_within(gcv, 5, 3; ball = 3.5))
        Test.@test sort(C.neighbors_within(gcv, 5, 3; ball = 3.5, reach = rc,
                                           topology = C.indexed(gcv))) == cc

        # A node set's adjacency is the one it stores, so a stencil is meaningless there and is refused
        # rather than ignored.
        gu = C.unstructured_grid(FG.SphericalSampling.HEALPixSampling(4))
        R = FG.Geometry.radius(GD.grid_geometry(gu))
        Test.@test issubset(sort(C.neighbors_within(gu, 1; ball = 0.4R, reach = C.Connected())),
                            sort(C.neighbors_within(gu, 1; ball = 0.4R)))
        Test.@test_throws ArgumentError C.neighbors_within(
            gu, 1; ball = 0.4R, reach = C.Connected(St.Axial(1)),
        )
        # And a graph query needs each cell to be one node, so image summing is refused.
        Test.@test_throws ArgumentError C.fold_within(
            (a, _, _) -> a, 0, g, 5, 5; ball = 2.5, reach = C.Connected(), images = C.AllImages(),
        )
    end

    Test.@testset "Ball connectivity builders agree with the per-cell query, on every architecture" begin
        C = FG.Connectivity
        GD = FG.Grids
        geo = FG.Geometry.CartesianGeometry{Float64}()
        xs = [x for x in range(0.0, 8.0; length = 9), _ in 1:9]
        ys = [y for _ in 1:9, y in range(0.0, 8.0; length = 9)]
        mask = trues(9, 9); mask[3, 7] = false
        gcv = GD.CurvilinearGrid(geo, xs, ys, mask; measure = fill(1.0, 9, 9))
        gu = C.unstructured_grid(FG.SphericalSampling.HEALPixSampling(4))
        R = FG.Geometry.radius(GD.grid_geometry(gu))

        for (grid, r, idxs) in ((gcv, 3.5, CartesianIndices((9, 9))),
                                (gu, 0.35R, CartesianIndices((length(GD.mask(gu)),))))
            conn = C.build_connectivity_within(grid; ball = r)
            Test.@test C.is_symmetric_adjacency(conn)
            for ci in idxs
                I = Tuple(ci)
                Test.@test sort(collect(GD.neighbors(conn, C._linidx(size(idxs), I...)))) ==
                           sort(C.neighbors_within(grid, I...; ball = r))
            end
            # The same graph whether an index was used. Row order follows whatever enumerated the
            # candidates, so the rows are compared as the sets they are.
            scanned = C.build_connectivity_within(grid; ball = r, topology = C.MetricTopology(grid))
            Test.@test scanned.ptr == conn.ptr
            for k in 1:(length(conn.ptr) - 1)
                Test.@test sort(conn.nbrs[conn.ptr[k]:(conn.ptr[k + 1] - 1)]) ==
                           sort(scanned.nbrs[scanned.ptr[k]:(scanned.ptr[k + 1] - 1)])
            end
        end
    end

    Test.@testset "Connected follows a periodic seam" begin
        C = FG.Connectivity
        geo = FG.Geometry.CartesianGeometry{Float64}()
        ring(mask) = FG.Grids.StructuredGrid(geo, collect(0.0:7.0), mask;
                                             periodic = true, period = 8.0)

        # A ring of 8 at unit spacing: a ball of 2.5 reaches ±2, and the wrap makes it contiguous, so
        # the component is the whole ball.
        g = ring(trues(8))
        Test.@test sort(C.neighbors_within(g, 1; ball = 2.5)) == [2, 3, 7, 8]
        Test.@test sort(C.neighbors_within(g, 1; ball = 2.5, reach = C.Connected())) == [2, 3, 7, 8]

        # Masking cell 8 cuts the walk one way round. Cell 7 stays in the ball — two cells away across
        # the seam — but every path to it now leaves the ball or passes through an inactive cell.
        gm = ring([true, true, true, true, true, true, true, false])
        Test.@test sort(C.neighbors_within(gm, 1; ball = 2.5)) == [2, 3, 7]
        Test.@test sort(C.neighbors_within(gm, 1; ball = 2.5, reach = C.Connected())) == [2, 3]

        # Seeded on the far side of the cut, the component runs the other way through the seam.
        Test.@test sort(C.neighbors_within(gm, 7; ball = 2.5)) == [1, 5, 6]
        Test.@test sort(C.neighbors_within(gm, 7; ball = 2.5, reach = C.Connected())) == [5, 6]
    end

    Test.@testset "A ball fold allocates nothing, and nothing that grows with the grid" begin
        C = FG.Connectivity
        GD = FG.Grids
        geo = FG.Geometry.CartesianGeometry{Float64}()
        function curv(n)
            x = [t for t in range(0.0, 1.0 * (n - 1); length = n), _ in 1:n]
            y = [t for _ in 1:n, t in range(0.0, 1.0 * (n - 1); length = n)]
            return GD.CurvilinearGrid(geo, x, y, trues(n, n); measure = fill(1.0, n, n))
        end
        gu = C.unstructured_grid(FG.SphericalSampling.HEALPixSampling(4))
        R = FG.Geometry.radius(GD.grid_geometry(gu))

        # Unindexed, the candidates are a range, so the traversal touches no heap at all.
        function nalloc_curv(g, top)
            FG.Connectivity.nneighbors_within(g, 8, 8; ball = 3.0, topology = top)
            return @allocated FG.Connectivity.nneighbors_within(g, 8, 8; ball = 3.0, topology = top)
        end
        function nalloc_node(g, top, r)
            FG.Connectivity.nneighbors_within(g, 1; ball = r, topology = top)
            return @allocated FG.Connectivity.nneighbors_within(g, 1; ball = r, topology = top)
        end
        g16 = curv(16)
        Test.@test nalloc_curv(g16, C.MetricTopology(g16)) == 0
        Test.@test nalloc_node(gu, C.MetricTopology(gu), 0.3R) == 0

        # Indexed with a buffer, nothing is left to allocate: the tree is queried in its own point type,
        # so it converts nothing per call, and the candidates land in the caller's buffer.
        function nalloc_ix(g, ix, s)
            FG.Connectivity.nneighbors_within(g, 8, 8; ball = 3.0, topology = ix, scratch = s)
            return @allocated FG.Connectivity.nneighbors_within(
                g, 8, 8; ball = 3.0, topology = ix, scratch = s,
            )
        end
        a16 = nalloc_ix(g16, C.indexed(g16), C.ball_scratch())
        a64 = nalloc_ix(curv(64), C.indexed(curv(64)), C.ball_scratch())
        Test.@test a16 == a64                  # 16× the cells, the same bytes per query
        Test.@test a16 == 0

        # The point-seeded form goes through the same index and must be free too, on both index kinds
        # and with images to deduplicate.
        gp = curv(32)
        pt = (15.5, 16.25)
        Test.@test _alloc(q_fold_at, gp, pt, 3.0, C.indexed(gp), C.ball_scratch()) == 0
        Test.@test _alloc(q_fold_at, gp, pt, 3.0,
                          C.MetricTopology(gp; index = GD.cell_list(gp; ball = 3.0)),
                          C.ball_scratch()) == 0
        Test.@test _alloc(q_fold_at, gu, (0.4, 0.1), 0.3R, C.indexed(gu), C.ball_scratch()) == 0

        # Locating a point is the same traversal, so it carries the same guarantee: per axis on a
        # rectilinear grid, and through the index elsewhere.
        Test.@test _alloc(q_locate, GD.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(),
                                                      collect(0.0:31.0), collect(0.0:31.0)),
                          (7.3, 4.2)) == 0
        Test.@test _alloc(q_locate_top, gp, pt, C.indexed(gp), C.ball_scratch()) == 0
        Test.@test _alloc(q_locate_top, gu, (0.4, 0.1), C.indexed(gu), C.ball_scratch()) == 0
    end

    Test.@testset "Adjacency symmetry is decided by comparison with the transpose" begin
        C = FG.Connectivity
        geo = FG.Geometry.CartesianGeometry{Float64}()
        xs = [x for x in range(0.0, 15.0; length = 16), _ in 1:16]
        ys = [y for _ in 1:16, y in range(0.0, 15.0; length = 16)]
        g = FG.Grids.CurvilinearGrid(geo, xs, ys, trues(16, 16); measure = fill(1.0, 16, 16))
        gs = FG.Grids.StructuredGrid(geo, collect(0.0:15.0), collect(0.0:15.0))

        # Symmetric graphs are accepted at any degree, which is the case a ball graph exercises.
        Test.@test C.is_symmetric_adjacency(C.build_connectivity(gs))
        for r in (2.0, 6.0)
            Test.@test C.is_symmetric_adjacency(C.build_connectivity_within(g; ball = r))
        end

        base = C.build_connectivity_within(g; ball = 2.0)
        # A one-way edge: overwrite `i` in row `j` with another of row `j`'s own neighbours, so the row
        # keeps its length and only the reciprocity is broken.
        oneway = C.csr_connectivity(copy(base.nbrs), copy(base.ptr); validate = false)
        i = 1
        j = oneway.nbrs[oneway.ptr[i]]
        slot = findfirst(q -> oneway.nbrs[q] == i, oneway.ptr[j]:(oneway.ptr[j + 1] - 1))
        oneway.nbrs[oneway.ptr[j] + slot - 1] =
            oneway.nbrs[oneway.ptr[j]] == i ? oneway.nbrs[oneway.ptr[j] + 1] : oneway.nbrs[oneway.ptr[j]]
        Test.@test !C.is_symmetric_adjacency(oneway)

        # An out-of-range neighbour is rejected, not used as an index.
        oob = C.csr_connectivity(copy(base.nbrs), copy(base.ptr); validate = false)
        oob.nbrs[1] = 10^6
        Test.@test !C.is_symmetric_adjacency(oob)
        Test.@test C.is_symmetric_adjacency(C.csr_connectivity(Int[], Int[1]; validate = false))
    end

    Test.@testset "A curvilinear measure may be given positionally or by keyword" begin
        # The docstring offers both forms, so both must build the same grid.
        geo = FG.Geometry.CartesianGeometry{Float64}()
        xs = [x for x in range(0.0, 3.0; length = 4), _ in 1:4]
        ys = [y for _ in 1:4, y in range(0.0, 3.0; length = 4)]
        a = fill(2.5, 4, 4)
        bykw = FG.Grids.CurvilinearGrid(geo, xs, ys, trues(4, 4); measure = a)
        bypos = FG.Grids.CurvilinearGrid(geo, xs, ys, a, trues(4, 4))
        Test.@test FG.Grids.measure(bykw) == FG.Grids.measure(bypos) == a
    end


    Test.@testset "Index-parallel loops run as kernels and give the same answer" begin
        D = FG.Discretization
        C = FG.Connectivity
        GD = FG.Grids
        cpu = KernelAbstractions.CPU()

        n, m = 48, 33
        x = collect(range(0.0, 2π; length = n))
        y = collect(range(0.0, 1.0; length = m))
        f = [sin(xi) * cos(yj) for xi in x, yj in y]
        ref = similar(f); dev = similar(f)
        for (ax, dim, ord) in ((x, 1, 1), (y, 2, 2))
            D.apply_stencil!(ref, f, ax, dim; order = ord, nodes = 5)
            D.apply_stencil!(dev, f, ax, dim; order = ord, nodes = 5, backend = cpu)
            Test.@test ref == dev
        end
        msk = trues(n, m); msk[10:14, :] .= false
        D.apply_stencil!(ref, f, x, 1; order = 1, nodes = 5, mask = msk)
        D.apply_stencil!(dev, f, x, 1; order = 1, nodes = 5, mask = msk, backend = cpu)
        Test.@test ref == dev

        cart = FG.Geometry.CartesianGeometry{Float64}()
        for g in (GD.StructuredGrid(cart, collect(0.0:31.0), collect(0.0:31.0)),
                  GD.StructuredGrid(cart, collect(0.0:31.0), collect(0.0:31.0);
                                    periodic = (true, false), period = (32.0, 0.0)))
            for sten in (FG.Stencils.Axial(1), FG.Stencils.Moore(2))
                a = C.build_connectivity(g; stencil = sten)
                b = C.build_connectivity(g; stencil = sten, backend = cpu)
                Test.@test a.ptr == b.ptr
                Test.@test a.nbrs == b.nbrs
            end
        end

        acc = zeros(Int, 100)
        FG.Execution.run_indices(i -> (acc[i] = i * i), 100, cpu)
        Test.@test acc == [i * i for i in 1:100]

        # A ball query is device-shaped once the topology carries no index: it reads coordinates and the
        # mask, and allocates nothing, so it runs inside the launch.
        gball = GD.StructuredGrid(cart, collect(0.0:31.0), collect(0.0:31.0))
        function counts(g, r, backend)
            sz = GD.size_tuple(g)
            out = zeros(Int, sz)
            mt = C.MetricTopology(g)
            ci = CartesianIndices(sz)
            FG.Execution.run_indices(length(ci), backend) do lin
                I = Tuple(@inbounds ci[lin])
                @inbounds out[lin] = C.nneighbors_within(g, I...; ball = r, topology = mt)
            end
            return out
        end
        Test.@test counts(gball, 3.0, cpu) == counts(gball, 3.0, nothing)
        Test.@test counts(gball, 3.0, cpu) ==
                   [C.nneighbors_within(gball, Tuple(ci)...; ball = 3.0)
                    for ci in CartesianIndices(size(GD.mask(gball)))]
        Test.@test isbits(C.MetricTopology(gball))

        # The sweep follows the same rule: unindexed it is one body per cell, so it launches.
        function sweep_counts(g, r, backend)
            out = zeros(Int, GD.size_tuple(g))
            C.foreach_within(g; ball = r, topology = C.MetricTopology(g), backend = backend) do I, J, d
                @inbounds out[I[1], I[2]] += 1
            end
            return out
        end
        Test.@test sweep_counts(gball, 3.0, cpu) == sweep_counts(gball, 3.0, nothing)
        Test.@test sum(sweep_counts(gball, 3.0, cpu)) ==
                   C.mapreduce_within((I, J, d) -> 1, +, 0, gball; ball = 3.0)

        # A chunked body accumulates across its range, so a device backend refuses it rather than
        # quietly running on the host.
        Test.@test_throws ArgumentError FG.Execution.run_chunks(r -> nothing, 4, cpu)
        Test.@test_throws ArgumentError FG.Execution.map_chunks(r -> 1, 4, cpu)

        # An unindexed topology is device-safe; one holding a k-d tree is refused, not silently dropped.
        gs = GD.StructuredGrid(cart, collect(0.0:7.0), collect(0.0:7.0))
        Test.@test Adapt.adapt(Array, C.MetricTopology(gs)) === C.MetricTopology(gs)
        gu = C.unstructured_grid(FG.SphericalSampling.HEALPixSampling(2))
        Test.@test_throws ArgumentError Adapt.adapt(Array, C.indexed(gu))
        Test.@test !isbits(C.indexed(gu))

        # An indexed sweep does stay on the host: each task needs its own candidate buffer, which a
        # launch has nowhere to put.
        Test.@test_throws ArgumentError C.foreach_within(
            (I, J, d) -> nothing, gu; ball = 2.0e6, topology = C.indexed(gu), backend = cpu,
        )
    end

    Test.@testset "k nearest is exact under the geometry's own metric" begin
        C = FG.Connectivity
        GD = FG.Grids
        GE = FG.Geometry
        cart = GE.CartesianGeometry{Float64}()
        sph = GE.SphericalGeometry(6.371e6)

        # Every candidate ranked by (distance, index), which is what the query promises.
        function brute(grid, I, k)
            sz = size(GD.mask(grid))
            node = grid isa GD.UnstructuredGrid
            p0 = node ? GD._raw_coords(grid, I[1]) : GD._raw_coords(grid, I...)
            geo = GD.grid_geometry(grid)
            prd = ntuple(d -> GD.isperiodic(grid, d) ? GD.period(grid, d) : 0.0, length(sz))
            pairs = Tuple{Float64,Int}[]
            for (lin, ci) in enumerate(CartesianIndices(sz))
                J = Tuple(ci)
                (node ? lin == I[1] : J == Tuple(I)) && continue
                (node ? GD.isactive(grid, lin) : GD.isactive(grid, J...)) || continue
                pt = node ? GD._raw_coords(grid, lin) : GD._raw_coords(grid, J...)
                q = ntuple(length(pt)) do d
                    p = d ≤ length(prd) ? prd[d] : 0.0
                    p > 0 ? p0[d] + (pt[d] - p0[d] - p * round((pt[d] - p0[d]) / p)) : pt[d]
                end
                push!(pairs, (GE.distance(geo, p0, q), lin))
            end
            sort!(pairs)
            m = min(k, length(pairs))
            return [p[2] for p in pairs[1:m]], [p[1] for p in pairs[1:m]]
        end

        n = 16
        v = collect(0.0:1.0:(n - 1.0)); rg = 0.0:1.0:(n - 1.0)
        cx = [x for x in 0.0:1.0:(n - 1.0), _ in 1:n]
        cy = [y for _ in 1:n, y in 0.0:1.0:(n - 1.0)]
        cases = (
            (GD.StructuredGrid(cart, rg, v), (8, 8)),
            (GD.StructuredGrid(cart, v), (8,)),
            (GD.StructuredGrid(cart, v, v; periodic = (true, false), period = (16.0, 0.0)), (1, 8)),
            (GD.StructuredGrid(sph, collect(range(0, 2π * (1 - 1 / n); length = n)),
                               collect(range(-1.2, 1.2; length = n))), (4, 8)),
            (GD.CurvilinearGrid(cart, cx, cy, trues(n, n); measure = fill(1.0, n, n)), (8, 8)),
        )
        for (g, I) in cases, k in (1, 5, 12)
            gi, gd = C.k_nearest(g, I...; k = k)
            wi, wd = brute(g, I, k)
            Test.@test gi == wi
            Test.@test gd ≈ wd
            Test.@test issorted(gd)
        end

        gu = C.unstructured_grid(FG.SphericalSampling.HEALPixSampling(4))
        for k in (1, 6, 20)
            gi, gd = C.k_nearest(gu, 1; k = k)
            wi, wd = brute(gu, (1,), k)
            Test.@test gi == wi
            Test.@test gd ≈ wd
        end
        # An index changes the cost, never the ranking.
        Test.@test C.k_nearest(gu, 100; k = 8) == C.k_nearest(gu, 100; k = 8, topology = C.indexed(gu))

        # Equal distances resolve by linear index, so the answer does not depend on visit order.
        gt = GD.StructuredGrid(cart, collect(0.0:6.0), collect(0.0:6.0))
        ti, td = C.k_nearest(gt, 4, 4; k = 4)
        Test.@test all(td .≈ 1.0)
        Test.@test issorted(ti)
        Test.@test ti == C.k_nearest(gt, 4, 4; k = 4, topology = C.MetricTopology(gt))[1]

        # A Cartesian node set indexes by one integer while carrying two coordinate directions, so the
        # radius the search may widen to has to come from the coordinates, not from the index space.
        # Tall and narrow, where taking only the first direction would stop the search far too early.
        nn = 200
        gcn = GD.UnstructuredGrid(cart, (collect(range(0.0, 1.0; length = nn)),
                                         collect(range(0.0, 500.0; length = nn))),
                                  trues(nn); k = 4, areas = fill(1.0, nn))
        p0 = GD._raw_coords(gcn, 1)
        for k in (1, 5, 30)
            _, dst = C.k_nearest(gcn, 1; k = k)
            want = sort([GE.distance(cart, p0, GD._raw_coords(gcn, j)) for j in 2:nn])[1:k]
            Test.@test dst ≈ want
        end

        # Asking for more than exists returns what exists, and k = 0 returns nothing.
        Test.@test length(C.k_nearest(GD.StructuredGrid(cart, [0.0, 1.0, 2.0]), 2; k = 99)[1]) == 2
        Test.@test isempty(C.k_nearest(gt, 4, 4; k = 0)[1])
    end

    Test.@testset "Per-cell entry points allocate nothing, on every grid shape" begin
        cart = FG.Geometry.CartesianGeometry{Float64}()
        sph = FG.Geometry.SphericalGeometry(6.371e6)
        n = 24
        v = collect(0.0:1.0:(n - 1.0))
        rg = 0.0:1.0:(n - 1.0)
        cx = [x for x in 0.0:1.0:(n - 1.0), _ in 1:n]
        cy = [y for _ in 1:n, y in 0.0:1.0:(n - 1.0)]

        # A range in one direction and a vector in another makes the coordinate tuple heterogeneous. That
        # shape was absent from the suite, and every coordinate read on it used to allocate.
        check_shape("structured 1-D vector",  FG.Grids.StructuredGrid(cart, v), (12,))
        check_shape("structured 1-D range",   FG.Grids.StructuredGrid(cart, rg), (12,))
        check_shape("structured 2-D vectors", FG.Grids.StructuredGrid(cart, v, v), (12, 12))
        check_shape("structured 2-D ranges",  FG.Grids.StructuredGrid(cart, rg, rg), (12, 12))
        check_shape("structured 2-D MIXED",   FG.Grids.StructuredGrid(cart, rg, v), (12, 12))
        check_shape("structured 2-D MIXED'",  FG.Grids.StructuredGrid(cart, v, rg), (12, 12))
        check_shape("structured 3-D MIXED",   FG.Grids.StructuredGrid(cart, rg, v, rg), (12, 12, 12))
        check_shape("spherical 2-D MIXED",    FG.Grids.StructuredGrid(sph,
                        range(0, 2π * (1 - 1 / n); length = n),
                        collect(range(-1.2, 1.2; length = n))), (12, 12))
        check_shape("curvilinear 2-D", FG.Grids.CurvilinearGrid(cart, cx, cy, trues(n, n);
                        measure = fill(1.0, n, n)), (12, 12))

        # `apply_stencil!` drives the field loop through the same index-parallel entry point the device
        # path uses, so it is gated here rather than trusted.
        for (nx, ny, dim) in ((9, 5, 1), (9, 5, 2), (17, 3, 1))
            ax = collect(0.0:1.0:((dim == 1 ? nx : ny) - 1.0))
            fld = [xi + 2yj for xi in 0.0:1.0:(nx - 1.0), yj in 0.0:1.0:(ny - 1.0)]
            o = similar(fld)
            si, sw = FG.Discretization.axis_stencils(ax, 1, 3)
            msk = trues(nx, ny); msk[2, :] .= false
            a1 = _alloc(q_stencil!, o, fld, si, sw, dim)
            a2 = _alloc(q_stencil_m!, o, fld, si, sw, dim, msk)
            a1 == 0 || println("    apply_stencil!(dim=", dim, ") -> ", a1, " B")
            a2 == 0 || println("    apply_stencil! masked(dim=", dim, ") -> ", a2, " B")
            Test.@test a1 == 0
            Test.@test a2 == 0
        end

        # The radius conversion sits on the per-query path, and dispatches on the embedding type rather
        # than branching on a stored tag, so it must be free.
        for emb in (FG.Grids.CartesianEmbedding(), FG.Grids.ChordEmbedding(),
                    FG.Grids.ArcEmbedding(6.371e6))
            a = _alloc(FG.Grids.embedded_radius, emb, 1.0e6)
            a == 0 || println("    embedded_radius(", typeof(emb), ") -> ", a, " B")
            Test.@test a == 0
            Test.@test (Test.@inferred FG.Grids.embedded_radius(emb, 1.0e6)) isa Float64
        end
        # An arc past the antipode saturates at the diameter instead of turning back down.
        Test.@test FG.Grids.embedded_radius(FG.Grids.ArcEmbedding(2.0), 100.0) == 4.0
        Test.@test FG.Grids.embedded_radius(FG.Grids.ArcEmbedding(2.0), 2.0 * π) == 4.0
        Test.@test FG.Grids.embedded_radius(FG.Grids.CartesianEmbedding(), 3.0) == 3.0

        # Axis- and geometry-level primitives that sit inside per-cell work. They take an axis or a
        # geometry rather than a grid, so they are checked here rather than in the shape matrix.
        let ax = collect(range(0.0, 1.0; length = 64)), rg = range(0.0, 1.0; length = 64),
            nd = [0.0, 0.7, 1.9, 3.1, 4.0], wv = Vector{Float64}(undef, 5),
            cv = Matrix{Float64}(undef, 5, 3),
            sph = FG.Geometry.SphericalGeometry(6.371e6)
            for (name, a) in (
                ("fd_weights!",          _alloc(FG.Discretization.fd_weights!, wv, cv, nd, 2.0, 2)),
                ("nearest_index/vector", _alloc(FG.Discretization.nearest_index, ax, 0.37)),
                ("nearest_index/range",  _alloc(FG.Discretization.nearest_index, rg, 0.37)),
                ("locate/vector",        _alloc(FG.Discretization.locate, ax, 0.37)),
                ("locate/range",         _alloc(FG.Discretization.locate, rg, 0.37)),
                ("interpolation_weights", _alloc(FG.Discretization.interpolation_weights, ax, 0.37)),
                ("scale_factors",        _alloc(FG.Discretization.scale_factors, sph, (0.3, 0.4))),
                ("jacobian",             _alloc(FG.Discretization.jacobian, sph, (0.3, 0.4))),
            )
                a == 0 || println("    ", name, " -> ", a, " B")
                Test.@test a == 0
            end
        end

        # Same answers, not merely the same speed.
        gm = FG.Grids.StructuredGrid(cart, rg, v)
        gh = FG.Grids.StructuredGrid(cart, v, v)
        for I in ((1, 1), (12, 12), (24, 24), (1, 24))
            Test.@test FG.Grids.coords(gm, I...) == FG.Grids.coords(gh, I...)
            Test.@test FG.Connectivity.neighbors_within(gm, I...; ball = 2.5) ==
                       FG.Connectivity.neighbors_within(gh, I...; ball = 2.5)
        end
    end

    Test.@testset "A sweep's allocation does not grow with the grid, beyond the index's own" begin
        C = FG.Connectivity
        GD = FG.Grids
        cart = FG.Geometry.CartesianGeometry{Float64}()
        function curv(n)
            x = [t for t in range(0.0, 1.0 * (n - 1); length = n), _ in 1:n]
            y = [t for _ in 1:n, t in range(0.0, 1.0 * (n - 1); length = n)]
            return GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
        end
        # Through the const global, not the testset-local alias: a captured `Module` makes the call a
        # dynamic lookup and adds a fixed 144 bytes that has nothing to do with the sweep.
        sweep(g, r, top) =
            FG.Connectivity.mapreduce_within((I, J, d) -> 1, +, 0, g; ball = r, topology = top)
        touch(g, r, top, out) =
            FG.Connectivity.foreach_within(g; ball = r, topology = top) do I, J, d
                @inbounds out[I[1], I[2]] += 1
            end
        one_query(buf, tree, v, r) =
            NearestNeighbors.inrange!(empty!(buf), tree, v, r, false)

        for n in (24, 48, 96)
            g = curv(n)
            scan = C.MetricTopology(g)
            out = zeros(Int, n, n)
            sweep(g, 2.5, scan); touch(g, 2.5, scan, out)
            # Without an index the whole sweep touches no heap at all, whatever the cell count: the
            # scratch and the topology are per-sweep, and the traversal itself is allocation-free.
            Test.@test _alloc(sweep, g, 2.5, scan) == 0
            Test.@test _alloc(touch, g, 2.5, scan, out) == 0
        end

        # With an index the only growth is one `NearestNeighbors.inrange!` per cell, which allocates
        # inside its own setup. Asserted against that measured constant, so the sweep cannot start
        # allocating anything of its own without this failing.
        per_query = let g = curv(48), ix = C.indexed(g)
            buf = Int[]
            v = view(ix.index.pts, :, 100)
            one_query(buf, ix.index.tree, v, 2.5)
            _alloc(one_query, buf, ix.index.tree, v, 2.5)
        end
        for n in (24, 48)
            g = curv(n)
            ix = C.indexed(g)
            sweep(g, 2.5, ix)
            Test.@test _alloc(sweep, g, 2.5, ix) ≤ (per_query + 8) * n^2
        end
    end

    Test.@testset "A stencil at a mask edge can degrade instead of blanking" begin
        D = FG.Discretization
        GD = FG.Grids
        geo = FG.Geometry.CartesianGeometry()
        x = collect(0.0:1.0:6.0)
        msk = trues(7, 1); msk[4, 1] = false
        grid = GD.StructuredGrid(geo, x, [0.0], msk)
        nomask = GD.StructuredGrid(geo, x, [0.0])
        f = reshape(collect(0.0:6.0), 7, 1)          # f = x, so df/dx == 1 wherever it is defined
        active = [i for i in 1:7 if msk[i, 1]]

        # Blanking loses every active cell within `nodes - 1` of the mask; at five nodes it loses the
        # whole axis. That is what the other policies exist to avoid.
        for nodes in (2, 3, 5)
            o = zeros(7, 1)
            D.apply_stencil!(o, f, grid, 1; order = 1, nodes = nodes)
            Test.@test iszero(o[3, 1])               # active, blanked by its masked neighbour
            nodes == 5 && Test.@test all(iszero, o)
        end

        # Runs here are [1,3] and [5,7], so five nodes do not fit and only ReduceInRun can fill them.
        for nodes in (2, 3)
            o = zeros(7, 1)
            D.apply_stencil!(o, f, grid, 1; order = 1, nodes = nodes, policy = D.ShiftWithinRun())
            Test.@test all(isapprox.(o[active, 1], 1.0))
            Test.@test iszero(o[4, 1])
        end
        for nodes in (2, 3, 5)
            o = zeros(7, 1)
            D.apply_stencil!(o, f, grid, 1; order = 1, nodes = nodes, policy = D.ReduceInRun())
            Test.@test all(isapprox.(o[active, 1], 1.0))
            Test.@test iszero(o[4, 1])
        end

        # Nothing anyone gets today changes: no mask, or the default policy, is bit-identical.
        for nodes in (2, 3, 5), pol in (D.BlankMasked(), D.ShiftWithinRun(), D.ReduceInRun())
            a = zeros(7, 1); b = zeros(7, 1)
            D.apply_stencil!(a, f, nomask, 1; order = 1, nodes = nodes)
            D.apply_stencil!(b, f, nomask, 1; order = 1, nodes = nodes, policy = pol)
            Test.@test a == b
        end
        for nodes in (2, 3, 5)
            a = zeros(7, 1); b = zeros(7, 1)
            D.apply_stencil!(a, f, grid, 1; order = 1, nodes = nodes)
            D.apply_stencil!(b, f, grid, 1; order = 1, nodes = nodes, policy = D.BlankMasked())
            Test.@test a == b
        end

        # In the interior of a run the weights are the unmasked ones, so the result is bit-for-bit.
        n = 60
        xl = collect(range(0.0, 1.0; length = n))
        m2 = trues(n, 1); m2[30, 1] = false
        gl = GD.StructuredGrid(geo, xl, [0.0], m2)
        gn = GD.StructuredGrid(geo, xl, [0.0])
        fl = reshape([sin(3xi) for xi in xl], n, 1)
        a = zeros(n, 1); b = zeros(n, 1)
        D.apply_stencil!(a, fl, gn, 1; order = 1, nodes = 5)
        D.apply_stencil!(b, fl, gl, 1; order = 1, nodes = 5, policy = D.ShiftWithinRun())
        for i in 1:n
            abs(i - 30) > 4 && 2 < i < n - 1 && Test.@test a[i, 1] === b[i, 1]
        end
        Test.@test any(a[i, 1] != b[i, 1] for i in 26:34 if i != 30)

        # And the accuracy order survives, which is the reason to shift rather than clip.
        errs = Float64[]
        for m in (80, 160, 320)
            xx = collect(range(0.0, 1.0; length = m))
            mm = trues(m, 1); mm[m ÷ 2, 1] = false
            gg = GD.StructuredGrid(geo, xx, [0.0], mm)
            ff = reshape([sin(3xi) for xi in xx], m, 1)
            oo = zeros(m, 1)
            D.apply_stencil!(oo, ff, gg, 1; order = 1, nodes = 5, policy = D.ShiftWithinRun())
            j = m ÷ 2 - 1                            # the cell against the mask edge
            push!(errs, abs(oo[j, 1] - 3cos(3xx[j])))
        end
        Test.@test all(log2(errs[i] / errs[i + 1]) > 3.5 for i in 1:2)

        # A run that wraps the seam. With no hole every policy is the centred periodic stencil exactly;
        # with one, the wrapped run still converges at the scheme's rate.
        np = 16; L = 16.0
        xp = collect(range(0.0, 15.0; length = np))
        fp = reshape([sin(2π * xi / L) for xi in xp], np, 1)
        gfull = GD.StructuredGrid(geo, xp, [0.0]; periodic = (true, false), period = (L, 0.0))
        pa = zeros(np, 1); pb = zeros(np, 1)
        D.apply_stencil!(pa, fp, gfull, 1; order = 1, nodes = 5)
        D.apply_stencil!(pb, fp, gfull, 1; order = 1, nodes = 5, policy = D.ShiftWithinRun())
        Test.@test pa == pb
        seam = Float64[]
        for m in (64, 128, 256)
            xx = collect(range(0.0, m - 1.0; length = m)); Lm = Float64(m)
            mm = trues(m, 1); mm[m ÷ 2, 1] = false
            gg = GD.StructuredGrid(geo, xx, [0.0], mm; periodic = (true, false), period = (Lm, 0.0))
            ff = reshape([sin(2π * xi / Lm) for xi in xx], m, 1)
            oo = zeros(m, 1)
            D.apply_stencil!(oo, ff, gg, 1; order = 1, nodes = 5, policy = D.ShiftWithinRun())
            Test.@test oo[1, 1] != 0                 # the seam cell is reached through the wrap
            push!(seam, abs(oo[1, 1] - 2π / Lm * cos(2π * xx[1] / Lm)))
        end
        Test.@test all(log2(seam[i] / seam[i + 1]) > 3.5 for i in 1:2)

        # Convergence fixes the order but not the node set, and the wrapped window is where the
        # `mod1`/`fld` unwrapping could be subtly wrong. So the weights themselves are checked against
        # first-derivative weights obtained by differentiating the Lagrange basis — a different
        # derivation from the Fornberg recursion under test.
        function lagrange_dw(nodes, z)
            m = length(nodes)
            return [sum(b == a ? 0.0 :
                        prod((c == a || c == b) ? 1.0 : (z - nodes[c]) / (nodes[a] - nodes[c])
                             for c in 1:m) / (nodes[a] - nodes[b]) for b in 1:m) for a in 1:m]
        end
        let np2 = 16, L2 = 16.0, k2 = 5
            x2 = collect(range(0.0, 15.0; length = np2))
            m2 = trues(np2, 1); m2[8, 1] = false
            g2 = GD.StructuredGrid(geo, x2, [0.0], m2; periodic = (true, false), period = (L2, 0.0))
            f2 = reshape([sin(2π * xi / L2) for xi in x2], np2, 1)
            o2 = zeros(np2, 1)
            D.apply_stencil!(o2, f2, g2, 1; order = 1, nodes = k2, policy = D.ShiftWithinRun())
            for i in 1:np2
                m2[i, 1] || continue
                back = 0
                while back < k2 && m2[mod1(i - back - 1, np2), 1]
                    back += 1
                end
                fwd = 0
                while fwd < k2 && m2[mod1(i + fwd + 1, np2), 1] && back + fwd + 1 < np2
                    fwd += 1
                end
                s = clamp(back - (k2 - 1) ÷ 2, 0, back + fwd + 1 - k2)
                raws = [(i - back) + s + q - 1 for q in 1:k2]
                # Unwrapped across the seam, so the spacing the weights see is the true one.
                nds = [x2[mod1(r, np2)] + fld(r - 1, np2) * L2 for r in raws]
                vals = [f2[mod1(r, np2), 1] for r in raws]
                Test.@test o2[i, 1] ≈ sum(lagrange_dw(nds, x2[i]) .* vals) rtol = 1e-12
                Test.@test length(unique(map(r -> mod1(r, np2), raws))) == k2   # no sample reused
                Test.@test all(m2[mod1(r, np2), 1] for r in raws)               # and all active
            end
        end

        # A run too short for the nodes: blanked by one policy, reduced by the other.
        ms = trues(9, 1); ms[4, 1] = false; ms[8, 1] = false        # runs of 3, 3 and 1
        gs = GD.StructuredGrid(geo, collect(0.0:8.0), [0.0], ms)
        fs = reshape(collect(0.0:8.0), 9, 1)
        os = zeros(9, 1); orr = zeros(9, 1)
        D.apply_stencil!(os, fs, gs, 1; order = 1, nodes = 5, policy = D.ShiftWithinRun())
        D.apply_stencil!(orr, fs, gs, 1; order = 1, nodes = 5, policy = D.ReduceInRun())
        Test.@test all(iszero, os)
        Test.@test all(isapprox.(orr[[1, 2, 3, 5, 6, 7], 1], 1.0))
        Test.@test iszero(orr[9, 1])                 # a run of one holds no first derivative

        # The matrix form has no axis to rebuild from, and says so rather than ignoring the policy.
        idx, w = D.axis_stencils(x, 1, 3)
        Test.@test_throws ArgumentError D.apply_stencil!(zeros(7, 1), f, idx, w, 1;
                                                         mask = msk, policy = D.ShiftWithinRun())

        # In-place weights match the allocating form and carry no state between calls.
        nd = [0.0, 0.7, 1.9, 3.1, 4.0]
        for ord in (0, 1, 2, 3)
            want = D.fd_weights(nd, 2.0, ord)
            wv = similar(want); cv = Matrix{Float64}(undef, length(nd), ord + 1)
            D.fd_weights!(wv, cv, nd, 2.0, ord)
            Test.@test wv == want
            D.fd_weights!(wv, cv, [0.0, 1.0, 2.0, 3.0, 4.0], 2.0, ord)
            D.fd_weights!(wv, cv, nd, 2.0, ord)
            Test.@test wv == want
        end
    end

    Test.@testset "Queries can be seeded by a point, not just a cell" begin
        C = FG.Connectivity
        GD = FG.Grids
        GE = FG.Geometry
        cart = GE.CartesianGeometry{Float64}()
        sph = GE.SphericalGeometry(6.371e6)

        function brute(grid, p, r)
            sz = size(GD.mask(grid)); node = grid isa GD.UnstructuredGrid
            geo = GD.grid_geometry(grid); D = length(GD.coordinates(grid))
            prd = ntuple(d -> GD.isperiodic(grid, d) ? GD.period(grid, d) : 0.0, D)
            out = Int[]
            for (lin, ci) in enumerate(CartesianIndices(sz))
                J = Tuple(ci)
                (node ? GD.isactive(grid, lin) : GD.isactive(grid, J...)) || continue
                pt = node ? GD._raw_coords(grid, lin) : GD._raw_coords(grid, J...)
                q = ntuple(D) do d
                    L = prd[d]
                    L > 0 ? p[d] + (pt[d] - p[d] - L * round((pt[d] - p[d]) / L)) : pt[d]
                end
                GE.distance(geo, p, q) ≤ r && push!(out, lin)
            end
            return sort(out)
        end

        n = 16
        v = collect(0.0:1.0:(n - 1.0))
        cx = [xx for xx in v, _ in 1:n]; cy = [yy for _ in 1:n, yy in v]
        λ = [l for l in range(0.0, 2π * (1 - 1 / n); length = n), _ in 1:n]
        φ = [fp for _ in 1:n, fp in range(-1.2, 1.2; length = n)]
        R = 6.371e6
        cases = (
            (GD.StructuredGrid(cart, v, v), ((7.3, 4.2), (0.0, 0.0), (15.0, 15.0)), 3.0),
            (GD.StructuredGrid(cart, v, v; periodic = (true, false), period = (16.0, 0.0)),
             ((0.3, 4.2), (15.7, 8.0)), 3.0),
            (GD.CurvilinearGrid(cart, cx, cy, trues(n, n); measure = fill(1.0, n, n)),
             ((7.3, 4.2), (1.5, 13.5)), 3.0),
            (GD.CurvilinearGrid(sph, λ, φ, trues(n, n)), ((0.4, 0.1), (6.0, -1.0)), 0.25R),
        )
        for (g, pts, r) in cases, p in pts
            want = brute(g, p, r)
            for top in (C.MetricTopology(g), C.MetricTopology(g; index = GD.cell_list(g; ball = r)))
                Test.@test sort(C.neighbors_within(g, p; ball = r, topology = top)) == want
            end
            Test.@test C.nneighbors_within(g, p; ball = r) == length(want)
        end

        gu = C.unstructured_grid(FG.SphericalSampling.HEALPixSampling(4))
        for i in (1, 100)
            base = GD._raw_coords(gu, i)
            p = (base[1] + 0.01, base[2] + 0.01)
            want = brute(gu, p, 0.3R)
            for top in (C.MetricTopology(gu), C.indexed(gu),
                        C.MetricTopology(gu; index = GD.cell_list(gu; ball = 0.3R)))
                Test.@test sort(C.neighbors_within(gu, p; ball = 0.3R, topology = top)) == want
            end
        end

        # A point AT a cell centre is the cell query plus the cell itself, which the cell form excludes.
        for g in (GD.StructuredGrid(cart, v, v),
                  GD.CurvilinearGrid(cart, cx, cy, trues(n, n); measure = fill(1.0, n, n)))
            I = (7, 5)
            p = GD._raw_coords(g, I...)
            cell = sort(C.neighbors_within(g, I...; ball = 3.0))
            Test.@test sort(C.neighbors_within(g, p; ball = 3.0)) ==
                       sort(vcat(cell, C._linidx(GD.size_tuple(g), I...)))
        end

        gs = GD.StructuredGrid(cart, v, v)
        gp = GD.StructuredGrid(cart, v, v; periodic = (true, false), period = (16.0, 0.0))
        Test.@test GD.locate(gs, (7.3, 4.2)) == (8, 5)
        Test.@test GD.locate(gs, (0.0, 0.0)) == (1, 1)
        Test.@test GD.locate(gp, (16.4, 3.0)) == GD.locate(gp, (0.4, 3.0))   # wraps
        Test.@test GD.locate(gp, (15.7, 3.0)) == (1, 4)                      # the seam cell
        Test.@test GD.locate(gs, (-5.0, 3.0))[1] == 0                        # outside is 0

        # k nearest to a point, against a full ranking
        for g in (gs, GD.CurvilinearGrid(cart, cx, cy, trues(n, n); measure = fill(1.0, n, n)))
            p = (7.3, 4.2)
            idx, dst = C.k_nearest(g, p; k = 6)
            all_cells = brute(g, p, 1000.0)
            ranked = sort([(GE.distance(cart, p, GD._raw_coords(g,
                            Tuple(CartesianIndices(size(GD.mask(g)))[l])...)), l) for l in all_cells])
            Test.@test dst ≈ [d for (d, _) in ranked[1:6]]
            Test.@test issorted(dst)
        end

        # Off a rectilinear grid there is no axis to bracket along, so `locate` is the nearest centre.
        # Every route to it — scan, tree, cell list — must name the same cell as an exhaustive search.
        function nearest_centre(g, p)
            sz = g isa GD.UnstructuredGrid ? nothing : GD.size_tuple(g)
            geo = GD.grid_geometry(g)
            best, bd = 0, Inf
            for l in 1:length(GD.mask(g))
                J = sz === nothing ? l : Tuple(CartesianIndices(sz)[l])
                d = GE.distance(geo, p, sz === nothing ? GD.coords(g, l) : GD.coords(g, J...))
                (d < bd || (d == bd && l < best)) && (bd = d; best = l)
            end
            return sz === nothing ? best : Tuple(CartesianIndices(sz)[best])
        end
        cgl = GD.CurvilinearGrid(cart, cx, cy, trues(n, n); measure = fill(1.0, n, n))
        sph = FG.Geometry.SphericalGeometry(6.371e6)
        nu = 200
        ugl = GD.UnstructuredGrid(sph, (range(0.0, 6.0; length = nu) |> collect,
                                        range(-1.2, 1.2; length = nu) |> collect),
                                  trues(nu); k = 6, areas = ones(nu))
        for (g, ps, bin) in ((cgl, ((7.3, 4.2), (0.0, 0.0), (-3.0, 20.0)), 3.0),
                             (ugl, ((0.4, 0.1), (3.0, -1.0), (6.2, 1.3)), 0.1 * 6.371e6))
            tops = (C.MetricTopology(g), C.indexed(g),
                    C.MetricTopology(g; index = GD.cell_list(g; ball = bin)))
            for p in ps
                want = nearest_centre(g, p)
                for top in tops
                    Test.@test GD.locate(g, p; topology = top, scratch = C.ball_scratch()) == want
                end
            end
        end
        # A masked cell still has a location; `active_only` is what excludes it.
        mk = trues(n, n); mk[6, 6] = false
        cgm = GD.CurvilinearGrid(cart, cx, cy, mk; measure = fill(1.0, n, n))
        Test.@test GD.locate(cgm, GD._raw_coords(cgm, 6, 6)) == (6, 6)
        Test.@test GD.locate(cgm, GD._raw_coords(cgm, 6, 6); active_only = true) != (6, 6)

        # A point is a point however it is written, here as everywhere else.
        let pt = (7.3, 4.2),
            reps = ((x = 7.3, y = 4.2), [7.3, 4.2], StaticArrays.SVector(7.3, 4.2))
            for q in reps
                Test.@test GD.locate(gs, q) == GD.locate(gs, pt)
                Test.@test C.nneighbors_within(gs, q; ball = 3.0) ==
                           C.nneighbors_within(gs, pt; ball = 3.0)
                Test.@test sort(C.neighbors_within(gs, q; ball = 3.0)) ==
                           sort(C.neighbors_within(gs, pt; ball = 3.0))
                Test.@test C.k_nearest(gs, q; k = 4)[1] == C.k_nearest(gs, pt; k = 4)[1]
                Test.@test C.fold_at(0, gs, q; ball = 3.0) do a, _J, _d
                    a + 1
                end == C.nneighbors_within(gs, pt; ball = 3.0)
            end
        end

        # A cell list is binned at one radius and may be queried at another. Past the point where the
        # window covers more bins than the lattice has buckets, walking the bins costs more than
        # offering every cell — and grows as `(r/h)^D` while the answer does not.
        let fine = C.MetricTopology(cgl; index = GD.cell_list(cgl; ball = 0.25)),
            plain = C.MetricTopology(cgl)
            for rq in (0.25, 2.5, 25.0, 250.0)
                t0 = time()
                got = sort(C.neighbors_within(cgl, (7.3, 4.2); ball = rq, topology = fine,
                                              scratch = C.ball_scratch()))
                el = time() - t0
                Test.@test got == sort(C.neighbors_within(cgl, (7.3, 4.2); ball = rq, topology = plain))
                Test.@test el < 1.0        # a `(r/h)^D` walk at r/h = 1000 would not return at all
            end
        end
    end

    Test.@testset "A wide ball on a wrapping domain reports each cell once" begin
        # A tree searches replicated points, so one cell can come back through several images — but only
        # when two of them fit inside the ball, which needs `2r` to reach the period. Below that the
        # dedup is skipped as unnecessary, so this sweeps `r` across the threshold to exercise both sides.
        C = FG.Connectivity
        GD = FG.Grids
        cart = FG.Geometry.CartesianGeometry{Float64}()
        n, L = 12, 12.0
        cx = [x for x in range(0.0, 11.0; length = n), _ in 1:n]
        cy = [y for _ in 1:n, y in range(0.0, 11.0; length = n)]
        g = GD.CurvilinearGrid(cart, cx, cy, trues(n, n); measure = fill(1.0, n, n),
                               periodic = (true, true), period = (L, L))
        for r in (2.0, 5.9, 6.0, 8.0, 15.0)          # 2r crosses the period at r = 6
            scan = sort(C.neighbors_within(g, 6, 6; ball = r))
            for top in (C.indexed(g), C.MetricTopology(g; index = GD.cell_list(g; ball = r)))
                got = C.neighbors_within(g, 6, 6; ball = r, topology = top)
                Test.@test allunique(got)
                Test.@test sort(got) == scan
                Test.@test C.nneighbors_within(g, 6, 6; ball = r, topology = top) == length(scan)
            end
        end
    end

    Test.@testset "Per-query cost does not grow with the grid" begin
        # The allocation gate cannot see this class: work that scans the grid but allocates nothing
        # passes it and is still linear per query. Every defect of that kind found here — a rescanned
        # minimum spacing, an `extrema` under a search radius — was allocation-free.
        #
        # Fixed work per query, growing grid, and the cost must stay flat. Bounds are loose because this
        # is wall clock on a shared machine; a linear regression would blow through them by 16×.
        C = FG.Connectivity
        GD = FG.Grids
        cart = FG.Geometry.CartesianGeometry{Float64}()

        function curv(n)
            x = [t for t in range(0.0, 1.0 * (n - 1); length = n), _ in 1:n]
            y = [t for _ in 1:n, t in range(0.0, 1.0 * (n - 1); length = n)]
            return GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
        end
        stretched(n) = GD.StructuredGrid(cart, cumsum(1.0 .+ 0.5 .* sin.(range(0, 3π; length = n))),
                                         collect(0.0:3.0))

        function percall(f, g, state)
            f(g, state)
            best = Inf
            for _ in 1:3
                t = @elapsed for _ in 1:200
                    f(g, state)
                end
                best = min(best, t / 200)
            end
            return best
        end

        # `setup` is everything built once per grid — an index, a buffer — and is deliberately outside
        # the timed call. Timing it too would measure construction, which is `O(n)` by right.
        function ratio(build, setup, op, small, big)
            gs, gb = build(small), build(big)
            ss, sb = setup(gs), setup(gb)
            return percall(op, gb, sb) / max(percall(op, gs, ss), eps())
        end

        indexed_setup(g) = (C.MetricTopology(g; index = GD.cell_list(g; ball = 2.5)),
                            Vector{Int}(undef, 8), Vector{Float64}(undef, 8))
        knn_op(g, s) = C.k_nearest!(s[2], s[3], g, 8, 8; k = 8, topology = s[1])
        ball_op(g, s) = C.nneighbors_within(g, 8, 8; ball = 2.5, topology = s[1])
        nothing_setup(_g) = nothing
        top_op(g, _s) = C.MetricTopology(g)
        span_op(g, _s) = (GD.extent(g, 1), GD.bounds(g, 2), GD.origin(g, 1))
        for (name, setup, op) in (("k_nearest!", indexed_setup, knn_op),
                                  ("indexed ball query", indexed_setup, ball_op),
                                  ("MetricTopology", nothing_setup, top_op),
                                  ("extent/bounds/origin", nothing_setup, span_op))
            r = ratio(curv, setup, op, 24, 96)          # 16× the cells
            r < 4 || println("    curvilinear ", name, " grew ", round(r; digits = 1), "× over 16× cells")
            Test.@test r < 4
        end

        # The separable architecture, where the window bound is what must stay O(1) per direction.
        win_op(g, _s) = C.nneighbors_within(g, 128, 2; ball = 3.0)
        r = ratio(stretched, nothing_setup, win_op, 256, 4096)
        r < 4 || println("    stretched-axis window grew ", round(r; digits = 1), "× over 16× samples")
        Test.@test r < 4
    end

    Test.@testset "Every public name is allocation-checked or has a stated reason not to be" begin
        public_of(m) = Set(s for s in (Symbol(b.var) for b in keys(Base.Docs.meta(m)))
                           if !startswith(String(s), "_"))

        ALLOCATION_CHECKED = Set([
            :coords, :measure, :isactive, :displacement, :neighbors, :neighbors!, :nneighbors,
            :nneighbors_within, :neighbors_within!, :fold_within, :metric_window, :k_nearest!,
            :MetricTopology, :minimum_spacing, :maximum_spacing, :axis_stats, :AxisStats,
            :area, :coords!, :axis, :spacing, :origin, :extent, :bounds, :isperiodic, :isuniform,
            :period, :size_tuple, :mask, :coordinate_names, :periodic_flags, :topology, :coordinates,
            :apply_stencil!, :foreach_within, :mapreduce_within, :embedded_radius, :fold_candidates,
            :fold_candidates_at, :locate, :embed_point, :fold_at,
            :fd_weights!, :nearest_index, :interpolation_weights, :scale_factors, :jacobian,
        ])

        # Adding a public name without putting it in one of the two sets fails this test. That is the
        # point: what is covered is derived from what the module documents, not from a list someone
        # remembered to update.
        NOT_CHECKED_BECAUSE = Set(Iterators.flatten((
            # types and traits
            [:AbstractGrid, :AbstractStructuredGrid, :AbstractCurvilinearGrid, :AbstractUnstructuredGrid,
             :StructuredGrid, :CurvilinearGrid, :UnstructuredGrid, :AbstractTopology, :Bounded, :Periodic,
             :AllActive, :SeparableMeasure, :PoleRotation, :AbstractImageConvention, :AbstractReach,
             :NearestImage, :AllImages, :Unrestricted, :Connected, :CSRConnectivity, :IndexTopology,
             :StencilNeighbors, :AbstractEmbedding, :CartesianEmbedding, :ChordEmbedding,
             :ArcEmbedding, :CellListIndex, :AbstractMaskPolicy, :BlankMasked, :ShiftWithinRun,
             :ReduceInRun, :AbstractLocation, :Center, :Face],
            # bulk or one-off operations, not per-cell hot paths
            [:build_connectivity, :build_connectivity_within, :foreach_within, :mapreduce_within,
             :adjacency_matrix, :adjacency_matrix!, :sparse_adjacency_matrix, :sparse_adjacency_matrix!,
             :sparse_adjacency_coo!, :sparse_adjacency_csc!, :sort_neighbors!, :is_symmetric_adjacency,
             :connected_components, :count_holes, :interior, :boundary_cells, :csr_connectivity,
             :empty_csr, :healpix_neighbors!, :indexed, :ball_scratch, :structured_grid,
             :rotate, :measure_array, :measure_factors,
             :corners, :corner_coords, :neighbor_nbrs, :distance, :embedded_points, :cell_list],
        # allocating forms, whose whole job is to return a fresh array
        [:neighbors_within, :k_nearest, :fd_weights, :lagrange_weights, :axis_stencils,
         :centers, :faces, :nodes],
        # grid constructors
        [:structured_grid, :unstructured_grid],
            # extension hooks, which throw until their trigger package is loaded
            [:spatial_index, :index_within!, :has_spatial_index],
            # returns the axis itself, so its type varies by direction: a runtime `d` cannot be stable
            [:axis],
        )))

        for m in (FG.Grids, FG.Connectivity, FG.Discretization)
            unclassified = setdiff(public_of(m), ALLOCATION_CHECKED, NOT_CHECKED_BECAUSE)
            isempty(unclassified) ||
                println("  unclassified in ", nameof(m), ": ", join(sort!(collect(unclassified)), ", "))
            Test.@test isempty(unclassified)
        end
    end

end
