# Dispatch gates over the derivative paths, on grids whose axis, closure and mask types differ by
# direction: building the plans, the grid forms at a runtime direction, and the applies make no runtime
# dispatch. `JET` refuses pre-release Julia.
using JET: JET

Test.@testset "Derivative plans and their applies make no runtime dispatch" begin
    if isempty(VERSION.prerelease)
        O, D, GD = FG.Operators, FG.Discretization, FG.Grids
        cart = FG.Geometry.CartesianGeometry{Float64}()
        sph = FG.Geometry.SphericalGeometry(6.371e6)
        mk = trues(16, 9)
        mk[5, 4] = false
        λ = FG.Axes.PeriodicUniformAxis(0.0, 2π, 16)
        φ = collect(range(-1.2, 1.2; length = 9))
        grids = (
            ("box, ranges", GD.StructuredGrid(cart, range(0.0, 1.0; length = 16), range(0.0, 2.0; length = 9))),
            ("box, range and vector", GD.StructuredGrid(cart, range(0.0, 1.0; length = 16), collect(range(0.0, 2.0; length = 9)))),
            ("masked box", GD.StructuredGrid(cart, collect(range(0.0, 1.0; length = 16)), collect(range(0.0, 2.0; length = 9)), mk)),
            ("sphere", GD.StructuredGrid(sph, λ, φ)),
            ("masked sphere", GD.StructuredGrid(sph, λ, φ, mk)),
            ("one latitude", GD.StructuredGrid(sph, λ, [0.4])),
            ("shell", GD.StructuredGrid(sph, FG.Axes.PeriodicUniformAxis(0.0, 2π, 8), range(-1.0, 1.0; length = 6),
                                        collect(range(6.371e6, 6.4e6; length = 5)))),
        )
        for (name, g) in grids
            sz = GD.size_tuple(g)
            N = length(sz)
            out, f = zeros(sz), rand(sz...)
            d = 1
            p = O.stencil_plans(g; nodes = 3)
            pl = D.stencil_plan(g, d; nodes = 3)
            u = ntuple(_ -> rand(sz...), N)
            S = NamedTuple{O.strain_components(g)}(ntuple(_ -> zeros(sz), N * (N + 1) ÷ 2))
            sc = ntuple(_ -> zeros(sz), N)
            Test.@testset "$name" begin
                JET.@test_opt target_modules = (FG,) O.stencil_plans(g; nodes = 3)
                JET.@test_opt target_modules = (FG,) D.stencil_plan(g, d; nodes = 3)
                JET.@test_opt target_modules = (FG,) D.axis_stencils(g, d; nodes = 3)
                JET.@test_opt target_modules = (FG,) O.derivative!(out, f, g, d; nodes = 3, masked = NaN)
                JET.@test_opt target_modules = (FG,) O.apply_stencil!(out, f, g, d; nodes = 3, masked = NaN)
                JET.@test_opt target_modules = (FG,) O.derivative!(out, f, g, pl, d; masked = NaN)
                JET.@test_opt target_modules = (FG,) O.derivative!(out, f, g, p, d; masked = NaN)
                JET.@test_opt target_modules = (FG,) O.strain_rate!(S, u, g, p; scratch = sc)
                JET.@test_opt target_modules = (FG,) O.divergence!(out, u, g, p; scratch = sc)
                GD.mask(g) isa GD.AllActive ||
                    JET.@test_opt target_modules = (FG,) O.masked_stencil_plan(g, d; nodes = 3)
                if GD.grid_geometry(g) isa FG.Geometry.SphericalGeometry
                    fp = O.frame_plan(g)
                    X = ntuple(_ -> zeros(sz), 3)
                    JET.@test_opt target_modules = (FG,) O.frame_plan(g)
                    JET.@test_opt target_modules = (FG,) O.to_planetary!(X..., u[1], u[2], nothing, fp)
                    JET.@test_opt target_modules = (FG,) O.to_local!(X..., u[1], u[2], f, fp)
                    JET.@test_opt target_modules = (FG,) O.tensor_to_local!(X..., nothing, nothing, nothing,
                                                                            ntuple(_ -> f, 6)..., fp)
                end
            end
        end
    end
end

Test.@testset "Builders, indices, queries and the geodesic make no runtime dispatch" begin
    if isempty(VERSION.prerelease)
        C, GD, GE, O, SS = FG.Connectivity, FG.Grids, FG.Geometry, FG.Operators, FG.SphericalSampling
        cart = GE.CartesianGeometry{Float64}()
        sph = GE.SphericalGeometry(6.371e6)
        sg = GD.StructuredGrid(cart, collect(0.0:15.0), collect(0.0:9.0))
        hp = GD.HEALPixGrid(sph, 4)
        xs, ys = rand(60), rand(60)
        un = GD.UnstructuredGrid(cart, (xs, ys), trues(60); k = 4, areas = ones(60))
        poles = GD.StructuredGrid(sph, FG.Axes.PeriodicUniformAxis(0.0, 2π, 16), collect(range(-π / 2, π / 2; length = 9)))
        for T in (Float64, Float32)
            spd = GE.SpheroidGeometry(T(6378137), T(1 / 298.257223563))
            JET.@test_opt target_modules = (FG,) GE.distance(spd, (T(0.1), T(0.2)), (T(2.0), T(-0.3)))
            JET.@test_opt target_modules = (FG,) GE.normal_coordinates(spd, (T(0.1), T(0.2)), (T(2.0), T(-0.3)))
            JET.@test_opt target_modules = (FG,) GE._geodesic_direct(spd, (T(0.1), T(0.2)), T(2.0), T(1e6))
        end
        # A spheroid's ball windows along a ring, a lattice and one with heights.
        spd = GE.SpheroidGeometry(6378137.0, 1 / 298.257223563)
        for g in (GD.StructuredGrid(spd, FG.Axes.PeriodicUniformAxis(0.0, 2π, 24)),
                  GD.StructuredGrid(spd, FG.Axes.PeriodicUniformAxis(0.0, 2π, 24), collect(-1.0:0.25:1.0)),
                  GD.StructuredGrid(spd, FG.Axes.PeriodicUniformAxis(0.0, 2π, 24), collect(-1.0:0.25:1.0), collect(0.0:1e4:5e4)))
            JET.@test_opt target_modules = (FG,) C.metric_window(g, 5e5)
            JET.@test_opt target_modules = (FG,) C.metric_window(g, map(_ -> 2, GD.size_tuple(g)), 5e5)
        end
        JET.@test_opt target_modules = (FG,) C.build_connectivity(sg)
        JET.@test_opt target_modules = (FG,) C.build_connectivity(Int, hp)
        JET.@test_opt target_modules = (FG,) C.build_connectivity_within(Int32, sg; ball = 1.5)
        JET.@test_opt target_modules = (FG,) C.build_connectivity(SS.HEALPixSampling(4))
        JET.@test_opt target_modules = (FG,) GD.YinYangGrid(sph, 16, 12)
        JET.@test_opt target_modules = (FG,) GD.UnstructuredGrid(cart, (xs, ys), trues(60); k = 4, areas = ones(60))
        JET.@test_opt target_modules = (FG,) GD.spatial_index(un)
        for layout in (GD.DenseBins(), GD.HashedBins())
            JET.@test_opt target_modules = (FG,) GD.binned_index(sg; layout)
        end
        JET.@test_opt target_modules = (FG,) C.nneighbors_within(sg, 5, 5; ball = 2.0)
        JET.@test_opt target_modules = (FG,) C.nneighbors_within(un, 5; ball = 0.2, topology = C.indexed(un))
        JET.@test_opt target_modules = (FG,) FG.Execution.reduce_indices(i -> i, +, 0, 100, CB.AutoBackend())
        JET.@test_opt target_modules = (FG,) FG.Execution.reduce_cells(I -> I[1], +, 0, (16, 9), CB.AutoBackend())
        fm = O.face_metrics(poles)
        JET.@test_opt target_modules = (FG,) O.face_metrics(poles)
        JET.@test_opt target_modules = (FG,) O.laplacian_coefficients(fm)
    end
end

Test.@testset "A block of a split grid is typed by the grid alone" begin
    if isempty(VERSION.prerelease)
        D, GD = FG.Decomposition, FG.Grids
        mk = trues(16, 9)
        mk[5, 4] = false
        y = cumsum(1.0 .+ 0.3 .* sin.(1:9))
        for g in (GD.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6), FG.Axes.PeriodicUniformAxis(0.0, 2π, 16),
                                    collect(range(-1.2, 1.2; length = 9)), mk),
                  GD.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), range(0.0, 1.0; length = 16), y;
                                    topology = (GD.Bounded(), GD.Periodic()), period = (0.0, y[end] - y[1] + 1.0)))
            # A block at a seam and one inside: the same type, inferred.
            halo = (3, 3)
            seam = D._block(g, (1:8, 1:9), halo, CB.SerialBackend())
            inner = D._block(g, (5:12, 1:9), halo, CB.SerialBackend())
            Test.@test typeof(seam) === typeof(inner)
            JET.@test_opt target_modules = (FG,) D._block(g, (1:8, 1:9), halo, CB.SerialBackend())
        end
    end
end
