# The checks a process backend passes, shared by the Distributed topic and the MPI script. Every
# operation's answer under `backend` equals its serial answer exactly: the shares write disjoint spans of
# the same arrays, and the reductions combine integer partials, or strings in an order that shows.
function check_process_backend(backend)
    GD = FG.Grids
    C = FG.Connectivity
    O = FG.Operators
    E = FG.Execution
    SS = FG.SphericalSampling
    cart = FG.Geometry.CartesianGeometry{Float64}()
    sph = FG.Geometry.SphericalGeometry(6.371e6)

    # Count, scan and fill, the degrees varying from cell to cell so a share boundary landing mid-row
    # shows up as a wrong offset.
    mk = [isodd(i * 7 + j * 3) for i in 1:31, j in 1:29]
    for g in (GD.StructuredGrid(cart, 0.0:30.0, 0.0:28.0, mk; periodic = (true, false)),
              GD.HEALPixGrid(sph, 4), GD.IcosahedralGrid(sph, 3), GD.CubedSphereGrid(sph, 4),
              GD.RingGrid(sph, SS.OctahedralGaussianSampling(8)))
        a = C.build_connectivity(g)
        b = C.build_connectivity(g; backend = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
    for s in (SS.HEALPixSampling(4), SS.IcosahedralSampling(3))
        a = C.build_connectivity(s)
        b = C.build_connectivity(s; backend = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
    let st = FG.Stencils.Moore(1)
        a = C.build_connectivity(SS.CubedSphereSampling(), 5; stencil = st)
        b = C.build_connectivity(SS.CubedSphereSampling(), 5; stencil = st, backend = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
    gball = GD.StructuredGrid(sph, range(0, 2π; length = 25)[1:24], range(-π / 2, π / 2; length = 13))
    let a = C.build_connectivity_within(gball; ball = 2.0e6),
        b = C.build_connectivity_within(gball; ball = 2.0e6, backend = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end

    # A stencil writes one value per cell: masked, batched, on every policy's path.
    x = collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = 24))))
    y = range(0.0, 2π * (1 - 1 / 20); length = 20)
    msk = trues(24, 20); msk[7, 9] = false; msk[13, 4] = false
    gs = GD.StructuredGrid(cart, x, y, msk; periodic = (false, true))
    f = [sin(3a) * cos(b) * c for a in x, b in y, c in 1:3]
    for dim in 1:2, pol in (O.BlankMasked(), O.ShiftWithinRun(), O.ReduceInRun())
        a = fill(NaN, size(f)); b = fill(NaN, size(f))
        O.apply_stencil!(a, f, gs, dim; order = 1, nodes = 5, masked = -3.0, policy = pol)
        O.apply_stencil!(b, f, gs, dim; order = 1, nodes = 5, masked = -3.0, policy = pol,
                         backend = backend)
        Test.@test isequal(a, b)
    end
    λ = collect(range(0, 2π * (1 - 1 / 16); length = 16))
    φ = collect(range(-π / 2, π / 2; length = 13))
    gsph = GD.StructuredGrid(sph, λ, φ)
    fs = [sin(l) * cos(p) for l in λ, p in φ]
    for dim in 1:2
        a = fill(NaN, size(fs)); b = fill(NaN, size(fs))
        O.derivative!(a, fs, gsph, dim; order = 1, nodes = 5, masked = -3.0)
        O.derivative!(b, fs, gsph, dim; order = 1, nodes = 5, masked = -3.0, backend = backend)
        Test.@test isequal(a, b)
    end
    Test.@test C.interior(gs) == C.interior(gs; backend = backend)
    Test.@test C.boundary_cells(gs; stencil = FG.Stencils.Moore(1)) ==
               C.boundary_cells(gs; stencil = FG.Stencils.Moore(1), backend = backend)

    # Point and area builders, one value per index and one column per row index.
    let a = SS.cubed_sphere_points(6), b = SS.cubed_sphere_points(6; backend = backend)
        Test.@test a.λ == b.λ && a.φ == b.φ && a.panel == b.panel
    end
    let Λ = [2π * (i - 1) / 12 for i in 1:12, j in 1:10], Φ = [asin(2 * (j - 0.5) / 10 - 1) for i in 1:12, j in 1:10]
        Test.@test GD.measure(GD.CurvilinearGrid(sph, Λ, Φ, trues(12, 10))) ==
                   GD.measure(GD.CurvilinearGrid(sph, Λ, Φ, trues(12, 10); backend = backend))
    end

    # Reductions, combined in share order.
    for n in (0, 1, 13, 10_000)
        Test.@test E.reduce_indices(i -> i * i, +, 0, n, backend) == sum(i * i for i in 1:n; init = 0)
    end
    Test.@test E.reduce_indices(string, *, "", 7, backend) == "1234567"
    Test.@test reduce(vcat, E.map_chunks(r -> collect(r), 1000, backend)) == collect(1:1000)
    gm = GD.StructuredGrid(cart, 0.0:1.0:19.0, 0.0:1.0:19.0)
    Test.@test C.mapreduce_within((I, J, d) -> 1, +, 0, gm; ball = 2.5, backend = backend) ==
               C.mapreduce_within((I, J, d) -> 1, +, 0, gm; ball = 2.5)
    # A sweep's own writes come back through the outputs it declares.
    out = zeros(Int, 20, 20)
    C.foreach_within(gm; ball = 2.5, backend = backend, outputs = (E.Written(out),)) do I, J, d
        @inbounds out[I[1], I[2]] += 1
    end
    Test.@test out == [C.nneighbors_within(gm, i, j; ball = 2.5) for i in 1:20, j in 1:20]
    return nothing
end
