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
    for g in (GD.StructuredGrid(cart, 0.0:30.0, 0.0:28.0, mk; topology = GD.Periodic()),
              GD.HEALPixGrid(sph, 4), GD.IcosahedralGrid(sph, 3), GD.CubedSphereGrid(sph, 4),
              GD.RingGrid(sph, SS.OctahedralGaussianSampling(8)))
        a = C.build_connectivity(g)
        b = C.build_connectivity(g; execution = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
    for s in (SS.HEALPixSampling(4), SS.IcosahedralSampling(3))
        a = C.build_connectivity(s)
        b = C.build_connectivity(s; execution = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
    let st = FG.Stencils.Moore(1)
        a = C.build_connectivity(SS.CubedSphereSampling(), 5; stencil = st)
        b = C.build_connectivity(SS.CubedSphereSampling(), 5; stencil = st, execution = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
        a = C.build_connectivity(SS.GaussLegendreSampling(), 9; stencil = st)
        b = C.build_connectivity(SS.GaussLegendreSampling(), 9; stencil = st, execution = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
    gball = GD.StructuredGrid(sph, FG.Axes.PeriodicUniformAxis(0.0, 2π, 24), range(-π / 2, π / 2; length = 13))
    let a = C.build_connectivity_within(gball; ball = 2.0e6),
        b = C.build_connectivity_within(gball; ball = 2.0e6, execution = backend)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end

    # A stencil writes one value per cell: masked, batched, on every policy's path.
    x = collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = 24))))
    y = range(0.0, 2π * (1 - 1 / 20); length = 20)
    msk = trues(24, 20); msk[7, 9] = false; msk[13, 4] = false
    gs = GD.StructuredGrid(cart, x, y, msk; topology = (GD.Bounded(), GD.Periodic()))
    f = [sin(3a) * cos(b) * c for a in x, b in y, c in 1:3]
    for dim in 1:2, pol in (O.BlankMasked(), O.ShiftWithinRun(), O.ReduceInRun())
        a = fill(NaN, size(f)); b = fill(NaN, size(f))
        O.apply_stencil!(a, f, gs, dim; order = 1, nodes = 5, masked = -3.0, policy = pol)
        O.apply_stencil!(b, f, gs, dim; order = 1, nodes = 5, masked = -3.0, policy = pol,
                         execution = backend)
        Test.@test isequal(a, b)
    end
    λ = collect(range(0, 2π * (1 - 1 / 16); length = 16))
    φ = collect(range(-π / 2, π / 2; length = 13))
    gsph = GD.StructuredGrid(sph, λ, φ; topology = GD.Periodic())
    fs = [sin(l) * cos(p) for l in λ, p in φ]
    for dim in 1:2
        a = fill(NaN, size(fs)); b = fill(NaN, size(fs))
        O.derivative!(a, fs, gsph, dim; order = 1, nodes = 5, masked = -3.0)
        O.derivative!(b, fs, gsph, dim; order = 1, nodes = 5, masked = -3.0, execution = backend)
        Test.@test isequal(a, b)
    end
    Test.@test C.interior(gs) == C.interior(gs; execution = backend)
    Test.@test C.boundary_cells(gs; stencil = FG.Stencils.Moore(1)) ==
               C.boundary_cells(gs; stencil = FG.Stencils.Moore(1), execution = backend)

    # Point and area builders, one value per index and one column per row index.
    let a = SS.cubed_sphere_points(6), b = SS.cubed_sphere_points(6; execution = backend)
        Test.@test a.λ == b.λ && a.φ == b.φ && a.panel == b.panel
    end
    let Λ = [2π * (i - 1) / 12 for i in 1:12, j in 1:10], Φ = [asin(2 * (j - 0.5) / 10 - 1) for i in 1:12, j in 1:10]
        Test.@test GD.measure(GD.CurvilinearGrid(sph, Λ, Φ, trues(12, 10))) ==
                   GD.measure(GD.CurvilinearGrid(sph, Λ, Φ, trues(12, 10); execution = backend))
    end

    # Reductions, combined in share order.
    for n in (0, 1, 13, 10_000)
        Test.@test E.reduce_indices(i -> i * i, +, 0, n, backend) == sum(i * i for i in 1:n; init = 0)
    end
    Test.@test E.reduce_indices(string, *, "", 7, backend) == "1234567"
    Test.@test reduce(vcat, E.map_chunks(r -> collect(r), 1000, backend)) == collect(1:1000)
    gm = GD.StructuredGrid(cart, 0.0:1.0:19.0, 0.0:1.0:19.0)
    Test.@test C.mapreduce_within((I, J, d) -> 1, +, 0, gm; ball = 2.5, execution = backend) ==
               C.mapreduce_within((I, J, d) -> 1, +, 0, gm; ball = 2.5)
    # A sweep's own writes come back through the outputs it declares.
    out = zeros(Int, 20, 20)
    C.foreach_within(gm; ball = 2.5, execution = backend, outputs = (E.Written(out),)) do I, J, d
        @inbounds out[I[1], I[2]] += 1
    end
    Test.@test out == [C.nneighbors_within(gm, i, j; ball = 2.5) for i in 1:20, j in 1:20]
    return nothing
end

# The checks a split grid passes on `nproc` processes, shared by the Distributed topic and the MPI script:
# each operation on a split field, gathered, against the same operation on the whole grid. `gather`
# returns the whole array where it lands (rank 0 under MPI, the caller under Distributed) and `nothing`
# elsewhere, so the comparisons are made where it lands.
function check_split(backend, nproc::Int)
    GD, O, D = FG.Grids, FG.Operators, FG.Decomposition
    landed(f, G) = G === nothing || f(G)

    # A masked sphere wrapping in longitude, split along it; and a box wrapping in both directions, its
    # second stretched, split along that one, its first held whole and wrapped onto itself, so a block's
    # corners come from a neighbour's halo.
    msk = trues(24, 17); msk[5, 8] = false; msk[20, 3] = false
    gs = GD.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6), FG.Axes.PeriodicUniformAxis(0.0, 2π, 24),
                           collect(range(-1.3, 1.3; length = 17)), msk)
    y = cumsum(1.0 .+ 0.3 .* sin.(1:18))
    gp = GD.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), FG.Axes.PeriodicUniformAxis(0.0, 4.0, 12), y;
                           topology = (GD.Periodic(), GD.Periodic()), period = (4.0, y[end] - y[1] + 1.0))
    for (g, dim) in ((gs, 1), (gp, 2))
        s = D.split_grid(g, backend; dims = (dim,), blocks = (nproc,), halo = 5)
        A = [sin(3a) * cos(b) + 0.1c for a in GD.coordinates(g, 1), b in GD.coordinates(g, 2), c in 1:2]
        f = D.distribute(s, A)
        landed(G -> Test.@test(G == A), D.gather(f))

        # After an exchange every block holds the whole grid's values at the cells it stands for.
        D.exchange_halo!(f)
        sz = size(g)
        Test.@test all(D.map_blocks(s, f) do b, a
            lo = map((r, i) -> first(r) - first(i), b.ranges, b.interior)
            all(a[I, c] == A[mod1(I[1] + lo[1], sz[1]), mod1(I[2] + lo[2], sz[2]), c]
                for I in CartesianIndices(size(b.grid)), c in 1:2)
        end)

        # Each block's own cells take the whole grid's derivative, along the split direction and the
        # other, under a policy whose windows depend on the mask. A window rebuilt beside the mask reads
        # the block's coordinates, which along a periodic uniform axis are a slice rounding as its own:
        # an ulp of the largest coordinate apart, so its weights are that over the spacing apart.
        for d in 1:2
            want = O.derivative!(similar(A), A, g, d; nodes = 5, masked = NaN, policy = O.ReduceInRun())
            o = D.split_field(s, Float64, 2)
            O.derivative!(o, f, d; nodes = 5, masked = NaN, policy = O.ReduceInRun())
            tol = 4 * eps(maximum(abs, GD.coordinates(g, d))) / GD.minimum_spacing(g, d)
            landed(D.gather(o)) do G
                ok = .!isnan.(want)
                Test.@test isnan.(G) == isnan.(want)
                Test.@test maximum(abs, G[ok] .- want[ok]) ≤ tol * maximum(abs, want[ok])
            end
        end
        let want = O.apply_stencil!(similar(A), A, g, dim; nodes = 4, masked = -1.0), o = D.split_field(s, Float64, 2)
            landed(G -> Test.@test(G == want), D.gather(O.apply_stencil!(o, f, dim; nodes = 4, masked = -1.0)))
        end
        Test.@test GD.integrate(f) ≈ GD.integrate(A, g) rtol = 1e-13

        # A stencil wider than the halo is refused.
        Test.@test_throws ArgumentError O.derivative!(D.split_field(s, Float64, 2), f, dim; nodes = 6)
    end

    # The same field split along the other direction, by the transpose.
    let s1 = D.split_grid(gp, backend; dims = (2,), blocks = (nproc,), halo = 2)
        s2 = D.split_grid(s1; dims = (1,))
        A = [a + 10b for a in 1:12, b in 1:18]
        o = D.redistribute!(D.split_field(s2, Float64), D.distribute(s1, Float64.(A)))
        landed(G -> Test.@test(G == A), D.gather(o))
    end
    return nothing
end
