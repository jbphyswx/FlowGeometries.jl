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
    gs = GD.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6), FG.Axes.PeriodicAxis(0.0, 2π, 24),
                           collect(range(-1.3, 1.3; length = 17)), msk)
    y = cumsum(1.0 .+ 0.3 .* sin.(1:18))
    gp = GD.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), FG.Axes.PeriodicAxis(0.0, 4.0, 12), y;
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

    # A whole-grid loop does not take a process backend.
    Test.@test_throws ArgumentError O.derivative!(zeros(24, 17), zeros(24, 17), gs, 1; execution = backend)
    return nothing
end
