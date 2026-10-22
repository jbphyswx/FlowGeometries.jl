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

    # A non-multiplicative `f` does not factor, so it falls through to the dense path and stays
    # correct there.
    for f in (exp, log, sin)
        Test.@test sum(f, m) ≈ sum(f, dense) rtol = 1e-9
    end

    # Reducing a direction away.
    for dims in (1, 2, (1, 2))
        Test.@test collect(sum(m; dims = dims)) ≈ sum(dense; dims = dims) rtol = 1e-12
    end

    # max/min of a product are taken from the per-axis endpoints, exact for factors of either sign.
    # A factor list holding a negative entry is the case that separates this from `∏ maximum`.
    ms = FG.Grids.SeparableMeasure(([-2.0, 1.0, 3.0], [-1.0, 4.0]))
    d2 = [ms[i, j] for i in 1:3, j in 1:2]
    Test.@test extrema(ms) == extrema(d2)
    Test.@test maximum(ms) == maximum(d2)
    Test.@test minimum(ms) == minimum(d2)
    Test.@test prod(ms) ≈ prod(d2) rtol = 1e-12
    # A product in range whose per-axis partial products are not: (1e-200)² underflows and (1e200)²
    # overflows, while every cell is 1.
    let mx = FG.Grids.SeparableMeasure(([1e-200, 1e-200], [1e200, 1e200])),
        mc = FG.Grids.SeparableMeasure((FG.Axes.ConstantVector(1e-200, 3), FG.Axes.ConstantVector(1e200, 4)))
        Test.@test prod(mx) ≈ 1.0 rtol = 1e-14
        Test.@test prod(mc) ≈ 1.0 rtol = 1e-14
        Test.@test prod(sqrt, mx) ≈ 1.0 rtol = 1e-14
        # …and one genuinely out of range saturates as the dense product does.
        Test.@test prod(FG.Grids.SeparableMeasure(([1e200, 1e200], [1e200]))) == Inf
    end

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
    end
    Test.@test S.reach(S.Moore(3), Val(2)) == (3, 3)
    Test.@test S.reach(S.Anisotropic((3, 1)), Val(2)) == (3, 1)
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
    Test.@test C.nneighbors(g, 1, 1; stencil = S.Moore(2)) == 8
end

Test.@testset "derivative! is with respect to distance, and masks where the metric dies" begin
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    GE = FG.Geometry
    R = 6.371e6
    sph = GE.SphericalGeometry(R)
    nλ, nφ = 48, 25
    λ = collect(range(0.0, 2π * (1 - 1 / nλ); length = nλ))
    φ = collect(range(-π / 2, π / 2; length = nφ))         # both poles are grid rows
    g = GD.StructuredGrid(sph, λ, φ; periodic = true)

    # ∂/∂north of sin φ is cos φ / R: the physical derivative, carrying the metric.
    f = [sin(fj) for _ in λ, fj in φ]
    o = zeros(nλ, nφ)
    O.derivative!(o, f, g, 2; order = 1, nodes = 5, masked = NaN)
    want = [cos(fj) / R for _ in λ, fj in φ]
    int = 4:(nφ - 3)
    Test.@test maximum(abs.(o[:, int] .- want[:, int])) < 1e-3 * maximum(abs.(want[:, int]))
    # …and it is exactly the coordinate derivative divided by the metric, cell by cell.
    co = zeros(nλ, nφ)
    O.apply_stencil!(co, f, g, 2; order = 1, nodes = 5)
    Test.@test all(o[i, j] ≈ co[i, j] / GE.scale_factors(sph, (λ[i], φ[j]))[2]
                   for i in 1:nλ, j in int)

    # ∂/∂east of sin λ cos φ is cos λ / R, and the poles have no east at all.
    f2 = [sin(l) * cos(fj) for l in λ, fj in φ]
    o2 = zeros(nλ, nφ)
    O.derivative!(o2, f2, g, 1; order = 1, nodes = 5, masked = NaN)
    j0 = nφ ÷ 2 + 1
    Test.@test maximum(abs.(o2[:, j0] .- [cos(l) / R for l in λ])) < 1e-3 / R
    Test.@test all(isnan, o2[:, 1]) && all(isnan, o2[:, nφ])
    Test.@test !any(isnan, o2[:, 2:(nφ - 1)])

    # The guard scales with the element type. At a Float32 pole `|h_λ|` is about 0.28, well above an
    # absolute `1e-12`, so only a type-relative floor marks those rows `masked`.
    let R32 = Float32(R), n1 = 16, n2 = 9
        s32 = GE.SphericalGeometry(R32)
        λ32 = collect(range(0.0f0, Float32(2π) * (1 - 1.0f0 / n1); length = n1))
        φ32 = collect(range(-Float32(π) / 2, Float32(π) / 2; length = n2))
        g32 = GD.StructuredGrid(s32, λ32, φ32; periodic = true)
        f32 = [sin(l) * cos(fj) for l in λ32, fj in φ32]
        o32 = zeros(Float32, n1, n2)
        O.derivative!(o32, f32, g32, 1; order = 1, nodes = 3, masked = Float32(NaN))
        Test.@test abs(GE.scale_factors(s32, (0.0f0, φ32[1]))[1]) > 1.0f-12   # the trap
        Test.@test all(isnan, o32[:, 1]) && all(isnan, o32[:, n2])
        Test.@test D.metric_floor(s32, 1) > abs(GE.scale_factors(s32, (0.0f0, φ32[1]))[1])
    end

    # A radius's factor is one, whatever the sphere's size: its floor is one's, and a shell at Earth's
    # radius in Float32 (`R·√eps ≈ 2200`) differences along it. `f = r` has derivative one, exact for
    # three nodes on a linear field.
    let R32 = Float32(R), r32 = collect(range(R32, 1.01f0 * R32; length = 5))
        s32 = GE.SphericalGeometry(R32)
        g3 = GD.StructuredGrid(s32, FG.Axes.PeriodicAxis(0.0f0, Float32(2π), 8),
                               collect(range(-1.0f0, 1.0f0; length = 5)), r32)
        o3 = zeros(Float32, 8, 5, 5)
        O.derivative!(o3, [r for _ in 1:8, _ in 1:5, r in r32], g3, 3; order = 1, nodes = 3, masked = NaN32)
        Test.@test D.metric_floor(s32, 3) < 1 < D.metric_floor(s32, 1)
        Test.@test maximum(abs, o3 .- 1) ≤ 64 * eps(Float32) * R32 / (r32[2] - r32[1])
    end

    # A Cartesian metric is the identity, so this must be `apply_stencil!` untouched.
    let cart = FG.Geometry.CartesianGeometry{Float64}(), x = collect(range(0.0, 1.0; length = 12))
        gc = GD.StructuredGrid(cart, x, x)
        fc = [xi^2 + yi for xi in x, yi in x]
        a = zeros(12, 12); b = zeros(12, 12)
        O.apply_stencil!(a, fc, gc, 1; order = 1, nodes = 3)
        O.derivative!(b, fc, gc, 1; order = 1, nodes = 3)
        Test.@test a == b
        Test.@test D.metric_floor(cart, 1) == 0.0
    end

    # On a spheroid `h_φ = M(φ)` varies with φ, so the factor varies along the direction being
    # differenced and stays inside the loop.
    let spd = GE.SpheroidGeometry(), n1 = 12, n2 = 21
        λs = collect(range(0.0, 2π * (1 - 1 / n1); length = n1))
        φs = collect(range(-1.2, 1.2; length = n2))
        gs = GD.StructuredGrid(spd, λs, φs; periodic = true)
        fs = [sin(2fj) for _ in λs, fj in φs]
        os = zeros(n1, n2); cs = zeros(n1, n2)
        O.derivative!(os, fs, gs, 2; order = 1, nodes = 5, masked = NaN)
        O.apply_stencil!(cs, fs, gs, 2; order = 1, nodes = 5)
        Test.@test all(os[i, j] ≈ cs[i, j] / GE.scale_factors(spd, (λs[i], φs[j]))[2]
                       for i in 1:n1, j in 1:n2)
        Test.@test GE.scale_factors(spd, (0.0, 0.0))[2] != GE.scale_factors(spd, (0.0, 1.2))[2]
    end

    # The structural fact the scaling sweep hoists on: no scale factor depends on longitude.
    for geo in (sph, GE.SpheroidGeometry()), fj in (-1.0, 0.0, 0.7)
        Test.@test all(GE.scale_factors(geo, (0.0, fj)) .== GE.scale_factors(geo, (2.5, fj)))
    end
end

Test.@testset "A held stencil table serves every mask policy" begin
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    n = 24
    x = collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = n))))
    y = collect(range(0.0, 2π * (1 - 1 / n); length = n))
    mk = trues(n, n); mk[7, 9] = false; mk[8, 9] = false; mk[13, 4] = false
    g = GD.StructuredGrid(cart, x, y, mk; periodic = (false, true), period = (0.0, 2π))
    f = [sin(3xi) * cos(yj) for xi in x, yj in y]

    # A table handed with its grid serves every policy, the grid supplying the axis a window at a mask
    # edge is rebuilt from.
    for dim in 1:2, pol in (O.BlankMasked(), O.ShiftWithinRun(), O.ReduceInRun()), k in (3, 5)
        a = zeros(n, n); b = zeros(n, n)
        O.apply_stencil!(a, f, g, dim; order = 1, nodes = k, masked = NaN, policy = pol)
        idx, w = D.axis_stencils(g, dim; order = 1, nodes = k)
        O.apply_stencil!(b, f, g, idx, w, dim; order = 1, masked = NaN, policy = pol)
        Test.@test all(isequal(a[i], b[i]) for i in eachindex(a))
    end


    # The bare form has no axis to rebuild a stencil from, and refuses the policy.
    let idx = D.axis_stencils(g, 1; order = 1, nodes = 3)
        Test.@test_throws ArgumentError O.apply_stencil!(zeros(n, n), f, idx[1], idx[2], 1;
                                                         mask = mk, policy = O.ReduceInRun())
    end

    # `derivative!` is the form a geometry-aware caller uses, so it takes a table too.
    let sph = FG.Geometry.SphericalGeometry(6.371e6),
        lam = collect(range(0.0, 2π * (1 - 1 / 16); length = 16)),
        phi = collect(range(-1.2, 1.2; length = 13))
        gs = GD.StructuredGrid(sph, lam, phi; periodic = true)
        fs = [sin(fj) for _ in lam, fj in phi]
        a = zeros(16, 13); b = zeros(16, 13)
        O.derivative!(a, fs, gs, 2; order = 1, nodes = 3, masked = NaN)
        i2, w2 = D.axis_stencils(gs, 2; order = 1, nodes = 3)
        O.derivative!(b, fs, gs, i2, w2, 2; order = 1, masked = NaN)
        Test.@test all(isequal(a[i], b[i]) for i in eachindex(a))
    end
end

Test.@testset "The host sweep is a different loop shape, and the same answer" begin
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()

    # The host loop and the index-parallel device form do identical arithmetic in identical order, so
    # they agree bit for bit across masks, node counts and both dimensions.
    bad = 0
    for n in (7, 16, 23), dim in 1:2, k in (2, 3, 4, 5, 6, 8), msk in (false, true)
        xs = cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = n)))
        ys = collect(range(0.0, 2.0; length = n + 1))
        ax = dim == 1 ? xs : ys
        k > length(ax) && continue
        f = [sin(xi) * cos(yi) for xi in xs, yi in ys]
        m = trues(n, n + 1)
        msk && (m[3, 4] = false; m[n, 2] = false)
        idx, w = D.axis_stencils(ax, 1, k)
        a = zeros(n, n + 1); b = zeros(n, n + 1)
        mm = msk ? m : nothing
        O.apply_stencil!(a, f, idx, w, dim; mask = mm, masked = NaN)
        O.apply_stencil!(b, f, idx, w, dim; mask = mm, masked = NaN,
                         execution = CB.GPUBackend(KernelAbstractions.CPU()))
        all(isequal(a[i], b[i]) for i in eachindex(a)) || (bad += 1)
    end
    Test.@test bad == 0

    # Every direction of a 3-D field, including the last, where the split leaves a contiguous span.
    let n = 9
        xs = collect(range(0.0, 1.0; length = n))
        ys = collect(range(0.0, 2.0; length = n))
        zs = collect(range(0.0, 3.0; length = n))
        f3 = [xi + 2yi + 4zi for xi in xs, yi in ys, zi in zs]
        for (dim, ax, want) in ((1, xs, 1.0), (2, ys, 2.0), (3, zs, 4.0))
            idx, w = D.axis_stencils(ax, 1, 3)
            o = zeros(n, n, n); ob = zeros(n, n, n)
            O.apply_stencil!(o, f3, idx, w, dim)
            O.apply_stencil!(ob, f3, idx, w, dim; execution = CB.GPUBackend(KernelAbstractions.CPU()))
            Test.@test maximum(abs.(o .- want)) < 1e-10
            Test.@test o == ob
        end
    end

    # A stencil wider than the specialization cap keeps the runtime-node-count path, and agrees.
    let n = 32
        xs = collect(range(0.0, 1.0; length = n))
        f = [sin(4xi) * yi for xi in xs, yi in xs]
        for k in (10, 12)
            idx, w = D.axis_stencils(xs, 1, k)
            a = zeros(n, n); b = zeros(n, n)
            O.apply_stencil!(a, f, idx, w, 1)
            O.apply_stencil!(b, f, idx, w, 1; execution = CB.GPUBackend(KernelAbstractions.CPU()))
            Test.@test a == b
        end
    end
end

Test.@testset "apply_stencil! differentiates a field along one direction" begin
    D = FG.Discretization
    O = FG.Operators
    geo = FG.Geometry.CartesianGeometry()

    # Exact for any polynomial the node count spans, at every sample including the ends, where the
    # stencil shifts inward and keeps its order.
    x = collect(range(0.0, 2.0; length = 11))
    f = @. 3x^2 - 2x + 5
    out = similar(f)
    O.apply_stencil!(out, f, x, 1; order = 1, nodes = 3)
    Test.@test maximum(abs, out .- (6 .* x .- 2)) < 1e-12
    O.apply_stencil!(out, f, x, 1; order = 2, nodes = 3)
    Test.@test maximum(abs, out .- 6.0) < 1e-11

    # A stretched axis is equally exact, its weights derived per sample.
    xs = [0.0, 0.11, 0.37, 0.9, 1.05, 1.6, 1.62, 2.0]
    outs = similar(xs)
    O.apply_stencil!(outs, (@. 3xs^2 - 2xs + 5), xs, 1; order = 1, nodes = 3)
    Test.@test maximum(abs, outs .- (6 .* xs .- 2)) < 1e-11
    # 3 nodes cannot span a cubic; 4 can.
    O.apply_stencil!(outs, xs .^ 3, xs, 1; order = 1, nodes = 3)
    Test.@test maximum(abs, outs .- 3 .* xs .^ 2) > 1e-3
    O.apply_stencil!(outs, xs .^ 3, xs, 1; order = 1, nodes = 4)
    Test.@test maximum(abs, outs .- 3 .* xs .^ 2) < 1e-11

    # Periodic: the stencil stays centred and wraps, so the seam matches the interior, and a 5-node
    # stencil converges at 4th order.
    perr(m) = begin
        lm = collect(range(0, 2π; length = m + 1)[1:m])
        o = similar(lm)
        O.apply_stencil!(o, sin.(lm), lm, 1; order = 1, nodes = 5, period = 2π)
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
    O.apply_stencil!(od, sin.(λd), λd, 1; order = 1, nodes = 5, period = 2π)
    Test.@test maximum(abs, od .- cos.(λd)) < 1e-5
    Test.@test FG.Axes.wrap_sign(λd) == -1.0 && FG.Axes.wrap_sign(-λd) == 1.0

    # Only the named direction is differenced.
    X = collect(range(0.0, 1.0; length = 9))
    Y = collect(range(0.0, 2.0; length = 7))
    F = [xi^2 + 3yi for xi in X, yi in Y]
    Od = similar(F)
    O.apply_stencil!(Od, F, X, 1; order = 1, nodes = 3)
    Test.@test maximum(abs, Od .- [2xi for xi in X, _ in Y]) < 1e-11
    O.apply_stencil!(Od, F, Y, 2; order = 1, nodes = 3)
    Test.@test maximum(abs, Od .- 3.0) < 1e-11
    F3 = [xi^2 + 3yi + 2zi for xi in X, yi in Y, zi in 0.0:0.5:1.0]
    O3 = similar(F3)
    O.apply_stencil!(O3, F3, collect(0.0:0.5:1.0), 3; order = 1, nodes = 3)
    Test.@test maximum(abs, O3 .- 2.0) < 1e-11

    # The grid form supplies axis, wrap period and mask, so none of it is restated.
    gx = FG.Grids.StructuredGrid(geo, X, Y)
    Og = similar(F)
    O.apply_stencil!(Og, F, gx, 1; order = 1, nodes = 3)
    Test.@test Og ≈ [2xi for xi in X, _ in Y]
    λ = collect(range(0, 2π; length = 65)[1:64])
    gp = FG.Grids.StructuredGrid(geo, λ, [0.0, 1.0]; periodic = true, period = 2π)
    Op = similar([sin(l) for l in λ, _ in 1:2])
    O.apply_stencil!(Op, [sin(l) for l in λ, _ in 1:2], gp, 1; order = 1, nodes = 5)
    Test.@test maximum(abs, Op[:, 1] .- cos.(λ)) < 1e-5

    # A derivative whose stencil reaches an inactive cell reports `masked`.
    mk = trues(9, 7)
    mk[5, 3] = false
    gm = FG.Grids.StructuredGrid(geo, X, Y, mk)
    Om = fill(NaN, 9, 7)
    O.apply_stencil!(Om, F, gm, 1; order = 1, nodes = 3, masked = -1.0)
    Test.@test Om[5, 3] == -1.0                       # the inactive cell itself
    Test.@test Om[4, 3] == -1.0 && Om[6, 3] == -1.0   # its stencil neighbours
    Test.@test Om[2, 3] ≈ 2X[2] && Om[8, 3] ≈ 2X[8]   # cells that never read it
    Test.@test Om[5, 4] ≈ 2X[5]                       # a different row is unaffected
    O.apply_stencil!(Om, F, gm, 1; order = 1, nodes = 3, active_only = false)
    Test.@test Om[5, 3] ≈ 2X[5]

    # A precomputed weight set gives the same answer.
    idx, w = D.axis_stencils(X, 1, 3)
    O2 = similar(F)
    O.apply_stencil!(O2, F, idx, w, 1)
    Test.@test O2 ≈ [2xi for xi in X, _ in Y]

    Test.@test_throws ArgumentError D.axis_stencils(X, 2, 2)          # too few nodes for order 2
    Test.@test_throws ArgumentError D.axis_stencils([0.0, 1.0], 1, 5) # more nodes than samples
    Test.@test_throws ArgumentError O.apply_stencil!(Od, F, X, 3)      # no direction 3 in a matrix
    Test.@test_throws DimensionMismatch O.apply_stencil!(Od, F, X[1:5], 1)
    Test.@test_throws DimensionMismatch O.apply_stencil!(similar(F, 3, 3), F, idx, w, 1)
end

Test.@testset "A field can be evaluated at a coordinate, on every architecture" begin
    C = FG.Connectivity
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()

    # Structured: the tensor product of the per-axis weights, so a bilinear field is exact.
    nx, ny = 11, 9
    x = collect(range(0.0, 2.0; length = nx))
    y = collect(range(-1.0, 3.0; length = ny))
    g = GD.StructuredGrid(cart, x, y)
    f = [2xi - 3yj + 0.5 * xi * yj + 7 for xi in x, yj in y]
    exact(px, py) = 2px - 3py + 0.5 * px * py + 7
    for px in range(0.05, 1.95; length = 9), py in range(-0.95, 2.95; length = 7)
        Test.@test O.interpolate(f, g, (px, py)) ≈ exact(px, py) atol = 1e-10
    end
    # And a cell's own value at its centre, which multilinear interpolation must reproduce.
    Test.@test all(O.interpolate(f, g, (x[i], y[j])) ≈ f[i, j] for i in 1:nx, j in 1:ny)

    # A periodic direction interpolates across its seam, pairing the last sample with the first.
    let nλ = 24, λ = collect(range(0.0, 2π * (1 - 1 / 24); length = 24)), z = [0.0, 1.0]
        gp = GD.StructuredGrid(cart, λ, z; periodic = (true, false), period = (2π, 0.0))
        fp = [sin(l) for l in λ, _ in z]
        mid = λ[end] + (2π - λ[end]) / 2         # strictly between the last sample and the first
        Test.@test O.interpolate(fp, gp, (mid, 0.0)) ≈ (sin(λ[end]) + sin(λ[1])) / 2
        Test.@test O.interpolate(fp, gp, (2π + 0.3, 0.0)) ≈ O.interpolate(fp, gp, (0.3, 0.0))
    end
    # A descending axis has its seam above its first sample, between that sample and the last one's
    # image one period up, and the pair is linear across it.
    let λd = reverse(collect(range(0.0, 2π * (1 - 1 / 24); length = 24))), z = [0.0, 1.0]
        gd = GD.StructuredGrid(cart, λd, z; periodic = (true, false), period = (2π, 0.0))
        fd = [sin(l) for l in λd, _ in z]
        gap = 2π - λd[1]
        Test.@test O.interpolate(fd, gd, (λd[1] + gap / 2, 0.0)) ≈ (sin(λd[1]) + sin(λd[end])) / 2
        Test.@test O.interpolate(fd, gd, (λd[1] + gap / 4, 0.0)) ≈
                   0.75 * sin(λd[1]) + 0.25 * sin(λd[end])
        Test.@test O.interpolate(fd, gd, (2π + 0.3, 0.0)) ≈ O.interpolate(fd, gd, (0.3, 0.0))
    end

    # Scattered and curvilinear: a least-squares plane, so a linear field is exact.
    let npt = 400, xs = 10.0 .* rand(npt), ys = 6.0 .* rand(npt)
        gu = GD.UnstructuredGrid(cart, (xs, ys), trues(npt); k = 8, areas = ones(npt))
        fu = 2.0 .* xs .- 3.0 .* ys .+ 5.0
        for _ in 1:20
            px, py = 1.0 + 8.0 * rand(), 1.0 + 4.0 * rand()
            Test.@test O.interpolate(fu, gu, (px, py); k = 8) ≈ 2px - 3py + 5 atol = 1e-8
        end
        Test.@test all(O.interpolate(fu, gu, (xs[i], ys[i]); k = 8) ≈ fu[i] for i in 1:20)
        # The fit's rank test is in the matrix's own units, so the length unit changes nothing.
        let su = GD.UnstructuredGrid(cart, (1e6 .* xs, 1e6 .* ys), trues(npt); k = 8, areas = ones(npt))
            Test.@test O.interpolate(fu, su, (5e6, 3e6); k = 8) ≈ 2 * 5.0 - 3 * 3.0 + 5 atol = 1e-8
        end
    end
    let n = 14
        xc = [t + 0.35u for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
        yc = [u - 0.2t for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
        cg = GD.CurvilinearGrid(cart, xc, yc, trues(n, n); measure = fill(1.0, n, n))
        fc = 2.0 .* xc .- 3.0 .* yc .+ 5.0
        for _ in 1:15
            px, py = 2.0 + 5.0 * rand(), 0.0 + 3.0 * rand()
            Test.@test O.interpolate(fc, cg, (px, py); k = 8) ≈ 2px - 3py + 5 atol = 1e-8
        end
    end

    # A node set on a sphere, through every way of finding the neighbours: a scan, a tree and a cell
    # list name the same cells in the same order, so the fit agrees to the bit, and a held buffer pair
    # changes nothing. A field linear in the tangent plane at the query point is reproduced there.
    let gh = healpix_node_grid(8)
        geo = GD.grid_geometry(gh)
        nh = length(GD.mask(gh))
        R = FG.Geometry.radius(geo)
        λv = [GD.coords(gh, j).λ for j in 1:nh]
        φv = [GD.coords(gh, j).φ for j in 1:nh]
        tops = (C.MetricTopology(gh), C.indexed(gh),
                C.MetricTopology(gh; index = GD.cell_list(gh; ball = 0.3R)))
        nb = (Vector{Int}(undef, 8), Vector{Float64}(undef, 8))
        for q in ((0.4, 0.1), (3.9, -0.7), (6.1, 1.4))
            fq = [(Δ = FG.Geometry.project_to_tangent_plane(geo, q, (λv[j], φv[j]));
                   2.0 + 3.0 * Δ.λ / R - 1.5 * Δ.φ / R) for j in 1:nh]
            ref = O.interpolate(fq, gh, q; k = 8)
            Test.@test abs(ref - 2.0) < 1e-10
            for top in tops
                Test.@test O.interpolate(fq, gh, q; k = 8, topology = top,
                                         scratch = C.ball_scratch(top), nearest = nb) == ref
            end
        end
        Test.@test_throws ArgumentError O.interpolate(ones(nh), gh, (0.4, 0.1); k = 8,
                                                      nearest = (zeros(Int, 4), zeros(8)))
        Test.@test_throws ArgumentError O.interpolate(ones(nh), gh, (0.4, 0.1); k = -1)

        # `BlankMasked` refuses a neighbourhood an inactive cell lies in: one at least as near as the
        # farthest of the `k` nearest active cells. Against a full ranking, on probes either side of the
        # edge of a masked cap.
        mk = φv .< 0.9
        gm = GD.UnstructuredGrid(geo, (λv, φv), mk; k = 8, areas = ones(nh))
        tops_m = (C.MetricTopology(gm), C.indexed(gm),
                  C.MetricTopology(gm; index = GD.cell_list(gm; ball = 0.3R)))
        wants = Bool[]
        for λq in (0.3, 2.0, 4.4), φq in range(0.55, 1.2; length = 9)
            q = (λq, φq)
            d = [FG.Geometry.distance(geo, q, (λv[j], φv[j])) for j in 1:nh]
            kth = sort(d[mk])[8]
            want = minimum(d[.!mk]) ≤ kth
            push!(wants, want)
            for top in tops_m
                Test.@test isnan(O.interpolate(ones(nh), gm, q; k = 8, masked = NaN, topology = top,
                                               scratch = C.ball_scratch(top))) == want
            end
            Test.@test O.interpolate(ones(nh), gm, q; k = 8, masked = NaN,
                                     policy = O.ReduceInRun()) ≈ 1.0
        end
        Test.@test any(wants) && !all(wants)
    end

    # A point near an edge can lie nearer the vertex across it than any vertex of its own triangle, so
    # the triangles around the nearest node need not hold it. Four nodes where that happens: the value
    # is the containing triangle's, 1 for a field equal to 1 at its three vertices.
    let x = [0.0, -sqrt(3) / 2, sqrt(3) / 2, 0.0], y = [1.0, -0.5, -0.5, -1.01]
        tm = GD.UnstructuredGrid(cart, (x, y), ones(4), trues(4);
                                 mesh = GD.CellMesh([1, 4, 7], [1, 2, 3, 2, 4, 3], 4))
        Test.@test O.interpolate(x .^ 2 .+ y .^ 2, tm, (0.0, -0.49); k = 4) ≈ 1.0
    end
    # On a Delaunay mesh of scattered nodes, every probe inside it takes the linear value of the
    # triangle holding it, found by testing every triangle.
    let n = 300, xs = 10.0 .* rand(n), ys = 6.0 .* rand(n)
        gd = GD.UnstructuredGrid(cart, (xs, ys), trues(n); k = 8)
        mesh = GD.cell_mesh(gd)
        fd = sin.(xs) .* cos.(ys)
        walked = 0
        for _ in 1:2000
            p = (10.0 * rand(), 6.0 * rand())
            want = nothing
            for c in 1:GD.ncells(mesh)
                a, b, e = GD.cell_nodes(mesh, c)
                det = (ys[b] - ys[e]) * (xs[a] - xs[e]) + (xs[e] - xs[b]) * (ys[a] - ys[e])
                w1 = ((ys[b] - ys[e]) * (p[1] - xs[e]) + (xs[e] - xs[b]) * (p[2] - ys[e])) / det
                w2 = ((ys[e] - ys[a]) * (p[1] - xs[e]) + (xs[a] - xs[e]) * (p[2] - ys[e])) / det
                w3 = 1 - w1 - w2
                if min(w1, w2, w3) > 1e-9
                    want = w1 * fd[a] + w2 * fd[b] + w3 * fd[e]
                    walked += argmin(hypot.(xs .- p[1], ys .- p[2])) ∉ (a, b, e)
                    break
                end
            end
            want === nothing && continue                 # outside the mesh
            Test.@test O.interpolate(fd, gd, p; k = 8) ≈ want atol = 1e-12
        end
        Test.@test walked > 0
    end

    # The mask policies mean here what they mean for a stencil.
    let mk = trues(nx, ny)
        mk[5, 4] = false
        gm = GD.StructuredGrid(cart, x, y, mk)
        pin = ((x[5] + x[6]) / 2, (y[4] + y[5]) / 2)      # its corners include the hole
        Test.@test isnan(O.interpolate(f, gm, pin; masked = NaN))
        Test.@test !isnan(O.interpolate(f, gm, pin; masked = NaN, policy = O.ReduceInRun()))
        Test.@test !isnan(O.interpolate(f, gm, (x[1], y[1]); masked = NaN))
        Test.@test_throws ArgumentError O.interpolate(f, gm, pin; policy = O.ShiftWithinRun())
    end

    # A point is a point however it is written.
    let a = O.interpolate(f, g, (0.7, 1.1))
        Test.@test a == O.interpolate(f, g, [0.7, 1.1])
        Test.@test a == O.interpolate(f, g, (x = 0.7, y = 1.1))
        Test.@test a == O.interpolate(f, g, StaticArrays.SVector(0.7, 1.1))
    end
    Test.@test_throws DimensionMismatch O.interpolate(zeros(3, 3), g, (0.5, 0.5))
end

Test.@testset "A least-squares gradient where there is no separable axis" begin
    C = FG.Connectivity
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    GE = FG.Geometry
    cart = FG.Geometry.CartesianGeometry{Float64}()

    # A sheared mesh, where a linear field is reproduced exactly on a skewed stencil.
    n = 14
    x = [t + 0.35u for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
    y = [u - 0.2t for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
    cg = GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
    plan = O.gradient_plan(cg)
    for (a, b) in ((2.0, -3.0), (0.0, 1.0), (-1.5, 0.0))
        f = a .* x .+ b .* y .+ 7.0
        g1 = zeros(n, n); g2 = zeros(n, n)
        O.gradient!(g1, g2, f, plan)
        # Every cell, boundaries and corners included — a one-sided stencil is still exact for a
        # linear field as long as it spans both tangent directions.
        Test.@test maximum(abs.(g1 .- a)) < 1e-10
        Test.@test maximum(abs.(g2 .- b)) < 1e-10
    end
    # Every cell is its own work item, so a threaded or device apply is the serial one to the bit,
    # trailing batch axis included.
    let f = cat(sin.(x) .* y, cos.(y) .- x; dims = 3)
        s1 = zeros(n, n, 2); s2 = zeros(n, n, 2)
        O.gradient!(s1, s2, f, plan)
        for be in (CB.ThreadedBackend(), CB.GPUBackend(KernelAbstractions.CPU()))
            t1 = fill(NaN, n, n, 2); t2 = fill(NaN, n, n, 2)
            O.gradient!(t1, t2, f, plan; execution = be)
            Test.@test t1 == s1 && t2 == s2
        end
    end

    # Where the stencil is separable and orthogonal, `A` is diagonal and this must reduce to the
    # centred difference — so it agrees with `apply_stencil!` wherever both apply.
    let m = 12
        xs = collect(range(0.0, 1.0; length = m)); ys = collect(range(0.0, 2.0; length = m))
        og = GD.CurvilinearGrid(cart, [xi for xi in xs, _ in ys], [yj for _ in xs, yj in ys],
                                trues(m, m); measure = fill(1.0, m, m))
        sg = GD.StructuredGrid(cart, xs, ys)
        ff = [sin(3xi) * cos(2yj) for xi in xs, yj in ys]
        q1 = zeros(m, m); q2 = zeros(m, m)
        O.gradient!(q1, q2, ff, O.gradient_plan(og))
        s1 = zeros(m, m); s2 = zeros(m, m)
        O.apply_stencil!(s1, ff, sg, 1; order = 1, nodes = 3)
        O.apply_stencil!(s2, ff, sg, 2; order = 1, nodes = 3)
        int = 2:(m - 1)
        Test.@test maximum(abs.(q1[int, int] .- s1[int, int])) < 1e-12
        Test.@test maximum(abs.(q2[int, int] .- s2[int, int])) < 1e-12
    end

    # A node set, whose neighbours come from its stored connectivity.
    let gu = C.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(8)),
        R = FG.Geometry.radius(GD.grid_geometry(C.unstructured_grid(
            FG.SphericalSampling.IcosahedralSampling(8))))
        φu = GD.coordinates(gu, 2)
        fu = [sin(φ) for φ in φu]                    # ∂/∂north = cos φ / R, ∂/∂east = 0
        h1 = zeros(length(fu)); h2 = zeros(length(fu))
        O.gradient!(h1, h2, fu, O.gradient_plan(gu))
        want = [cos(φ) / R for φ in φu]
        Test.@test maximum(abs.(h2 .- want)) < 0.05 * maximum(abs.(want))
        Test.@test maximum(abs.(h1)) < 0.05 * maximum(abs.(want))
    end

    # Rank deficiency: every neighbour on one line leaves the across-line component undetermined
    # by the data, and the pseudo-inverse zeroes it.
    let xl = collect(range(0.0, 5.0; length = 6)), yl = zeros(6)
        lg = GD.UnstructuredGrid(cart, (xl, yl), trues(6); k = 2, areas = ones(6))
        l1 = zeros(6); l2 = zeros(6)
        O.gradient!(l1, l2, 3.0 .* xl, O.gradient_plan(lg))
        Test.@test all(isapprox.(l1, 3.0; atol = 1e-10))     # the resolved direction is exact
        Test.@test all(iszero, l2)                           # the undetermined one is zero
    end

    # A mask: the hole has no coefficients and reads zero, and its neighbours are still exact,
    # having simply not been offered it.
    let mk = trues(n, n)
        mk[5, 5] = false; mk[6, 5] = false
        mg = GD.CurvilinearGrid(cart, x, y, mk; measure = fill(1.0, n, n))
        mf = 2.0 .* x .- 3.0 .* y
        m1 = zeros(n, n); m2 = zeros(n, n)
        O.gradient!(m1, m2, mf, O.gradient_plan(mg))
        Test.@test iszero(m1[5, 5]) && iszero(m2[5, 5])
        Test.@test all(abs(m1[i, j] - 2.0) < 1e-10 && abs(m2[i, j] + 3.0) < 1e-10
                       for i in 1:n, j in 1:n if mk[i, j])
    end

    # A spheroid. The projection is written against `embed` and the local frame, so the
    # least-squares gradient and the scattered fit hold here too. The frame matches the sphere's at
    # the same (λ, φ), and the two differ in where each (λ, φ) sits.
    let spd = GE.SpheroidGeometry(), nx = 7, ny = 6, np = 7 * 6
        λs = [0.4 + 0.01 * (i - 1) for j in 1:ny for i in 1:nx]
        φs = [0.6 + 0.011 * (j - 1) for j in 1:ny for i in 1:nx]
        sg = GD.UnstructuredGrid(spd, (λs, φs), trues(np); k = 6, areas = ones(np))
        sp = O.gradient_plan(sg)
        i0 = 3 + nx * 2
        p0 = (λs[i0], φs[i0])
        α, β = 3.5, -1.25
        # Linear in the tangent plane AT `i0`, which is where exactness is claimed: a field linear in
        # one cell's plane is not linear in another's, the surface being curved.
        fs = [(Δ = GE.project_to_tangent_plane(spd, p0, (λs[j], φs[j])); 10.0 + α * Δ.λ + β * Δ.φ)
              for j in 1:np]
        s1 = zeros(np); s2 = zeros(np)
        O.gradient!(s1, s2, fs, sp)
        Test.@test abs(s1[i0] - α) < 1e-9 * abs(α)
        Test.@test abs(s2[i0] - β) < 1e-9 * abs(β)
        # Scattered interpolation is the same projection under a 3-parameter fit, so its constant term
        # recovers the value at the query point exactly.
        q = (0.4 + 0.025, 0.6 + 0.0275)
        fq = [(Δ = GE.project_to_tangent_plane(spd, q, (λs[j], φs[j])); -4.0 + α * Δ.λ + β * Δ.φ)
              for j in 1:np]
        Test.@test abs(O.interpolate(fq, sg, q; k = 8) + 4.0) < 1e-8
    end

    Test.@test O.ncomponents(plan) == 2
    Test.@test_throws DimensionMismatch O.gradient!(zeros(3), zeros(3), zeros(3), plan)

    # Three coordinates resolve three directions, on the full local frame.
    let m = 4, np = 4^3
        xs = [1.0 * (i - 1) for k in 1:m for j in 1:m for i in 1:m]
        ys = [1.0 * (j - 1) for k in 1:m for j in 1:m for i in 1:m]
        zs = [1.0 * (k - 1) for k in 1:m for j in 1:m for i in 1:m]
        g3 = GD.UnstructuredGrid(cart, (xs, ys, zs), trues(np); k = 6, areas = ones(np))
        p3 = O.gradient_plan(g3)
        Test.@test O.ncomponents(p3) == 3
        α, β, γ = 2.0, -3.0, 0.5
        f3 = α .* xs .+ β .* ys .+ γ .* zs .+ 7.0
        u1 = zeros(np); u2 = zeros(np); u3 = zeros(np)
        O.gradient!(u1, u2, u3, f3, p3)
        # Interior nodes, whose six nearest neighbours span all three directions.
        lin = LinearIndices((m, m, m))
        int3 = [lin[i, j, k] for k in 2:(m - 1), j in 2:(m - 1), i in 2:(m - 1)]
        Test.@test all(abs(u1[t] - α) < 1e-10 && abs(u2[t] - β) < 1e-10 && abs(u3[t] - γ) < 1e-10
                       for t in int3)

        # Rank deficiency in 3-D: every neighbour in one plane leaves the normal component
        # undetermined by the data, and the pseudo-inverse zeroes it.
        xp = [1.0 * i for j in 1:4 for i in 1:4]
        yp = [1.0 * j for j in 1:4 for i in 1:4]
        gp = GD.UnstructuredGrid(cart, (xp, yp, zeros(16)), trues(16); k = 4, areas = ones(16))
        w1 = zeros(16); w2 = zeros(16); w3 = zeros(16)
        O.gradient!(w1, w2, w3, 3.0 .* xp .- 2.0 .* yp, O.gradient_plan(gp))
        lin2 = LinearIndices((4, 4))
        Test.@test all(abs(w1[lin2[i, j]] - 3.0) < 1e-10 && abs(w2[lin2[i, j]] + 2.0) < 1e-10
                       for j in 2:3, i in 2:3)
        Test.@test all(iszero, w3)
    end

    # A fourth coordinate has no fixed-size solve behind it, and is refused at construction.
    let g4 = GD.UnstructuredGrid(cart, (rand(8), rand(8), rand(8), rand(8)), trues(8); k = 4,
                                 areas = ones(8))
        Test.@test_throws ArgumentError O.gradient_plan(g4)
    end

    # A node set given positions and areas alone has no adjacency, so no cell has a neighbour to fit
    # from and the plan is refused. Passing the adjacency as `conn` gives the grid's own gradient.
    let xs = [0.0, 1.0, 0.0, 1.0, 0.5, 2.0], ys = [0.0, 0.0, 1.0, 1.0, 0.5, 1.5]
        bare = GD.UnstructuredGrid(cart, (xs, ys), ones(6))
        knn = GD.UnstructuredGrid(cart, (xs, ys), trues(6); k = 3, areas = ones(6))
        Test.@test_throws ArgumentError O.gradient_plan(bare)
        Test.@test O.gradient_plan(bare; conn = C.build_connectivity(knn)).coeffs ==
                   O.gradient_plan(knn).coeffs
        # With every cell masked there is nothing to fit, which is not an error.
        dark = GD.UnstructuredGrid(cart, (xs, ys), ones(6), falses(6))
        Test.@test isempty(O.gradient_plan(dark).nbr)
    end

    # A formula layout's adjacency is arithmetic, and the plan is still built.
    let sphere = GE.SphericalGeometry()
        for g in (GD.HEALPixGrid(4), GD.CubedSphereGrid(8), GD.IcosahedralGrid(4),
                  GD.YinYangGrid(16, 8),
                  GD.RingGrid(FG.SphericalSampling.OctahedralGaussianSampling(8)))
            p = O.gradient_plan(g)
            # Every cell keeps the neighbours its formula gives it.
            Test.@test minimum(p.ptr[i + 1] - p.ptr[i] for i in 1:length(GD.mask(g))) ≥ 2
            # The keyword names offsets this architecture has none of, so it selects nothing.
            Test.@test O.gradient_plan(g; stencil = FG.Stencils.Moore(2)).nbr == p.nbr
        end
        # `sin φ` has gradient magnitude `cos(φ)/R`, recovered to the grid's own resolution.
        g = GD.HEALPixGrid(32)
        n = length(GD.mask(g))
        lat = [GD.coords(g, i).φ for i in 1:n]
        q1 = zeros(n); q2 = zeros(n)
        O.gradient!(q1, q2, sin.(lat), O.gradient_plan(g))
        R = GE.radius(sphere)
        Test.@test maximum(abs.(sqrt.(q1 .^ 2 .+ q2 .^ 2) .* R .- cos.(lat))) < 0.02
    end
end

Test.@testset "The symmetric pseudo-inverse satisfies the Moore-Penrose conditions" begin
    O = FG.Operators
    # `_sympinv3` against the four Moore–Penrose identities, at every rank.
    mul3(A, B) = ntuple(i -> ntuple(j -> sum(A[i][k] * B[k][j] for k in 1:3), Val(3)), Val(3))
    transp(A) = ntuple(i -> ntuple(j -> A[j][i], Val(3)), Val(3))
    maxdiff(A, B) = maximum(abs(A[i][j] - B[i][j]) for i in 1:3, j in 1:3)
    maxabs(A) = maximum(abs(A[i][j]) for i in 1:3, j in 1:3)
    gram(vs) = ntuple(i -> ntuple(j -> sum(v[i] * v[j] for v in vs; init = 0.0), Val(3)), Val(3))

    for (nm, vs) in (
        ("rank 3", [(1.0, 0.0, 0.0), (0.3, 1.0, 0.0), (-0.2, 0.4, 1.0), (1.0, 1.0, 1.0)]),
        ("rank 3 skew", [(3.0, 1.0, 0.2), (0.1, 2.0, -0.4), (0.0, 0.05, 0.5)]),
        ("rank 3 near-degenerate", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0 + 1e-9)]),
        ("rank 2 coplanar", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]),
        ("rank 2 tilted", [(1.0, 1.0, 1.0), (1.0, -1.0, 0.0)]),
        ("rank 1 collinear", [(0.6, 0.8, 0.0), (1.2, 1.6, 0.0)]),
        ("rank 0", Tuple{Float64,Float64,Float64}[]),
        ("isotropic", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]),
    )
        A = gram(vs)
        tol = max(A[1][1] + A[2][2] + A[3][3], 1.0) * sqrt(eps(Float64))
        P = O._sympinv3(A, tol)
        AP = mul3(A, P)
        PA = mul3(P, A)
        Test.@test maxdiff(mul3(AP, A), A) < 1e-9 * max(maxabs(A), 1.0)          # A A⁺ A = A
        Test.@test maxdiff(mul3(PA, P), P) < 1e-9 * max(maxabs(P), 1.0)          # A⁺ A A⁺ = A⁺
        Test.@test maxdiff(AP, transp(AP)) < 1e-12                               # A A⁺ symmetric
        Test.@test maxdiff(PA, transp(PA)) < 1e-12                               # A⁺ A symmetric
        # A⁺ annihilates the null space, leaving the unspanned directions at zero.
        for v in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            Av = ntuple(i -> sum(A[i][j] * v[j] for j in 1:3), Val(3))
            if sqrt(sum(abs2, Av)) < 1e-12
                Pv = ntuple(i -> sum(P[i][j] * v[j] for j in 1:3), Val(3))
                Test.@test sqrt(sum(abs2, Pv)) < 1e-9
            end
        end
    end
    # `D = 2`: the 2×2 closed form inverts a full-rank matrix, symmetrically.
    let A = ((4.0, 1.0), (1.0, 9.0))
        P = O._sympinv(A, 1e-12)
        Test.@test P[1][2] == P[2][1]
        Test.@test all(abs(sum(A[i][k] * P[k][j] for k in 1:2) - (i == j ? 1.0 : 0.0)) < 1e-12
                       for i in 1:2, j in 1:2)
    end
end

Test.@testset "Staggering, point location and interpolation weights" begin
    D = FG.Discretization
    O = FG.Operators
    A = FG.Axes
    # Faces bracket the centres, one more of them, at the midpoints.
    x = [0.0, 1.0, 3.0, 6.0]
    f = D.faces(x)
    Test.@test length(f) == length(x) + 1
    Test.@test f[2] ≈ 0.5 && f[3] ≈ 2.0 && f[4] ≈ 4.5
    Test.@test f[1] ≈ -0.5 && f[5] ≈ 7.5          # extrapolated a half-cell beyond the ends
    # `centers` inverts `faces` on a uniform axis.
    Test.@test length(D.centers(f)) == length(x)
    u = A.UniformAxis(0.0, 0.5, 6)
    Test.@test A.isuniform(D.faces(u))            # a uniform axis keeps its guarantee
    Test.@test collect(D.faces(u)) ≈ D.faces(collect(u))
    Test.@test collect(D.centers(D.faces(u))) ≈ collect(u)
    Test.@test length(D.nodes(u, D.Face())) == length(u) + 1
    # The faces of a uniform axis are exactly the cell boundaries: their gaps are the spacing.
    Test.@test A.spacing(D.faces(u)) === A.spacing(u)

    # On a wrapping axis the first face sits midway across the seam and the last is it one period on, so
    # the face gaps are the cell widths there too, whichever way the axis is stored.
    P = 10.0
    for xp in ([0.5, 1.0, 3.0, 6.0, 8.5], reverse([0.5, 1.0, 3.0, 6.0, 8.5]))
        fp = D.faces(xp, P)
        s = sign(xp[end] - xp[1])
        Test.@test fp[1] ≈ ((xp[end] - s * P) + xp[1]) / 2 && fp[end] ≈ fp[1] + s * P
        Test.@test abs.(diff(fp)) ≈ D.cell_widths(xp, P)
    end
    Test.@test D.faces([0.5, 1.0, 3.0], nothing) == D.faces([0.5, 1.0, 3.0])
    # A periodic axis's seam is one step wide by its type, so its faces are a uniform axis; a range given
    # a period takes the per-cell form, whatever seam the period leaves.
    let pa = A.PeriodicAxis(0.0, 5.0, 5)
        Test.@test A.isuniform(D.faces(pa, 5.0))
        Test.@test collect(D.faces(pa, 5.0)) ≈ D.faces(collect(0.0:1.0:4.0), 5.0)
        Test.@test D.faces(0.0:1.0:4.0, 5.0) ≈ D.faces(collect(0.0:1.0:4.0), 5.0)
        Test.@test_throws ArgumentError D.faces(pa, 7.0)
    end
    Test.@test D.faces(0.0:1.0:4.0, 7.0) ≈ D.faces(collect(0.0:1.0:4.0), 7.0)
    sgp = FG.Grids.StaggeredGrid(FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), [0.5, 1.0, 3.0, 6.0, 8.5],
                                                         [0.0, 1.0]; periodic = (true, false), period = (P, 0.0)))
    Test.@test collect(sgp.faces[1]) ≈ D.faces([0.5, 1.0, 3.0, 6.0, 8.5], P)[1:5]

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

    # Both are searches: the elements a query on a stretched axis reads grow as `log n`. `locate` reads
    # two faces per bisection step plus the two end faces.
    for pow in (6, 18)
        n = 1 << pow
        c = CountingAxis(collect(range(0.0, 1.0; length = n)))
        D.locate(c, 0.37); D.nearest_index(c, 0.37)          # warm, then count
        Test.@test reads(() -> D.locate(c, 0.37), c) ≤ 8 * pow + 16
        Test.@test reads(() -> D.nearest_index(c, 0.37), c) ≤ 8 * pow + 16
    end

    # Linear weights sum to 1 and reproduce a linear function exactly.
    for t in range(0.0, 2.5; length = 37)
        i, w = D.interpolation_weights(v, t)
        Test.@test sum(w) ≈ 1.0
        Test.@test w[1] * v[i] + w[2] * v[i+1] ≈ t atol = 1e-12
    end
    # Outside the axis it clamps to the end sample.
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

Test.@testset "Per-index gaps and widths are public, exact and free of allocation" begin
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    GE = FG.Geometry
    cart = FG.Geometry.CartesianGeometry{Float64}()
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    vec1 = cumsum([0.0, 1.0, 0.3, 2.5, 0.7, 4.0])
    rng1 = range(0.0, 1.0; length = 8)

    # `cell_width` is the gap between a cell's two faces, answered without materializing the axis
    # the way `faces` does.
    for x in (vec1, reverse(vec1), collect(rng1), [3.0])
        f = D.faces(x)
        for i in eachindex(x)
            Test.@test D.cell_width(x, i) ≈ abs(f[i+1] - f[i]) rtol = 1e-14
        end
    end

    # Gaps keep their sign, so a stencil carries the index-versus-coordinate direction.
    Test.@test all(D.local_spacing(vec1, i) == (vec1[i] - vec1[i-1], vec1[i+1] - vec1[i])
                   for i in 2:(length(vec1) - 1))
    # Reversing the axis moves cell `i` to `n+1-i`, where the gaps come back negated and swapped
    # — the same two neighbours, from the other side — while the width, being a length, does not.
    let n = length(vec1), rv = reverse(vec1)
        for i in 2:(n - 1)
            h_m, h_p = D.local_spacing(vec1, i)
            Test.@test D.local_spacing(rv, n + 1 - i) == (-h_p, -h_m)
            Test.@test D.cell_width(rv, n + 1 - i) == D.cell_width(vec1, i)
        end
    end
    # A bounded end has no gap on the outside; a period supplies the wrapped one.
    Test.@test D.local_spacing(vec1, 1)[1] == 0.0
    Test.@test D.local_spacing(vec1, length(vec1))[2] == 0.0
    λ = collect(range(0.0, 2π * (1 - 1 / 16); length = 16))
    Test.@test D.local_spacing(λ, 16, 2π)[2] ≈ 2π / 16
    Test.@test D.local_spacing(λ, 1, 2π)[1] ≈ 2π / 16
    Test.@test D.local_spacing(reverse(λ), 1, 2π)[1] ≈ -2π / 16   # orientation follows the axis

    # The bulk form is the per-index one at every index, and costs nothing on a uniform axis.
    for (x, p) in ((vec1, nothing), (collect(rng1), nothing), (λ, 2π), (rng1, nothing))
        Test.@test collect(D.cell_widths(x, p)) ≈ [D.cell_width(x, i, p) for i in eachindex(x)]
    end
    # A periodic axis wraps at its own period and no other.
    let pa = FG.Axes.PeriodicAxis(0.0, 2π, 16)
        Test.@test collect(D.cell_widths(pa, 2π)) ≈ D.cell_widths(λ, 2π)
        Test.@test_throws ArgumentError D.cell_widths(pa, π)
    end

    # The grid forms take the period from the grid.
    grids = (
        ("range x range",   GD.StructuredGrid(cart, rng1, rng1)),
        ("range x Vector",  GD.StructuredGrid(cart, rng1, vec1)),
        ("Vector x range",  GD.StructuredGrid(cart, vec1, rng1)),
        ("Vector x Vector", GD.StructuredGrid(cart, vec1, vec1)),
        ("3D mixed",        GD.StructuredGrid(cart, rng1, vec1, collect(0.0:2.0))),
        ("periodic sphere", GD.StructuredGrid(sph, λ, collect(range(-1.0, 1.0; length = 9));
                                              periodic = true)),
        ("PeriodicAxis sphere", GD.StructuredGrid(sph, FG.Axes.PeriodicAxis(0.0, 2π, 16),
                                              collect(range(-1.0, 1.0; length = 9)))),
    )
    for (nm, g) in grids, d in 1:length(GD.coordinates(g))
        x = GD.coordinates(g, d)
        p = GD.isperiodic(g, d) ? GD.period(g, d) : nothing
        for i in 1:length(x)
            Test.@test GD.local_spacing(g, d, i) == D.local_spacing(x, i, p)
            Test.@test GD.cell_width(g, d, i) == D.cell_width(x, i, p)
        end
        Test.@test collect(GD.cell_widths(g, d)) == [D.cell_width(x, i, p) for i in 1:length(x)]
    end
    Test.@test_throws ArgumentError GD.local_spacing(first(grids)[2], 3, 1)

    # The three-point first derivative assembled from the gaps is exact for a quadratic, in either
    # storage order.
    fq(x) = 3x^2 - 2x + 5
    dfq(x) = 6x - 2
    for x in (vec1, reverse(vec1))
        for i in 2:(length(x) - 1)
            h_m, h_p = D.local_spacing(x, i)
            Test.@test GE.nonuniform_first_derivative(fq(x[i-1]), fq(x[i]), fq(x[i+1]), h_m, h_p) ≈
                       dfq(x[i]) atol = 1e-12
        end
    end
    # Across a periodic seam the gap wraps. The centred difference of `sin` is `cos λ·sin(h)/h`, so the
    # error is `|cos λ|(1 − sin h/h)`, largest at the seam cell `λ = 0`.
    np = 32
    λ2 = collect(range(0.0, 2π * (1 - 1 / np); length = np))
    gper = GD.StructuredGrid(cart, λ2, [0.0]; periodic = (true, false), period = (2π, 0.0))
    errs = map(1:np) do i
        h_m, h_p = GD.local_spacing(gper, 1, i)
        return abs(GE.nonuniform_first_derivative(sin(λ2[mod1(i - 1, np)]), sin(λ2[i]),
                                                  sin(λ2[mod1(i + 1, np)]), h_m, h_p) - cos(λ2[i]))
    end
    h = 2π / np
    Test.@test errs[1] ≈ 1 - sin(h) / h rtol = 1e-10
    Test.@test errs[1] ≈ maximum(errs) rtol = 1e-10
end

Test.@testset "Fornberg finite-difference weights are exact to their stated order" begin
    D = FG.Discretization
    O = FG.Operators
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

    # The defining property, on nodes at arbitrary spacing: with m nodes the weights differentiate
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

    # The axis form keeps the node count at a boundary by shifting inward, holding the accuracy
    # order everywhere.
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

Test.@testset "A stencil at a mask edge can degrade instead of blanking" begin
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    geo = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1.0:6.0)
    msk = trues(7, 1); msk[4, 1] = false
    grid = GD.StructuredGrid(geo, x, [0.0], msk)
    nomask = GD.StructuredGrid(geo, x, [0.0])
    f = reshape(collect(0.0:6.0), 7, 1)          # f = x, so df/dx == 1 wherever it is defined
    active = [i for i in 1:7 if msk[i, 1]]

    # Blanking loses every active cell within `nodes - 1` of the mask; at five nodes it loses the
    # whole axis. The other two policies fill those cells instead.
    for nodes in (2, 3, 5)
        o = zeros(7, 1)
        O.apply_stencil!(o, f, grid, 1; order = 1, nodes = nodes)
        Test.@test iszero(o[3, 1])               # active, blanked by its masked neighbour
        nodes == 5 && Test.@test all(iszero, o)
    end

    # Runs here are [1,3] and [5,7], so five nodes do not fit and only ReduceInRun can fill them.
    for nodes in (2, 3)
        o = zeros(7, 1)
        O.apply_stencil!(o, f, grid, 1; order = 1, nodes = nodes, policy = O.ShiftWithinRun())
        Test.@test all(isapprox.(o[active, 1], 1.0))
        Test.@test iszero(o[4, 1])
    end
    for nodes in (2, 3, 5)
        o = zeros(7, 1)
        O.apply_stencil!(o, f, grid, 1; order = 1, nodes = nodes, policy = O.ReduceInRun())
        Test.@test all(isapprox.(o[active, 1], 1.0))
        Test.@test iszero(o[4, 1])
    end

    # Without a mask every policy is the plain stencil, bit for bit.
    for nodes in (2, 3, 5), pol in (O.BlankMasked(), O.ShiftWithinRun(), O.ReduceInRun())
        a = zeros(7, 1); b = zeros(7, 1)
        O.apply_stencil!(a, f, nomask, 1; order = 1, nodes = nodes)
        O.apply_stencil!(b, f, nomask, 1; order = 1, nodes = nodes, policy = pol)
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
    O.apply_stencil!(a, fl, gn, 1; order = 1, nodes = 5)
    O.apply_stencil!(b, fl, gl, 1; order = 1, nodes = 5, policy = O.ShiftWithinRun())
    for i in 1:n
        abs(i - 30) > 4 && 2 < i < n - 1 && Test.@test a[i, 1] === b[i, 1]
    end
    Test.@test any(a[i, 1] != b[i, 1] for i in 26:34 if i != 30)

    # And the accuracy order survives the shift, converging at the interior rate.
    errs = Float64[]
    for m in (80, 160, 320)
        xx = collect(range(0.0, 1.0; length = m))
        mm = trues(m, 1); mm[m ÷ 2, 1] = false
        gg = GD.StructuredGrid(geo, xx, [0.0], mm)
        ff = reshape([sin(3xi) for xi in xx], m, 1)
        oo = zeros(m, 1)
        O.apply_stencil!(oo, ff, gg, 1; order = 1, nodes = 5, policy = O.ShiftWithinRun())
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
    O.apply_stencil!(pa, fp, gfull, 1; order = 1, nodes = 5)
    O.apply_stencil!(pb, fp, gfull, 1; order = 1, nodes = 5, policy = O.ShiftWithinRun())
    Test.@test pa == pb
    seam = Float64[]
    for m in (64, 128, 256)
        xx = collect(range(0.0, m - 1.0; length = m)); Lm = Float64(m)
        mm = trues(m, 1); mm[m ÷ 2, 1] = false
        gg = GD.StructuredGrid(geo, xx, [0.0], mm; periodic = (true, false), period = (Lm, 0.0))
        ff = reshape([sin(2π * xi / Lm) for xi in xx], m, 1)
        oo = zeros(m, 1)
        O.apply_stencil!(oo, ff, gg, 1; order = 1, nodes = 5, policy = O.ShiftWithinRun())
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
        O.apply_stencil!(o2, f2, g2, 1; order = 1, nodes = k2, policy = O.ShiftWithinRun())
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
    O.apply_stencil!(os, fs, gs, 1; order = 1, nodes = 5, policy = O.ShiftWithinRun())
    O.apply_stencil!(orr, fs, gs, 1; order = 1, nodes = 5, policy = O.ReduceInRun())
    Test.@test all(iszero, os)
    Test.@test all(isapprox.(orr[[1, 2, 3, 5, 6, 7], 1], 1.0))
    Test.@test iszero(orr[9, 1])                 # a run of one holds no first derivative

    # The end of the axis bounds a window as the end of a run does, so under `ReduceInRun` `nodes`
    # is a ceiling there too. These are ordinary degenerate grids: a single-latitude strip, a
    # two-level column, a one-cell channel.
    let xa = collect(0.0:5.0)
        # An axis with fewer samples than `nodes`: use what there is, and stay exact for the
        # degree the reduced window still supports.
        for (ny, want) in ((2, 5.0), (3, 5.0))
            ys = collect(range(0.0, 1.0; length = ny))
            g2 = GD.StructuredGrid(geo, xa, ys)
            f2 = [xi + 5yi for xi in xa, yi in ys]
            o2 = zeros(length(xa), ny)
            O.apply_stencil!(o2, f2, g2, 2; order = 1, nodes = 5, policy = O.ReduceInRun(),
                             masked = NaN)
            Test.@test all(isapprox.(o2, want))
        end
        # Fewer than `order + 1` samples: no derivative of that order exists anywhere on the axis.
        g1 = GD.StructuredGrid(geo, xa, [3.0])
        f1 = reshape([2xi for xi in xa], length(xa), 1)
        o1 = zeros(length(xa), 1)
        O.apply_stencil!(o1, f1, g1, 2; order = 1, nodes = 3, policy = O.ReduceInRun(),
                         masked = NaN)
        Test.@test all(isnan, o1)
        # …while the other direction of the very same grid is untouched by any of it.
        od = zeros(length(xa), 1)
        O.apply_stencil!(od, f1, g1, 1; order = 1, nodes = 3, policy = O.ReduceInRun())
        Test.@test all(isapprox.(od, 2.0))
        # The policies that do not claim to degrade keep the error.
        for pol in (O.BlankMasked(), O.ShiftWithinRun())
            Test.@test_throws ArgumentError O.apply_stencil!(zeros(length(xa), 1), f1, g1, 2;
                                                             order = 1, nodes = 3, policy = pol)
        end
    end

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

# Every window a cell can get spans at least `order + 1` samples of its own active run, so a field that is
# a polynomial of degree `order` along the differenced direction is differenced exactly wherever the
# plan writes, and the cells it writes are the active ones whose run holds `order + 1` samples.
Test.@testset "A masked stencil plan resolves each cell's window once, on any backend" begin
    O = FG.Operators
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    nx, ny = 20, 11
    x = collect(cumsum(1.0 .+ 0.4 .* sin.(range(0, 3π; length = nx))))
    y = collect(cumsum(0.8 .+ 0.3 .* cos.(range(0, 2π; length = ny))))
    mk = trues(nx, ny)
    mk[6:8, 3] .= false; mk[12, 2:9] .= false; mk[17, 7] = false; mk[2, 5] = false; mk[4:5, 5] .= false
    g = GD.StructuredGrid(cart, x, y, mk)
    yp = collect(range(0.0, 2π * (1 - 1 / ny); length = ny))
    gp = GD.StructuredGrid(cart, x, yp, mk; periodic = (false, true), period = (0.0, 2π))
    # The run of active samples through cell `I` along `d`, walked without the package.
    function runlen(m, I, d, wrap)
        n = size(m, d)
        at(j) = m[ntuple(e -> e == d ? mod1(j, n) : I[e], 2)...]
        back = 0
        while back < n - 1 && (wrap || I[d] - back - 1 ≥ 1) && at(I[d] - back - 1)
            back += 1
        end
        fwd = 0
        while back + fwd + 1 < n && (wrap || I[d] + fwd + 1 ≤ n) && at(I[d] + fwd + 1)
            fwd += 1
        end
        return back + fwd + 1
    end
    # `p` of degree `ord` along `d`, with `dp` its `ord`-th derivative; the other direction adds a term the
    # derivative along `d` does not see.
    poly(ord) = ord == 1 ? (s -> 1.5 + 2.0s, 2.0) : (s -> 0.7 - 0.3s^2, -0.6)
    for d in 1:2, (ord, nodes) in ((1, 3), (1, 5), (2, 5)), pol in (O.ShiftWithinRun(), O.ReduceInRun())
        p, dp = poly(ord)
        f = [d == 1 ? p(xi) + sin(yj) : p(yj) + sin(xi) for xi in x, yj in y]
        plan = O.masked_stencil_plan(g, d; order = ord, nodes = nodes, policy = pol)
        o = fill(NaN, nx, ny)
        O.apply_stencil!(o, f, plan; masked = NaN)
        k = pol isa O.ReduceInRun ? ord + 1 : nodes
        written = [mk[I] && runlen(mk, Tuple(I), d, false) ≥ k for I in CartesianIndices(mk)]
        Test.@test (plan.row .!= 0) == written
        Test.@test all(isnan, o[.!written])
        Test.@test maximum(abs, o[written] .- dp) < 1e-10
        # The plan-free apply rebuilds each window per cell on the host and takes a plan on a device;
        # both are the held plan's answer.
        oh = fill(NaN, nx, ny)
        od1 = fill(NaN, nx, ny)
        ot = fill(NaN, nx, ny)
        O.apply_stencil!(oh, f, g, d; order = ord, nodes = nodes, masked = NaN, policy = pol)
        O.apply_stencil!(od1, f, g, d; order = ord, nodes = nodes, masked = NaN, policy = pol,
                         execution = CB.GPUBackend(KernelAbstractions.CPU()))
        O.apply_stencil!(ot, f, g, d; order = ord, nodes = nodes, masked = NaN, policy = pol,
                         execution = FG.Execution.BalancedThreads(3))
        Test.@test all(isequal.(oh, o)) && all(isequal.(od1, o)) && all(isequal.(ot, o))
        # A trailing batch axis reads the same row per cell; the device body is the host's.
        fb = cat(f, 2 .* f, -f; dims = 3)
        ob = fill(NaN, nx, ny, 3)
        O.apply_stencil!(ob, fb, plan; masked = NaN)
        Test.@test all(isequal.(ob[:, :, 2], 2 .* o))
        od = fill(NaN, nx, ny, 3)
        O.apply_stencil!(od, fb, plan; masked = NaN, execution = CB.GPUBackend(KernelAbstractions.CPU()))
        Test.@test all(isequal.(od, ob))
    end
    # A periodic direction's runs continue across the seam, each cell counted once: column 12's run of
    # three wraps from row 10 to row 1.
    fy = [sin(yj) + xi for xi in x, yj in yp]
    for pol in (O.ShiftWithinRun(), O.ReduceInRun())
        plan = O.masked_stencil_plan(gp, 2; order = 1, nodes = 5, policy = pol)
        k = pol isa O.ReduceInRun ? 2 : 5
        written = [mk[I] && runlen(mk, Tuple(I), 2, true) ≥ k for I in CartesianIndices(mk)]
        Test.@test (plan.row .!= 0) == written
        o = fill(NaN, nx, ny)
        O.apply_stencil!(o, fy, gp, 2; order = 1, nodes = 5, masked = NaN, policy = pol)
        Test.@test .!isnan.(o) == written
    end
    # Away from the mask a cell keeps the axis table's own row, bit for bit.
    let plan = O.masked_stencil_plan(g, 1; order = 1, nodes = 3), f = [sin(xi) * cos(yj) for xi in x, yj in y]
        a = zeros(nx, ny)
        b = zeros(nx, ny)
        O.apply_stencil!(a, f, plan)
        O.apply_stencil!(b, f, GD.StructuredGrid(cart, x, y), 1; order = 1, nodes = 3)
        interior = [plan.row[I] == I[1] for I in CartesianIndices(mk)]
        Test.@test count(interior) > 100
        Test.@test a[interior] == b[interior]
        Test.@test _alloc(q_mplan!, a, f, plan) == 0
    end
    # On a sphere the derivative along λ of `a + bλ` is `b / (R cos φ)` wherever it is written.
    let R = 6.371e6, lam = collect(range(0.0, 1.4; length = 15)), phi = collect(range(-1.0, 1.0; length = 9)),
        ms = trues(15, 9)
        ms[5:6, 4] .= false
        gs = GD.StructuredGrid(FG.Geometry.SphericalGeometry(R), lam, phi, ms)
        plan = O.masked_stencil_plan(gs, 1; order = 1, nodes = 5)
        o = fill(NaN, 15, 9)
        O.derivative!(o, [0.4 + 3.0 * a for a in lam, _ in phi], gs, plan; masked = NaN)
        w = plan.row .!= 0
        Test.@test count(w) == count(ms)
        Test.@test maximum(abs, (o .- [3.0 / (R * cos(p)) for _ in lam, p in phi])[w]) < 1e-12 / R
        # Rows resolved for one mask refuse a grid with another, an equal copy of the mask included.
        Test.@test O.derivative!(similar(o), o, GD.StructuredGrid(FG.Geometry.SphericalGeometry(R), lam, phi, copy(ms)),
                                 plan; masked = NaN) isa Matrix
        m2 = copy(ms)
        m2[9, 7] = false
        for other in (GD.StructuredGrid(FG.Geometry.SphericalGeometry(R), lam, phi, m2),
                      GD.StructuredGrid(FG.Geometry.SphericalGeometry(R), lam, phi))
            Test.@test_throws ArgumentError O.derivative!(similar(o), o, other, plan)
        end
    end
end

# The `p`-th derivative is with respect to distance. A polynomial of degree below the node count in the
# distance itself is differenced exactly: on a sphere distance is `R cosφ·λ` and `R·φ`, on a spheroid's
# latitude it is the meridian distance plus `h·φ`, here integrated in BigFloat independently of the package.
Test.@testset "derivative! of any order is with respect to distance" begin
    O = FG.Operators
    GD = FG.Grids
    GE = FG.Geometry
    D = FG.Discretization
    R = 6.371e6
    lam = collect(range(0.2, 1.4; length = 15))
    phi = collect(range(-1.1, 1.2; length = 13))
    gs = GD.StructuredGrid(GE.SphericalGeometry(R), lam, phi)
    o = zeros(15, 13)
    for (ord, nodes) in ((2, 3), (3, 4))
        O.derivative!(o, [a^ord for a in lam, _ in phi], gs, 1; order = ord, nodes = nodes)
        Test.@test maximum(abs, o .* [(R * cos(p))^ord for _ in lam, p in phi] ./ factorial(ord) .- 1) < 1e-10
        O.derivative!(o, [p^ord for _ in lam, p in phi], gs, 2; order = ord, nodes = nodes)
        Test.@test maximum(abs, o .* R^ord ./ factorial(ord) .- 1) < 1e-10
    end

    spd = GE.SpheroidGeometry()
    function merid(φ)
        a = big(spd.a)
        e² = big(spd.f) * (2 - big(spd.f))
        M(t) = a * (1 - e²) / (1 - e² * sin(t)^2)^(big(3) / 2)
        n = 4000
        h = big(φ) / n
        return Float64(h / 3 * (M(big(0)) + M(big(φ)) + sum((isodd(j) ? 4 : 2) * M(j * h) for j in 1:(n - 1))))
    end
    # Polynomials in `u = (distance − s₀)/L`, so the values stay of order one.
    s = merid.(phi)
    L = 1.0e6
    u(x) = (x - s[7]) / L
    hs = [0.0, 5.0e3, 2.0e4]
    g2 = GD.StructuredGrid(spd, lam, phi)
    g3 = GD.StructuredGrid(spd, lam, phi, hs)
    for (nodes, F, want) in ((3, x -> x^2, _ -> 2.0), (4, x -> x^3, x -> 6x))
        O.derivative!(o, [F(u(s[j])) for _ in lam, j in eachindex(phi)], g2, 2; order = 2, nodes = nodes)
        Test.@test maximum(abs, o .* L^2 .- [want(u(s[j])) for _ in lam, j in eachindex(phi)]) < 1e-9
        o3 = zeros(15, 13, 3)
        f3 = [F(u(s[j] + hs[k] * phi[j])) for _ in lam, j in eachindex(phi), k in eachindex(hs)]
        O.derivative!(o3, f3, g3, 2; order = 2, nodes = nodes)
        w3 = [want(u(s[j] + hs[k] * phi[j])) for _ in lam, j in eachindex(phi), k in eachindex(hs)]
        Test.@test maximum(abs, o3 .* L^2 .- w3) < 1e-9
    end
    # The third derivative over a mid-latitude band, where `M` changes along every window.
    let band = collect(0.75:0.01:0.85), sb = merid.(band), Lb = 1.0e5
        gb = GD.StructuredGrid(spd, lam, band)
        ob = zeros(15, 11)
        O.derivative!(ob, [((sb[j] - sb[1]) / Lb)^3 for _ in lam, j in eachindex(band)], gb, 2; order = 3, nodes = 4)
        Test.@test maximum(abs, ob .* Lb^3 .- 6) < 1e-9
    end
    # Longitude keeps the power of the scale factor: `λ²` has `2/(N cosφ)²`.
    O.derivative!(o, [a^2 for a in lam, _ in phi], g2, 1; order = 2, nodes = 3)
    Test.@test maximum(abs, o .* [GE.scale_factors(spd, (0.0, p))[1]^2 for _ in lam, p in phi] ./ 2 .- 1) < 1e-10

    # Every form reaches the same weights: plan-free, a held table or plan, the masked plan, a device.
    f = [s[j]^2 + sin(a) for a in lam, j in eachindex(phi)]
    ref = O.derivative!(zeros(15, 13), f, g2, 2; order = 2, nodes = 3)
    idx, wts = D.axis_stencils(g2, 2; order = 2, nodes = 3)
    mp = O.masked_stencil_plan(g2, 2; order = 2, nodes = 3)
    Test.@test O.derivative!(zeros(15, 13), f, g2, idx, wts, 2; order = 2) == ref
    Test.@test O.derivative!(zeros(15, 13), f, g2, D.stencil_plan(g2, 2; order = 2, nodes = 3), 2) == ref
    Test.@test O.derivative!(zeros(15, 13), f, g2, mp) == ref
    Test.@test O.derivative!(zeros(15, 13), f, g2, 2; order = 2, nodes = 3,
                             execution = CB.GPUBackend(KernelAbstractions.CPU())) == ref
    Test.@test _alloc(q_mderiv!, zeros(15, 13), f, g2, mp) == 0

    # On a masked grid the window shrinks inside the cell's run along φ, and a run of fewer than three
    # cells has no second derivative.
    mk = trues(15, 13)
    mk[4, 5:6] .= false; mk[9, 3] = false; mk[9, 5] = false; mk[12, 2:11] .= false
    gm = GD.StructuredGrid(spd, lam, phi, mk)
    om = fill(NaN, 15, 13)
    O.derivative!(om, [u(s[j])^2 for _ in lam, j in eachindex(phi)], gm, 2; order = 2, nodes = 5,
                  masked = NaN, policy = O.ReduceInRun())
    reach(I, t) = (k = 0; J = I + t; while checkbounds(Bool, mk, J) && mk[J]; k += 1; J += t; end; k)
    run3 = [mk[I] && reach(I, CartesianIndex(0, 1)) + reach(I, CartesianIndex(0, -1)) ≥ 2 for I in CartesianIndices(mk)]
    Test.@test .!isnan.(om) == run3
    Test.@test maximum(abs, om[run3] .* L^2 .- 2) < 1e-9

    # A coordinate-weight plan cannot serve that derivative, and a geometry that varies along a direction
    # it gives no distance for says so.
    Test.@test_throws ArgumentError O.derivative!(zeros(15, 13), f, g2,
                                                  O.masked_stencil_plan(phi, trues(15, 13), 2, 2, 3))
    gt = GD.StructuredGrid(TiltedSphere{Float64}(), lam, phi)
    Test.@test_throws ArgumentError O.derivative!(zeros(15, 13), f, gt, 1; order = 2, nodes = 3)
end

Test.@testset "A trailing batch axis is differenced in one pass, identically to a slice loop" begin
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    GE = FG.Geometry
    cart = GE.CartesianGeometry{Float64}()
    nx, ny, nb, nb2 = 14, 9, 4, 3
    x = collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = nx))))
    mk = trues(nx, ny); mk[5, 4] = false; mk[6, 4] = false; mk[11, 2] = false
    idx, w = D.axis_stencils(x, 1, 3)

    # Zero, one and two trailing batch axes, under every mask policy. The mask spans the grid's axes
    # only, and is indexed by the leading components of each cell index.
    for dims in ((nx, ny), (nx, ny, nb), (nx, ny, nb, nb2))
        f = reshape([sin(3i) * cos(2j) * (k + 1)
                     for i in 1:dims[1], j in 1:dims[2], k in 1:prod(dims[3:end]; init = 1)], dims)
        for pol in (O.BlankMasked(), O.ShiftWithinRun(), O.ReduceInRun())
            o = fill(NaN, dims); ref = fill(NaN, dims)
            O.apply_stencil!(o, f, x, 1; order = 1, nodes = 3, mask = mk, masked = NaN, policy = pol)
            for c in CartesianIndices(dims[3:end])
                O.apply_stencil!(view(ref, :, :, Tuple(c)...), view(f, :, :, Tuple(c)...), x, 1;
                                 order = 1, nodes = 3, mask = mk, masked = NaN, policy = pol)
            end
            Test.@test all(isequal(o[i], ref[i]) for i in eachindex(o))
        end
        # The device body walks the whole output, batch included, and agrees bit for bit.
        o1 = fill(NaN, dims); o2 = fill(NaN, dims)
        O.apply_stencil!(o1, f, x, idx, w, 1; mask = mk, masked = NaN)
        O.apply_stencil!(o2, f, x, idx, w, 1; mask = mk, masked = NaN,
                         execution = CB.GPUBackend(KernelAbstractions.CPU()))
        Test.@test all(isequal(o1[i], o2[i]) for i in eachindex(o1))
    end

    # `derivative!` divides by a spatial metric factor; the batched result equals the slice loop
    # exactly.
    sph = GE.SphericalGeometry(6.371e6)
    λ = collect(range(0.0, 2π * (1 - 1 / nx); length = nx))
    φ = collect(range(-1.1, 1.1; length = ny))
    for g in (GD.StructuredGrid(sph, λ, φ; periodic = true), GD.StructuredGrid(sph, λ, φ, mk; periodic = true),
              GD.StructuredGrid(cart, collect(1.0:nx), collect(1.0:ny)))
        f = [sin(2a) * cos(b) * (k + 1) for a in λ, b in φ, k in 1:nb]
        for dim in 1:2, pol in (O.BlankMasked(), O.ReduceInRun())
            o = fill(NaN, nx, ny, nb); ref = fill(NaN, nx, ny, nb)
            O.derivative!(o, f, g, dim; order = 1, nodes = 3, masked = NaN, policy = pol)
            for b in 1:nb
                O.derivative!(view(ref, :, :, b), view(f, :, :, b), g, dim;
                              order = 1, nodes = 3, masked = NaN, policy = pol)
            end
            Test.@test all(isequal(o[i], ref[i]) for i in eachindex(o))
        end
        let (i2, w2) = D.axis_stencils(g, 2; order = 1, nodes = 3)
            o = fill(NaN, nx, ny, nb); ref = fill(NaN, nx, ny, nb)
            O.derivative!(o, f, g, i2, w2, 2; order = 1, masked = NaN)
            for b in 1:nb
                O.derivative!(view(ref, :, :, b), view(f, :, :, b), g, i2, w2, 2;
                              order = 1, masked = NaN)
            end
            Test.@test all(isequal(o[i], ref[i]) for i in eachindex(o))
        end
    end

    # The metric floor masks a `Float32` pole row in every batch element.
    let λ32 = collect(range(0.0f0, 2π * (1 - 1 / 8); length = 8)), φ32 = Float32[-π/2, 0, π/2]
        g32 = GD.StructuredGrid(GE.SphericalGeometry(6.371f6), λ32, φ32; periodic = true)
        f32 = [sin(a) * b for a in λ32, b in φ32, _ in 1:2]
        o32 = fill(NaN32, 8, 3, 2)
        O.derivative!(o32, f32, g32, 1; order = 1, nodes = 3, masked = NaN32)
        Test.@test all(isnan, view(o32, :, 1, :)) && all(isnan, view(o32, :, 3, :))
        Test.@test !any(isnan, view(o32, :, 2, :))
    end

    # Implicit-by-rank must not swallow a real mistake.
    g2 = GD.StructuredGrid(cart, collect(1.0:nx), collect(1.0:ny))
    Test.@test_throws DimensionMismatch O.derivative!(zeros(nx, ny + 1, nb), zeros(nx, ny + 1, nb), g2, 1)
    Test.@test_throws ArgumentError O.derivative!(zeros(nx, ny, nb), zeros(nx, ny, nb), g2, 3)
    Test.@test_throws DimensionMismatch O.derivative!(zeros(nx, ny, nb), zeros(nx, ny, 2), g2, 1)
    Test.@test_throws DimensionMismatch O.apply_stencil!(zeros(nx, ny, nb), zeros(nx, ny, nb), x,
                                                         idx, w, 1; mask = trues(nx, ny + 1))
end

Test.@testset "A batch is evaluated at a coordinate, and gradients take one too" begin
    D = FG.Discretization
    O = FG.Operators
    C = FG.Connectivity
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    nx, ny, nb = 12, 9, 5
    x = collect(range(0.0, 11.0; length = nx)); y = collect(range(0.0, 8.0; length = ny))
    mk = trues(nx, ny); mk[4, 4] = false

    for g in (GD.StructuredGrid(cart, x, y), GD.StructuredGrid(cart, x, y, mk))
        f = [2.0xi - 3.0yj + 100.0b for xi in x, yj in y, b in 1:nb]
        for p in ((3.7, 2.4), (8.4, 6.1)), pol in (O.BlankMasked(), O.ReduceInRun())
            out = Vector{Float64}(undef, nb)
            O.interpolate!(out, f, g, p; masked = NaN, policy = pol)
            ref = [O.interpolate(view(f, :, :, b), g, p; masked = NaN, policy = pol) for b in 1:nb]
            Test.@test all(isequal(out[b], ref[b]) for b in 1:nb)
        end
    end
    Test.@test_throws DimensionMismatch O.interpolate!(
        Vector{Float64}(undef, 2), zeros(nx, ny, nb), GD.StructuredGrid(cart, x, y), (1.0, 1.0))

    # Off a rectilinear grid the neighbour set and the tangent-plane fit depend on the point alone,
    # so they are solved once, and the values still match a call per slice.
    n = 12
    X = [0.7i + 0.05j for i in 1:n, j in 1:n]; Y = [0.9j - 0.03i for i in 1:n, j in 1:n]
    cg = GD.CurvilinearGrid(cart, X, Y, trues(n, n); measure = fill(1.0, n, n))
    fb = [2.0X[i, j] - 3.0Y[i, j] + 50.0b for i in 1:n, j in 1:n, b in 1:4]
    vb = O.interpolate(fb, cg, (4.2, 5.1))
    Test.@test all(isapprox(vb[b], O.interpolate(view(fb, :, :, b), cg, (4.2, 5.1)); atol = 1e-10)
                   for b in 1:4)
    Test.@test all(abs(vb[b] - (2.0 * 4.2 - 3.0 * 5.1 + 50.0b)) < 1e-8 for b in 1:4)

    gu = healpix_node_grid(4)
    m = length(GD.mask(gu))
    fu = [Float64(i) + 1000.0b for i in 1:m, b in 1:3]
    vu = O.interpolate(fu, gu, (0.4, 0.1))
    Test.@test all(isapprox(vu[b], O.interpolate(view(fu, :, b), gu, (0.4, 0.1)); atol = 1e-8)
                   for b in 1:3)

    # A gradient plan is geometry, so one plan serves the whole batch.
    plan = O.gradient_plan(cg)
    G1 = zeros(n, n, 4); G2 = zeros(n, n, 4)
    O.gradient!(G1, G2, fb, plan)
    R1 = zeros(n, n, 4); R2 = zeros(n, n, 4)
    for b in 1:4
        O.gradient!(view(R1, :, :, b), view(R2, :, :, b), view(fb, :, :, b), plan)
    end
    Test.@test G1 == R1 && G2 == R2
    Test.@test all(abs(G1[i, j, b] - 2.0) < 1e-10 && abs(G2[i, j, b] + 3.0) < 1e-10
                   for i in 2:(n - 1), j in 2:(n - 1), b in 1:4)
    Test.@test_throws DimensionMismatch O.gradient!(zeros(n, n, 4), zeros(n, n, 4), zeros(n * 4 + 1), plan)
end

Test.@testset "The metric hoist is the geometry's claim, not an assumption" begin
    GE = FG.Geometry
    GD = FG.Grids
    O = FG.Operators
    D = FG.Discretization

    # The built-ins declare longitude; the abstracts declare nothing, so a subtype that writes its own
    # scale_factors inherits no claim about them.
    Test.@test GE.metric_invariant_directions(GE.SphericalGeometry()) == (1,)
    Test.@test GE.metric_invariant_directions(GE.SpheroidGeometry()) == (1,)
    Test.@test GE.metric_invariant_directions(GE.CartesianGeometry()) == ()
    Test.@test GE.metric_invariant_directions(OneSphere{Float64}()) == ()

    # A geometry whose h_λ varies with λ, so the factor cannot be hoisted out of a row:
    # `derivative!` divides each cell by its own.
    nx, ny = 16, 12
    λ = collect(range(0, 2π * (1 - 1 / nx); length = nx))
    φ = collect(range(-1.0, 1.0; length = ny))
    f = [sin(2li) * cos(pj) for li in λ, pj in φ]
    g = GD.StructuredGrid(TiltedSphere{Float64}(), λ, φ; periodic = (true, false),
                          period = (2π, 0.0))
    for dim in (1, 2)
        got = similar(f)
        O.derivative!(got, f, g, dim; order = 1, nodes = 5)
        idx, w = D.axis_stencils(dim == 1 ? λ : φ, 1, 5; period = dim == 1 ? 2π : nothing)
        raw = similar(f)
        O.apply_stencil!(raw, f, dim == 1 ? λ : φ, idx, w, dim)
        want = [raw[i, j] / GE.scale_factors(TiltedSphere{Float64}(), (λ[i], φ[j]))[dim]
                for i in 1:nx, j in 1:ny]
        Test.@test maximum(abs.(got .- want)) < 1e-12
    end
end

Test.@testset "An 8-node stencil keeps its unrolled inner loop" begin
    O = FG.Operators
    # k nodes span degree ≤ k−1: 7 miss a degree-7 polynomial, 8 and 9 reproduce it.
    xs = collect(range(0.0, 1.0; length = 24))
    p7 = @. xs^7 - 2xs^3 + 1
    d7 = @. 7xs^6 - 6xs^2
    errs = map((7, 8, 9)) do k
        o = similar(xs)
        O.apply_stencil!(o, p7, xs, 1; order = 1, nodes = k)
        maximum(abs.(o .- d7))
    end
    Test.@test errs[1] > 1e-8
    Test.@test errs[2] < 1e-10
    Test.@test errs[3] < 1e-10
end

Test.@testset "A held stencil plan is the axis's weights, register-resident where it can be" begin
    D = FG.Discretization
    O = FG.Operators
    A = FG.Axes

    # A periodic axis wraps at its own period and no other.
    Test.@test_throws ArgumentError D.stencil_plan(A.PeriodicAxis(0.0, 2π, 32), 1, 5; period = π)

    # A plan reads exactly the samples the table reads, and its weights are the same numbers to
    # round-off, the table computing each row from its own window.
    for (x, per) in ((0.0:0.5:10.0, nothing),
                     (A.PeriodicAxis(0.0, 2π, 32), 2π),
                     (collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = 40)))), nothing))
        for (ord, k) in ((1, 3), (1, 4), (1, 5), (2, 5), (3, 6))
            pl = D.stencil_plan(x, ord, k; period = per)
            idx, w = D.axis_stencils(x, ord, k; period = per)
            for j in 1:length(x)
                nodes, wts = D.plan_row(pl, j)
                Test.@test collect(nodes) == Int.(idx[j, :])
                Test.@test all(abs(wts[q] - w[j, q]) / max(abs(w[j, q]), 1) < 1e-11 for q in 1:k)
            end
        end
    end

    # A uniform plan's storage does not depend on the axis length.
    su = [Base.summarysize(D.stencil_plan(range(0.0, 1.0; length = m), 1, 5))
          for m in (64, 256, 1024, 4096)]
    Test.@test allequal(su)

    # Applying a plan agrees with applying the table, on every shape the sweep splits on: the
    # differenced direction contiguous or not, masked, batched, wrapping, and an even node count.
    for (nx, ny, nb) in ((17, 9, 0), (32, 32, 0), (9, 17, 3))
        for per in (nothing, 2π)
            ax = per === nothing ? range(0.0, 4.0; length = nx) : A.PeriodicAxis(0.0, 2π, nx)
            xs = collect(ax)
            ys = collect(range(0.0, 2.0; length = ny))
            fld = nb == 0 ? [sin(a) * cos(b) for a in xs, b in ys] :
                            [sin(a) * cos(b) + 0.1c for a in xs, b in ys, c in 1:nb]
            for k in (3, 4, 5), ord in (1, 2)
                (k ≥ ord + 1 && k ≤ nx) || continue
                pl = D.stencil_plan(ax, ord, k; period = per)
                idx, w = D.axis_stencils(ax, ord, k; period = per)
                o1 = fill(NaN, size(fld))
                o2 = fill(NaN, size(fld))
                O.apply_stencil!(o1, fld, pl, 1)
                O.apply_stencil!(o2, fld, idx, w, 1)
                Test.@test maximum(abs.(o1 .- o2)) / max(maximum(abs, o2), 1) < 1e-10
                msk = trues(nx, ny)
                msk[3, 2] = false
                msk[nx ÷ 2, ny ÷ 2] = false
                o3 = fill(NaN, size(fld))
                o4 = fill(NaN, size(fld))
                O.apply_stencil!(o3, fld, pl, 1; mask = msk, masked = -7.0)
                O.apply_stencil!(o4, fld, idx, w, 1; mask = msk, masked = -7.0)
                # the same cells are blanked, and the rest agree
                Test.@test (o3 .== -7.0) == (o4 .== -7.0)
                Test.@test maximum(abs.(o3 .- o4)) / max(maximum(abs, o4), 1) < 1e-10
            end
            aly = range(0.0, 2.0; length = ny)
            o1 = fill(NaN, size(fld))
            o2 = fill(NaN, size(fld))
            O.apply_stencil!(o1, fld, D.stencil_plan(aly, 1, 3), 2)
            O.apply_stencil!(o2, fld, D.axis_stencils(aly, 1, 3)..., 2)
            Test.@test maximum(abs.(o1 .- o2)) / max(maximum(abs, o2), 1) < 1e-10
        end
    end

    # K nodes span degree K−1, and the assertion sits at that degree.
    xu = collect(range(0.0, 1.0; length = 40))
    for k in (3, 4, 5, 6)
        pl = D.stencil_plan(range(0.0, 1.0; length = 40), 1, k)
        deg = k - 1
        o = similar(xu)
        O.apply_stencil!(o, [a^deg for a in xu], pl, 1)
        Test.@test maximum(abs.(o .- [deg * a^(deg - 1) for a in xu])) < 1e-9
    end

    # Applying a held plan allocates nothing.
    let xs = range(0.0, 1.0; length = 64), fld = [sin(20a) * b for a in xs, b in 1:64]
        out = similar(fld)
        for k in (3, 4, 5, 7)
            pl = D.stencil_plan(xs, 1, k)
            Test.@test _alloc(q_plan!, out, fld, pl, 1) == 0
            Test.@test _alloc(q_plan!, out, fld, pl, 2) == 0
        end
    end

    # Threads sweep the host nest over each chunk of cells, and a device launches one cell per work item
    # reading its own row from the plan; both give the serial value. A threaded apply allocates the same
    # at any axis length.
    let dev = CB.GPUBackend(KernelAbstractions.CPU())
        for (ax, per) in ((range(0.0, 4.0; length = 33), nothing), (A.PeriodicAxis(0.0, 2π, 33), 2π),
                          (collect(cumsum(1.0 .+ 0.3 .* sin.(1:33))), nothing))
            pl = D.stencil_plan(ax, 1, 5; period = per)
            fld = [sin(a) * b + 0.1c for a in collect(ax), b in 1:7, c in 1:2]
            msk = trues(33, 7)
            msk[5, 3] = false
            ref = fill(NaN, size(fld))
            O.apply_stencil!(ref, fld, pl, 1; mask = msk, masked = -7.0)
            for be in (CB.ThreadedBackend(), dev)
                o = fill(NaN, size(fld))
                O.apply_stencil!(o, fld, pl, 1; mask = msk, masked = -7.0, execution = be)
                Test.@test o == ref
            end
        end
        small = (zeros(64, 8), ones(64, 8), D.stencil_plan(range(0.0, 1.0; length = 64), 1, 5))
        large = (zeros(4096, 8), ones(4096, 8), D.stencil_plan(range(0.0, 1.0; length = 4096), 1, 5))
        thr = CB.ThreadedBackend()
        Test.@test _alloc(q_plan_be!, small[1], small[2], small[3], 1, thr) ==
                   _alloc(q_plan_be!, large[1], large[2], large[3], 1, thr)
    end

    # Along either direction, plain or written with a metric factor per run or per cell: 33·7·3 cells
    # split over the threads put chunk boundaries inside slabs and rows, and every piece is the serial
    # sweep's.
    let λ = A.PeriodicAxis(0.0, 2π, 33), φ = collect(range(-1.2, 1.2; length = 7))
        fld = [sin(a) * cos(b) + 0.1c for a in collect(λ), b in φ, c in 1:3]
        msk = trues(33, 7)
        msk[5, 3] = false
        for geo in (FG.Geometry.CartesianGeometry(), FG.Geometry.SphericalGeometry(), TiltedSphere{Float64}()),
            m in (nothing, msk), d in (1, 2)
            g = m === nothing ? FG.Grids.StructuredGrid(geo, λ, φ) : FG.Grids.StructuredGrid(geo, λ, φ, m)
            ref = fill(NaN, size(fld))
            o = fill(NaN, size(fld))
            O.derivative!(ref, fld, g, d; nodes = 3, masked = -7.0)
            O.derivative!(o, fld, g, d; nodes = 3, masked = -7.0, execution = CB.ThreadedBackend())
            Test.@test isequal(o, ref)
        end
    end

    # A plan describes one axis length, and a field of another length raises.
    Test.@test_throws DimensionMismatch O.apply_stencil!(
        zeros(8, 4), zeros(8, 4), D.stencil_plan(range(0.0, 1.0; length = 9), 1, 3), 1)
end

Test.@testset "derivative! writes each value with its metric factor, and a masked cell keeps masked" begin
    GE = FG.Geometry
    GD = FG.Grids
    O = FG.Operators
    D = FG.Discretization

    # The reference: the coordinate derivative times `inv(h)` at the cell's own point, and the sentinel
    # wherever the stencil wrote it or the metric degenerates. The sentinel is finite and nonzero, the
    # value a scaling after the store changes.
    s = -3.0
    function reference(raw, gg, geo, dim)
        floor_ = D.metric_floor(geo, dim)
        out = similar(raw)
        for I in CartesianIndices(raw)
            h = GE.scale_factors(geo, ntuple(d -> GD.coordinates(gg, d)[I[d]], ndims(gg)))[dim]
            out[I] = raw[I] == s || abs(h) ≤ floor_ ? s : raw[I] * inv(h)
        end
        return out
    end
    nkept = 0
    for geo in (GE.SphericalGeometry(), GE.SpheroidGeometry(), TiltedSphere{Float64}())
        for (nx, ny, nz, nb) in ((16, 12, 0, 0), (9, 7, 0, 3), (8, 6, 5, 0), (8, 6, 5, 2))
            geo isa TiltedSphere && nz > 0 && continue       # its scale factors are 2-D
            λ = FG.Axes.PeriodicAxis(0.0, 2π, nx)
            # A uniform and a stretched latitude axis, and one with a row on each pole, where the
            # longitude factor degenerates.
            for φ in (range(-1.4, 1.4; length = ny), [-1.4 + 2.8 * (t / (ny - 1))^1.3 for t in 0:(ny - 1)],
                      range(-π / 2, π / 2; length = ny))
                axs = nz == 0 ? (λ, φ) : (λ, φ, range(1.0, 2.0; length = nz))
                szs = nz == 0 ? (nx, ny) : (nx, ny, nz)
                fsz = nb == 0 ? szs : (szs..., nb)
                fld = reshape([sin(0.3k) + 0.1k for k in 1:prod(fsz)], fsz)
                msk = trues(szs)
                msk[CartesianIndices(szs)[2]] = false
                per = (periodic = (true, false, false)[1:length(axs)], period = (2π, 0.0, 0.0)[1:length(axs)])
                for gg in (GD.StructuredGrid(geo, axs...; per...),
                           GD.StructuredGrid(geo, axs...; mask = msk, per...)), dim in 1:length(axs)
                    mk = GD.mask(gg) isa GD.AllActive ? nothing : GD.mask(gg)
                    for pol in (O.BlankMasked(), O.ReduceInRun())
                        raw = fill(NaN, fsz); a = fill(NaN, fsz)
                        O.apply_stencil!(raw, fld, gg, dim; order = 1, nodes = 5, masked = s, policy = pol)
                        O.derivative!(a, fld, gg, dim; order = 1, nodes = 5, masked = s, policy = pol)
                        Test.@test isequal(a, reference(raw, gg, geo, dim))
                        nkept += count(==(s), raw)
                    end
                    # A held plan and a held table, each against its own coordinate derivative.
                    pl = D.stencil_plan(gg, dim; order = 1, nodes = 5)
                    raw = fill(NaN, fsz); c = fill(NaN, fsz)
                    O.apply_stencil!(raw, fld, pl, dim; mask = mk, masked = s)
                    O.derivative!(c, fld, gg, pl, dim; masked = s)
                    Test.@test isequal(c, reference(raw, gg, geo, dim))
                    ix, wt = D.axis_stencils(gg, dim; order = 1, nodes = 5)
                    raw = fill(NaN, fsz); e = fill(NaN, fsz)
                    O.apply_stencil!(raw, fld, gg, ix, wt, dim; order = 1, masked = s)
                    O.derivative!(e, fld, gg, ix, wt, dim; order = 1, masked = s)
                    Test.@test isequal(e, reference(raw, gg, geo, dim))
                    # The launched kernel writes the same values the host sweep does.
                    a = fill(NaN, fsz); k = fill(NaN, fsz)
                    O.derivative!(a, fld, gg, dim; order = 1, nodes = 5, masked = s)
                    O.derivative!(k, fld, gg, dim; order = 1, nodes = 5, masked = s,
                                  execution = CB.GPUBackend(KernelAbstractions.CPU()))
                    Test.@test isequal(k, a)
                end
            end
        end
    end
    Test.@test nkept > 0                 # the sentinel was written, so its survival was tested
end

Test.@testset "The ! forms write into the caller's buffers and allocate nothing" begin
    D = FG.Discretization

    for x in (collect(range(0.0, 4.0; length = 40)),
              collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = 40)))))
        # With a scratch, `axis_stencils!` allocates nothing.
        let sc = D.stencil_scratch(2, 5),
            gi = Matrix{Int}(undef, length(x), 5), gw = Matrix{Float64}(undef, length(x), 5)
            Test.@test _alloc(q_axst!, gi, gw, x, 2, 5, sc) == 0
            # a scratch too small for the table is refused at entry
            Test.@test_throws ArgumentError D.axis_stencils!(gi, gw, x, 2, 5;
                                                             scratch = D.stencil_scratch(1, 2))
        end

        f = D.faces(x)
        Test.@test _alloc(q_faces!, similar(x, length(x) + 1), x) == 0
        Test.@test _alloc(q_centers!, similar(x, length(f) - 1), f) == 0

        # `lagrange_weights` at every node count and on both sides of the axis.
        for k in (2, 3, 5)
            for v in (x[1], x[end], (x[3] + x[4]) / 2, x[1] - 1.0)
                _, gw = D.lagrange_weights(x, v, k)
                # A partition of unity. The weights alternate in sign and grow with the Lebesgue
                # constant once `v` sits outside the axis, so the cancellation error scales with `Σ|w|`.
                Test.@test sum(gw) ≈ 1.0 atol = 8 * eps(Float64) * sum(abs, gw)
            end
            Test.@test _alloc(q_lagr!, Vector{Float64}(undef, k), x, 1.7, k) == 0
        end
    end

    Test.@test D.centers!(zeros(1), [0.0, 2.0]) == [1.0]

    # A wrongly sized buffer raises before anything is written.
    Test.@test_throws DimensionMismatch D.faces!(zeros(3), collect(1.0:5.0))
    Test.@test_throws DimensionMismatch D.centers!(zeros(9), collect(1.0:5.0))
    Test.@test_throws DimensionMismatch D.axis_stencils!(
        Matrix{Int}(undef, 3, 3), Matrix{Float64}(undef, 3, 3), collect(1.0:5.0), 1, 3)
    Test.@test_throws ArgumentError D.lagrange_weights!(zeros(2), collect(1.0:5.0), 2.0, 4)

    Test.@test _alloc(q_iw, collect(1.0:5.0), 2.4) == 0
end

# A mildly warped quadrilateral mesh given by its cell vertices, its centres their averages.
function mimetic_warped(nx, ny)
    kx = [(i - 1) / nx + 0.06 * sin(π * (j - 1) / ny) / nx for i in 1:(nx + 1), j in 1:(ny + 1)]
    ky = [(j - 1) / ny + 0.05 * sin(2π * (i - 1) / nx) / ny for i in 1:(nx + 1), j in 1:(ny + 1)]
    avg(k) = [(k[i, j] + k[i + 1, j] + k[i, j + 1] + k[i + 1, j + 1]) / 4 for i in 1:nx, j in 1:ny]
    return FG.Grids.CurvilinearGrid(FG.Geometry.CartesianGeometry{Float64}(), avg(kx), avg(ky); corners = (kx, ky))
end

# The mimetic family, gated by identities of the operators themselves. Each identity is a sum over the
# face–cell or face–corner incidences of products each carrying a few roundings, so it holds to the
# number of incidences `M` times `eps` times the sum of the products' magnitudes.
Test.@testset "The mimetic operators are one adjoint family on every face-indexed grid" begin
    GD = FG.Grids
    GE = FG.Geometry
    O = FG.Operators
    cart = GE.CartesianGeometry{Float64}()
    sph = GE.SphericalGeometry(1.0)
    warped = mimetic_warped
    xs = range(0.0, 1.0; length = 12)
    ys = range(0.0, 0.8; length = 9)
    stretched = cumsum(vcat(0.0, 0.05 .+ 0.03 .* sin.(range(0, 3π; length = 11))))
    hole = trues(12, 9)
    hole[5:7, 4:5] .= false
    kinds = (
        ("box, Neumann", GD.StructuredGrid(cart, xs, ys), O.Neumann()),
        ("box, Dirichlet", GD.StructuredGrid(cart, xs, ys), O.Dirichlet()),
        ("stretched, descending", GD.StructuredGrid(cart, stretched, reverse(collect(ys))), O.Neumann()),
        ("channel", GD.StructuredGrid(cart, FG.Axes.PeriodicAxis(0.0, 1.0, 12), ys), O.Dirichlet()),
        ("masked hole", GD.StructuredGrid(cart, xs, ys, hole), O.Neumann()),
        ("latitude band", GD.StructuredGrid(sph, FG.Axes.PeriodicAxis(0.0, 2π, 16),
                                            range(-π / 2 + π / 20, π / 2 - π / 20; length = 10)), O.Neumann()),
        ("3-D periodic", GD.StructuredGrid(cart, FG.Axes.PeriodicAxis(0.0, 1.0, 6), FG.Axes.PeriodicAxis(0.0, 0.8, 5),
                                           FG.Axes.PeriodicAxis(0.0, 0.5, 4)), O.Neumann()),
        ("3-D box", GD.StructuredGrid(cart, range(0.0, 1.0; length = 6), stretched[1:5],
                                      range(0.5, 0.0; length = 4)), O.Dirichlet()),
        ("3-D shell", GD.StructuredGrid(sph, FG.Axes.PeriodicAxis(0.0, 2π, 8), collect(range(-1.0, 1.0; length = 6)),
                                        collect(range(1.0, 1.6; length = 4))), O.Neumann()),
        ("curvilinear, Neumann", warped(11, 9), O.Neumann()),
        ("curvilinear, Dirichlet", warped(11, 9), O.Dirichlet()),
    )
    for (name, grid, bc) in kinds
        N = GD.ncoordinates(grid)
        dims = GD.size_tuple(grid)
        cells = CartesianIndices(dims)
        act = [GD.isactive(grid, Tuple(I)...) for I in cells]
        V = [act[I] ? GD.measure(grid, Tuple(I)...) : 0.0 for I in cells]
        fm = O.face_metrics(grid, bc)
        faces(d) = CartesianIndices(GD.face_dims(grid, d))
        side(a, C) = C === nothing ? 0.0 : abs(a[C])
        incident(a, d, F) = side(a, GD.cell_below(grid, F, d)) + side(a, GD.cell_above(grid, F, d))
        fsum(f) = sum(d -> sum(F -> f(d, F), faces(d)), 1:N)
        M = 2 * sum(d -> length(faces(d)), 1:N)
        χ = rand(dims...) .* act
        ψ = rand(dims...) .* act
        v = map(a -> rand(size(a)...), O.allocate_faces(Float64, grid))
        g = O.gradient!(O.allocate_faces(Float64, grid), χ, fm)

        # D = −G* under the cell measure and the face weight `A g`: both sides sum `A (±χ) v`.
        Dv = O.divergence!(zeros(dims), v, fm)
        lhs = fsum((d, F) -> fm.area_op[d][F] * fm.gap[d][F] * g[d][F] * v[d][F])
        Test.@test abs(lhs + sum(V .* χ .* Dv)) ≤
                   M * eps() * fsum((d, F) -> fm.area_op[d][F] * abs(v[d][F]) * incident(χ, d, F))

        # `L` assembled is `D G`, zero on every inactive cell, symmetric and negative definite. A cell's
        # value is `2N` terms of size at most `2 max|χ|` times its own coefficients, `|diag|`.
        c = O.laplacian_coefficients(fm)
        Lχ = O.apply_laplacian!(fill(NaN, dims), χ, c)
        Lψ = O.apply_laplacian!(zeros(dims), ψ, c)
        DG = O.divergence!(zeros(dims), g, fm)
        Test.@test all(iszero, Lχ[.!act])
        Test.@test all(abs(Lχ[I] - DG[I]) ≤ 8(N + 3) * eps() * maximum(abs, χ) * abs(c.diag[I])
                       for I in cells if act[I])
        Test.@test abs(sum(V .* Lχ .* ψ) - sum(V .* χ .* Lψ)) ≤
                   2M * eps() * fsum((d, F) -> c.coef[d][F] * incident(χ, d, F) * incident(ψ, d, F))
        Test.@test sum(V .* Lχ .* χ) < 0
        # Under `Neumann` the constants on the active cells are its null space, removed by the projection:
        # its mean is two sums of `n` terms, each within `(n − 1)·eps` of its magnitude.
        Test.@test c.singular == (bc isa O.Neumann)
        if c.singular
            Test.@test all(iszero, O.apply_laplacian!(zeros(dims), Float64.(act), c))
            Test.@test maximum(abs, O.project_out_constant!(3.0 .* act, c)) ≤ 6 * length(χ) * eps()
        end

        # curl ∘ G = 0: around a corner the four differences of χ cancel, each a few roundings of χ.
        pairs = O.rotation_pairs(Val(N))
        W = O.curl!(O.allocate_corners(Float64, grid), g, fm)
        Test.@test all(all(abs(W[p][Q]) ≤ 24eps() * maximum(abs, χ) / fm.corner[p][Q] for Q in CartesianIndices(W[p]))
                       for p in eachindex(W))

        # On the corners of the complex, D ∘ rot = 0, and rot = −curl* under `A g` and `ℓ μ`.
        ℓ(p, Q) = fm.edge[p] === nothing ? 1.0 : fm.edge[p][Q]
        R = ntuple(p -> rand(GD.corner_dims(grid, pairs[p]...)...) .* O.corner_mask(fm, pairs[p]...), length(pairs))
        Rℓ = maximum(p -> maximum(Q -> abs(R[p][Q] * ℓ(p, Q)), CartesianIndices(R[p])), eachindex(R))
        u = O.rotational_velocity!(O.allocate_faces(Float64, grid), R, fm)
        Du = O.divergence!(zeros(dims), u, fm)
        Test.@test all(abs(Du[I]) * V[I] ≤ 16N^2 * eps() * Rℓ for I in cells if act[I])
        Wv = O.curl!(O.allocate_corners(Float64, grid), v, fm)
        around(p, Qc) = (Q = GD.corner_to_face(grid, Qc, pairs[p]...); (a, b) = pairs[p];
                         Cla = GD.cell_below(grid, Q, a); Clb = GD.cell_below(grid, Q, b);
                         R[p][Qc] == 0 ? 0.0 : abs(R[p][Qc] * ℓ(p, Qc)) *
                         (fm.gap[b][Q] * abs(v[b][Q]) + fm.gap[b][Cla] * abs(v[b][Cla]) +
                          fm.gap[a][Q] * abs(v[a][Q]) + fm.gap[a][Clb] * abs(v[a][Clb])))
        Mc = 4 * sum(length, R; init = 0)
        rhs = sum(p -> sum(Q -> ℓ(p, Q) * fm.corner[p][Q] * R[p][Q] * Wv[p][Q], CartesianIndices(R[p])),
                  eachindex(R); init = 0.0)
        Test.@test abs(fsum((d, F) -> fm.area_op[d][F] * fm.gap[d][F] * u[d][F] * v[d][F]) + rhs) ≤
                   2Mc * eps() * sum(p -> sum(Q -> around(p, Q), CartesianIndices(R[p])), eachindex(R); init = 0.0)

        # A metric held as its axis factors is its dense evaluation.
        if fm.area_op[1] isa GD.SeparableMeasure
            dense = O.face_metrics(GD.StructuredGrid(GD.grid_geometry(grid), GD.coordinates(grid)..., trues(dims)), bc)
            for (s, t) in Iterators.flatten((zip(fm.area_op, dense.area_op), zip(fm.area_flux, dense.area_flux),
                                             zip(fm.gap, dense.gap), zip(fm.corner, dense.corner),
                                             zip(filter(!isnothing, fm.edge), filter(!isnothing, dense.edge))))
                Test.@test all(abs(s[F] - t[F]) ≤ 8eps() * abs(t[F]) for F in CartesianIndices(t))
            end
        end
    end
end

# Closed forms. On a stretched, descending Cartesian mesh the operators are exact for a linear field, a
# difference of two values of size `U` over a length `ℓ` rounding within a few `eps·U/ℓ`; on a sphere and a
# shell they converge at second order to the continuous divergence and curl.
Test.@testset "The mimetic operators are exact for linear fields and second order on a sphere" begin
    GD = FG.Grids
    GE = FG.Geometry
    O = FG.Operators
    D = FG.Discretization
    cart = GE.CartesianGeometry{Float64}()
    nc(grid) = GD.ncoordinates(grid)
    # `f` at each face normal to `d`, each `(a, b)` corner and each centre.
    on_faces(f, grid, d) = (x = GD.face_coordinates(grid, d);
                            [f(ntuple(e -> e == d ? x[F[d]] : GD.coordinates(grid, e)[F[e]], nc(grid))...)
                             for F in CartesianIndices(GD.face_dims(grid, d))])
    on_corners(f, grid, a, b) = (xa = GD.corner_coordinates(grid, a); xb = GD.corner_coordinates(grid, b);
                                 [f(ntuple(e -> e == a ? xa[Q[a]] : e == b ? xb[Q[b]] : GD.coordinates(grid, e)[Q[e]],
                                           nc(grid))...) for Q in CartesianIndices(GD.corner_dims(grid, a, b))])
    on_cells(f, grid) = [f(GD.coords(Tuple, grid, Tuple(I)...)...) for I in CartesianIndices(GD.size_tuple(grid))]

    xs = cumsum(vcat(0.0, 0.1 .+ 0.05 .* (1:19)))
    ys = collect(range(6.0, 0.0; length = 16))
    grid = GD.StructuredGrid(cart, xs, ys)
    n1, n2 = GD.size_tuple(grid)
    fm = O.face_metrics(grid, O.Dirichlet())
    ℓmin = min(minimum(diff(xs)), minimum(abs, diff(ys)))
    χ = on_cells((a, b) -> 3.0a - 2.0b + 7.0, grid)
    g = O.gradient!(O.allocate_faces(Float64, grid), χ, fm)
    for (d, slope) in ((1, 3.0), (2, -2.0))
        Test.@test all(abs(g[d][F] - slope) ≤ 4eps() * maximum(abs, χ) / fm.gap[d][F]
                       for F in CartesianIndices(GD.face_dims(grid, d)) if 1 < F[d] ≤ GD.size_tuple(grid)[d])
    end
    v = (on_faces((a, b) -> 2.0a + 0.5b, grid, 1), on_faces((a, b) -> -0.25a + 3.0b, grid, 2))
    U = maximum(a -> maximum(abs, a), v)
    Test.@test maximum(abs, O.divergence!(zeros(n1, n2), v, fm) .- 5.0) ≤ 16eps() * U / ℓmin
    W = O.curl!(O.allocate_corners(Float64, grid), v, fm)
    Test.@test maximum(abs, W[1] .+ 0.75) ≤ 16eps() * U / ℓmin
    # `R = 0.7x − 1.1y` gives `(−∂_y R, ∂_x R)` on every face whose two corners are interior.
    R = (on_corners((a, b) -> 0.7a - 1.1b, grid, 1, 2),)
    u = O.rotational_velocity!(O.allocate_faces(Float64, grid), R, fm)
    Rmax = maximum(abs, R[1])
    Test.@test all(abs(u[1][i, j] - 1.1) ≤ 8eps() * Rmax / ℓmin for i in 2:n1, j in 2:(n2 - 1))
    Test.@test all(abs(u[2][i, j] - 0.7) ≤ 8eps() * Rmax / ℓmin for i in 2:(n1 - 1), j in 2:n2)
    # A centred field and its faces: along the uniform axis the round trip of a linear field is exact away
    # from the ends, its centres the midpoints of their faces.
    uc = (on_cells((a, b) -> 2.0a + 0.5b, grid), on_cells((a, b) -> -0.25a + 3.0b, grid))
    back = O.to_centres!(map(similar, uc), O.to_faces!(O.allocate_faces(Float64, grid), uc, fm), fm)
    Test.@test all(abs(back[2][i, j] - uc[2][i, j]) ≤ 4eps() * U for i in 1:n1, j in 2:(n2 - 1))

    # Every operator writes each value from its own index, so threads and a device launch give the serial
    # values.
    let c = O.laplacian_coefficients(fm), serial = CB.SerialBackend()
        runs(be) = (O.gradient!(O.allocate_faces(Float64, grid), χ, fm; execution = be),
                    O.divergence!(zeros(n1, n2), v, fm; execution = be),
                    O.curl!(O.allocate_corners(Float64, grid), v, fm; execution = be),
                    O.rotational_velocity!(O.allocate_faces(Float64, grid), R, fm; execution = be),
                    O.to_faces!(O.allocate_faces(Float64, grid), uc, fm; execution = be),
                    O.to_centres!(map(similar, uc), v, fm; execution = be),
                    O.apply_laplacian!(zeros(n1, n2), χ, c; execution = be))
        for be in (CB.ThreadedBackend(), CB.GPUBackend(KernelAbstractions.CPU()))
            Test.@test isequal(runs(be), runs(serial))
        end
    end

    # A hole: its faces carry nothing, and the corners around it leave the complex.
    let mk = trues(n1, n2)
        mk[8, 6] = false
        gm = GD.StructuredGrid(cart, xs, ys, mk)
        fh = O.face_metrics(gm)
        Test.@test fh.area_op[1][8, 6] == 0 && fh.area_op[1][9, 6] == 0
        Test.@test fh.area_op[2][8, 6] == 0 && fh.area_op[2][8, 7] == 0
        cm = O.corner_mask(fh, 1, 2)
        Test.@test count(!, cm) == 4 && !cm[7, 5] && !cm[8, 5] && !cm[7, 6] && !cm[8, 6]
        gh = O.gradient!(O.allocate_faces(Float64, gm), χ .* mk, fh)
        Test.@test gh[1][8, 6] == 0 && gh[1][9, 6] == 0 && gh[1][10, 6] ≈ 3.0
    end

    # The face axes are the staggered grid's, and the staggered grid answers for its centres.
    let sg = GD.StaggeredGrid(grid)
        Test.@test all(collect(GD.face_coordinates(grid, d)) == collect(GD.axis_at(sg, d, D.Face())) for d in 1:2)
        Test.@test O.face_metrics(sg, O.Dirichlet()).gap == fm.gap
        Test.@test GD.face_dims(grid, 1) == (n1 + 1, n2) && GD.corner_dims(grid, 1, 2) == (n1 - 1, n2 - 1)
        wrapped = GD.StructuredGrid(cart, FG.Axes.PeriodicAxis(0.0, 1.0, 8), ys)
        Test.@test GD.face_dims(wrapped, 1) == (8, n2) && GD.corner_dims(wrapped, 1, 2) == (8, n2 - 1)
    end

    # On a sphere: `u_λ = sin λ cos φ` has divergence `cos λ / R`, `u_φ = sin φ` has `cos 2φ/(R cos φ)`, and
    # `u_λ = cos φ` has vorticity `2 sin φ / R`. On a shell the zonal flow also has `W_λr = −cos φ / r`, exact.
    # Each refinement halves both spacings, so their ratio, which weighs the two `O(h²)` terms of the first
    # divergence against each other, stays fixed.
    let Rs = 6.371e6, sph = GE.SphericalGeometry(Rs)
        rel(a, b) = maximum(abs, a .- b) / maximum(abs, b)
        errs = map((24, 48, 96)) do nlon
            gs = GD.StructuredGrid(sph, FG.Axes.PeriodicAxis(0.0, 2π, nlon), collect(range(-1.3, 1.3; length = nlon ÷ 2 + 1)))
            fs = O.face_metrics(gs, O.Dirichlet())
            z = zeros(GD.face_dims(gs, 2))
            d1 = O.divergence!(zeros(GD.size_tuple(gs)), (on_faces((l, p) -> sin(l) * cos(p), gs, 1), z), fs)
            d2 = O.divergence!(zeros(GD.size_tuple(gs)),
                               (zeros(GD.face_dims(gs, 1)), on_faces((l, p) -> sin(p), gs, 2)), fs)
            w = O.curl!(O.allocate_corners(Float64, gs), (on_faces((l, p) -> cos(p), gs, 1), z), fs)[1]
            return (rel(d1, on_cells((l, p) -> cos(l) / Rs, gs)),
                    rel(d2, on_cells((l, p) -> cos(2p) / (Rs * cos(p)), gs)),
                    rel(w, on_corners((l, p) -> 2sin(p) / Rs, gs, 1, 2)))
        end
        for k in 1:3
            Test.@test errs[1][k] / errs[2][k] > 3.5 && errs[2][k] / errs[3][k] > 3.5
        end
        errs3 = map((16, 32, 64)) do nlon
            gs = GD.StructuredGrid(sph, FG.Axes.PeriodicAxis(0.0, 2π, nlon), collect(range(-1.2, 1.2; length = nlon ÷ 2 + 1)),
                                   collect(range(Rs, 1.01Rs; length = 5)))
            fs = O.face_metrics(gs)
            us = (on_faces((l, p, r) -> cos(p), gs, 1), zeros(GD.face_dims(gs, 2)), zeros(GD.face_dims(gs, 3)))
            ws = O.curl!(O.allocate_corners(Float64, gs), us, fs)
            # Two products `u_λ g_λ` and their difference, over the corner area.
            Test.@test maximum(abs, ws[2] .- on_corners((l, p, r) -> -cos(p) / r, gs, 1, 3)) ≤
                       8eps() * maximum(abs, us[1] .* fs.gap[1]) / minimum(fs.corner[2])
            return rel(ws[1], on_corners((l, p, r) -> 2sin(p) / r, gs, 1, 2))
        end
        Test.@test errs3[1] / errs3[2] > 3.5 && errs3[2] / errs3[3] > 3.5
    end

    # Shapes are checked, and a curvilinear grid without its vertices has no face metric.
    Test.@test_throws DimensionMismatch O.gradient!((zeros(3, 3), zeros(3, 3)), χ, fm)
    Test.@test_throws DimensionMismatch O.divergence!(zeros(3, 3), v, fm)
    Test.@test_throws DimensionMismatch O.curl!((zeros(2, 2),), v, fm)
    let n = 6, x = [a + 0.2b for a in 1.0:n, b in 1.0:n], y = [b - 0.1a for a in 1.0:n, b in 1.0:n]
        Test.@test_throws ArgumentError O.face_metrics(GD.CurvilinearGrid(cart, x, y))
    end
end

# The next level down is `R L P` for injection `P` and the measure-weighted mean `R`, taken here as dense
# matrices. A coarse entry sums at most `(2^N)²(2N+1)` fine products, each a few roundings.
Test.@testset "A coarse level is the Galerkin product of the fine one" begin
    GD = FG.Grids
    O = FG.Operators
    cart = FG.Geometry.CartesianGeometry{Float64}()
    hole = trues(9, 9)
    hole[4:5, 3:6] .= false
    odd = range(0.0, 1.0; length = 9)
    dense(c) = (n = length(c.diag); A = zeros(n, n); e = zeros(size(c.diag));
                for j in 1:n; fill!(e, 0); e[j] = 1; A[:, j] .= vec(O.apply_laplacian!(zeros(size(e)), e, c)); end; A)
    for (name, grid, bc) in (("odd, Dirichlet", GD.StructuredGrid(cart, odd, odd), O.Dirichlet()),
                             ("odd, Neumann", GD.StructuredGrid(cart, odd, odd), O.Neumann()),
                             ("odd periodic", GD.StructuredGrid(cart, FG.Axes.PeriodicAxis(0.0, 1.0, 9),
                                                                FG.Axes.PeriodicAxis(0.0, 1.0, 9)), O.Neumann()),
                             ("even periodic", GD.StructuredGrid(cart, FG.Axes.PeriodicAxis(0.0, 1.0, 8), odd),
                              O.Dirichlet()),
                             ("odd masked, Dirichlet", GD.StructuredGrid(cart, odd, odd, hole), O.Dirichlet()),
                             ("curvilinear, Dirichlet", mimetic_warped(9, 7), O.Dirichlet()))
        c = O.laplacian_coefficients(O.face_metrics(grid, bc))
        cg = O.coarsen(grid)
        cc = O.galerkin_coefficients(c, cg)
        fd, cd = GD.size_tuple(grid), GD.size_tuple(cg)
        Test.@test cd == map(n -> n < 4 ? n : cld(n, 2), fd)
        P = zeros(prod(fd), prod(cd))
        LI = LinearIndices(cd)
        for (j, J) in enumerate(CartesianIndices(fd))
            P[j, LI[CartesianIndex(ntuple(d -> fd[d] < 4 ? J[d] : cld(J[d], 2), 2))]] = 1
        end
        Vf = vec(collect(c.measure))
        Vc = vec(collect(cc.measure))
        on = Vc .> 0
        Ac = dense(cc)[on, on]
        RAP = (1 ./ Vc[on]) .* (P[:, on]' * (Vf .* dense(c) * P[:, on]))
        Test.@test maximum(abs, RAP .- Ac) ≤ 16 * 5 * 4eps() * maximum(abs, Ac)
        # `restrict!` and `prolong_add!` are adjoint under the two measures.
        f = rand(fd...)
        x = rand(cd...)
        rf = O.restrict!(zeros(cd), f, c, cc)
        px = O.prolong_add!(zeros(fd), x, c, cc)
        terms = vec(Vf) .* vec(f) .* vec(px)
        Test.@test abs(sum(Vc .* vec(rf) .* vec(x)) - sum(terms)) ≤ 4length(f) * eps() * sum(abs, terms)
        # The residual restricted as it is formed equals the stored residual restricted.
        b, xf = rand(fd...), rand(fd...)
        Lx = O.apply_laplacian!(zeros(fd), xf, c)
        rr = O.restrict!(zeros(cd), Lx .+ b, c, cc)
        Test.@test maximum(abs, O.restrict_residual!(zeros(cd), b, xf, c, cc) .- rr) ≤ 64eps() * maximum(abs, Lx)
        Test.@test GD.isperiodic(cg, 1) == GD.isperiodic(grid, 1)
    end
    let g = GD.StructuredGrid(cart, collect(odd), collect(odd))
        Test.@test GD.coordinates(O.coarsen(g), 1) ≈ [(odd[2i - 1] + odd[min(2i, 9)]) / 2 for i in 1:5]
        Test.@test O.coarsen(GD.StructuredGrid(cart, 0.0:1.0:2.0, 0.0:1.0:2.0)) === nothing
    end
end

# Strain rate, curl and divergence along the local frame, against the orthogonal-curvilinear closed forms
# (Batchelor 1967, Appendix 2). Every field is at most quadratic along each direction, which three nodes
# difference exactly, so what remains to check is the frame's own terms.
Test.@testset "Strain rate, curl and divergence carry the turning of the local frame" begin
    O = FG.Operators
    GD = FG.Grids
    GE = FG.Geometry
    D = FG.Discretization
    A = FG.Axes
    R = 6.371e6
    sph = GE.SphericalGeometry(R)
    spd = GE.SpheroidGeometry()
    e² = spd.f * (2 - spd.f)
    Nsp(φ) = spd.a / sqrt(1 - e² * sin(φ)^2)
    Msp(φ) = spd.a * (1 - e²) / (1 - e² * sin(φ)^2)^1.5
    P(φ) = 0.3 + 0.7φ - 0.4φ^2
    dP(φ) = 0.7 - 0.8φ
    Q(φ) = -0.5 + 0.2φ + 0.9φ^2
    dQ(φ) = 0.2 + 1.8φ
    nλ, nφ = 16, 13
    λs = A.PeriodicAxis(0.0, 2π, nλ)
    φs = [-1.3 + 2.6 * (t / 12)^1.2 for t in 0:12]
    u = ([P(φ) for _ in λs, φ in φs], [Q(φ) for _ in λs, φ in φs])
    # Each derivative sums three products of a value of size `U` and a weight of a few `1/Δφ`, each
    # rounded at `eps`, and `1/M` makes it a derivative per unit distance.
    Uof(φs) = max(maximum(abs ∘ P, φs), maximum(abs ∘ Q, φs))
    tol(U, M, Δ) = 64eps() * U / (M * Δ)
    # `u = (P(φ), Q(φ))` on a surface, `N` and `M` the prime-vertical and meridional radii.
    surface(N, M, φ) = (λλ = -Q(φ) * tan(φ) / N, φφ = dQ(φ) / M, λφ = (dP(φ) / M + P(φ) * tan(φ) / N) / 2,
                        ω = -dP(φ) / M + P(φ) * tan(φ) / N, δ = dQ(φ) / M - Q(φ) * tan(φ) / N)
    pick(S, ω, δ) = ((:λλ, S.λλ), (:φφ, S.φφ), (:λφ, S.λφ), (:ω, ω), (:δ, δ))

    for (geo, N_, M_) in ((sph, _ -> R, _ -> R), (spd, Nsp, Msp))
        g = GD.StructuredGrid(geo, λs, φs)
        plans = O.stencil_plans(g; nodes = 3)
        S = O.strain_rate(u, g, plans)
        Test.@test keys(S) == O.strain_components(g) == (:λλ, :φφ, :λφ)
        w = [surface(N_(φ), M_(φ), φ) for _ in λs, φ in φs]
        t = tol(Uof(φs), minimum(M_, φs), minimum(diff(φs)))
        for (k, got) in pick(S, O.curl(u, g, plans), O.divergence(u, g, plans))
            Test.@test maximum(abs, got .- getfield.(w, k)) ≤ t
        end
    end

    # A shell `(λ, φ, r)`, where `γ_λφ = −tan φ/r` and `γ_λr = γ_φr = 1/r`, with
    # `u = ((r/R)P(φ), Q(φ), (r/R)²/10)`.
    let λ3 = A.PeriodicAxis(0.0, 2π, 8), φ3 = φs[1:2:end], rs = collect(range(R, 1.5R; length = 5))
        g3 = GD.StructuredGrid(sph, λ3, φ3, rs)
        p3 = O.stencil_plans(g3; nodes = 3)
        cell(f) = [f(φ, r) for _ in λ3, φ in φ3, r in rs]
        vλ(φ, r) = r / R * P(φ)
        vr(φ, r) = (r / R)^2 / 10
        u3 = (cell(vλ), cell((φ, r) -> Q(φ)), cell(vr))
        S3 = O.strain_rate(u3, g3, p3)
        Test.@test keys(S3) == (:λλ, :φφ, :rr, :λφ, :λr, :φr)
        Sw = (cell((φ, r) -> (vr(φ, r) - Q(φ) * tan(φ)) / r), cell((φ, r) -> (dQ(φ) + vr(φ, r)) / r),
              cell((φ, r) -> r / (5R^2)), cell((φ, r) -> (dP(φ) / R + vλ(φ, r) * tan(φ) / r) / 2),
              cell((φ, r) -> (P(φ) / R - vλ(φ, r) / r) / 2), cell((φ, r) -> -Q(φ) / (2r)))
        ωw = (cell((φ, r) -> -Q(φ) / r), cell((φ, r) -> P(φ) / R + vλ(φ, r) / r),
              cell((φ, r) -> -dP(φ) / R + vλ(φ, r) * tan(φ) / r))
        δw = cell((φ, r) -> (dQ(φ) - Q(φ) * tan(φ) + 2vr(φ, r)) / r + r / (5R^2))
        t3 = 64eps() * maximum(abs, vcat(vec.(u3)...)) / min(rs[1] * minimum(diff(φ3)), rs[2] - rs[1])
        ω3 = O.curl(u3, g3, p3)
        Test.@test all(maximum(abs, S3[c] .- Sw[c]) ≤ t3 for c in 1:6)
        Test.@test all(maximum(abs, ω3[c] .- ωw[c]) ≤ t3 for c in 1:3)
        Test.@test maximum(abs, O.divergence(u3, g3, p3) .- δw) ≤ t3
        # A radius's factor is one, far above its own floor in Float32, so nothing is masked there.
        g32 = GD.StructuredGrid(GE.SphericalGeometry(Float32(R)), A.PeriodicAxis(0.0f0, Float32(2π), 8),
                                Float32.(φ3), Float32.(rs))
        S32 = O.strain_rate(map(a -> Float32.(a), u3), g32, O.stencil_plans(g32; nodes = 3))
        Test.@test all(c -> all(isfinite, c), S32)
    end

    # A node set through its least-squares gradient. A uniform frame field `(c₁, c₂)` differences to zero,
    # so every value is a frame term alone, one rounding from its closed form; a pole has no east.
    let gi = GD.IcosahedralGrid(8), c1 = 3.0, c2 = -2.0
        n = length(GD.mask(gi))
        φi = [GD.coords(gi, i).φ for i in 1:n]
        tφ = tan.(φi) ./ GE.radius(GD.grid_geometry(gi))
        pole = abs.(cos.(φi)) .≤ sqrt(eps())
        pl = O.gradient_plan(gi)
        ui = (fill(c1, n), fill(c2, n))
        Si = O.strain_rate(ui, gi, pl)
        ωi = O.curl(ui, gi, pl)
        δi = O.divergence(ui, gi, pl)
        Test.@test count(pole) == 2
        Test.@test all(k -> all(isnan, k[pole]), (Si.λλ, Si.φφ, Si.λφ, ωi, δi))
        near(a, b) = abs(a - b) ≤ 4eps() * abs(b)
        q = .!pole
        Test.@test all(iszero, Si.φφ[q])
        Test.@test all(near.(Si.λλ[q], -c2 .* tφ[q])) && all(near.(Si.λφ[q], c1 .* tφ[q] ./ 2))
        Test.@test all(near.(ωi[q], c1 .* tφ[q])) && all(near.(δi[q], -c2 .* tφ[q]))
    end

    # A sheared Cartesian mesh, whose least-squares gradient differences a linear field exactly.
    let n = 14, cart = GE.CartesianGeometry{Float64}()
        x = [s + 0.35t for s in range(0.0, 10.0; length = n), t in range(0.0, 6.0; length = n)]
        y = [t - 0.2s for s in range(0.0, 10.0; length = n), t in range(0.0, 6.0; length = n)]
        cg = GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
        pl = O.gradient_plan(cg)
        ax, bx, ay, by = 0.7, -1.3, 2.1, 0.4
        uc = (ax .* x .+ bx .* y .+ 5.0, ay .* x .+ by .* y .- 2.0)
        Sc = O.strain_rate(uc, cg, pl)
        Test.@test keys(Sc) == (:xx, :yy, :xy)
        hmin = min(minimum(hypot(x[i + 1, j] - x[i, j], y[i + 1, j] - y[i, j]) for i in 1:(n - 1), j in 1:n),
                   minimum(hypot(x[i, j + 1] - x[i, j], y[i, j + 1] - y[i, j]) for i in 1:n, j in 1:(n - 1)))
        tc = 64eps() * maximum(abs, vcat(vec.(uc)...)) / hmin
        Test.@test maximum(abs, Sc.xx .- ax) ≤ tc && maximum(abs, Sc.yy .- by) ≤ tc
        Test.@test maximum(abs, Sc.xy .- (bx + ay) / 2) ≤ tc
        Test.@test maximum(abs, O.curl(uc, cg, pl) .- (ay - bx)) ≤ tc
        Test.@test maximum(abs, O.divergence(uc, cg, pl) .- (ax + by)) ≤ tc
    end

    # A hole. Under `BlankMasked` it takes every cell whose window reads it along either direction, and
    # under `ReduceInRun` itself alone; a finite `masked` lands exactly, and the rest is the closed form.
    let mk = trues(nλ, nφ)
        mk[6, 7] = false
        gm = GD.StructuredGrid(sph, λs, φs, mk)
        w = [surface(R, R, φ) for _ in λs, φ in φs]
        t = tol(Uof(φs), R, minimum(diff(φs)))
        reads = falses(nλ, nφ)
        reads[5:7, 7] .= true
        reads[6, 6:8] .= true
        for (pol, hit) in ((O.BlankMasked(), reads), (O.ReduceInRun(), .!mk))
            pm = O.stencil_plans(gm; nodes = 3, policy = pol)
            for (k, got) in pick(O.strain_rate(u, gm, pm; masked = -7.0), O.curl(u, gm, pm; masked = -7.0),
                                 O.divergence(u, gm, pm; masked = -7.0))
                Test.@test (got .== -7.0) == hit
                Test.@test maximum(abs, (got .- getfield.(w, k))[.!hit]) ≤ t
            end
            # Threads and a device launch write every cell the serial pass does.
            for be in (CB.ThreadedBackend(), CB.GPUBackend(KernelAbstractions.CPU()))
                Test.@test isequal(O.strain_rate(u, gm, pm; execution = be), O.strain_rate(u, gm, pm))
                Test.@test isequal(O.curl(u, gm, pm; execution = be), O.curl(u, gm, pm))
                Test.@test isequal(O.divergence(u, gm, pm; execution = be), O.divergence(u, gm, pm))
            end
        end
    end

    # Rows on the poles, where `h_λ` vanishes and the frame has no east: every component there is masked.
    let φp = collect(range(-π / 2, π / 2; length = 9))
        gp = GD.StructuredGrid(sph, λs, φp)
        up = ([P(φ) for _ in λs, φ in φp], [Q(φ) for _ in λs, φ in φp])
        pp = O.stencil_plans(gp; nodes = 3)
        w = [surface(R, R, φ) for _ in λs, φ in φp]
        t = tol(Uof(φp), R, step(range(-π / 2, π / 2; length = 9)))
        for (k, got) in pick(O.strain_rate(up, gp, pp; masked = -7.0), O.curl(up, gp, pp; masked = -7.0),
                             O.divergence(up, gp, pp; masked = -7.0))
            Test.@test all(==(-7.0), got[:, [1, 9]])
            Test.@test maximum(abs, got[:, 2:8] .- getfield.(w[:, 2:8], k)) ≤ t
        end
    end

    # A direction with one sample carries no derivative: every value reading it is masked.
    let g1 = GD.StructuredGrid(sph, λs, [0.4])
        p1 = O.stencil_plans(g1; nodes = 3)
        f1 = reshape([sin(a) for a in λs], nλ, 1)
        Test.@test all(isnan, O.derivative!(zeros(nλ, 1), f1, g1, p1, 2; masked = NaN))
        Test.@test all(isnan, O.curl((f1, f1), g1, p1))
        Test.@test_throws ArgumentError O.derivative!(zeros(nλ, 1), f1, g1, p1, 3)
    end

    # A field with the wrong number of components, or names that are not the frame's, is refused.
    let g = GD.StructuredGrid(sph, λs, φs), plans = O.stencil_plans(g; nodes = 3)
        Test.@test_throws DimensionMismatch O.strain_rate((u[1],), g, plans)
        Test.@test_throws DimensionMismatch O.curl((u[1], u[2], u[1]), g, plans)
        Test.@test_throws ArgumentError O.strain_rate!(
            (xx = similar(u[1]), yy = similar(u[1]), xy = similar(u[1])), u, g, plans)
    end

    # In place, with scratch, nothing is allocated: on held stencil plans, masked ones and a least-squares
    # plan.
    let g = GD.StructuredGrid(sph, λs, φs), mk = trues(nλ, nφ)
        mk[6, 7] = false
        gm = GD.StructuredGrid(sph, λs, φs, mk)
        S = map(_ -> zeros(nλ, nφ), (λλ = 1, φφ = 2, λφ = 3))
        sc = (zeros(nλ, nφ), zeros(nλ, nφ))
        o = zeros(nλ, nφ)
        for (gg, pp) in ((g, O.stencil_plans(g; nodes = 3)),
                         (gm, O.stencil_plans(gm; nodes = 3, policy = O.ReduceInRun())))
            Test.@test _alloc(q_strain!, S, u, gg, pp, sc) == 0
            Test.@test _alloc(q_vcurl!, o, u, gg, pp, sc) == 0
            Test.@test _alloc(q_vdiv!, o, u, gg, pp, sc) == 0
        end
        Test.@test _alloc(q_strain_names, g) == 0
        gi = GD.IcosahedralGrid(8)
        n = length(GD.mask(gi))
        pl = O.gradient_plan(gi)
        ui = (rand(n), rand(n))
        sci = (zeros(n), zeros(n))
        Test.@test _alloc(q_strain!, map(_ -> zeros(n), (λλ = 1, φφ = 2, λφ = 3)), ui, gi, pl, sci) == 0
        Test.@test _alloc(q_vcurl!, zeros(n), ui, gi, pl, sci) == 0
        Test.@test _alloc(q_vdiv!, zeros(n), ui, gi, pl, sci) == 0
    end

    # `γ[i][k] = (1/(hᵢhₖ)) ∂hᵢ/∂xₖ` against a central difference of the scale factors themselves. Along an
    # angle a step `δ` leaves `δ²/6` of `max|h|` in truncation; along a radius or a height each factor is
    # linear and none; each factor is a handful of roundings, `eps/δ` of `max|h|` in the difference.
    for (geo, p) in ((sph, (0.3, 0.7)), (sph, (0.3, 0.7, 1.2R)), (spd, (0.3, 0.7)), (spd, (0.3, 0.7, 2.0e4)),
                     (GE.CartesianGeometry{Float64}(), (0.3, 0.7, 1.2)))
        h = GE.scale_factors(geo, p)
        γ = GE.log_scale_gradients(geo, p)
        Test.@test _alloc(q_lsg, geo, p) == 0
        for i in eachindex(p), k in eachindex(p)
            i == k && (Test.@test iszero(γ[i][k]); continue)
            δ = k ≤ 2 ? 1.0e-5 : 1.0e3
            trunc = k ≤ 2 ? δ^2 / 6 : 0.0
            e = ntuple(j -> j == k ? δ : 0.0, length(p))
            fd = (GE.scale_factors(geo, p .+ e)[i] - GE.scale_factors(geo, p .- e)[i]) / (2δ)
            Test.@test abs(γ[i][k] - fd / (h[i] * h[k])) ≤
                       8 * (trunc + eps() / δ) * maximum(abs, h) / abs(h[i] * h[k])
        end
    end
end
