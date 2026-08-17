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
end

Test.@testset "derivative! is with respect to distance, and masks where the metric dies" begin
    D = FG.Discretization
    GD = FG.Grids
    GE = FG.Geometry
    R = 6.371e6
    sph = GE.SphericalGeometry(R)
    nλ, nφ = 48, 25
    λ = collect(range(0.0, 2π * (1 - 1 / nλ); length = nλ))
    φ = collect(range(-π / 2, π / 2; length = nφ))         # both poles are grid rows
    g = GD.StructuredGrid(sph, λ, φ)

    # ∂/∂north of sin φ is cos φ / R — the physical derivative, not the coordinate one.
    f = [sin(fj) for _ in λ, fj in φ]
    o = zeros(nλ, nφ)
    D.derivative!(o, f, g, 2; order = 1, nodes = 5, masked = NaN)
    want = [cos(fj) / R for _ in λ, fj in φ]
    int = 4:(nφ - 3)
    Test.@test maximum(abs.(o[:, int] .- want[:, int])) < 1e-3 * maximum(abs.(want[:, int]))
    # …and it is exactly the coordinate derivative divided by the metric, cell by cell.
    co = zeros(nλ, nφ)
    D.apply_stencil!(co, f, g, 2; order = 1, nodes = 5)
    Test.@test all(o[i, j] ≈ co[i, j] / GE.scale_factors(sph, (λ[i], φ[j]))[2]
                   for i in 1:nλ, j in int)

    # ∂/∂east of sin λ cos φ is cos λ / R, and the poles have no east at all.
    f2 = [sin(l) * cos(fj) for l in λ, fj in φ]
    o2 = zeros(nλ, nφ)
    D.derivative!(o2, f2, g, 1; order = 1, nodes = 5, masked = NaN)
    j0 = nφ ÷ 2 + 1
    Test.@test maximum(abs.(o2[:, j0] .- [cos(l) / R for l in λ])) < 1e-3 / R
    Test.@test all(isnan, o2[:, 1]) && all(isnan, o2[:, nφ])
    Test.@test !any(isnan, o2[:, 2:(nφ - 1)])

    # The guard is relative to the element type, which is the whole point: at a Float32 pole
    # `|h_λ|` is about 0.28, so an absolute `1e-12` never fires and a large finite number would be
    # written on the pole rows instead of `masked`.
    let R32 = Float32(R), n1 = 16, n2 = 9
        s32 = GE.SphericalGeometry(R32)
        λ32 = collect(range(0.0f0, Float32(2π) * (1 - 1.0f0 / n1); length = n1))
        φ32 = collect(range(-Float32(π) / 2, Float32(π) / 2; length = n2))
        g32 = GD.StructuredGrid(s32, λ32, φ32)
        f32 = [sin(l) * cos(fj) for l in λ32, fj in φ32]
        o32 = zeros(Float32, n1, n2)
        D.derivative!(o32, f32, g32, 1; order = 1, nodes = 3, masked = Float32(NaN))
        Test.@test abs(GE.scale_factors(s32, (0.0f0, φ32[1]))[1]) > 1.0f-12   # the trap
        Test.@test all(isnan, o32[:, 1]) && all(isnan, o32[:, n2])
        Test.@test eltype(o32) === Float32
        Test.@test D.metric_floor(s32) > abs(GE.scale_factors(s32, (0.0f0, φ32[1]))[1])
    end

    # A Cartesian metric is the identity, so this must be `apply_stencil!` untouched.
    let cart = FG.Geometry.CartesianGeometry{Float64}(), x = collect(range(0.0, 1.0; length = 12))
        gc = GD.StructuredGrid(cart, x, x)
        fc = [xi^2 + yi for xi in x, yi in x]
        a = zeros(12, 12); b = zeros(12, 12)
        D.apply_stencil!(a, fc, gc, 1; order = 1, nodes = 3)
        D.derivative!(b, fc, gc, 1; order = 1, nodes = 3)
        Test.@test a == b
        Test.@test D.metric_floor(cart) == 0.0
    end

    # On a spheroid `h_φ = M(φ)` varies with φ, so the factor is NOT constant along the direction
    # being differenced and must not be hoisted that way.
    let spd = GE.SpheroidGeometry(), n1 = 12, n2 = 21
        λs = collect(range(0.0, 2π * (1 - 1 / n1); length = n1))
        φs = collect(range(-1.2, 1.2; length = n2))
        gs = GD.StructuredGrid(spd, λs, φs)
        fs = [sin(2fj) for _ in λs, fj in φs]
        os = zeros(n1, n2); cs = zeros(n1, n2)
        D.derivative!(os, fs, gs, 2; order = 1, nodes = 5, masked = NaN)
        D.apply_stencil!(cs, fs, gs, 2; order = 1, nodes = 5)
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
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    n = 24
    x = collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = n))))
    y = collect(range(0.0, 2π * (1 - 1 / n); length = n))
    mk = trues(n, n); mk[7, 9] = false; mk[8, 9] = false; mk[13, 4] = false
    g = GD.StructuredGrid(cart, x, y, mk; periodic = (false, true), period = (0.0, 2π))
    f = [sin(3xi) * cos(yj) for xi in x, yj in y]

    # The bare `(indices, weights)` form cannot degrade — it has no axis to rebuild a window from
    # at a mask edge — so a caller wanting `ReduceInRun` had to give up the table entirely and pay
    # its rebuild per call. Handing the axis alongside the table serves every policy.
    for dim in 1:2, pol in (D.BlankMasked(), D.ShiftWithinRun(), D.ReduceInRun()), k in (3, 5)
        a = zeros(n, n); b = zeros(n, n)
        D.apply_stencil!(a, f, g, dim; order = 1, nodes = k, masked = NaN, policy = pol)
        idx, w = D.axis_stencils(g, dim; order = 1, nodes = k)
        D.apply_stencil!(b, f, g, idx, w, dim; order = 1, masked = NaN, policy = pol)
        Test.@test all(isequal(a[i], b[i]) for i in eachindex(a))
    end


    # The bare form still refuses rather than silently ignoring the policy.
    let idx = D.axis_stencils(g, 1; order = 1, nodes = 3)
        Test.@test_throws ArgumentError D.apply_stencil!(zeros(n, n), f, idx[1], idx[2], 1;
                                                         mask = mk, policy = D.ReduceInRun())
    end

    # `derivative!` is the form a geometry-aware caller uses, so it takes a table too.
    let sph = FG.Geometry.SphericalGeometry(6.371e6),
        lam = collect(range(0.0, 2π * (1 - 1 / 16); length = 16)),
        phi = collect(range(-1.2, 1.2; length = 13))
        gs = GD.StructuredGrid(sph, lam, phi)
        fs = [sin(fj) for _ in lam, fj in phi]
        a = zeros(16, 13); b = zeros(16, 13)
        D.derivative!(a, fs, gs, 2; order = 1, nodes = 3, masked = NaN)
        i2, w2 = D.axis_stencils(gs, 2; order = 1, nodes = 3)
        D.derivative!(b, fs, gs, i2, w2, 2; order = 1, masked = NaN)
        Test.@test all(isequal(a[i], b[i]) for i in eachindex(a))
    end
end

Test.@testset "The host sweep is a different loop shape, and the same answer" begin
    D = FG.Discretization
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()

    # The host reshapes the loop — Cartesian rather than linear, split at `dim`, node count in the
    # type, address arithmetic where the layout allows it — while the index-parallel form stays for
    # a device launch. Different shape, identical arithmetic in identical order, so the two must
    # agree BIT for bit, not merely to a tolerance. Masks, node counts and both dimensions.
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
        D.apply_stencil!(a, f, idx, w, dim; mask = mm, masked = NaN)
        D.apply_stencil!(b, f, idx, w, dim; mask = mm, masked = NaN, backend = KernelAbstractions.CPU())
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
            D.apply_stencil!(o, f3, idx, w, dim)
            D.apply_stencil!(ob, f3, idx, w, dim; backend = KernelAbstractions.CPU())
            Test.@test maximum(abs.(o .- want)) < 1e-10
            Test.@test o == ob
        end
    end

    # The address-arithmetic path assumes a linear, one-based layout. An offset array has neither,
    # so it must take the Cartesian nest and still get the same answer.
    let n = 12
        xs = collect(range(0.0, 1.0; length = n))
        f = [xi^2 + yi for xi in xs, yi in xs]
        idx, w = D.axis_stencils(xs, 1, 3)
        o = zeros(n, n)
        D.apply_stencil!(o, f, idx, w, 1)
        v = view(f, :, :)                                  # a view is still linear here
        ov = zeros(n, n)
        D.apply_stencil!(ov, v, idx, w, 1)
        Test.@test o == ov
    end

    # A stencil wider than the specialization cap keeps the runtime-node-count path, and agrees.
    let n = 32
        xs = collect(range(0.0, 1.0; length = n))
        f = [sin(4xi) * yi for xi in xs, yi in xs]
        for k in (10, 12)
            idx, w = D.axis_stencils(xs, 1, k)
            a = zeros(n, n); b = zeros(n, n)
            D.apply_stencil!(a, f, idx, w, 1)
            D.apply_stencil!(b, f, idx, w, 1; backend = KernelAbstractions.CPU())
            Test.@test a == b
        end
    end

    # A table built from the grid carries the grid's period, and applying it is the same answer as
    # letting the grid form build one per call — which is the point of having the entry point.
    let n = 24
        xs = collect(range(0.0, 1.0; length = n))
        ys = collect(range(0.0, 2π * (1 - 1 / n); length = n))
        mk = trues(n, n); mk[5, 6] = false
        g = GD.StructuredGrid(cart, xs, ys, mk; periodic = (false, true), period = (0.0, 2π))
        f = [sin(3xi) * cos(yi) for xi in xs, yi in ys]
        for dim in 1:2
            a = zeros(n, n); b = zeros(n, n)
            D.apply_stencil!(a, f, g, dim; order = 1, nodes = 5, masked = NaN)
            idx, w = D.axis_stencils(g, dim; order = 1, nodes = 5)
            D.apply_stencil!(b, f, g, idx, w, dim; masked = NaN)
            Test.@test all(isequal(a[i], b[i]) for i in eachindex(a))
        end
        # The periodic direction's table is genuinely the wrapped one.
        Test.@test D.axis_stencils(g, 2; order = 1, nodes = 5)[1] !=
                   D.axis_stencils(ys, 1, 5)[1]
    end
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

    # A precomputed weight set gives the same answer.
    idx, w = D.axis_stencils(X, 1, 3)
    Test.@test size(idx) == (9, 3) && size(w) == (9, 3)
    O2 = similar(F)
    D.apply_stencil!(O2, F, idx, w, 1)
    Test.@test O2 ≈ [2xi for xi in X, _ in Y]

    Test.@test_throws ArgumentError D.axis_stencils(X, 2, 2)          # too few nodes for order 2
    Test.@test_throws ArgumentError D.axis_stencils([0.0, 1.0], 1, 5) # more nodes than samples
    Test.@test_throws ArgumentError D.apply_stencil!(O, F, X, 3)      # no direction 3 in a matrix
    Test.@test_throws DimensionMismatch D.apply_stencil!(O, F, X[1:5], 1)
    Test.@test_throws DimensionMismatch D.apply_stencil!(similar(F, 3, 3), F, idx, w, 1)
end

Test.@testset "A field can be evaluated at a coordinate, on every architecture" begin
    C = FG.Connectivity
    D = FG.Discretization
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
        Test.@test D.interpolate(f, g, (px, py)) ≈ exact(px, py) atol = 1e-10
    end
    # And a cell's own value at its centre, which multilinear interpolation must reproduce.
    Test.@test all(D.interpolate(f, g, (x[i], y[j])) ≈ f[i, j] for i in 1:nx, j in 1:ny)

    # A periodic direction interpolates ACROSS its seam rather than clamping at the last sample,
    # which is the case a caller composing per-axis weights by hand gets wrong.
    let nλ = 24, λ = collect(range(0.0, 2π * (1 - 1 / 24); length = 24)), z = [0.0, 1.0]
        gp = GD.StructuredGrid(cart, λ, z; periodic = (true, false), period = (2π, 0.0))
        fp = [sin(l) for l in λ, _ in z]
        mid = λ[end] + (2π - λ[end]) / 2         # strictly between the last sample and the first
        Test.@test D.interpolate(fp, gp, (mid, 0.0)) ≈ (sin(λ[end]) + sin(λ[1])) / 2
        Test.@test D.interpolate(fp, gp, (2π + 0.3, 0.0)) ≈ D.interpolate(fp, gp, (0.3, 0.0))
    end

    # Scattered and curvilinear: a least-squares plane, so a linear field is exact — a plain
    # inverse-distance average would not be.
    let npt = 400, xs = 10.0 .* rand(npt), ys = 6.0 .* rand(npt)
        gu = GD.UnstructuredGrid(cart, (xs, ys), trues(npt); k = 8, areas = ones(npt))
        fu = 2.0 .* xs .- 3.0 .* ys .+ 5.0
        for _ in 1:20
            px, py = 1.0 + 8.0 * rand(), 1.0 + 4.0 * rand()
            Test.@test D.interpolate(fu, gu, (px, py); k = 8) ≈ 2px - 3py + 5 atol = 1e-8
        end
        Test.@test all(D.interpolate(fu, gu, (xs[i], ys[i]); k = 8) ≈ fu[i] for i in 1:20)
    end
    let n = 14
        xc = [t + 0.35u for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
        yc = [u - 0.2t for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
        cg = GD.CurvilinearGrid(cart, xc, yc, trues(n, n); measure = fill(1.0, n, n))
        fc = 2.0 .* xc .- 3.0 .* yc .+ 5.0
        for _ in 1:15
            px, py = 2.0 + 5.0 * rand(), 0.0 + 3.0 * rand()
            Test.@test D.interpolate(fc, cg, (px, py); k = 8) ≈ 2px - 3py + 5 atol = 1e-8
        end
    end

    # The mask policies mean here what they mean for a stencil.
    let mk = trues(nx, ny)
        mk[5, 4] = false
        gm = GD.StructuredGrid(cart, x, y, mk)
        pin = ((x[5] + x[6]) / 2, (y[4] + y[5]) / 2)      # its corners include the hole
        Test.@test isnan(D.interpolate(f, gm, pin; masked = NaN))
        Test.@test !isnan(D.interpolate(f, gm, pin; masked = NaN, policy = D.ReduceInRun()))
        Test.@test !isnan(D.interpolate(f, gm, (x[1], y[1]); masked = NaN))
        Test.@test_throws ArgumentError D.interpolate(f, gm, pin; policy = D.ShiftWithinRun())
    end

    # A point is a point however it is written.
    let a = D.interpolate(f, g, (0.7, 1.1))
        Test.@test a == D.interpolate(f, g, [0.7, 1.1])
        Test.@test a == D.interpolate(f, g, (x = 0.7, y = 1.1))
        Test.@test a == D.interpolate(f, g, StaticArrays.SVector(0.7, 1.1))
    end
    Test.@test_throws DimensionMismatch D.interpolate(zeros(3, 3), g, (0.5, 0.5))
end

Test.@testset "A least-squares gradient where there is no separable axis" begin
    C = FG.Connectivity
    D = FG.Discretization
    GD = FG.Grids
    GE = FG.Geometry
    cart = FG.Geometry.CartesianGeometry{Float64}()

    # Deliberately SHEARED. Exactness for a linear field on a skewed stencil is the property that
    # distinguishes this from inverting an index-space Jacobian, which does not have it.
    n = 14
    x = [t + 0.35u for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
    y = [u - 0.2t for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
    cg = GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
    plan = C.gradient_plan(cg)
    Test.@test length(plan) == n * n
    for (a, b) in ((2.0, -3.0), (0.0, 1.0), (-1.5, 0.0))
        f = a .* x .+ b .* y .+ 7.0
        g1 = zeros(n, n); g2 = zeros(n, n)
        D.gradient!(g1, g2, f, plan)
        # Every cell, boundaries and corners included — a one-sided stencil is still exact for a
        # linear field as long as it spans both tangent directions.
        Test.@test maximum(abs.(g1 .- a)) < 1e-10
        Test.@test maximum(abs.(g2 .- b)) < 1e-10
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
        D.gradient!(q1, q2, ff, C.gradient_plan(og))
        s1 = zeros(m, m); s2 = zeros(m, m)
        D.apply_stencil!(s1, ff, sg, 1; order = 1, nodes = 3)
        D.apply_stencil!(s2, ff, sg, 2; order = 1, nodes = 3)
        int = 2:(m - 1)
        Test.@test maximum(abs.(q1[int, int] .- s1[int, int])) < 1e-12
        Test.@test maximum(abs.(q2[int, int] .- s2[int, int])) < 1e-12
    end

    # A node set, where neighbours come from connectivity rather than an index offset.
    let gu = C.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(8)),
        R = FG.Geometry.radius(GD.grid_geometry(C.unstructured_grid(
            FG.SphericalSampling.IcosahedralSampling(8))))
        φu = GD.coordinates(gu, 2)
        fu = [sin(φ) for φ in φu]                    # ∂/∂north = cos φ / R, ∂/∂east = 0
        h1 = zeros(length(fu)); h2 = zeros(length(fu))
        D.gradient!(h1, h2, fu, C.gradient_plan(gu))
        want = [cos(φ) / R for φ in φu]
        Test.@test maximum(abs.(h2 .- want)) < 0.05 * maximum(abs.(want))
        Test.@test maximum(abs.(h1)) < 0.05 * maximum(abs.(want))
    end

    # Rank deficiency: every neighbour on one line leaves the across-line component undetermined
    # by the data, so it is zeroed rather than produced by inverting a nudged matrix.
    let xl = collect(range(0.0, 5.0; length = 6)), yl = zeros(6)
        lg = GD.UnstructuredGrid(cart, (xl, yl), trues(6); k = 2, areas = ones(6))
        l1 = zeros(6); l2 = zeros(6)
        D.gradient!(l1, l2, 3.0 .* xl, C.gradient_plan(lg))
        Test.@test all(isapprox.(l1, 3.0; atol = 1e-10))     # the resolved direction is exact
        Test.@test all(iszero, l2)                           # the other is zero, not enormous
    end

    # A mask: the hole has no coefficients and reads zero, and its neighbours are still exact,
    # having simply not been offered it.
    let mk = trues(n, n)
        mk[5, 5] = false; mk[6, 5] = false
        mg = GD.CurvilinearGrid(cart, x, y, mk; measure = fill(1.0, n, n))
        mf = 2.0 .* x .- 3.0 .* y
        m1 = zeros(n, n); m2 = zeros(n, n)
        D.gradient!(m1, m2, mf, C.gradient_plan(mg))
        Test.@test iszero(m1[5, 5]) && iszero(m2[5, 5])
        Test.@test all(abs(m1[i, j] - 2.0) < 1e-10 && abs(m2[i, j] + 3.0) < 1e-10
                       for i in 1:n, j in 1:n if mk[i, j])
    end

    Test.@test plan.names == (:x, :y)
    Test.@test_throws DimensionMismatch D.gradient!(zeros(3), zeros(3), zeros(3), plan)
    # The tangent plane is two-dimensional, so a 3-coordinate grid is refused rather than guessed.
    let g3 = GD.UnstructuredGrid(cart, (rand(6), rand(6), rand(6)), trues(6); k = 3,
                                 areas = ones(6))
        Test.@test_throws ArgumentError C.gradient_plan(g3)
    end
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

    # Both are searches, not sweeps: a stretched axis must not be walked, nor materialize its
    # faces, on a query. Asserted by COUNTING the elements read — the claim itself — rather than
    # by a clock, which a collection can decide. `locate` reads two faces per bisection step plus
    # the two end faces, so the bound is generous but still far below `n`.
    for pow in (6, 18)
        n = 1 << pow
        c = CountingAxis(collect(range(0.0, 1.0; length = n)))
        D.locate(c, 0.37); D.nearest_index(c, 0.37)          # warm, then count
        Test.@test reads(() -> D.locate(c, 0.37), c) ≤ 8 * pow + 16
        Test.@test reads(() -> D.nearest_index(c, 0.37), c) ≤ 8 * pow + 16
    end
    # …and the bound really does discriminate: a scan of the larger axis would read `n` of them.
    let c = CountingAxis(collect(range(0.0, 1.0; length = 1 << 18)))
        Test.@test reads(() -> D.locate(c, 0.37), c) < 1000     # against n = 262144
        Test.@test reads(() -> sum(c), c) == 1 << 18            # the wrapper does count reads
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

Test.@testset "Per-index gaps and widths are public, exact and free of allocation" begin
    D = FG.Discretization
    GD = FG.Grids
    GE = FG.Geometry
    cart = FG.Geometry.CartesianGeometry{Float64}()
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    vec1 = cumsum([0.0, 1.0, 0.3, 2.5, 0.7, 4.0])
    rng1 = range(0.0, 1.0; length = 8)

    # `cell_width` IS the gap between faces — that is what it means, and the reason it exists
    # separately is that `faces` materializes the whole axis to answer for one cell.
    for x in (vec1, reverse(vec1), collect(rng1), [3.0])
        f = D.faces(x)
        for i in eachindex(x)
            Test.@test D.cell_width(x, i) ≈ abs(f[i+1] - f[i]) rtol = 1e-14
        end
    end

    # Gaps are SIGNED, and that is what lets a stencil keep the index-vs-coordinate direction.
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
    Test.@test D.cell_widths(rng1) isa FG.Axes.ConstantVector

    # The grid forms take the period from the grid, which is the part a caller composing the
    # axis form by hand gets wrong at a seam.
    grids = (
        ("range x range",   GD.StructuredGrid(cart, rng1, rng1)),
        ("range x Vector",  GD.StructuredGrid(cart, rng1, vec1)),
        ("Vector x range",  GD.StructuredGrid(cart, vec1, rng1)),
        ("Vector x Vector", GD.StructuredGrid(cart, vec1, vec1)),
        ("3D mixed",        GD.StructuredGrid(cart, rng1, vec1, collect(0.0:2.0))),
        ("periodic sphere", GD.StructuredGrid(sph, λ, collect(range(-1.0, 1.0; length = 9)))),
    )
    for (nm, g) in grids, d in 1:length(GD.coordinates(g))
        x = GD.coordinates(g, d)
        p = GD.isperiodic(g, d) ? GD.period(g, d) : nothing
        for i in 1:length(x)
            Test.@test GD.local_spacing(g, d, i) == D.local_spacing(x, i, p)
            Test.@test GD.cell_width(g, d, i) == D.cell_width(x, i, p)
        end
        Test.@test collect(GD.cell_widths(g, d)) == [D.cell_width(x, i, p) for i in 1:length(x)]
        # A runtime `d` indexes a tuple whose entries have different types on the mixed grids;
        # the result must still infer.
        Test.@inferred GD.local_spacing(g, d, 2)
        Test.@inferred GD.cell_width(g, d, 2)
    end
    Test.@test_throws ArgumentError GD.local_spacing(first(grids)[2], 3, 1)

    # The point of making them public: assembling the operator the module header says is the
    # caller's to assemble. A quadratic must come back exactly, in either storage order.
    fq(x) = 3x^2 - 2x + 5
    dfq(x) = 6x - 2
    for x in (vec1, reverse(vec1))
        for i in 2:(length(x) - 1)
            h_m, h_p = D.local_spacing(x, i)
            Test.@test GE.nonuniform_first_derivative(fq(x[i-1]), fq(x[i]), fq(x[i+1]), h_m, h_p) ≈
                       dfq(x[i]) atol = 1e-12
        end
    end
    # And across a periodic seam, where the wrapped gap is the whole point: the seam cell's error
    # is the interior truncation error, not the O(1) one a zero boundary gap would give.
    np = 32
    λ2 = collect(range(0.0, 2π * (1 - 1 / np); length = np))
    gper = GD.StructuredGrid(cart, λ2, [0.0]; periodic = (true, false), period = (2π, 0.0))
    errs = map(1:np) do i
        h_m, h_p = GD.local_spacing(gper, 1, i)
        return abs(GE.nonuniform_first_derivative(sin(λ2[mod1(i - 1, np)]), sin(λ2[i]),
                                                  sin(λ2[mod1(i + 1, np)]), h_m, h_p) - cos(λ2[i]))
    end
    Test.@test errs[1] ≈ maximum(errs) rtol = 0.5      # the seam is no worse than the interior
    Test.@test maximum(errs) < 0.01
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

    # The END OF THE AXIS bounds a window exactly as the end of a run does, so under `ReduceInRun`
    # `nodes` is a ceiling there too. These are ordinary degenerate grids — a single-latitude
    # strip, a two-level column, a one-cell channel — not caller mistakes.
    let xa = collect(0.0:5.0)
        # An axis with fewer samples than `nodes`: use what there is, and stay exact for the
        # degree the reduced window still supports.
        for (ny, want) in ((2, 5.0), (3, 5.0))
            ys = collect(range(0.0, 1.0; length = ny))
            g2 = GD.StructuredGrid(geo, xa, ys)
            f2 = [xi + 5yi for xi in xa, yi in ys]
            o2 = zeros(length(xa), ny)
            D.apply_stencil!(o2, f2, g2, 2; order = 1, nodes = 5, policy = D.ReduceInRun(),
                             masked = NaN)
            Test.@test all(isapprox.(o2, want))
        end
        # Fewer than `order + 1` samples: no derivative of that order exists anywhere on the axis.
        g1 = GD.StructuredGrid(geo, xa, [3.0])
        f1 = reshape([2xi for xi in xa], length(xa), 1)
        o1 = zeros(length(xa), 1)
        D.apply_stencil!(o1, f1, g1, 2; order = 1, nodes = 3, policy = D.ReduceInRun(),
                         masked = NaN)
        Test.@test all(isnan, o1)
        # …while the other direction of the very same grid is untouched by any of it.
        od = zeros(length(xa), 1)
        D.apply_stencil!(od, f1, g1, 1; order = 1, nodes = 3, policy = D.ReduceInRun())
        Test.@test all(isapprox.(od, 2.0))
        # The policies that do not claim to degrade keep the error.
        for pol in (D.BlankMasked(), D.ShiftWithinRun())
            Test.@test_throws ArgumentError D.apply_stencil!(zeros(length(xa), 1), f1, g1, 2;
                                                             order = 1, nodes = 3, policy = pol)
        end
        # And `axis_stencils` itself still means exactly `nodes`: it is handed no policy.
        Test.@test_throws ArgumentError D.axis_stencils([0.0, 1.0], 1, 3)
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

Test.@testset "A trailing batch axis is differenced in one pass, identically to a slice loop" begin
    D = FG.Discretization
    GD = FG.Grids
    GE = FG.Geometry
    cart = GE.CartesianGeometry{Float64}()
    nx, ny, nb, nb2 = 14, 9, 4, 3
    x = collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = nx))))
    mk = trues(nx, ny); mk[5, 4] = false; mk[6, 4] = false; mk[11, 2] = false
    idx, w = D.axis_stencils(x, 1, 3)

    # `K = 0` proves the unbatched path is untouched; `K = 1` and `K = 2` that a batch is the same
    # answer. Every mask policy, because the mask is the part a widened signature alone gets wrong: it
    # spans the grid's axes only, and is indexed by the leading components of each cell index.
    for dims in ((nx, ny), (nx, ny, nb), (nx, ny, nb, nb2))
        f = reshape([sin(3i) * cos(2j) * (k + 1)
                     for i in 1:dims[1], j in 1:dims[2], k in 1:prod(dims[3:end]; init = 1)], dims)
        for pol in (D.BlankMasked(), D.ShiftWithinRun(), D.ReduceInRun())
            o = fill(NaN, dims); ref = fill(NaN, dims)
            D.apply_stencil!(o, f, x, 1; order = 1, nodes = 3, mask = mk, masked = NaN, policy = pol)
            for c in CartesianIndices(dims[3:end])
                D.apply_stencil!(view(ref, :, :, Tuple(c)...), view(f, :, :, Tuple(c)...), x, 1;
                                 order = 1, nodes = 3, mask = mk, masked = NaN, policy = pol)
            end
            Test.@test all(isequal(o[i], ref[i]) for i in eachindex(o))
        end
        # The device body walks the whole output, batch included, so it must agree bit for bit too.
        o1 = fill(NaN, dims); o2 = fill(NaN, dims)
        D.apply_stencil!(o1, f, x, idx, w, 1; mask = mk, masked = NaN)
        D.apply_stencil!(o2, f, x, idx, w, 1; mask = mk, masked = NaN, backend = KernelAbstractions.CPU())
        Test.@test all(isequal(o1[i], o2[i]) for i in eachindex(o1))
    end

    # `derivative!` divides by a metric factor, which is spatial-only and so is solved once per spatial
    # index and reused across the batch — the batched result must still equal the slice loop exactly.
    sph = GE.SphericalGeometry(6.371e6)
    λ = collect(range(0.0, 2π * (1 - 1 / nx); length = nx))
    φ = collect(range(-1.1, 1.1; length = ny))
    for g in (GD.StructuredGrid(sph, λ, φ), GD.StructuredGrid(sph, λ, φ, mk),
              GD.StructuredGrid(cart, collect(1.0:nx), collect(1.0:ny)))
        f = [sin(2a) * cos(b) * (k + 1) for a in λ, b in φ, k in 1:nb]
        for dim in 1:2, pol in (D.BlankMasked(), D.ReduceInRun())
            o = fill(NaN, nx, ny, nb); ref = fill(NaN, nx, ny, nb)
            D.derivative!(o, f, g, dim; order = 1, nodes = 3, masked = NaN, policy = pol)
            for b in 1:nb
                D.derivative!(view(ref, :, :, b), view(f, :, :, b), g, dim;
                              order = 1, nodes = 3, masked = NaN, policy = pol)
            end
            Test.@test all(isequal(o[i], ref[i]) for i in eachindex(o))
        end
        let (i2, w2) = D.axis_stencils(g, 2; order = 1, nodes = 3)
            o = fill(NaN, nx, ny, nb); ref = fill(NaN, nx, ny, nb)
            D.derivative!(o, f, g, i2, w2, 2; order = 1, masked = NaN)
            for b in 1:nb
                D.derivative!(view(ref, :, :, b), view(f, :, :, b), g, i2, w2, 2;
                              order = 1, masked = NaN)
            end
            Test.@test all(isequal(o[i], ref[i]) for i in eachindex(o))
        end
    end

    # The metric floor still masks a pole row, in every batch element — the `Float32` case an absolute
    # threshold gets wrong.
    let λ32 = collect(range(0.0f0, 2π * (1 - 1 / 8); length = 8)), φ32 = Float32[-π/2, 0, π/2]
        g32 = GD.StructuredGrid(GE.SphericalGeometry(6.371f6), λ32, φ32)
        f32 = [sin(a) * b for a in λ32, b in φ32, _ in 1:2]
        o32 = fill(NaN32, 8, 3, 2)
        D.derivative!(o32, f32, g32, 1; order = 1, nodes = 3, masked = NaN32)
        Test.@test all(isnan, view(o32, :, 1, :)) && all(isnan, view(o32, :, 3, :))
        Test.@test !any(isnan, view(o32, :, 2, :))
    end

    # Implicit-by-rank must not swallow a real mistake.
    g2 = GD.StructuredGrid(cart, collect(1.0:nx), collect(1.0:ny))
    Test.@test_throws DimensionMismatch D.derivative!(zeros(nx, ny + 1, nb), zeros(nx, ny + 1, nb), g2, 1)
    Test.@test_throws ArgumentError D.derivative!(zeros(nx, ny, nb), zeros(nx, ny, nb), g2, 3)
    Test.@test_throws DimensionMismatch D.derivative!(zeros(nx, ny, nb), zeros(nx, ny, 2), g2, 1)
    Test.@test_throws DimensionMismatch D.apply_stencil!(zeros(nx, ny, nb), zeros(nx, ny, nb), x,
                                                         idx, w, 1; mask = trues(nx, ny + 1))
end

Test.@testset "A batch is evaluated at a coordinate, and gradients take one too" begin
    D = FG.Discretization
    C = FG.Connectivity
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    nx, ny, nb = 12, 9, 5
    x = collect(range(0.0, 11.0; length = nx)); y = collect(range(0.0, 8.0; length = ny))
    mk = trues(nx, ny); mk[4, 4] = false

    for g in (GD.StructuredGrid(cart, x, y), GD.StructuredGrid(cart, x, y, mk))
        f = [2.0xi - 3.0yj + 100.0b for xi in x, yj in y, b in 1:nb]
        for p in ((3.7, 2.4), (8.4, 6.1)), pol in (D.BlankMasked(), D.ReduceInRun())
            out = Vector{Float64}(undef, nb)
            D.interpolate!(out, f, g, p; masked = NaN, policy = pol)
            ref = [D.interpolate(view(f, :, :, b), g, p; masked = NaN, policy = pol) for b in 1:nb]
            Test.@test all(isequal(out[b], ref[b]) for b in 1:nb)
            Test.@test all(isequal(D.interpolate(f, g, p; masked = NaN, policy = pol)[b], ref[b])
                           for b in 1:nb)          # the allocating form is the same values
        end
        # An unbatched field still answers with a scalar: the rank decides, not a length.
        Test.@test D.interpolate(view(f, :, :, 1), g, (8.4, 6.1)) isa Float64
    end
    Test.@test_throws DimensionMismatch D.interpolate!(
        Vector{Float64}(undef, 2), zeros(nx, ny, nb), GD.StructuredGrid(cart, x, y), (1.0, 1.0))

    # Off a rectilinear grid the neighbour set and the tangent-plane fit are the point's, not the
    # data's, so they are solved once; the values must still match a call per slice.
    n = 12
    X = [0.7i + 0.05j for i in 1:n, j in 1:n]; Y = [0.9j - 0.03i for i in 1:n, j in 1:n]
    cg = GD.CurvilinearGrid(cart, X, Y, trues(n, n); measure = fill(1.0, n, n))
    fb = [2.0X[i, j] - 3.0Y[i, j] + 50.0b for i in 1:n, j in 1:n, b in 1:4]
    vb = D.interpolate(fb, cg, (4.2, 5.1))
    Test.@test vb isa Vector{Float64} && length(vb) == 4
    Test.@test all(isapprox(vb[b], D.interpolate(view(fb, :, :, b), cg, (4.2, 5.1)); atol = 1e-10)
                   for b in 1:4)
    Test.@test all(abs(vb[b] - (2.0 * 4.2 - 3.0 * 5.1 + 50.0b)) < 1e-8 for b in 1:4)
    Test.@test D.interpolate(view(fb, :, :, 1), cg, (4.2, 5.1)) isa Float64

    gu = C.unstructured_grid(FG.SphericalSampling.HEALPixSampling(4))
    m = length(GD.mask(gu))
    fu = [Float64(i) + 1000.0b for i in 1:m, b in 1:3]
    vu = D.interpolate(fu, gu, (0.4, 0.1))
    Test.@test vu isa Vector{Float64}
    Test.@test all(isapprox(vu[b], D.interpolate(view(fu, :, b), gu, (0.4, 0.1)); atol = 1e-8)
                   for b in 1:3)

    # A gradient plan is geometry, so one plan serves the whole batch.
    plan = C.gradient_plan(cg)
    G1 = zeros(n, n, 4); G2 = zeros(n, n, 4)
    D.gradient!(G1, G2, fb, plan)
    R1 = zeros(n, n, 4); R2 = zeros(n, n, 4)
    for b in 1:4
        D.gradient!(view(R1, :, :, b), view(R2, :, :, b), view(fb, :, :, b), plan)
    end
    Test.@test G1 == R1 && G2 == R2
    Test.@test all(abs(G1[i, j, b] - 2.0) < 1e-10 && abs(G2[i, j, b] + 3.0) < 1e-10
                   for i in 2:(n - 1), j in 2:(n - 1), b in 1:4)
    Test.@test_throws DimensionMismatch D.gradient!(zeros(n, n, 4), zeros(n, n, 4), zeros(n * 4 + 1), plan)
end
