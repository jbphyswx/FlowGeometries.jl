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
    O = FG.Operators
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
        O.derivative!(o32, f32, g32, 1; order = 1, nodes = 3, masked = Float32(NaN))
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
        O.apply_stencil!(a, fc, gc, 1; order = 1, nodes = 3)
        O.derivative!(b, fc, gc, 1; order = 1, nodes = 3)
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

    # The bare `(indices, weights)` form cannot degrade — it has no axis to rebuild a window from
    # at a mask edge — so a caller wanting `ReduceInRun` had to give up the table entirely and pay
    # its rebuild per call. Handing the axis alongside the table serves every policy.
    for dim in 1:2, pol in (O.BlankMasked(), O.ShiftWithinRun(), O.ReduceInRun()), k in (3, 5)
        a = zeros(n, n); b = zeros(n, n)
        O.apply_stencil!(a, f, g, dim; order = 1, nodes = k, masked = NaN, policy = pol)
        idx, w = D.axis_stencils(g, dim; order = 1, nodes = k)
        O.apply_stencil!(b, f, g, idx, w, dim; order = 1, masked = NaN, policy = pol)
        Test.@test all(isequal(a[i], b[i]) for i in eachindex(a))
    end


    # The bare form still refuses rather than silently ignoring the policy.
    let idx = D.axis_stencils(g, 1; order = 1, nodes = 3)
        Test.@test_throws ArgumentError O.apply_stencil!(zeros(n, n), f, idx[1], idx[2], 1;
                                                         mask = mk, policy = O.ReduceInRun())
    end

    # `derivative!` is the form a geometry-aware caller uses, so it takes a table too.
    let sph = FG.Geometry.SphericalGeometry(6.371e6),
        lam = collect(range(0.0, 2π * (1 - 1 / 16); length = 16)),
        phi = collect(range(-1.2, 1.2; length = 13))
        gs = GD.StructuredGrid(sph, lam, phi)
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
        O.apply_stencil!(a, f, idx, w, dim; mask = mm, masked = NaN)
        O.apply_stencil!(b, f, idx, w, dim; mask = mm, masked = NaN, backend = KernelAbstractions.CPU())
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
            O.apply_stencil!(ob, f3, idx, w, dim; backend = KernelAbstractions.CPU())
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
        O.apply_stencil!(o, f, idx, w, 1)
        v = view(f, :, :)                                  # a view is still linear here
        ov = zeros(n, n)
        O.apply_stencil!(ov, v, idx, w, 1)
        Test.@test o == ov
    end

    # A stencil wider than the specialization cap keeps the runtime-node-count path, and agrees.
    let n = 32
        xs = collect(range(0.0, 1.0; length = n))
        f = [sin(4xi) * yi for xi in xs, yi in xs]
        for k in (10, 12)
            idx, w = D.axis_stencils(xs, 1, k)
            a = zeros(n, n); b = zeros(n, n)
            O.apply_stencil!(a, f, idx, w, 1)
            O.apply_stencil!(b, f, idx, w, 1; backend = KernelAbstractions.CPU())
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
            O.apply_stencil!(a, f, g, dim; order = 1, nodes = 5, masked = NaN)
            idx, w = D.axis_stencils(g, dim; order = 1, nodes = 5)
            O.apply_stencil!(b, f, g, idx, w, dim; masked = NaN)
            Test.@test all(isequal(a[i], b[i]) for i in eachindex(a))
        end
        # The periodic direction's table is genuinely the wrapped one.
        Test.@test D.axis_stencils(g, 2; order = 1, nodes = 5)[1] !=
                   D.axis_stencils(ys, 1, 5)[1]
    end
end

Test.@testset "apply_stencil! differentiates a field along one direction" begin
    D = FG.Discretization
    O = FG.Operators
    geo = FG.Geometry.CartesianGeometry()

    # Exact for any polynomial the node count spans, at EVERY sample — the ends included, because
    # the stencil shifts inward rather than clipping to a lower order.
    x = collect(range(0.0, 2.0; length = 11))
    f = @. 3x^2 - 2x + 5
    out = similar(f)
    O.apply_stencil!(out, f, x, 1; order = 1, nodes = 3)
    Test.@test maximum(abs, out .- (6 .* x .- 2)) < 1e-12
    O.apply_stencil!(out, f, x, 1; order = 2, nodes = 3)
    Test.@test maximum(abs, out .- 6.0) < 1e-11

    # A stretched axis is equally exact: the weights are per-sample, not one set reused.
    xs = [0.0, 0.11, 0.37, 0.9, 1.05, 1.6, 1.62, 2.0]
    outs = similar(xs)
    O.apply_stencil!(outs, (@. 3xs^2 - 2xs + 5), xs, 1; order = 1, nodes = 3)
    Test.@test maximum(abs, outs .- (6 .* xs .- 2)) < 1e-11
    # 3 nodes cannot span a cubic; 4 can. Both statements matter — the first shows the test bites.
    O.apply_stencil!(outs, xs .^ 3, xs, 1; order = 1, nodes = 3)
    Test.@test maximum(abs, outs .- 3 .* xs .^ 2) > 1e-3
    O.apply_stencil!(outs, xs .^ 3, xs, 1; order = 1, nodes = 4)
    Test.@test maximum(abs, outs .- 3 .* xs .^ 2) < 1e-11

    # Periodic: the stencil stays centred and wraps, so the seam is no worse than the interior,
    # and a 5-node stencil converges at 4th order rather than to machine precision.
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

    # A derivative that would read an inactive cell is not invented.
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
    Test.@test size(idx) == (9, 3) && size(w) == (9, 3)
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

    # A periodic direction interpolates ACROSS its seam rather than clamping at the last sample,
    # which is the case a caller composing per-axis weights by hand gets wrong.
    let nλ = 24, λ = collect(range(0.0, 2π * (1 - 1 / 24); length = 24)), z = [0.0, 1.0]
        gp = GD.StructuredGrid(cart, λ, z; periodic = (true, false), period = (2π, 0.0))
        fp = [sin(l) for l in λ, _ in z]
        mid = λ[end] + (2π - λ[end]) / 2         # strictly between the last sample and the first
        Test.@test O.interpolate(fp, gp, (mid, 0.0)) ≈ (sin(λ[end]) + sin(λ[1])) / 2
        Test.@test O.interpolate(fp, gp, (2π + 0.3, 0.0)) ≈ O.interpolate(fp, gp, (0.3, 0.0))
    end

    # Scattered and curvilinear: a least-squares plane, so a linear field is exact — a plain
    # inverse-distance average would not be.
    let npt = 400, xs = 10.0 .* rand(npt), ys = 6.0 .* rand(npt)
        gu = GD.UnstructuredGrid(cart, (xs, ys), trues(npt); k = 8, areas = ones(npt))
        fu = 2.0 .* xs .- 3.0 .* ys .+ 5.0
        for _ in 1:20
            px, py = 1.0 + 8.0 * rand(), 1.0 + 4.0 * rand()
            Test.@test O.interpolate(fu, gu, (px, py); k = 8) ≈ 2px - 3py + 5 atol = 1e-8
        end
        Test.@test all(O.interpolate(fu, gu, (xs[i], ys[i]); k = 8) ≈ fu[i] for i in 1:20)
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

    # Deliberately SHEARED. Exactness for a linear field on a skewed stencil is the property that
    # distinguishes this from inverting an index-space Jacobian, which does not have it.
    n = 14
    x = [t + 0.35u for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
    y = [u - 0.2t for t in range(0.0, 10.0; length = n), u in range(0.0, 6.0; length = n)]
    cg = GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
    plan = O.gradient_plan(cg)
    Test.@test length(plan) == n * n
    for (a, b) in ((2.0, -3.0), (0.0, 1.0), (-1.5, 0.0))
        f = a .* x .+ b .* y .+ 7.0
        g1 = zeros(n, n); g2 = zeros(n, n)
        O.gradient!(g1, g2, f, plan)
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
        O.gradient!(q1, q2, ff, O.gradient_plan(og))
        s1 = zeros(m, m); s2 = zeros(m, m)
        O.apply_stencil!(s1, ff, sg, 1; order = 1, nodes = 3)
        O.apply_stencil!(s2, ff, sg, 2; order = 1, nodes = 3)
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
        O.gradient!(h1, h2, fu, O.gradient_plan(gu))
        want = [cos(φ) / R for φ in φu]
        Test.@test maximum(abs.(h2 .- want)) < 0.05 * maximum(abs.(want))
        Test.@test maximum(abs.(h1)) < 0.05 * maximum(abs.(want))
    end

    # Rank deficiency: every neighbour on one line leaves the across-line component undetermined
    # by the data, so it is zeroed rather than produced by inverting a nudged matrix.
    let xl = collect(range(0.0, 5.0; length = 6)), yl = zeros(6)
        lg = GD.UnstructuredGrid(cart, (xl, yl), trues(6); k = 2, areas = ones(6))
        l1 = zeros(6); l2 = zeros(6)
        O.gradient!(l1, l2, 3.0 .* xl, O.gradient_plan(lg))
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
        O.gradient!(m1, m2, mf, O.gradient_plan(mg))
        Test.@test iszero(m1[5, 5]) && iszero(m2[5, 5])
        Test.@test all(abs(m1[i, j] - 2.0) < 1e-10 && abs(m2[i, j] + 3.0) < 1e-10
                       for i in 1:n, j in 1:n if mk[i, j])
    end

    # A spheroid. The projection is written against `embed` and the local frame rather than against the
    # sphere's formulas, so the least-squares gradient and the scattered fit hold there too: the frame
    # is the sphere's at the same (λ, φ), the positions each (λ, φ) sits at are not.
    let spd = GE.SpheroidGeometry(), nx = 7, ny = 6, np = 7 * 6
        λs = [0.4 + 0.01 * (i - 1) for j in 1:ny for i in 1:nx]
        φs = [0.6 + 0.011 * (j - 1) for j in 1:ny for i in 1:nx]
        sg = GD.UnstructuredGrid(spd, (λs, φs), trues(np); k = 6, areas = ones(np))
        sp = O.gradient_plan(sg)
        Test.@test sp.names == (:λ, :φ)
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

    Test.@test plan.names == (:x, :y)
    Test.@test O.ncomponents(plan) == 2
    Test.@test_throws DimensionMismatch O.gradient!(zeros(3), zeros(3), zeros(3), plan)

    # Three coordinates resolve three directions, on the local frame rather than a tangent plane.
    let m = 4, np = 4^3
        xs = [1.0 * (i - 1) for k in 1:m for j in 1:m for i in 1:m]
        ys = [1.0 * (j - 1) for k in 1:m for j in 1:m for i in 1:m]
        zs = [1.0 * (k - 1) for k in 1:m for j in 1:m for i in 1:m]
        g3 = GD.UnstructuredGrid(cart, (xs, ys, zs), trues(np); k = 6, areas = ones(np))
        p3 = O.gradient_plan(g3)
        Test.@test O.ncomponents(p3) == 3
        Test.@test p3.names == (:x, :y, :z)
        α, β, γ = 2.0, -3.0, 0.5
        f3 = α .* xs .+ β .* ys .+ γ .* zs .+ 7.0
        u1 = zeros(np); u2 = zeros(np); u3 = zeros(np)
        O.gradient!(u1, u2, u3, f3, p3)
        # Interior nodes, whose six nearest neighbours span all three directions.
        lin = LinearIndices((m, m, m))
        int3 = [lin[i, j, k] for k in 2:(m - 1), j in 2:(m - 1), i in 2:(m - 1)]
        Test.@test all(abs(u1[t] - α) < 1e-10 && abs(u2[t] - β) < 1e-10 && abs(u3[t] - γ) < 1e-10
                       for t in int3)
        # The tuple form is the same call, and the positional one forwards to it.
        v1 = zeros(np); v2 = zeros(np); v3 = zeros(np)
        O.gradient!((v1, v2, v3), f3, p3)
        Test.@test v1 == u1 && v2 == u2 && v3 == u3

        # Rank deficiency in 3-D: every neighbour in one plane leaves the normal component
        # undetermined by the data, so it is zeroed rather than produced by a nudged inverse.
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

    # A fourth coordinate has no fixed-size solve behind it, so it is refused rather than guessed.
    let g4 = GD.UnstructuredGrid(cart, (rand(8), rand(8), rand(8), rand(8)), trues(8); k = 4,
                                 areas = ones(8))
        Test.@test_throws ArgumentError O.gradient_plan(g4)
    end
end

Test.@testset "The symmetric pseudo-inverse satisfies the Moore-Penrose conditions" begin
    O = FG.Operators
    # `_sympinv3` is closed form and eigenvector-free, on the grounds that a pseudo-inverse of a
    # symmetric matrix is a polynomial in it. What makes that valid is exactly these four identities,
    # so they are what gets asserted — at every rank, the rank-deficient cases being the reason a
    # pseudo-inverse exists at all.
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
        # A⁺ must annihilate the null space rather than invent a direction the data never spanned.
        for v in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            Av = ntuple(i -> sum(A[i][j] * v[j] for j in 1:3), Val(3))
            if sqrt(sum(abs2, Av)) < 1e-12
                Pv = ntuple(i -> sum(P[i][j] * v[j] for j in 1:3), Val(3))
                Test.@test sqrt(sum(abs2, Pv)) < 1e-9
            end
        end
    end
    # `D = 2` goes through the 2×2 closed form unchanged, which is what keeps its results identical.
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
    O = FG.Operators
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
    # whole axis. That is what the other policies exist to avoid.
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

    # Nothing anyone gets today changes: no mask, or the default policy, is bit-identical.
    for nodes in (2, 3, 5), pol in (O.BlankMasked(), O.ShiftWithinRun(), O.ReduceInRun())
        a = zeros(7, 1); b = zeros(7, 1)
        O.apply_stencil!(a, f, nomask, 1; order = 1, nodes = nodes)
        O.apply_stencil!(b, f, nomask, 1; order = 1, nodes = nodes, policy = pol)
        Test.@test a == b
    end
    for nodes in (2, 3, 5)
        a = zeros(7, 1); b = zeros(7, 1)
        O.apply_stencil!(a, f, grid, 1; order = 1, nodes = nodes)
        O.apply_stencil!(b, f, grid, 1; order = 1, nodes = nodes, policy = O.BlankMasked())
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

    # And the accuracy order survives, which is the reason to shift rather than clip.
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

    # The matrix form has no axis to rebuild from, and says so rather than ignoring the policy.
    idx, w = D.axis_stencils(x, 1, 3)
    Test.@test_throws ArgumentError O.apply_stencil!(zeros(7, 1), f, idx, w, 1;
                                                     mask = msk, policy = O.ShiftWithinRun())

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
    O = FG.Operators
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
        for pol in (O.BlankMasked(), O.ShiftWithinRun(), O.ReduceInRun())
            o = fill(NaN, dims); ref = fill(NaN, dims)
            O.apply_stencil!(o, f, x, 1; order = 1, nodes = 3, mask = mk, masked = NaN, policy = pol)
            for c in CartesianIndices(dims[3:end])
                O.apply_stencil!(view(ref, :, :, Tuple(c)...), view(f, :, :, Tuple(c)...), x, 1;
                                 order = 1, nodes = 3, mask = mk, masked = NaN, policy = pol)
            end
            Test.@test all(isequal(o[i], ref[i]) for i in eachindex(o))
        end
        # The device body walks the whole output, batch included, so it must agree bit for bit too.
        o1 = fill(NaN, dims); o2 = fill(NaN, dims)
        O.apply_stencil!(o1, f, x, idx, w, 1; mask = mk, masked = NaN)
        O.apply_stencil!(o2, f, x, idx, w, 1; mask = mk, masked = NaN, backend = KernelAbstractions.CPU())
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

    # The metric floor still masks a pole row, in every batch element — the `Float32` case an absolute
    # threshold gets wrong.
    let λ32 = collect(range(0.0f0, 2π * (1 - 1 / 8); length = 8)), φ32 = Float32[-π/2, 0, π/2]
        g32 = GD.StructuredGrid(GE.SphericalGeometry(6.371f6), λ32, φ32)
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
            Test.@test all(isequal(O.interpolate(f, g, p; masked = NaN, policy = pol)[b], ref[b])
                           for b in 1:nb)          # the allocating form is the same values
        end
        # An unbatched field still answers with a scalar: the rank decides, not a length.
        Test.@test O.interpolate(view(f, :, :, 1), g, (8.4, 6.1)) isa Float64
    end
    Test.@test_throws DimensionMismatch O.interpolate!(
        Vector{Float64}(undef, 2), zeros(nx, ny, nb), GD.StructuredGrid(cart, x, y), (1.0, 1.0))

    # Off a rectilinear grid the neighbour set and the tangent-plane fit are the point's, not the
    # data's, so they are solved once; the values must still match a call per slice.
    n = 12
    X = [0.7i + 0.05j for i in 1:n, j in 1:n]; Y = [0.9j - 0.03i for i in 1:n, j in 1:n]
    cg = GD.CurvilinearGrid(cart, X, Y, trues(n, n); measure = fill(1.0, n, n))
    fb = [2.0X[i, j] - 3.0Y[i, j] + 50.0b for i in 1:n, j in 1:n, b in 1:4]
    vb = O.interpolate(fb, cg, (4.2, 5.1))
    Test.@test vb isa Vector{Float64} && length(vb) == 4
    Test.@test all(isapprox(vb[b], O.interpolate(view(fb, :, :, b), cg, (4.2, 5.1)); atol = 1e-10)
                   for b in 1:4)
    Test.@test all(abs(vb[b] - (2.0 * 4.2 - 3.0 * 5.1 + 50.0b)) < 1e-8 for b in 1:4)
    Test.@test O.interpolate(view(fb, :, :, 1), cg, (4.2, 5.1)) isa Float64

    gu = healpix_node_grid(4)
    m = length(GD.mask(gu))
    fu = [Float64(i) + 1000.0b for i in 1:m, b in 1:3]
    vu = O.interpolate(fu, gu, (0.4, 0.1))
    Test.@test vu isa Vector{Float64}
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

    # A geometry whose h_λ varies with λ: the hoist would divide every cell of a row by the factor at
    # λ = λ[1]. `derivative!` must instead divide each cell by its own.
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

    # The sphere's answers are unchanged by the routing: the hoisted and per-cell paths are the same
    # arithmetic where the factor genuinely is constant along the row.
    gs = GD.StructuredGrid(GE.SphericalGeometry(), λ, φ; periodic = (true, false),
                           period = (2π, 0.0))
    for dim in (1, 2)
        got = similar(f)
        O.derivative!(got, f, gs, dim; order = 1, nodes = 5)
        got2 = similar(f)
        O.apply_stencil!(got2, f, gs, dim; order = 1, nodes = 5)
        Test.@test all(isfinite, got[:, 2:(ny - 1)])
        # dividing the raw stencil by the factor at each cell reproduces it
        want = [got2[i, j] / GE.scale_factors(GE.SphericalGeometry(), (λ[i], φ[j]))[dim]
                for i in 1:nx, j in 1:ny]
        Test.@test got ≈ want rtol = 1e-14
    end
end

Test.@testset "An 8-node stencil keeps its unrolled inner loop" begin
    O = FG.Operators
    # k nodes span degree ≤ k−1, so 7 must miss a degree-7 polynomial and 8 must hit it — the pair
    # that shows the specialisation both bites and is right.
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

    # Which form a plan takes follows the axis's TYPE, never its values: a vector of equally spaced
    # numbers is still a stretched axis, because nothing in its type says otherwise.
    for (x, per, uniform) in ((0.0:0.5:10.0, nothing, true),
                              (range(0, 2π * (1 - 1 / 32); length = 32), 2π, true),
                              (A.UniformAxis(0.0, 0.25, 40), nothing, true),
                              (collect(0.0:0.5:19.5), nothing, false),
                              (collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = 40)))),
                               nothing, false))
        pl = D.stencil_plan(x, 1, 5; period = per)
        Test.@test (pl isa D.UniformStencilPlan) == uniform
        Test.@test D.axis_length(pl) == length(x)
        Test.@test D.nnodes(pl) == 5
        Test.@test D.derivative_order(pl) == 1
    end
    # A period that is not the one the spacing implies means the seam is not uniform, so the table is
    # what describes the axis.
    Test.@test D.stencil_plan(range(0.0, 1.0; length = 16), 1, 3; period = 99.0) isa
               D.TabulatedStencilPlan

    # A plan reads exactly the samples the table reads, and its weights are the same numbers to
    # round-off — the table recomputes each row from its own window, so it carries per-row noise.
    for (x, per) in ((0.0:0.5:10.0, nothing),
                     (range(0, 2π * (1 - 1 / 32); length = 32), 2π),
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

    # A uniform plan's storage does not depend on the axis length; a tabulated one is O(n·K).
    su = [Base.summarysize(D.stencil_plan(range(0.0, 1.0; length = m), 1, 5))
          for m in (64, 256, 1024, 4096)]
    st = [Base.summarysize(D.stencil_plan(collect(cumsum(fill(1.0, m))), 1, 5))
          for m in (64, 256, 1024, 4096)]
    Test.@test allequal(su)
    Test.@test issorted(st) && st[end] > 20 * st[1]

    # Applying a plan agrees with applying the table, on every shape the sweep splits on: the
    # differenced direction contiguous or not, masked, batched, wrapping, and an even node count.
    for (nx, ny, nb) in ((17, 9, 0), (32, 32, 0), (9, 17, 3))
        for per in (nothing, 2π)
            ax = per === nothing ? range(0.0, 4.0; length = nx) :
                                   range(0, 2π * (1 - 1 / nx); length = nx)
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

    # K nodes span degree K−1: exact there, and not above it.
    xu = collect(range(0.0, 1.0; length = 40))
    for k in (3, 4, 5, 6)
        pl = D.stencil_plan(range(0.0, 1.0; length = 40), 1, k)
        deg = k - 1
        o = similar(xu)
        O.apply_stencil!(o, [a^deg for a in xu], pl, 1)
        Test.@test maximum(abs.(o .- [deg * a^(deg - 1) for a in xu])) < 1e-9
    end

    # Applying a held plan allocates nothing, which is the point of holding it.
    let xs = range(0.0, 1.0; length = 64), fld = [sin(20a) * b for a in xs, b in 1:64]
        out = similar(fld)
        for k in (3, 4, 5, 7)
            pl = D.stencil_plan(xs, 1, k)
            Test.@test _alloc(q_plan!, out, fld, pl, 1) == 0
            Test.@test _alloc(q_plan!, out, fld, pl, 2) == 0
        end
    end

    # A plan describes one axis length, and a field of another is a mistake rather than a wrong answer.
    Test.@test_throws DimensionMismatch O.apply_stencil!(
        zeros(8, 4), zeros(8, 4), D.stencil_plan(range(0.0, 1.0; length = 9), 1, 3), 1)
end

Test.@testset "derivative! fuses the metric into the sweep, identically" begin
    GE = FG.Geometry
    GD = FG.Grids
    O = FG.Operators
    D = FG.Discretization

    # The factor has to be constant across each direction-1 run the sweep writes, which for a geometry
    # declaring direction 1 metric-invariant it is — in every direction, on any axis spacing.
    let cart = GE.CartesianGeometry(), sph = GE.SphericalGeometry()
        f2 = zeros(12, 12)
        f3 = zeros(12, 12, 12)
        # a Cartesian metric is the identity: nothing to fuse
        Test.@test !O._fusable(f2, f2, cart, nothing)
        Test.@test O._fusable(f2, f2, sph, nothing)
        Test.@test O._fusable(f3, f3, sph, nothing)
        # a geometry that does not declare direction 1 invariant
        Test.@test !O._fusable(f2, f2, TiltedSphere{Float64}(), nothing)
        # and the run addressing needs the linear layout
        Test.@test !O._fusable(view(f3, 1:2:11, :, :), f3, sph, nothing)
    end

    # Where it fuses, the answer must be what the two passes gave — bit for bit, being the same
    # multiplication by the same factor. Uniform and stretched axes, every direction, each with a mask
    # and without one, with a batch axis and without one, through all three entry points.
    fused_seen = 0
    for geo in (GE.SphericalGeometry(), GE.SpheroidGeometry())
        for (nx, ny, nz, nb) in ((16, 12, 0, 0), (9, 7, 0, 3), (8, 6, 5, 0), (8, 6, 5, 2))
            λ = range(0, 2π * (1 - 1 / nx); length = nx)
            for φ in (range(-1.4, 1.4; length = ny),
                      [-1.4 + 2.8 * (t / (ny - 1))^1.3 for t in 0:(ny - 1)])
                axs = nz == 0 ? (λ, φ) : (λ, φ, range(1.0, 2.0; length = nz))
                szs = nz == 0 ? (nx, ny) : (nx, ny, nz)
                fsz = nb == 0 ? szs : (szs..., nb)
                fld = reshape(collect(Float64, 1:prod(fsz)), fsz) ./ prod(fsz)
                msk = trues(szs)
                msk[CartesianIndices(szs)[2]] = false
                for gg in (GD.StructuredGrid(geo, axs...;
                                periodic = (true, false, false)[1:length(axs)],
                                period = (2π, 0.0, 0.0)[1:length(axs)]),
                           GD.StructuredGrid(geo, axs...; mask = msk,
                                periodic = (true, false, false)[1:length(axs)],
                                period = (2π, 0.0, 0.0)[1:length(axs)]))
                    for dim in 1:length(axs)
                        pl = D.stencil_plan(gg, dim; order = 1, nodes = 5)
                        mk = GD.mask(gg) isa GD.AllActive ? nothing : GD.mask(gg)
                        O._fusable(fld, fld, geo, mk) && (fused_seen += 1)
                        a = fill(NaN, fsz)
                        b = fill(NaN, fsz)
                        c = fill(NaN, fsz)
                        O.derivative!(a, fld, gg, dim; order = 1, nodes = 5, masked = -3.0)
                        O.apply_stencil!(b, fld, gg, dim; order = 1, nodes = 5, masked = -3.0)
                        O._scale_by_metric!(b, gg, dim, -3.0)
                        O.derivative!(c, fld, gg, pl, dim; masked = -3.0)
                        Test.@test isequal(a, b)
                        Test.@test isequal(c, b)

                        # The table entry point fuses too, against a table-built reference: on a
                        # uniform axis a plan's weights are translation-invariant where a table's are
                        # rebuilt per row, so the two differ in the last bits by construction.
                        ix, wt = D.axis_stencils(gg, dim; order = 1, nodes = 5)
                        e = fill(NaN, fsz)
                        t = fill(NaN, fsz)
                        O.derivative!(e, fld, gg, ix, wt, dim; order = 1, masked = -3.0)
                        O.apply_stencil!(t, fld, gg, ix, wt, dim; order = 1, masked = -3.0)
                        O._scale_by_metric!(t, gg, dim, -3.0)
                        Test.@test isequal(e, t)
                    end
                end
            end
        end
    end
    Test.@test fused_seen == 80           # every case above took the fused path
end

Test.@testset "The ! forms write into the caller's buffers and allocate nothing" begin
    D = FG.Discretization

    for x in (collect(range(0.0, 4.0; length = 40)),
              collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = 40)))))
        # `axis_stencils!` builds the same table, and with a scratch it allocates nothing: the two
        # matrices are the caller's and the Fornberg buffers come from the scratch.
        for (ord, k) in ((1, 3), (1, 4), (1, 5), (2, 5)), per in (nothing, 12.0)
            ri, rw = D.axis_stencils(x, ord, k; period = per)
            gi = Matrix{Int}(undef, size(ri))
            gw = Matrix{Float64}(undef, size(rw))
            D.axis_stencils!(gi, gw, x, ord, k; period = per)
            Test.@test gi == ri
            Test.@test gw == rw
        end
        let sc = D.stencil_scratch(2, 5),
            gi = Matrix{Int}(undef, length(x), 5), gw = Matrix{Float64}(undef, length(x), 5)
            Test.@test _alloc(q_axst!, gi, gw, x, 2, 5, sc) == 0
            # a scratch too small for the table says so rather than reading past it
            Test.@test_throws ArgumentError D.axis_stencils!(gi, gw, x, 2, 5;
                                                             scratch = D.stencil_scratch(1, 2))
        end

        # faces!/centers! agree with the allocating forms and are free.
        f = D.faces(x)
        of = similar(x, length(x) + 1)
        Test.@test D.faces!(of, x) == f
        oc = similar(x, length(f) - 1)
        Test.@test D.centers!(oc, f) == D.centers(f)
        Test.@test _alloc(q_faces!, of, x) == 0
        Test.@test _alloc(q_centers!, oc, f) == 0

        # lagrange_weights! likewise, at every node count and on both sides of the axis.
        for k in (2, 3, 5)
            for v in (x[1], x[end], (x[3] + x[4]) / 2, x[1] - 1.0)
                ri, rw = D.lagrange_weights(x, v, k)
                w = Vector{Float64}(undef, k)
                gi, gw = D.lagrange_weights!(w, x, v, k)
                Test.@test gi == ri
                Test.@test gw == rw
                # A partition of unity, so it interpolates. The weights alternate in sign and grow
                # with the Lebesgue constant once `v` sits outside the axis, so the cancellation
                # error scales with `Σ|w|`
                Test.@test sum(gw) ≈ 1.0 atol = 8 * eps(Float64) * sum(abs, gw)
            end
            Test.@test _alloc(q_lagr!, Vector{Float64}(undef, k), x, 1.7, k) == 0
        end
    end

    # An empty axis has no faces, which is what both forms say about it.
    Test.@test D.faces!(Float64[], Float64[]) == D.faces(Float64[])
    Test.@test D.faces!(zeros(2), [3.0]) == D.faces([3.0])
    Test.@test D.centers!(zeros(1), [0.0, 2.0]) == [1.0]

    # A wrongly sized buffer is a mistake, not a silent partial write.
    Test.@test_throws DimensionMismatch D.faces!(zeros(3), collect(1.0:5.0))
    Test.@test_throws DimensionMismatch D.centers!(zeros(9), collect(1.0:5.0))
    Test.@test_throws DimensionMismatch D.axis_stencils!(
        Matrix{Int}(undef, 3, 3), Matrix{Float64}(undef, 3, 3), collect(1.0:5.0), 1, 3)
    Test.@test_throws ArgumentError D.lagrange_weights!(zeros(2), collect(1.0:5.0), 2.0, 4)

    # `interpolation_weights` has no `!` form because it needs none: the pair comes back as a tuple.
    Test.@test _alloc(q_iw, collect(1.0:5.0), 2.4) == 0
    Test.@test (Test.@inferred D.interpolation_weights(collect(1.0:5.0), 2.4)) isa
               Tuple{Int,Tuple{Float64,Float64}}
end

Test.@testset "Staggered gradient, divergence and curl" begin
    GD = FG.Grids
    GE = FG.Geometry
    O = FG.Operators
    D = FG.Discretization
    C, F = D.Center(), D.Face()
    cart = GE.CartesianGeometry{Float64}()

    # Sample an analytic function at a location's own points.
    smp(sg, loc, f) = [f(ntuple(e -> GD.axis_at(sg, e, loc[e])[I[e]], ndims(sg))...)
                       for I in CartesianIndices(ntuple(e -> length(GD.axis_at(sg, e, loc[e])),
                                                        ndims(sg)))]

    x = collect(range(0.0, 4.0; length = 21))
    y = collect(range(0.0, 3.0; length = 16))
    sg = GD.StaggeredGrid(cart, x, y)

    # Each component is ONE difference across ONE cell, so a linear field is differentiated exactly.
    f = smp(sg, (C, C), (a, b) -> 3.0a - 2.0b + 7.0)
    g1, g2 = O.gradient(f, sg)
    Test.@test maximum(abs.(g1[2:(end - 1), :] .- 3.0)) < 1e-12
    Test.@test maximum(abs.(g2[:, 2:(end - 1)] .+ 2.0)) < 1e-12
    # An outer face of a bounded direction has a cell on one side only, so there is no difference
    # across it. It is where a boundary condition goes, and none is invented.
    Test.@test all(isnan, g1[1, :]) && all(isnan, g1[end, :])
    Test.@test all(isnan, g2[:, 1]) && all(isnan, g2[:, end])
    Test.@test all(iszero, O.gradient(f, sg; masked = 0.0)[1][1, :])

    u = smp(sg, (F, C), (a, b) -> 2.0a + 0.5b)
    v = smp(sg, (C, F), (a, b) -> -0.25a + 3.0b)
    Test.@test maximum(abs.(O.divergence((u, v), sg) .- 5.0)) < 1e-11
    Test.@test maximum(abs.(O.curl(u, v, sg)[2:(end - 1), 2:(end - 1)] .+ 0.75)) < 1e-11

    # The curl of a discrete gradient is zero to round-off, not merely small: both are the same
    # one-cell differences, so they cancel identically rather than to truncation order.
    let ψ = smp(sg, (C, C), (a, b) -> sin(1.3a) * cos(0.9b))
        p1, p2 = O.gradient(ψ, sg; masked = 0.0)
        z = O.curl(p1, p2, sg; masked = 0.0)
        Test.@test maximum(abs.(z[2:(end - 1), 2:(end - 1)])) < 1e-10
    end

    # A stretched mesh and a DESCENDING one: the signed gaps keep the derivative's sense either way.
    let xs = cumsum(vcat(0.0, 0.1 .+ 0.05 .* (1:19))), ys = collect(range(6.0, 0.0; length = 16))
        s = GD.StaggeredGrid(cart, xs, ys)
        ff = smp(s, (C, C), (a, b) -> 3.0a - 2.0b + 7.0)
        h1, h2 = O.gradient(ff, s)
        Test.@test maximum(abs.(h1[2:(end - 1), :] .- 3.0)) < 1e-11
        Test.@test maximum(abs.(h2[:, 2:(end - 1)] .+ 2.0)) < 1e-11
        uu = smp(s, (F, C), (a, b) -> 2.0a + 0.5b)
        vv = smp(s, (C, F), (a, b) -> -0.25a + 3.0b)
        Test.@test maximum(abs.(O.divergence((uu, vv), s) .- 5.0)) < 1e-10
    end

    # On a sphere the same call is the metric form, built from the geometry's own scale factors:
    # ∇·u = (1/(R cosφ))[∂u_λ/∂λ + ∂(cosφ·u_φ)/∂φ]. Gated by CONVERGENCE — second order — rather than
    # by a tolerance on one resolution, which would not distinguish it from a wrong constant factor.
    let R = 6.371e6, sph = GE.SphericalGeometry(R)
        errs = map((24, 48, 96)) do nlon
            λs = collect(range(0, 2π; length = nlon + 1)[1:nlon])
            φs = collect(range(-1.3, 1.3; length = nlon ÷ 2))
            s = GD.StaggeredGrid(sph, λs, φs)
            dv = O.divergence((smp(s, (F, C), (l, p) -> sin(l) * cos(p)),
                               smp(s, (C, F), (l, p) -> 0.0)), s)
            want = smp(s, (C, C), (l, p) -> cos(l) / R)
            return maximum(abs.(dv .- want)) / maximum(abs.(want))
        end
        Test.@test errs[3] < 1e-2
        Test.@test errs[1] / errs[2] > 3.5 && errs[2] / errs[3] > 3.5
        # The meridional term carries the cosφ inside the derivative; get that wrong and this diverges.
        errs2 = map((24, 48, 96)) do nlon
            λs = collect(range(0, 2π; length = nlon + 1)[1:nlon])
            φs = collect(range(-1.3, 1.3; length = nlon ÷ 2))
            s = GD.StaggeredGrid(sph, λs, φs)
            dv = O.divergence((smp(s, (F, C), (l, p) -> 0.0),
                               smp(s, (C, F), (l, p) -> sin(p))), s)
            want = smp(s, (C, C), (l, p) -> cos(2p) / (R * cos(p)))
            return maximum(abs.(dv .- want)) / maximum(abs.(want))
        end
        Test.@test errs2[1] / errs2[2] > 3.5 && errs2[2] / errs2[3] > 3.5
    end

    # Discretely conservative: the divergence is the net flux through a cell's own faces over its own
    # volume, so neighbouring cells' shared faces cancel EXACTLY and a wrapping domain integrates to
    # zero — to round-off, not to truncation order.
    let sph = GE.SphericalGeometry(6.371e6),
        λs = collect(range(0, 2π; length = 41)[1:40]),
        φs = collect(range(-1.4, 1.4; length = 25))
        s = GD.StaggeredGrid(sph, λs, φs)
        ctr = GD.center_grid(s)
        dv = O.divergence((smp(s, (F, C), (l, p) -> sin(3l) * cos(p) + 0.3),
                           smp(s, (C, F), (l, p) -> 0.0)), s)
        tot = sum(dv[i, j] * GD.measure(ctr, i, j) for i in 1:40, j in 1:25)
        scale = sum(abs(dv[i, j]) * GD.measure(ctr, i, j) for i in 1:40, j in 1:25)
        Test.@test abs(tot) < 1e-10 * scale
        # A meridional flux vanishing at both edges likewise integrates to nothing.
        dv2 = O.divergence((smp(s, (F, C), (l, p) -> 0.0),
                            smp(s, (C, F), (l, p) -> cos(l) * (p - φs[1]) * (p - φs[end]))), s)
        t2 = sum(dv2[i, j] * GD.measure(ctr, i, j) for i in 1:40, j in 1:25)
        s2 = sum(abs(dv2[i, j]) * GD.measure(ctr, i, j) for i in 1:40, j in 1:25)
        Test.@test abs(t2) < 1e-10 * s2
    end

    # A hole takes with it every value that depended on it, and nothing else.
    let mk = trues(21, 16)
        mk[10, 8] = false
        s = GD.StaggeredGrid(GD.StructuredGrid(cart, x, y, mk))
        dv = O.divergence((u, v), s)
        Test.@test isnan(dv[10, 8])
        Test.@test isnan(dv[9, 8]) && isnan(dv[11, 8]) && isnan(dv[10, 7]) && isnan(dv[10, 9])
        Test.@test !isnan(dv[12, 8]) && !isnan(dv[10, 10])
        gm = O.gradient(f, s)[1]
        Test.@test isnan(gm[10, 8]) && isnan(gm[11, 8]) && !isnan(gm[9, 8])
    end

    Test.@test_throws DimensionMismatch O.divergence!(zeros(3, 3), (u, v), sg)
    Test.@test_throws DimensionMismatch O.gradient!((zeros(3, 3), zeros(3, 3)), f, sg)
end

Test.@testset "In three directions the curl is a vector, at three vorticity points" begin
    GD = FG.Grids
    GE = FG.Geometry
    O = FG.Operators
    D = FG.Discretization
    C, F = D.Center(), D.Face()
    cart = GE.CartesianGeometry{Float64}()

    smp(sg, loc, f) = [f(ntuple(e -> GD.axis_at(sg, e, loc[e])[I[e]], ndims(sg))...)
                       for I in CartesianIndices(ntuple(e -> length(GD.axis_at(sg, e, loc[e])),
                                                        ndims(sg)))]

    x = collect(range(0.0, 2.0; length = 11))
    y = collect(range(0.0, 1.5; length = 9))
    z = collect(range(0.0, 3.0; length = 7))
    sg = GD.StaggeredGrid(cart, x, y, z)
    L = ((F, C, C), (C, F, C), (C, C, F))         # where each velocity component lives
    W = ((C, F, F), (F, C, F), (F, F, C))         # where each vorticity component lives

    # Each term is one difference across one cell, so a linear field is curled exactly.
    for (u, want) in ((((a, b, c) -> -b, (a, b, c) -> a, (a, b, c) -> 0.0), (0.0, 0.0, 2.0)),
                      (((a, b, c) -> 0.0, (a, b, c) -> 0.0, (a, b, c) -> b), (1.0, 0.0, 0.0)),
                      (((a, b, c) -> c, (a, b, c) -> 0.0, (a, b, c) -> 0.0), (0.0, 1.0, 0.0)),
                      (((a, b, c) -> 0.0, (a, b, c) -> c, (a, b, c) -> 0.0), (-1.0, 0.0, 0.0)))
        us = ntuple(d -> smp(sg, L[d], u[d]), Val(3))
        ws = O.curl(us, sg)
        for i in 1:3
            Test.@test size(ws[i]) == size(smp(sg, W[i], (a, b, c) -> 0.0))
            fin = filter(isfinite, vec(ws[i]))
            Test.@test !isempty(fin)
            Test.@test maximum(abs, fin .- want[i]) < 1e-11
        end
    end

    # ∇×∇f vanishes to round-off: the curl differences the very quantities the gradient built, so the
    # terms cancel identically.
    let ψ = smp(sg, (C, C, C), (a, b, c) -> sin(1.3a) * cos(0.9b) * exp(-0.2c))
        gr = O.gradient(ψ, sg; masked = 0.0)
        scale = maximum(abs, filter(isfinite, vcat(vec.(gr)...)))
        ws = O.curl(gr, sg; masked = 0.0)
        for i in 1:3
            Test.@test maximum(abs, filter(isfinite, vec(ws[i]))) < 1e-9 * scale
        end
    end

    # On a sphere the same call is the metric form, built from the geometry's own scale factors.
    let sph = GE.SphericalGeometry(6.371e6),
        λs = collect(range(0, 2π; length = 25)[1:24]),
        φs = collect(range(-1.2, 1.2; length = 13)),
        rs = collect(range(6.371e6, 6.4e6; length = 7))
        s = GD.StaggeredGrid(sph, λs, φs, rs)
        ψ = smp(s, (C, C, C), (l, p, r) -> sin(2l) * cos(p) * (r / 6.4e6))
        gr = O.gradient(ψ, s; masked = 0.0)
        scale = maximum(abs, filter(isfinite, vcat(vec.(gr)...)))
        ws = O.curl(gr, s; masked = 0.0)
        for i in 1:3
            Test.@test maximum(abs, filter(isfinite, vec(ws[i]))) < 1e-9 * scale
        end
        # A zonal flow `u_λ = cosφ` has `(∇×u)_r = 2 sinφ / r`. Gated by CONVERGENCE — second order —
        # at each point's OWN radius; a fixed one floors the error at the radial axis's width.
        errs = map((16, 32, 64)) do nlon
            λ2 = collect(range(0, 2π; length = nlon + 1)[1:nlon])
            φ2 = collect(range(-1.2, 1.2; length = nlon ÷ 2))
            r2 = collect(range(6.371e6, 6.4e6; length = 5))
            ss = GD.StaggeredGrid(sph, λ2, φ2, r2)
            us = (smp(ss, (F, C, C), (l, p, r) -> cos(p)),
                  smp(ss, (C, F, C), (l, p, r) -> 0.0),
                  smp(ss, (C, C, F), (l, p, r) -> 0.0))
            w = O.curl(us, ss)[3]
            want = smp(ss, (F, F, C), (l, p, r) -> 2 * sin(p) / r)
            fin = [(a, b) for (a, b) in zip(w, want) if isfinite(a)]
            return maximum(abs(a - b) for (a, b) in fin) / maximum(abs(b) for (_, b) in fin)
        end
        Test.@test errs[3] < 1e-2
        Test.@test errs[1] / errs[2] > 3.5 && errs[2] / errs[3] > 3.5
    end

    # A hole takes with it every value that depended on it.
    let mk = trues(11, 9, 7)
        mk[5, 4, 3] = false
        s = GD.StaggeredGrid(GD.StructuredGrid(cart, x, y, z, mk))
        us = ntuple(d -> smp(s, L[d], (a, b, c) -> a + b + c), Val(3))
        ws = O.curl(us, s)
        Test.@test isnan(ws[1][5, 4, 3]) && isnan(ws[1][5, 5, 4])
        Test.@test isfinite(ws[1][5, 7, 6])
    end

    # In two directions the tuple form is the two-argument one.
    let s2 = GD.StaggeredGrid(cart, x, y)
        u1 = smp(s2, (F, C), (a, b) -> -b)
        u2 = smp(s2, (C, F), (a, b) -> a)
        Test.@test isequal(O.curl((u1, u2), s2), O.curl(u1, u2, s2))
    end

    let us = ntuple(d -> smp(sg, L[d], (a, b, c) -> 0.0), Val(3))
        Test.@test_throws DimensionMismatch O.curl!((zeros(2, 2, 2), zeros(2, 2, 2),
                                                     zeros(2, 2, 2)), us, sg)
    end
end
