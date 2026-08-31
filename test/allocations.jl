Test.@testset "A degrading sweep's cost is O(1) in the grid, not O(n)" begin
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    # The degrade path keeps a Fornberg scratch, but it is `O(1)` in the grid, not `O(n)`.
    let sizes = (24, 96)
        allocs = map(sizes) do m
            xs = collect(range(0.0, 10.0; length = m))
            mm = trues(m, m); mm[7, 9] = false
            gm = GD.StructuredGrid(cart, xs, xs, mm)
            fm = [sin(xi) * cos(yj) for xi in xs, yj in xs]
            iw = D.axis_stencils(gm, 1; order = 1, nodes = 3)
            _alloc(q_tbl!, zeros(m, m), fm, gm, iw, 1, O.ReduceInRun())
        end
        Test.@test allocs[1] == allocs[2]          # 16x the cells, the same bytes
    end
end

Test.@testset "A precomputed weight set applies allocation-free" begin
    D = FG.Discretization
    O = FG.Operators
    X = collect(range(0.0, 8.0; length = 9))
    Y = collect(range(0.0, 4.0; length = 5))
    F = [xi^2 for xi in X, _ in Y]
    idx, w = D.axis_stencils(X, 1, 3)
    O2 = similar(F)
    ap() = FG.Operators.apply_stencil!(O2, F, idx, w, 1)
    ap()
    Test.@test @allocated(ap()) == 0
end

Test.@testset "A per-cell traversal with a caller's own stencil allocates nothing" begin
    geo = FG.Geometry.CartesianGeometry()
    g = FG.Grids.StructuredGrid(geo, range(0.0; step = 1.0, length = 6),
                                range(0.0; step = 1.0, length = 5))
    # A stencil defined outside the package drives the connectivity stack allocation-free, like the
    # built-ins. `Upwind` is defined in `runtests.jl`.
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

Test.@testset "Every stencil traversal allocates nothing, at any shape or dimension" begin
    S = FG.Stencils
    geo = FG.Geometry.CartesianGeometry()
    g = FG.Grids.StructuredGrid(geo, 0.0:1.0:9.0, 0.0:1.0:9.0, FG.Grids.AllActive((10, 10)))

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
    Test.@test first(counts) < 20end

Test.@testset "Curvilinear construction memory is its own stored content, nothing more" begin
    geo = FG.Geometry.SphericalGeometry()
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

    # …and corner reconstruction costs only its own output. Minimum of several samples: a single
    # `@allocated` can land on a collection or on the tail of compilation, and the true cost is a
    # floor, so the minimum converges to it.
    cc(A) = (FG.Grids._centers_to_corners(A);
             minimum(@allocated(FG.Grids._centers_to_corners(A)) for _ in 1:3))
    big = [1.0i + 2.0j for i in 1:120, j in 1:120]
    Test.@test cc(big) < 1.2 * 8 * 121^2
end

Test.@testset "The grid-free connectivity path does not scale its allocations with nlat" begin
    # The grid-free path must not scale its allocations with nlat the way building a grid does.
    nalloc(f) = (f(); Base.gc_alloc_count((@timed f()).gcstats))
    small = nalloc(() -> FG.Connectivity.build_connectivity(FG.SphericalSampling.GaussLegendreSampling(), 16))
    large = nalloc(() -> FG.Connectivity.build_connectivity(FG.SphericalSampling.GaussLegendreSampling(), 128))
    Test.@test large <= small + 2
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
end

Test.@testset "Neighbor traversal allocates nothing" begin
    geom = FG.Geometry.CartesianGeometry()
    n = 40
    xs = collect(0.0:1.0:(n - 1))
    grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(n, n))
    function sweep(g, m)
        c = 0
        for j in 1:m, i in 1:m
            for v in FG.Grids.neighbors(g, i, j)
                c += v
            end
        end
        return c
    end
    sweep(grid, 3)
    Test.@test @allocated(sweep(grid, n)) == 0
end

Test.@testset "Distance and displacement across a periodic seam allocate nothing" begin
    GR = FG.Grids
    geo = FG.Geometry.CartesianGeometry()
    n, Δ = 10, 1.0
    gp = GR.StructuredGrid(geo, range(0.0; step = Δ, length = n),
                           range(0.0; step = Δ, length = 6); periodic = true, period = n * Δ)
    dd() = FG.Geometry.distance(gp, (3, 2), (9, 5))
    pp() = FG.Grids.displacement(gp, (3, 2), (9, 5))
    dd(); pp()
    Test.@test @allocated(dd()) == 0
    Test.@test @allocated(pp()) == 0
end

Test.@testset "A whole-grid ball sweep allocates nothing" begin
    GR = FG.Grids
    Nx, Δx = 32, 62.5
    ax = range(0.0; step = Δx, length = Nx)
    g = GR.StructuredGrid(FG.Geometry.CartesianGeometry(), ax, ax;
                          periodic = (true, true), period = (Nx * Δx, Nx * Δx))
    # Both entry points stay allocation-free over a whole sweep. Called through the const `FG` path:
    # a captured non-const module local would defeat const-folding and charge dispatch to the sweep.
    cnt(gr, m) = (t = 0; for j in 1:m, i in 1:m
        t += FG.Connectivity.fold_within((a, J, d) -> a + 1, 0, gr, i, j; ball = 600.0) end; t)
    nnw(gr, m) = (t = 0; for j in 1:m, i in 1:m
        t += FG.Connectivity.nneighbors_within(gr, i, j; ball = 600.0) end; t)
    cnt(g, Nx); nnw(g, Nx)
    Test.@test @allocated(cnt(g, Nx)) == 0
    Test.@test @allocated(nnw(g, Nx)) == 0

    # And on a bounded rectilinear grid with unequal steps.
    gu = GR.StructuredGrid(FG.Geometry.CartesianGeometry(),
                           range(0.0; step = 1.0, length = 12), range(0.0; step = 0.7, length = 9))
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

Test.@testset "A traversal stays allocation-free past three dimensions" begin
    geo = FG.Geometry.CartesianGeometry()
    g4 = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:3.0)
    g5 = FG.Grids.StructuredGrid(geo, ntuple(_ -> 0.0:1.0:2.0, 5)...)

    # Coordinate names are numbered rather than lettered from N = 4 on, and a per-cell query reaches
    # them; building those symbols at run time would allocate on every call, so this covers more than
    # the loop itself.
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

    # A point given as a vector has a runtime width, so it arrives as a union of the widths. Each of
    # these reduces it to a value the geometry types, so the union splits and nothing is boxed through.
    vp, vv = [0.1, 0.2], [1.0, 2.0, 3.0]
    Test.@test _alloc(q_dist, sgeo, vp, vp) == 0
    Test.@test _alloc(q_uvec, vp) == 0
    Test.@test _alloc(q_s2c, sgeo, vp) == 0
    Test.@test _alloc(q_vfc, sgeo, vv, vp) == 0

    # The two primitives and everything routed through them, on both hierarchies. A spheroid pays no
    # more than a sphere: the frame is the same arithmetic and the position is one more division.
    spgeo = FG.Geometry.SpheroidGeometry()
    c2, n2, p3 = (0.4, 0.6), (0.41, 0.61), (4.0e6, 1.0e6, 4.5e6)
    τ6 = (0.3, -1.1, 2.0, 0.7, -0.2, 1.4)
    for geo in (sgeo, spgeo)
        Test.@test _alloc(q_embed, geo, c2) == 0
        Test.@test _alloc(q_proj, geo, c2, n2) == 0
        Test.@test _alloc(q_ltb, geo, c2) == 0
        Test.@test _alloc(q_vfc, geo, vv, c2) == 0
        Test.@test _alloc(q_tensor_local, geo, τ6, c2) == 0
        # The displacement at both widths: two components in a tangent plane, three on the frame.
        Test.@test _alloc(q_locdisp, geo, c2, n2) == 0
        Test.@test _alloc(q_locdisp, geo, (0.4, 0.6, 6.4e6), (0.41, 0.61, 6.5e6)) == 0
    end
    Test.@test _alloc(q_c2g, spgeo, p3) == 0
    Test.@test _alloc(q_locdisp, FG.Geometry.CartesianGeometry(), (1.0, 2.0, 3.0),
                      (1.5, 2.5, 3.5)) == 0
end

Test.@testset "Rotating a point set in place allocates nothing" begin
    rot = FG.Geometry.PoleRotation(0.7, 0.3)
    λ2 = [0.1, 1.2, 3.0, 5.5]
    φ2 = [0.0, -0.4, 0.9, 0.2]
    # Spelled through the const `FG`, not a local module alias: a captured non-const `Module` is a
    # dynamic lookup and charges its own bytes to the call under test.
    rr() = FG.Geometry.rotate!(λ2, φ2, rot)
    rr()
    Test.@test @allocated(rr()) == 0
end

Test.@testset "An @inbounds sweep over a grid pays nothing for the bounds check" begin
    geom = FG.Geometry.CartesianGeometry()
    g = FG.Grids.StructuredGrid(geom, 0.0:1.0:4.0, 0.0:1.0:3.0, trues(5, 4))
    # `@boundscheck` is elided at an `@inbounds` call site, so a hot loop still pays nothing.
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

Test.@testset "Icosahedral construction does not scale with a per-vertex hash table" begin
    # Construction cost must not scale with a per-vertex hash table.
    function allocs(f)
        f()
        r = @timed f()
        return Base.gc_alloc_count(r.gcstats)
    end
    Test.@test allocs(() -> FG.SphericalSampling.icosahedral_mesh(4)) < 200
    Test.@test allocs(() -> FG.SphericalSampling.icosahedral_mesh(32)) < 200
end

Test.@testset "sparse_adjacency_matrix! reuses the caller's arrays" begin
    conn = FG.Connectivity.build_connectivity(
        FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 8))
    n = FG.Connectivity.nnodes(conn)
    ne = FG.Connectivity.nedges(conn)
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, ne)
    nzval = Vector{Bool}(undef, ne)
    # Filling caller-owned arrays adds nothing beyond the matrix's own wrapper.
    FG.Connectivity.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)
    Test.@test @allocated(FG.Connectivity.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)) < 200
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
    # A layout whose coordinates, adjacency and measure are arithmetic must reach the same floor: the
    # arithmetic runs in registers, so there is nothing for a per-cell read to allocate.
    check_shape("HEALPix nside=8", FG.Grids.HEALPixGrid(sph, 8), (100,); axes = false)
    check_shape("octahedral N=16",
                FG.Grids.RingGrid(sph, FG.SphericalSampling.OctahedralGaussianSampling(16)),
                (100,); axes = false)

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
            ("local_spacing/vector", _alloc(FG.Discretization.local_spacing, ax, 3)),
            ("local_spacing/range",  _alloc(FG.Discretization.local_spacing, rg, 3)),
            ("cell_width/vector",    _alloc(FG.Discretization.cell_width, ax, 3)),
            ("cell_width/range",     _alloc(FG.Discretization.cell_width, rg, 3)),
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

Test.@testset "The per-point kernels outside Grids allocate nothing either" begin
    GE = FG.Geometry
    A = FG.Axes
    St = FG.Stencils
    SS = FG.SphericalSampling
    sph = GE.SphericalGeometry(6.371e6)
    spd = GE.SpheroidGeometry()
    P2 = (0.3, 0.4)
    V3 = (1.0, 2.0, 3.0)
    T6 = (1.3, -0.7, 2.1, 0.4, -1.1, 0.9)
    rot = GE.PoleRotation(0.3, 0.4)
    u = A.UniformAxis(0.0, 0.5, 64)
    v = collect(u)
    hp = SS.HEALPixSampling(4)
    gl = SS.GaussLegendreSampling()

    # These are called per point, per cell or per pixel, so an allocation here is paid on every
    # one. The failure mode this catches is an element type reaching one of them as a runtime
    # `Type` value: the return widens to `Any` and the kernel allocates on every call.
    for (name, a) in (
        ("radius",                   _alloc(GE.radius, sph)),
        ("semimajor_axis",           _alloc(GE.semimajor_axis, spd)),
        ("semiminor_axis",           _alloc(GE.semiminor_axis, spd)),
        ("flattening",               _alloc(GE.flattening, spd)),
        ("eccentricity²",            _alloc(GE.eccentricity², spd)),
        ("meridional_radius",        _alloc(GE.meridional_radius, spd, 0.4)),
        ("prime_vertical_radius",    _alloc(GE.prime_vertical_radius, spd, 0.4)),
        ("as_ntuple",                _alloc(GE.as_ntuple, P2)),
        ("as_tensor6",               _alloc(GE.as_tensor6, T6)),
        ("point_names",              _alloc(GE.point_names, sph, Val(2))),
        ("named_point",              _alloc(GE.named_point, sph, P2)),
        ("area_element",             _alloc(GE.area_element, spd, 0.4, 0.01, 0.01)),
        ("volume_element",           _alloc(GE.volume_element, sph, 0.4, 6.4e6, 0.01, 0.01, 10.0)),
        ("spherical_to_cartesian",   _alloc(GE.spherical_to_cartesian, sph, P2)),
        ("cartesian_to_spherical",   _alloc(GE.cartesian_to_spherical, sph, V3)),
        ("geodetic_to_cartesian",    _alloc(GE.geodetic_to_cartesian, spd, P2)),
        ("unit_vector",              _alloc(GE.unit_vector, Float64, P2)),
        ("local_tangent_basis",      _alloc(GE.local_tangent_basis, sph, P2)),
        ("project_to_tangent_plane", _alloc(GE.project_to_tangent_plane, sph, P2, (0.31, 0.41))),
        ("nonuniform_first_derivative",
                                     _alloc(GE.nonuniform_first_derivative, 1.0, 2.0, 4.0, 1.0, 1.5)),
        ("vector_to_cartesian",      _alloc(GE.vector_to_cartesian, sph, 1.0, 2.0, 0.3, 0.4)),
        ("vector_from_cartesian",    _alloc(GE.vector_from_cartesian, sph, V3, 0.3, 0.4)),
        ("tensor_to_local",          _alloc(GE.tensor_to_local, sph, T6, 0.3, 0.4)),
        ("tensor_from_local",        _alloc(GE.tensor_from_local, sph, T6, 0.3, 0.4)),
        ("spherical_excess",         _alloc(GE.spherical_excess, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                            (0.0, 0.0, 1.0))),
        ("triangle_area",            _alloc(GE.triangle_area, sph, P2, (0.4, 0.5), (0.5, 0.4))),
        ("rotate",                   _alloc(GE.rotate, rot, 0.5, 0.6)),
        ("unrotate",                 _alloc(GE.unrotate, rot, 0.5, 0.6)),
        ("similar_geometry",         _alloc(GE.similar_geometry, Float32, sph)),
        ("similar_rotation",         _alloc(GE.similar_rotation, Float32, rot)),
        ("metric_invariant_directions", _alloc(GE.metric_invariant_directions, sph)),
        ("float_type",               _alloc(GE.float_type, sph)),
        ("Axes.spacing_trait",       _alloc(A.spacing_trait, u)),
        ("Axes.wrap_sign",           _alloc(A.wrap_sign, v)),
        ("Axes.similar_axis",        _alloc(A.similar_axis, u, 0.0, 0.5, 32)),
        ("Axes.uniform_axis",        _alloc(A.uniform_axis, Float64, u)),
        ("Stencils.offsets",         _alloc(St.offsets, St.Moore(1), Val(2))),
        ("Stencils.nstencil",        _alloc(St.nstencil, St.Moore(1), Val(2))),
        ("Stencils.reach",           _alloc(St.reach, St.Moore(1), Val(2))),
        ("bandlimit",                _alloc(SS.bandlimit, gl, 16)),
        ("nlat_for_bandlimit",       _alloc(SS.nlat_for_bandlimit, gl, 15)),
        ("nlon_for_nlat",            _alloc(SS.nlon_for_nlat, gl, 16)),
        ("npoints",                  _alloc(SS.npoints, hp)),
        ("nrings",                   _alloc(SS.nrings, hp)),
        ("axes_lengths",             _alloc(SS.axes_lengths, gl, 16)),
        ("colatitude",               _alloc(SS.colatitude, 0.4)),
        ("geographic_latitude",      _alloc(SS.geographic_latitude, 0.4)),
        ("ang2pix",                  _alloc(SS.ang2pix, 4, 1.0, 2.0)),
        ("pix2ang",                  _alloc(SS.pix2ang, 4, 10)),
        ("pix2vec",                  _alloc(SS.pix2vec, 4, 10)),
        ("vec2pix",                  _alloc(SS.vec2pix, 4, V3)),
        ("ring2nest",                _alloc(SS.ring2nest, 4, 10)),
        ("ring_info",                _alloc(SS.ring_info, 4, 3)),
    )
        a == 0 || println("    ", name, " -> ", a, " B")
        Test.@test a == 0
    end

end

Test.@testset "A degrading sweep can be made to allocate nothing" begin
    D = FG.Discretization
    O = FG.Operators
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    n = 32
    x = collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = n))))
    mk = trues(n, n); mk[7, 9] = false; mk[8, 9] = false; mk[20, 3] = false
    g = GD.StructuredGrid(cart, x, x, mk)
    f = [sin(xi) * cos(yj) for xi in x, yj in x]
    idx, w = D.axis_stencils(g, 1; order = 1, nodes = 3)
    sc = D.stencil_scratch(1, 3)

    # Same answer, and the buffers carry nothing between calls.
    a = zeros(n, n); b = zeros(n, n)
    O.apply_stencil!(a, f, g, idx, w, 1; order = 1, masked = NaN, policy = O.ReduceInRun())
    for _ in 1:3
        O.apply_stencil!(b, f, g, idx, w, 1; order = 1, masked = NaN,
                         policy = O.ReduceInRun(), scratch = sc)
    end
    Test.@test all(isequal(a[i], b[i]) for i in eachindex(a))

    # The scratch is the difference between a per-call cost and none.
    Test.@test _alloc(q_tbl_sc!, zeros(n, n), f, g, (idx, w), 1, sc) == 0
    Test.@test _alloc(q_tbl_sc!, zeros(n, n), f, g, (idx, w), 1, nothing) > 0

    # A run walk must not allocate at all: `_run_reach` closed over a loop variable it also
    # reassigned, which Julia boxes — 288 bytes per call, once per cell adjacent to a mask, so it
    # grew with the length of a coastline rather than being a fixed cost.
    Test.@test _alloc(q_runreach, mk, (6, 9), 1, 6, n, 3) == 0
    # …and every cell away from a mask was already free, which is why this hid.
    Test.@test _alloc(q_tbl!, zeros(n, n), f, GD.StructuredGrid(cart, x, x), (idx, w), 1,
                      O.ReduceInRun()) == 0

    # A scratch too small for the stencil is refused rather than overrun.
    let small = D.stencil_scratch(1, 2), i5w5 = D.axis_stencils(g, 1; order = 1, nodes = 5)
        Test.@test_throws DimensionMismatch O.apply_stencil!(
            zeros(n, n), f, g, i5w5[1], i5w5[2], 1; order = 1, masked = NaN,
            policy = O.ReduceInRun(), scratch = small)
    end
end

Test.@testset "Rank-2 tensor rotation allocates nothing" begin
    GE = FG.Geometry
    geo = GE.SphericalGeometry(6.371e6)
    τ = (1.3, -0.7, 2.1, 0.4, -1.1, 0.9)              # xx yy zz xy xz yz
    # Meant for a hot loop, so it must not allocate — the reason a caller would hand-roll it.
    # Through `_alloc`, not a local closure: a closure over a testset local boxes what it captures
    # and would measure the harness rather than the function.
    Test.@test _alloc(q_tensor_local, geo, τ, (0.7, -0.4)) == 0
end

Test.@testset "The Gauss-Legendre solve keeps O(1) scratch, whatever n" begin
    nalloc(f) = (f(); minimum(Base.gc_alloc_count((@timed f()).gcstats) for _ in 1:5))
    SS = FG.SphericalSampling
    gl = SS.GaussLegendreSampling()
    n = 64
    sz = SS.axes_lengths(gl, n)
    λb = Vector{Float64}(undef, sz.nlon)
    φb = Vector{Float64}(undef, sz.nlat)
    wb = Vector{Float64}(undef, sz.nlat)
    SS.spherical_quadrature!(λb, φb, wb, gl, n)
    # Only the returned NamedTuple; the solve itself needs O(1) scratch, not the O(n²)
    # eigenvector matrix a Golub–Welsch decomposition would.
    Test.@test nalloc(() -> SS.spherical_quadrature!(λb, φb, wb, gl, n)) <= 1
    Test.@test nalloc(() -> SS._gauss_legendre_μ!(φb, wb)) <= 1
    # O(1) scratch means the count cannot grow with n.
    big = Vector{Float64}(undef, 4n)
    bigw = similar(big)
    Test.@test nalloc(() -> SS._gauss_legendre_μ!(big, bigw)) <= 1
end

Test.@testset "The Fibonacci fill allocates nothing per point" begin
    SS = FG.SphericalSampling
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
end

Test.@testset "A ring is reached in O(1), allocating nothing whatever the grid size" begin
    SS = FG.SphericalSampling
    # O(1) means allocation-free, whatever the grid size — the table form cannot be.
    big = SS.OctahedralGaussianSampling(400)
    Test.@test _alloc(q_nlon_in_ring, big, 500) == 0
    Test.@test _alloc(q_ring_range, big, 500) == 0
    Test.@test _alloc(q_ring_range, SS.HEALPixSampling(256), 700) == 0
    Test.@test _alloc(q_npoints, big) == 0

    # A caller-supplied buffer makes the reduced-Gaussian fill allocate nothing at all.
    let s = SS.OctahedralGaussianSampling(12), n = SS.npoints(s)
        λ = Vector{Float64}(undef, n); φ = Vector{Float64}(undef, n)
        sc = Vector{Float64}(undef, SS.nrings(s))
        SS.spherical_points!(λ, φ, s; scratch = sc)
        Test.@test _alloc(q_rg_points!, λ, φ, s, sc) == 0
    end
end

Test.@testset "A stencil plan's accessors allocate nothing" begin
    D = FG.Discretization
    # A plan is read per cell on the paths that do not specialize on its form, so every accessor has to
    # be free — including `plan_row`, whose two tuples are stack values.
    for (x, per) in ((range(0.0, 1.0; length = 64), nothing),
                     (range(0, 2π * (1 - 1 / 64); length = 64), 2π),
                     (collect(cumsum(fill(0.5, 64))), nothing))
        for k in (3, 4, 5)
            pl = D.stencil_plan(x, 1, k)
            for j in (1, 2, 32, 63, 64)
                Test.@test _alloc(D.plan_row, pl, j) == 0
            end
            Test.@test _alloc(D.nnodes, pl) == 0
            Test.@test _alloc(D.derivative_order, pl) == 0
            Test.@test _alloc(D.axis_length, pl) == 0
            Test.@test (Test.@inferred D.plan_row(pl, 3)) isa Tuple{NTuple{k,Int},NTuple{k,Float64}}
        end
    end
end

Test.@testset "The NESTED bit interleave allocates nothing" begin
    SS = FG.SphericalSampling
    Test.@test _alloc(SS._spread_bits, 12345) == 0
    Test.@test _alloc(SS._compress_bits, 12345) == 0
end

Test.@testset "The pixel neighbour walk returns a stack tuple, allocating nothing" begin
    C = FG.Connectivity
    # The tuple form is why a traversal over a formula layout needs no scratch threaded through it, so
    # it is measured directly and not only through the layout that calls it.
    for ns in (1, 4, 64)
        Test.@test _alloc(C.healpix_neighbor_ids, ns, 10) == 0
        Test.@test _alloc(C.healpix_neighbors!, Vector{Int}(undef, 8), ns, 10) == 0
    end
    ids, n = C.healpix_neighbor_ids(4, 10)
    Test.@test ids isa NTuple{8,Int}
    Test.@test (Test.@inferred C.healpix_neighbor_ids(4, 10)) isa Tuple{NTuple{8,Int},Int}
    buf = Vector{Int}(undef, 8)
    Test.@test C.healpix_neighbors!(buf, 4, 10) == n
    Test.@test sort(buf[1:n]) == sort(collect(ids)[1:n])
end

Test.@testset "Holding a stencil table removes the per-call allocation" begin
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

    # Holding the table is what removes the per-call allocation, which is the point of it.
    for dim in 1:2
        idx, w = D.axis_stencils(g, dim; order = 1, nodes = 3)
        Test.@test _alloc(q_tbl!, zeros(n, n), f, g, (idx, w), dim, O.BlankMasked()) == 0
    end
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
    gu = healpix_node_grid(4)
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
end

Test.@testset "A ball query on a stretched axis allocates nothing, at any length" begin
    C = FG.Connectivity
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
    for n in (256, 4096)
        g = stretched(n)
        Test.@test nalloc(g, C.MetricTopology(g)) == 0
    end
end

Test.@testset "A per-index gap or width lookup allocates nothing on every axis mix" begin
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    vec1 = cumsum([0.0, 1.0, 0.3, 2.5, 0.7, 4.0])
    rng1 = range(0.0, 1.0; length = 8)
    λ = collect(range(0.0, 2π * (1 - 1 / 8); length = 8))
    grids = (
        GD.StructuredGrid(cart, rng1, rng1),
        GD.StructuredGrid(cart, rng1, vec1),
        GD.StructuredGrid(cart, vec1, rng1),
        GD.StructuredGrid(cart, vec1, vec1),
        GD.StructuredGrid(cart, rng1, vec1, collect(0.0:2.0)),
        GD.StructuredGrid(sph, λ, collect(range(-1.0, 1.0; length = 9))),
    )
    # A runtime `d` indexes a tuple whose entries have different types on the mixed grids, so the
    # lookup must not allocate for any of them.
    for g in grids, d in 1:length(GD.coordinates(g))
        Test.@test _alloc(q_gap, g, d, 2) == 0
        Test.@test _alloc(q_width, g, d, 2) == 0
    end
end

Test.@testset "Applying a gradient plan allocates nothing" begin
    D = FG.Discretization
    O = FG.Operators
    C = FG.Connectivity
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    n = 12
    x = [0.7i + 0.05j for i in 1:n, j in 1:n]
    y = [0.9j - 0.03i for i in 1:n, j in 1:n]
    grid = GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
    plan = O.gradient_plan(grid)
    # Applying is one dot product per cell and must allocate nothing.
    let f = 2.0 .* x .- 3.0 .* y, g1 = zeros(n, n), g2 = zeros(n, n)
        O.gradient!(g1, g2, f, plan)
        Test.@test _alloc(q_gradient!, g1, g2, f, plan) == 0
        Test.@test _alloc(q_gradient_t!, (g1, g2), f, plan) == 0
    end
    # And at three directions, where the per-direction accumulator is a tuple: written as a local it
    # would be both reassigned each step and captured by the unrolling closure, which boxes.
    let m = 4, np = 4^3
        xs = [1.0 * (i - 1) for k in 1:m for j in 1:m for i in 1:m]
        ys = [1.0 * (j - 1) for k in 1:m for j in 1:m for i in 1:m]
        zs = [1.0 * (k - 1) for k in 1:m for j in 1:m for i in 1:m]
        g3d = GD.UnstructuredGrid(cart, (xs, ys, zs), trues(np); k = 6, areas = ones(np))
        p3 = O.gradient_plan(g3d)
        f3 = 2.0 .* xs .- 3.0 .* ys .+ 0.5 .* zs
        u1 = zeros(np); u2 = zeros(np); u3 = zeros(np)
        O.gradient!(u1, u2, u3, f3, p3)
        Test.@test _alloc(q_gradient_t!, (u1, u2, u3), f3, p3) == 0
    end
end

Test.@testset "A batch axis costs no allocation, and none that grows with it" begin
    D = FG.Discretization
    O = FG.Operators
    C = FG.Connectivity
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()
    nx, ny = 16, 11
    x = collect(cumsum(1.0 .+ 0.3 .* sin.(range(0, 3π; length = nx))))
    mk = trues(nx, ny); mk[5, 4] = false; mk[6, 4] = false
    g = GD.StructuredGrid(cart, x, collect(1.0:ny), mk)
    sph = GD.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6),
                            collect(range(0.0, 2π * (1 - 1 / nx); length = nx)),
                            collect(range(-1.1, 1.1; length = ny)))
    idx, w = D.axis_stencils(g, 1; order = 1, nodes = 3)
    plan = O.gradient_plan(GD.CurvilinearGrid(cart,
        [0.7i + 0.05j for i in 1:nx, j in 1:ny], [0.9j - 0.03i for i in 1:nx, j in 1:ny],
        trues(nx, ny); measure = fill(1.0, nx, ny)))

    # The batch adds output, not overhead: the same bytes at Nb = 1 and Nb = 8 is the claim, and zero is
    # the stronger one. Measured through the top-level `q_*` helpers so the harness is not what is seen.
    for (name, mk_args) in (
        ("apply_stencil! (held table)", nb -> (q_tbl!, zeros(nx, ny, nb),
                                              [sin(3i) * j * k for i in 1:nx, j in 1:ny, k in 1:nb],
                                              g, (idx, w), 1, O.BlankMasked())),
        ("derivative! (spherical, held)", nb -> (q_deriv_held!, zeros(nx, ny, nb),
                                              [sin(3i) * j * k for i in 1:nx, j in 1:ny, k in 1:nb],
                                              sph, D.axis_stencils(sph, 2; order = 1, nodes = 3), 2)),
        ("gradient!",                   nb -> (q_gradient!, zeros(nx, ny, nb), zeros(nx, ny, nb),
                                              [2.0i - 3.0j + k for i in 1:nx, j in 1:ny, k in 1:nb],
                                              plan)),
        ("interpolate!",                nb -> (q_interp!, Vector{Float64}(undef, nb),
                                              [2.0i - 3.0j + k for i in 1:nx, j in 1:ny, k in 1:nb],
                                              g, (4.5, 5.5))),
    )
        a1 = _alloc(mk_args(1)...)
        a8 = _alloc(mk_args(8)...)
        (a1 == 0 && a8 == 0) || println("    ", name, " -> ", a1, " B at Nb=1, ", a8, " B at Nb=8")
        Test.@test a1 == 0
        Test.@test a8 == a1                     # nothing scales with the batch
    end
end

Test.@testset "The staggered operators allocate nothing" begin
    GD = FG.Grids
    O = FG.Operators
    D = FG.Discretization
    cart = FG.Geometry.CartesianGeometry{Float64}()
    C, Fa = D.Center(), D.Face()
    x = collect(range(0.0, 4.0; length = 21))
    y = collect(range(0.0, 3.0; length = 16))
    sg = GD.StaggeredGrid(cart, x, y)
    f = [3.0a - 2.0b for a in x, b in y]
    fx = GD.axis_at(sg, 1, Fa)
    fy = GD.axis_at(sg, 2, Fa)
    u = [2.0a for a in fx, _ in y]
    v = [3.0b for _ in x, b in fy]
    g1 = similar(u); g2 = similar(v)
    dv = similar(f)
    z = zeros(length(fx), length(fy))
    # Warm each, then measure: the per-point work is scale factors and one difference, and the
    # per-direction loop is a tuple over a `Val`, so nothing here belongs on the heap.
    O.gradient!((g1, g2), f, sg)
    O.divergence!(dv, (u, v), sg)
    O.curl!(z, u, v, sg)
    Test.@test _alloc(q_stag_grad!, (g1, g2), f, sg) == 0
    Test.@test _alloc(q_stag_div!, dv, (u, v), sg) == 0
    Test.@test _alloc(q_stag_curl!, z, u, v, sg) == 0

    # …on a sphere too, where the scale factors are trigonometry rather than ones.
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    λ = collect(range(0, 2π; length = 25)[1:24])
    φ = collect(range(-1.2, 1.2; length = 13))
    sp = GD.StaggeredGrid(sph, λ, φ)
    su = zeros(length(GD.axis_at(sp, 1, Fa)), length(φ))
    sv = zeros(length(λ), length(GD.axis_at(sp, 2, Fa)))
    sd = zeros(length(λ), length(φ))
    O.divergence!(sd, (su, sv), sp)
    Test.@test _alloc(q_stag_div!, sd, (su, sv), sp) == 0
end
