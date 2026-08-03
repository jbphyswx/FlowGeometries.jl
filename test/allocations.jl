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
