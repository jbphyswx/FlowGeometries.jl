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

    # The sampling-level builder and the layout describe the same graph, cell for cell.
    cs = FG.Grids.CubedSphereGrid(3)
    csconn = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), 3)
    Test.@test FG.Connectivity.nnodes(csconn) == length(cs)
    for k in 1:length(cs)
        Test.@test sort(collect(FG.Grids.neighbors(csconn, k))) ==
                   sort(collect(FG.Grids.neighbors(cs, k)))
    end
end

Test.@testset "Connectivity is built into contiguous CSR, not per-node vectors" begin
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

Test.@testset "Node-set cell areas are the sampling's own tessellation" begin
    using Quickhull: Quickhull
    R = 6.371e6
    geo = FG.Geometry.SphericalGeometry(R)
    tot = 4π * R^2

    # Equal-area by construction: uniform is exact.
    gh = healpix_node_grid(4; geometry = geo)
    ah = FG.Grids.measure(gh)
    Test.@test all(≈(tot / length(ah)), ah)
    Test.@test sum(ah) ≈ tot rtol = 1e-12

    # A geodesic's dual cells are genuinely non-uniform, so a uniform `4πR²/N` would be silently
    # wrong: the real dual areas are what it is built with.
    let g = FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(4);
                                              geometry = geo)
        a = FG.Grids.measure(g)
        Test.@test sum(a) ≈ tot rtol = 1e-8      # still tiles the sphere exactly
        Test.@test all(>(0), a)                   # no degenerate zero-area cells
        Test.@test minimum(a) / maximum(a) < 0.95 # genuinely non-uniform
    end

    # An explicit `areas` always wins over any default.
    gx = FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(2); geometry = geo, areas = fill(1.0, 42))
    Test.@test all(==(1.0), FG.Grids.measure(gx))
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

Test.@testset "Neighbor traversal allocates nothing" begin
    geom = FG.Geometry.CartesianGeometry()
    n = 40
    xs = collect(0.0:1.0:(n - 1))
    grid = FG.Grids.StructuredGrid(geom, xs, xs, trues(n, n))

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

end

Test.@testset "k-d-tree knn adjacency is the k nearest by great-circle distance" begin
    geo = FG.Geometry.SphericalGeometry(1.0)
    # The k nearest by great-circle distance, nearest-first.
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
    p32 = FG.SphericalSampling.spherical_points(Float32, FG.SphericalSampling.HEALPixSampling(2))
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

    # The existing entry points are this fold.
    Test.@test C.nneighbors_within(g, 7, 11; ball = 600.0) ==
               C.fold_within((a, J, d) -> a + 1, 0, g, 7, 11; ball = 600.0)
end

Test.@testset "A window for the whole grid, and an exact extent for one row" begin
    C = FG.Connectivity
    GD = FG.Grids
    GE = FG.Geometry
    R = 6.371e6
    sph = GE.SphericalGeometry(R)
    nλ, nφ = 72, 37
    λ = collect(range(0.0, 2π * (1 - 1 / nλ); length = nλ))
    φ = collect(range(-π / 2, π / 2; length = nφ))          # poles are rows
    g = GD.StructuredGrid(sph, λ, φ)

    # `metric_band` is the EXACT extent, so it is checked both ways: it must cover every cell of
    # the row that is genuinely in range, and no cell beyond it may be in range either. A bound
    # would pass the first and fail the second, which is the whole difference from `metric_window`.
    dλ = λ[2] - λ[1]
    for frac in (0.02, 0.15, 0.4, 0.9, 1.2), jt in (1, 7, 19, 31, nφ), jn in (1, 5, 18, 30, nφ)
        r = frac * π * R
        band = C.metric_band(g, 1, φ[jt], φ[jn], r)
        inr = [abs(rem(l, 2π, RoundNearest)) for l in λ
               if GE.distance(sph, (0.0, φ[jt]), (l, φ[jn])) ≤ r]
        if isempty(inr)
            Test.@test band < 0 || band ≤ dλ / 2 + 1e-9
        else
            Test.@test band ≥ maximum(inr) - 1e-9
            Test.@test !any(GE.distance(sph, (0.0, φ[jt]), (l, φ[jn])) ≤ r for l in λ
                            if abs(rem(l, 2π, RoundNearest)) > band + 1e-9)
        end
    end

    # The cases that fall out of the same expression, each of which a caller would otherwise have
    # to special-case: a pole at either end (where the separation stops depending on longitude),
    # a band that reaches nothing, and a ball past the antipode.
    Test.@test C.metric_band(g, 1, 0.0, π / 2, 0.05R) < 0
    Test.@test C.metric_band(g, 1, π / 2 - 0.01, π / 2, 0.05R) ≈ π
    Test.@test C.metric_band(g, 1, π / 2, π / 2, 0.01R) ≈ π
    Test.@test C.metric_band(g, 1, 0.0, 1.4, 0.01R) < 0
    Test.@test C.metric_band(g, 1, 0.0, 0.0, 3.2R) ≈ π
    # The latitude extent is not the same closed form, and is refused rather than answered wrongly.
    Test.@test_throws ArgumentError C.metric_band(g, 2, 0.0, 0.1, 1.0e5)

    # Cartesian: the exact half-chord of a circle at that offset.
    let cart = FG.Geometry.CartesianGeometry{Float64}(), x = collect(range(0.0, 10.0; length = 41))
        gc = GD.StructuredGrid(cart, x, x)
        Test.@test C.metric_band(gc, 1, 3.0, 3.0, 2.0) ≈ 2.0
        Test.@test C.metric_band(gc, 1, 3.0, 4.0, 2.0) ≈ sqrt(3.0)
        Test.@test C.metric_band(gc, 1, 3.0, 6.0, 2.0) < 0
        # The grid-level window is the per-cell one at its worst cell — checked by taking that
        # maximum, which is the O(N) computation the O(1) form exists to avoid.
        Test.@test C.metric_window(gc, 2.0) ==
                   ntuple(d -> maximum(C.metric_window(gc, (i, j), 2.0)[d] for i in 1:41, j in 1:41), 2)
    end

    for r in (0.05R, 0.3R, 1.1R)
        w = C.metric_window(g, r)
        worst = ntuple(d -> maximum(C.metric_window(g, (i, j), r)[d] for i in 1:nλ, j in 1:nφ), 2)
        Test.@test all(w .>= worst)      # conservative: never under-covers any cell
    end
    # It reads the cached extremes, so it touches no coordinate of the latitude axis.
    let cx = CountingAxis(collect(range(-1.4, 1.4; length = 200))),
        gcount = GD.StructuredGrid(sph, λ, cx)
        C.metric_window(gcount, 0.2R)
        Test.@test reads(() -> C.metric_window(gcount, 0.2R), cx) == 0
    end
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
    # What "O(1) per direction rather than a scan" means is that the candidate WINDOW does not
    # grow with the axis, so that is what is asserted — the window itself, not how long it took
    # to bound. `metric_window` returns the half-width the traversal will walk.
    wins = map((256, 4096)) do n
        g = stretched(n)
        mt = C.MetricTopology(g)
        Test.@test mt isa C.MetricTopology
        Test.@test isbits(mt)                              # nothing heap-allocated to carry
        I = (length(FG.Grids.mask(g)) ÷ 2,)
        (C.metric_window(g, I, 3.0, mt), g, mt)
    end
    # A 16× longer axis at a fixed radius: the window is the same handful of cells, not 16× wider.
    Test.@test wins[1][1] == wins[2][1]
    Test.@test only(wins[1][1]) ≤ 8
    # And the count is the truth, whichever topology computed it.
    for (_, g, mt) in wins
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
    gu = healpix_node_grid(4)
    ixu = C.indexed(gu)
    su = C.ball_scratch()
    for (idx, r) in ((1, 0.3R), (100, 0.3R), (192, 0.3R), (1, 3.1R))
        scan = sort(C.neighbors_within(gu, idx; ball = r))
        Test.@test sort(C.neighbors_within(gu, idx; ball = r, topology = ixu, scratch = su)) == scan
        Test.@test C.nneighbors_within(gu, idx; ball = r, topology = ixu) == length(scan)
    end

    # Cost: fixed radius, growing n. The scan offers every cell as a candidate; the index must
    # offer a bounded set. That is the claim, and it is a COUNT — countable exactly through the
    # public fold, which hands the caller each candidate. A clock would answer the same question
    # nondeterministically and be decided by whether a collection landed in the sample.
    # Each index enumerates its candidates its own way — a cell list folds, a tree fills a buffer,
    # because it has to deduplicate the periodic images it searched. Count through whichever it is.
    candidates(g, ix::GD.CellListIndex, r, I) = GD.fold_candidates(0, ix, g, I, r) do acc, _k
        return acc + 1
    end
    candidates(g, ix, r, I) = length(GD.index_within!(Int[], ix, g, I, r))
    small, big = curv(24), curv(96)                      # 576 vs 9216 cells, 16× more
    # `curv` spans the same extent at every `n`, so the radius has to shrink with the spacing to
    # hold the ball at a fixed cell count — otherwise what grows is the answer, not the overhead.
    cells = 2.5
    r_small, r_big = cells * 10.0 / 23, cells * 10.0 / 95
    Test.@test C.nneighbors_within(small, 12, 12; ball = r_small) ==
               C.nneighbors_within(big, 48, 48; ball = r_big)
    # A cell list is binned at the radius it will be queried at — bin it wider and each bin holds
    # more cells as the grid refines, which is a property of the caller's choice, not the index.
    for mk in ((g, r) -> GD.spatial_index(g), (g, r) -> GD.cell_list(g; ball = r))
        cs = candidates(small, mk(small, r_small), r_small, (12, 12))
        cb = candidates(big, mk(big, r_big), r_big, (48, 48))
        # 16× the cells, the same ball: the candidate set must not grow with the grid.
        cb ≤ 3 * cs || println("    candidates grew ", cs, " → ", cb, " over 16× cells")
        Test.@test cb ≤ 3 * cs
        Test.@test cb < length(GD.mask(big)) ÷ 4         # and is nothing like the scan's every-cell
    end
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
    gu = healpix_node_grid(4)
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
    gu = healpix_node_grid(4)
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

    gu = healpix_node_grid(4)
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

    gu = healpix_node_grid(4)
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

    # "Recomputes no grid invariant" is a statement about how much of the grid a call READS, so it
    # is asserted by counting reads. A `CountingAxis` is kept by `_to_axis` unchanged, so a grid
    # can be built on one and then asked what each entry point touches. Every one of these must be
    # flat in `n`; the earlier defects — a rescanned minimum spacing, an `extrema` under a search
    # radius — would each show up here as `n` reads.
    for n in (256, 4096)
        cx = CountingAxis(cumsum(1.0 .+ 0.5 .* sin.(range(0, 3π; length = n))))
        g = GD.StructuredGrid(cart, cx, collect(0.0:3.0))
        Test.@test reads(() -> C.MetricTopology(g), cx) == 0
        Test.@test reads(() -> (GD.extent(g, 1), GD.bounds(g, 1), GD.origin(g, 1)), cx) == 0
        Test.@test reads(() -> GD.minimum_spacing(g, 1), cx) == 0
        Test.@test reads(() -> GD.maximum_spacing(g, 1), cx) == 0
        # A ball query reads only its own window, not the axis: bounded, and the same bound at
        # both sizes rather than growing with the axis.
        mt = C.MetricTopology(g)
        Test.@test reads(() -> C.nneighbors_within(g, n ÷ 2, 2; ball = 3.0, topology = mt), cx) ≤ 64
    end

    # The window itself is what must stay O(1) per direction, at both sizes.
    w256 = C.metric_window(stretched(256), (128, 2), 3.0, C.MetricTopology(stretched(256)))
    w4096 = C.metric_window(stretched(4096), (2048, 2), 3.0, C.MetricTopology(stretched(4096)))
    Test.@test w256 == w4096

    # On the architectures with an index, the bounded quantity is the candidate set. `k_nearest!`
    # and the ball query both walk it, and neither may grow with the grid.
    for (name, small, big) in (("curvilinear", curv(24), curv(96)),)
        for (g, I) in ((small, (12, 12)), (big, (48, 48)))
            ix = GD.cell_list(g; ball = 2.5)
            c = FG.Grids.fold_candidates(0, ix, g, I, 2.5) do acc, _k
                return acc + 1
            end
            c ≤ 64 || println("    ", name, " candidates at n=", length(GD.mask(g)), ": ", c)
            Test.@test c ≤ 64
        end
    end
end

Test.@testset "An index records its mask policy, and a sweep passes its own" begin
    C = FG.Connectivity
    GD = FG.Grids
    GE = FG.Geometry
    SS = FG.SphericalSampling
    cart = GE.CartesianGeometry{Float64}()
    sph = GE.SphericalGeometry(6.371e6)
    R = GE.radius(sph)

    n = 12
    X = [Float64(i) for i in 1:n, _ in 1:n]
    Y = [Float64(j) for _ in 1:n, j in 1:n]
    mk = trues(n, n); mk[3:9, 3:9] .= false          # mostly-masked interior
    cgm = GD.CurvilinearGrid(cart, X, Y, mk)

    # Narrowing is worth asking for: the index is the size of the active region, not the bounding box.
    full = GD.cell_list(cgm; ball = 1.5)
    act = GD.cell_list(cgm; ball = 1.5, active_only = true)
    Test.@test Base.summarysize(act) < Base.summarysize(full)
    # …and it answers only at that policy, rather than returning a short answer.
    Test.@test C.nneighbors_within(cgm, 1, 1; ball = 1.5,
                                   topology = C.MetricTopology(cgm; index = act)) ==
               C.nneighbors_within(cgm, 1, 1; ball = 1.5)
    Test.@test_throws ArgumentError C.nneighbors_within(
        cgm, 1, 1; ball = 1.5, active_only = false,
        topology = C.MetricTopology(cgm; index = act))
    # The default covers every cell, so it serves both policies and `locate` needs no special index.
    let top = C.MetricTopology(cgm; index = full)
        for ao in (true, false)
            for I in ((1, 1), (5, 5), (n, n))
                Test.@test sort(C.neighbors_within(cgm, I...; ball = 1.5, active_only = ao,
                                                   topology = top)) ==
                           sort(C.neighbors_within(cgm, I...; ball = 1.5, active_only = ao))
            end
        end
        Test.@test GD.locate(cgm, (5.2, 5.2); topology = top, scratch = C.ball_scratch()) ==
                   GD.locate(cgm, (5.2, 5.2))
    end

    # One `build_connectivity_within` body serves every layout: the cell naming comes from
    # `cell_address`, the candidate bound from `candidate_source`, and each row must equal what the
    # per-cell query returns for that cell — at both mask policies, on every architecture.
    msk = trues(6, 5); msk[3, 3] = false; msk[4, 4] = false
    nbrs = [2, 1, 3, 2, 4, 3]
    ptr = [1, 2, 4, 6, 7, 7]
    cases = (("structured periodic, masked",
              GD.StructuredGrid(cart, range(0.0; step = 1.0, length = 6),
                                range(0.0; step = 1.0, length = 5), msk;
                                topology = (GD.Periodic(), GD.Bounded()), period = (6.0, 0.0)), 1.5),
             ("curvilinear, masked", cgm, 1.5),
             ("node set", GD.UnstructuredGrid(sph, ([0.0, 0.1, 0.2, 0.3, 1.0],
                                                    [0.0, 0.0, 0.0, 0.0, 1.0]),
                                              fill(1.0, 5), GD.AllActive((5,)), nbrs, ptr), 0.15R),
             ("HEALPix", GD.HEALPixGrid(sph, 4), 2.0e6),
             ("ring", GD.RingGrid(sph, SS.OctahedralGaussianSampling(8)), 1.5e6))
    for (name, g, r) in cases, ao in (true, false)
        conn = C.build_connectivity_within(g; ball = r, active_only = ao)
        cs = GD.cells(g)
        for k in 1:length(cs)
            cell = GD.cell_at(g, cs[k])
            got = sort(collect(GD.neighbors(conn, k)))
            want = sort(C.neighbors_within(g, cell...; ball = r, active_only = ao))
            got == want || println("    ", name, " (active_only=", ao, ") row ", k,
                                   ": ", got, " ≠ ", want)
            Test.@test got == want
            Test.@test !(k in got)              # a ball adjacency is the neighbours, not the closed ball
        end
        # The metric is symmetric, so the graph it induces is, whichever cells are excluded.
        Test.@test C.is_symmetric_adjacency(conn)
    end

    # Which default topology a sweep builds is `candidate_source`: a separable window needs no index.
    Test.@test C.default_sweep_topology(GD.StructuredGrid(cart, 0.0:1.0:5.0, 0.0:1.0:4.0),
                                        1.5, true).index === nothing
    for g in (cgm, GD.HEALPixGrid(sph, 2), GD.RingGrid(sph, SS.OctahedralGaussianSampling(4)))
        Test.@test C.default_sweep_topology(g, 1.5e6, true).index isa GD.CellListIndex
    end
end

Test.@testset "A stencil's count and reach are properties of its shape" begin
    S = FG.Stencils
    # Both are generated for the built-in shapes, so they must still equal the offset walk they replace
    # — on every shape, at every dimension.
    for N in (1, 2, 3, 4)
        for st in (S.Axial(1), S.Axial(3), S.VonNeumann(2), S.Moore(1), S.Moore(2),
                   S.Diagonal(1), S.Anisotropic((3, 1, 2, 1)[1:N]))
            offs = S.offsets(st, Val(N))
            Test.@test S.nstencil(st, Val(N)) == length(offs)
            Test.@test S.reach(st, Val(N)) == ntuple(d -> maximum(abs(o[d]) for o in offs), N)
        end
        # A caller's own shape answers by the same generic route.
        Test.@test S.nstencil(Upwind(2), Val(N)) == length(S.offsets(Upwind(2), Val(N)))
        Test.@test S.reach(Upwind(2), Val(N)) ==
                   ntuple(d -> maximum(abs(o[d]) for o in S.offsets(Upwind(2), Val(N))), N)
    end
    # The widest built-in: the answers are literals, so neither needs the 2400-offset tuple.
    Test.@test S.nstencil(S.Moore(3), Val(4)) == 2400
    Test.@test S.reach(S.Moore(3), Val(4)) == (3, 3, 3, 3)
    Test.@test (Test.@inferred S.reach(S.Moore(3), Val(4))) isa NTuple{4,Int}
end

Test.@testset "A prefix scan and an index reduction are execution primitives" begin
    E = FG.Execution
    # The scan is what every CSR builder uses between counting and filling, so it has to be exact.
    for n in (0, 1, 7, 1000, 65_536)
        counts = rand(0:9, n)
        want = Vector{Int}(undef, n + 1)
        want[1] = 1
        for i in 1:n
            want[i + 1] = want[i] + counts[i]
        end
        Test.@test E.exclusive_scan!(Vector{Int}(undef, n + 1), counts) == want
        Test.@test E.exclusive_scan!(Vector{Int}(undef, n + 1), counts)[end] - 1 == sum(counts)
        # `init` moves the whole array, which is what a caller indexing from zero needs.
        Test.@test E.exclusive_scan!(Vector{Int}(undef, n + 1), counts; init = 0) == want .- 1
    end
    Test.@test_throws DimensionMismatch E.exclusive_scan!(Vector{Int}(undef, 3), [1, 2, 3])

    # The reduction is the per-index form, so it is the one a device can run.
    for n in (0, 1, 13, 10_000)
        Test.@test E.reduce_indices(i -> i * i, +, 0, n, nothing) ==
                   sum(i * i for i in 1:n; init = 0)
    end
    # `op` is applied left to right from `init`, so a non-commutative one is still well defined.
    Test.@test E.reduce_indices(string, *, "", 4, nothing) == "1234"
end
