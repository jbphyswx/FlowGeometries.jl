using FlowGeometries: FlowGeometries
using Test: Test

const FG = FlowGeometries

Test.@testset "FlowGeometries.jl" begin

    Test.@testset "Abstract hierarchy" begin
        Test.@test FG.AbstractCartesianGeometry <: FG.AbstractGeometry
        Test.@test FG.AbstractSphericalGeometry <: FG.AbstractGeometry
        Test.@test FG.CartesianGeometry <: FG.AbstractCartesianGeometry
        Test.@test FG.SphericalGeometry <: FG.AbstractSphericalGeometry
        Test.@test FG.AbstractStructuredGrid <: FG.AbstractGrid
        Test.@test FG.AbstractCurvilinearGrid <: FG.AbstractGrid
        Test.@test FG.AbstractUnstructuredGrid <: FG.AbstractGrid
        Test.@test FG.StructuredGrid <: FG.AbstractStructuredGrid
        Test.@test FG.CurvilinearGrid <: FG.AbstractCurvilinearGrid
        Test.@test FG.UnstructuredGrid <: FG.AbstractUnstructuredGrid
    end

    Test.@testset "User geometry subtype participates" begin
        struct StretchedCartesian{T} <: FG.AbstractCartesianGeometry{T}
            dx::T
            dy::T
            dz::T
        end
        g = StretchedCartesian(2.0, 3.0, 0.0)
        Test.@test g isa FG.AbstractGeometry
        Test.@test FG.area_element(g) ≈ 6.0
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        Test.@test FG.distance(g, p1, p2) ≈ 5.0
    end

    Test.@testset "Cartesian geometry" begin
        geom = FG.CartesianGeometry(1000.0, 1000.0)
        Test.@test geom.dz == 0.0
        p1 = (0.0, 0.0)
        p2 = (3000.0, 4000.0)
        Test.@test FG.distance(geom, p1, p2) ≈ 5000.0
        Test.@test FG.area_element(geom) ≈ 1.0e6
        Test.@test FG.volume_element(FG.CartesianGeometry(1.0, 2.0, 3.0)) ≈ 6.0
    end

    Test.@testset "Spherical geometry + planetary Cartesian" begin
        geom = FG.SphericalGeometry(6.371e6)
        london = (deg2rad(-0.1276), deg2rad(51.5074))
        paris = (deg2rad(2.3522), deg2rad(48.8566))
        d_km = FG.distance(geom, london, paris) / 1000.0
        Test.@test 300 < d_km < 400

        λ, φ = 0.3, 0.4
        uλ, uφ = 1.0, -0.5
        p = FG.to_planetary_cartesian(geom, uλ, uφ, λ, φ)
        back = FG.from_planetary_cartesian(geom, p, λ, φ)
        Test.@test back.λ ≈ uλ
        Test.@test back.φ ≈ uφ
        Test.@test abs(back.r) < 1e-12
    end

    Test.@testset "nonuniform_first_derivative" begin
        # Uniform: recovers (f_p - f_m)/(2h)
        Test.@test FG.nonuniform_first_derivative(1.0, 2.0, 3.0, 1.0, 1.0) ≈ 1.0
        # Exact for quadratic on nonuniform stencil
        # f(x)=x^2 at x=0 with neighbors -1 and 2: f'=2x=0 at center... use x=-1,0,2
        f_m, f_0, f_p = 1.0, 0.0, 4.0  # (-1)^2, 0, 2^2
        Test.@test FG.nonuniform_first_derivative(f_m, f_0, f_p, 1.0, 2.0) ≈ 0.0 atol = 1e-12
    end

    Test.@testset "StructuredGrid preserves Range axes" begin
        geom = FG.CartesianGeometry(2000.0, 2000.0)
        x = 0.0:2000.0:20_000.0
        y = 0.0:2000.0:10_000.0
        mask = trues(length(x), length(y))
        grid = FG.StructuredGrid(geom, x, y, mask)
        Test.@test grid.x isa AbstractRange
        Test.@test grid.y isa AbstractRange
        Test.@test FG.size_tuple(grid) == (length(x), length(y))
        Test.@test FG.area(grid, 2, 2) ≈ 2000.0 * 2000.0
        Test.@test FG.coords(grid, 2, 3) == (x = 2000.0, y = 4000.0)
        Test.@test FG.coords(NTuple{2,Float64}, grid, 2, 3) == (2000.0, 4000.0)
        out = zeros(2)
        FG.coords!(out, grid, 2, 3)
        Test.@test out == [2000.0, 4000.0]
        Test.@test FG.isactive(grid, 1, 1)
        Test.@test FG.grid_geometry(grid) === geom
        Test.@test !FG.isperiodic(grid, 1)  # Cartesian not auto-periodic
    end

    Test.@testset "StructuredGrid with Vector axes is still structured" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        x = collect(0.0:1.0:4.0)
        y = collect(0.0:1.0:3.0)
        grid = FG.StructuredGrid(geom, x, y, trues(length(x), length(y)))
        Test.@test grid isa FG.StructuredGrid
        Test.@test grid.x isa Vector
        Test.@test grid.y isa Vector
        Test.@test FG.size_tuple(grid) == (5, 4)
    end

    Test.@testset "Spherical StructuredGrid auto-periodic longitude" begin
        R = 6.371e6
        geom = FG.SphericalGeometry(R)
        # Full-circle longitude (closed to within one cell)
        nλ, nφ = 8, 5
        dλ = 2π / nλ
        lon = range(0.0; step = dλ, length = nλ)
        lat = range(-π / 4, π / 4; length = nφ)
        grid = FG.StructuredGrid(geom, lon, lat, trues(nλ, nφ))
        Test.@test FG.isperiodic(grid, 1)
        Test.@test !FG.isperiodic(grid, 2)
        Test.@test grid.x isa AbstractRange
        Test.@test all(FG.area(grid, i, j) > 0 for i in 1:nλ, j in 1:nφ)
        p = FG.coords(grid, 1, 1)
        Test.@test keys(p) == (:λ, :φ)
        Test.@test p.λ ≈ 0.0
        Test.@test p.φ ≈ -π / 4
    end

    Test.@testset "1D and 3D StructuredGrid" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        x = 0.0:1.0:9.0
        g1 = FG.StructuredGrid(geom, x, trues(length(x)))
        Test.@test FG.size_tuple(g1) == (10,)
        Test.@test FG.area(g1, 2) ≈ 1.0

        geom3 = FG.CartesianGeometry(1.0, 1.0, 1.0)
        z = 0.0:1.0:4.0
        g3 = FG.StructuredGrid(geom3, x, x, z, trues(length(x), length(x), length(z)))
        Test.@test FG.size_tuple(g3) == (10, 10, 5)
        Test.@test FG.area(g3, 2, 2, 2) ≈ 1.0
    end

    Test.@testset "CurvilinearGrid" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        nx, ny = 4, 3
        x = [Float64(i) for i in 1:nx, j in 1:ny]
        y = [Float64(j) for i in 1:nx, j in 1:ny]
        grid = FG.CurvilinearGrid(geom, x, y, trues(nx, ny))
        Test.@test FG.size_tuple(grid) == (nx, ny)
        Test.@test FG.coords(grid, 2, 2) == (x = 2.0, y = 2.0)
        Test.@test FG.area(grid, 2, 2) > 0
    end

    Test.@testset "UnstructuredGrid explicit adjacency" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        x = [0.0, 1.0, 0.0]
        y = [0.0, 0.0, 1.0]
        areas = [0.5, 0.5, 0.5]
        mask = trues(3)
        # CSR: node 1 → [2,3], node 2 → [1], node 3 → [1]
        nbrs = [2, 3, 1, 1]
        ptr = [1, 3, 4, 5]
        grid = FG.UnstructuredGrid(geom, x, y, areas, mask, nbrs, ptr)
        Test.@test FG.size_tuple(grid) == (3,)
        Test.@test collect(FG.neighbors(grid, 1)) == [2, 3]
        Test.@test FG.coords(grid, 2) == (x = 1.0, y = 0.0)

        # Convenience no-neighbor constructor
        g0 = FG.UnstructuredGrid(geom, x, y, areas, mask)
        Test.@test isempty(FG.neighbors(g0, 1))
    end

    Test.@testset "UnstructuredGrid auto-build throws without extensions" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        x = [0.0, 1.0, 0.5]
        y = [0.0, 0.0, 1.0]
        Test.@test_throws ArgumentError FG.UnstructuredGrid(geom, x, y, trues(3))
    end

    Test.@testset "local_tangent_basis / project_to_tangent_plane" begin
        cgeom = FG.CartesianGeometry(1.0, 1.0)
        c = (0.0, 0.0)
        n = (1.0, 2.0)
        Test.@test FG.project_to_tangent_plane(cgeom, c, n) == (; x = 1.0, y = 2.0)
        Test.@test FG.distance(cgeom, (x = 0.0, y = 0.0), (x = 3.0, y = 4.0)) ≈ 5.0

        sgeom = FG.SphericalGeometry(1.0)
        ê = FG.local_tangent_basis(sgeom, (0.0, 0.0))
        Test.@test abs(sqrt(sum(abs2, ê.λ)) - 1) < 1e-14
        Test.@test abs(sqrt(sum(abs2, ê.φ)) - 1) < 1e-14
        Test.@test FG.distance(sgeom, (λ = 0.0, φ = 0.0), (λ = 0.1, φ = 0.0)) ≈ sgeom.R * 0.1 atol = 1e-10
    end

    Test.@testset "Spherical sampling hierarchy" begin
        Test.@test FG.GaussLegendreSampling <: FG.AbstractGaussLegendreSampling
        Test.@test FG.DriscollHealySampling <: FG.AbstractDriscollHealySampling
        Test.@test FG.ClenshawCurtisSampling <: FG.AbstractClenshawCurtisSampling
        Test.@test FG.McEwenWiauxSampling <: FG.AbstractMcEwenWiauxSampling
        Test.@test FG.LatLonSampling <: FG.AbstractLatLonSampling
        Test.@test FG.HEALPixSampling <: FG.AbstractHEALPixSampling
        Test.@test FG.CubedSphereSampling <: FG.AbstractCubedSphereSampling
        Test.@test FG.IcosahedralSampling <: FG.AbstractIcosahedralSampling
        Test.@test FG.YinYangSampling <: FG.AbstractYinYangSampling
        Test.@test FG.ScatteredSphericalSampling <: FG.AbstractScatteredSphericalSampling

        Test.@test FG.is_tensor_product(FG.ClenshawCurtisSampling())
        Test.@test FG.is_iso_latitude(FG.HEALPixSampling(2))
        Test.@test FG.is_equal_area(FG.HEALPixSampling(2))
        Test.@test FG.admits_exact_bandlimited_quadrature(FG.GaussLegendreSampling())
        Test.@test !FG.admits_exact_bandlimited_quadrature(FG.LatLonSampling())
        Test.@test !FG.is_tensor_product(FG.HEALPixSampling(1))
    end

    Test.@testset "Clenshaw–Curtis = FastSphericalHarmonics sph_points" begin
        nθ = 8
        (; λ, φ) = FG.spherical_axes(FG.ClenshawCurtisSampling(), nθ)
        Test.@test length(φ) == nθ
        Test.@test length(λ) == 2nθ - 1
        Test.@test FG.bandlimit(FG.ClenshawCurtisSampling(), nθ) == nθ - 1
        # θ = π(i−1/2)/N ; φ_geo = π/2 − θ
        θ = [π * (i - 0.5) / nθ for i in 1:nθ]
        Test.@test φ ≈ (π / 2 .- θ)
        Test.@test λ[1] ≈ 0
        Test.@test λ[2] - λ[1] ≈ 2π / (2nθ - 1)
        # Open: no poles
        Test.@test all(abs.(φ) .< π / 2 - 1e-12)
    end

    Test.@testset "Gauss–Legendre nodes / weights" begin
        nθ = 16
        (; λ, φ) = FG.spherical_axes(FG.GaussLegendreSampling(), nθ)
        w = FG.latitude_weights(FG.GaussLegendreSampling(), nθ)
        Test.@test length(λ) == 2nθ - 1
        Test.@test length(φ) == nθ
        Test.@test sum(w) ≈ 2 atol = 1e-12          # ∫_{-1}^{1} dμ
        Test.@test FG.bandlimit(FG.GaussLegendreSampling(), nθ) == nθ - 1
        # μ = sin(φ) should be symmetric about 0
        μ = sin.(φ)
        Test.@test μ ≈ -reverse(μ) atol = 1e-12
    end

    Test.@testset "Driscoll–Healy DH1 / DH2" begin
        lmax = 5
        nlat = FG.nlat_for_bandlimit(FG.DriscollHealySampling(), lmax)
        Test.@test nlat == 2 * (lmax + 1)
        ax = FG.spherical_axes(FG.DriscollHealySampling(), nlat)
        λ2 = ax.λ
        φ = ax.φ
        ax = FG.spherical_axes(FG.DriscollHealyEqualSampling(), nlat)
        λ1 = ax.λ
        φ1 = ax.φ
        Test.@test length(λ2) == 2nlat
        Test.@test length(λ1) == nlat
        Test.@test φ ≈ φ1
        Test.@test φ[1] ≈ π / 2 atol = 1e-14       # north pole
        Test.@test φ[end] > -π / 2 + 1e-10         # south excluded
        w = FG.latitude_weights(FG.DriscollHealySampling(), nlat)
        Test.@test w[1] ≈ 0 atol = 1e-14           # north pole weight vanishes
        Test.@test FG.bandlimit(FG.DriscollHealySampling(), nlat) == lmax
    end

    Test.@testset "McEwen–Wiaux axes" begin
        lmax = 7
        nlat = FG.nlat_for_bandlimit(FG.McEwenWiauxSampling(), lmax)
        (; λ, φ) = FG.spherical_axes(FG.McEwenWiauxSampling(), nlat)
        L = lmax + 1
        Test.@test length(φ) == L
        Test.@test length(λ) == 2L - 1
        θ = FG.colatitude.(φ)
        Test.@test θ[1] ≈ π / (2L - 1)
        Test.@test θ[end] ≈ π
    end

    Test.@testset "LatLonSampling" begin
        (; λ, φ) = FG.spherical_axes(FG.LatLonSampling(), 5; nlon = 8)
        Test.@test length(λ) == 8
        Test.@test length(φ) == 5
        Test.@test φ[1] ≈ -π / 2
        Test.@test φ[end] ≈ π / 2
    end

    Test.@testset "HEALPix" begin
        s = FG.HEALPixSampling(1)
        Test.@test FG.healpix_npix(s) == 12
        Test.@test FG.healpix_nring(s) == 3
        Test.@test FG.healpix_pixel_area(s) ≈ 4π / 12
        (; λ, φ) = FG.spherical_points(s)
        Test.@test length(λ) == 12
        Test.@test length(φ) == 12
        s4 = FG.HEALPixSampling(4)
        Test.@test FG.healpix_npix(s4) == 12 * 16
        ax = FG.spherical_points(s4)
        λ4 = ax.λ
        φ4 = ax.φ
        Test.@test length(λ4) == 192
        # Equal-area: all pixels same area by construction; centers on |φ| < π/2
        Test.@test all(abs.(φ4) .≤ π / 2 + 1e-12)
        # nside=32 first ring: 4 pixels at same latitude, φ = 45° + 90°k
        ax = FG.spherical_points(FG.HEALPixSampling(32))
        λ32 = ax.λ
        φ32 = ax.φ
        Test.@test length(unique(round.(φ32[1:4]; digits = 12))) == 1
        Test.@test rad2deg.(λ32[1:4]) ≈ [45.0, 135.0, 225.0, 315.0] atol = 1e-8
        Test.@test rad2deg(FG.colatitude(φ32[1])) ≈ 1.46197116 atol = 1e-5
    end

    Test.@testset "Cubed sphere / Yin–Yang / icosahedral" begin
        (; λ, φ, panel) = FG.cubed_sphere_points(4)
        Test.@test length(λ) == 6 * 16
        Test.@test extrema(panel) == (1, 6)
        Test.@test all(abs.(φ) .≤ π / 2 + 1e-10)

        yy = FG.yin_yang_axes(6, 4)
        Test.@test length(yy.yin.λ) == 6
        Test.@test length(yy.yin.φ) == 4
        Test.@test length(yy.yang.λ) == 24

        ax = FG.icosahedral_vertices(1)

        λi = ax.λ

        φi = ax.φ
        Test.@test length(λi) == 12
        ax = FG.icosahedral_vertices(2)
        λ2 = ax.λ
        φ2 = ax.φ
        # ν=2 geodesic: 10ν²+2 = 42 vertices
        Test.@test length(λ2) == 42
    end

    Test.@testset "Sampling → StructuredGrid" begin
        geom = FG.SphericalGeometry(1.0)
        (; λ, φ) = FG.spherical_axes(FG.ClenshawCurtisSampling(), 6)
        grid = FG.StructuredGrid(geom, λ, φ, trues(length(λ), length(φ)))
        Test.@test FG.isperiodic(grid, 1)
        Test.@test FG.size_tuple(grid) == (length(λ), length(φ))
    end


    Test.@testset "bang spherical_axes! / points!" begin
        nθ = 8
        sz = FG.axes_lengths(FG.ClenshawCurtisSampling(), nθ)
        λ = Vector{Float64}(undef, sz.nlon)
        φ = Vector{Float64}(undef, sz.nlat)
        FG.spherical_axes!(λ, φ, FG.ClenshawCurtisSampling(), nθ)
        ax = FG.spherical_axes(FG.ClenshawCurtisSampling(), nθ)
        λ2 = ax.λ
        φ2 = ax.φ
        Test.@test λ == λ2
        Test.@test φ == φ2

        n = FG.npoints(FG.HEALPixSampling(2))
        Λ = Vector{Float64}(undef, n)
        Φ = Vector{Float64}(undef, n)
        FG.spherical_points!(Λ, Φ, FG.HEALPixSampling(2))
        ax = FG.spherical_points(FG.HEALPixSampling(2))
        Λ2 = ax.λ
        Φ2 = ax.φ
        Test.@test Λ == Λ2
        Test.@test Φ == Φ2

        w = Vector{Float64}(undef, 12)
        FG.latitude_weights!(w, FG.GaussLegendreSampling(), 12)
        Test.@test sum(w) ≈ 2 atol = 1e-12
    end

    Test.@testset "Connectivity CSR / structured neighbors" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        grid = FG.StructuredGrid(geom, 0.0:1.0:2.0, 0.0:1.0:2.0, trues(3, 3); periodic = (false, false))
        nbr = FG.neighbors(grid, 2, 2)
        Test.@test Set(nbr) == Set([
            FG.linear_index(grid, 1, 2),
            FG.linear_index(grid, 3, 2),
            FG.linear_index(grid, 2, 1),
            FG.linear_index(grid, 2, 3),
        ])
        Test.@test FG.nneighbors(grid, 2, 2) == 4
        Test.@test FG.nneighbors(grid, 1, 1) == 2
        out = Vector{Int}(undef, 4)
        Test.@test FG.neighbors!(out, grid, 2, 2) == 4
        Test.@test Set(out) == Set(nbr)

        conn = FG.build_connectivity(grid)
        Test.@test conn isa FG.CSRConnectivity
        Test.@test FG.nnodes(conn) == 9
        Test.@test Set(FG.Connectivity.neighbors(conn, FG.linear_index(grid, 2, 2))) == Set(nbr)
        Test.@test FG.nedges(conn) == sum(FG.nneighbors(grid, Tuple(ci)...) for ci in CartesianIndices((3, 3)))

        mask = trues(3, 3)
        mask[2, 2] = false
        g2 = FG.StructuredGrid(geom, 0.0:1.0:2.0, 0.0:1.0:2.0, mask)
        Test.@test FG.nneighbors(g2, 2, 2) == 0
        Test.@test FG.nneighbors(g2, 1, 2) == 2

        ug = FG.UnstructuredGrid(geom, [0.0, 1.0, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], trues(3),
            [2, 3, 1, 3, 1, 2], [1, 3, 5, 7])
        uc = FG.build_connectivity(ug)
        Test.@test Set(FG.neighbors(ug, 1)) == Set([2, 3])
        Test.@test Set(FG.Connectivity.neighbors(uc, 1)) == Set([2, 3])

        A = FG.adjacency_matrix(grid)
        Test.@test A isa Matrix{Bool}
        Test.@test size(A) == (9, 9)
        Test.@test A[FG.linear_index(grid, 2, 2), FG.linear_index(grid, 2, 3)]
        Abuf = falses(9, 9)
        FG.adjacency_matrix!(Abuf, grid)
        Test.@test Abuf == A
        FG.adjacency_matrix!(Abuf, conn)
        Test.@test Abuf == A
    end

    Test.@testset "Curvilinear periodicity" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        x = [Float64(i) for i in 1:3, j in 1:2]
        y = [Float64(j) for i in 1:3, j in 1:2]
        g = FG.CurvilinearGrid(geom, x, y, trues(3, 2); periodic = (true, false))
        Test.@test FG.isperiodic(g, 1)
        Test.@test !FG.isperiodic(g, 2)
        Test.@test FG.nneighbors(g, 1, 1) == 3  # wrap in x, open in y
        nbr = FG.neighbors(g, 1, 1)
        Test.@test FG.linear_index(g, 3, 1) in nbr  # periodic wrap
        Test.@test FG.linear_index(g, 1, 2) in nbr
    end

    Test.@testset "Spherical sampling connectivity" begin
        function _symmetric(conn)
            A = FG.adjacency_matrix(conn)
            return A == A'
        end
        function _min_degree(conn, dmin)
            return all(i -> FG.nneighbors(conn, i) ≥ dmin, 1:FG.nnodes(conn))
        end

        # Tensor-product → structured lon-periodic
        sg = FG.structured_grid(FG.ClenshawCurtisSampling(), 8)
        Test.@test FG.isperiodic(sg, 1)
        Test.@test !FG.isperiodic(sg, 2)
        cc = FG.build_connectivity(FG.ClenshawCurtisSampling(), 8)
        Test.@test FG.nnodes(cc) == length(sg.mask)
        Test.@test _symmetric(cc)

        # Cubed sphere: seams + symmetry
        n = 4
        csc = FG.build_connectivity(FG.CubedSphereSampling(), n)
        Test.@test FG.nnodes(csc) == 6 * n * n
        Test.@test _symmetric(csc)
        Test.@test _min_degree(csc, 2)
        # Interior panel cell (away from edges) has 4 face neighbors
        # face 1, i=j=2 → lin = (2-1)*n + 2 = n+2 when j=2,i=2 → (f-1)*n²+(j-1)*n+i
        lin_int = (1 - 1) * n * n + (2 - 1) * n + 2
        Test.@test FG.nneighbors(csc, lin_int) == 4
        # Edge (not corner): still 4 after seam fold
        lin_edge = (1 - 1) * n * n + (1 - 1) * n + 2  # j=1, i=2 on face 1
        Test.@test FG.nneighbors(csc, lin_edge) == 4

        # Yin–Yang: two disconnected panels
        nlon, nlat = 5, 4
        yyc = FG.build_connectivity(FG.YinYangSampling(), nlon, nlat)
        Test.@test FG.nnodes(yyc) == 2 * nlon * nlat
        Test.@test _symmetric(yyc)
        yin_nodes = 1:(nlon * nlat)
        yang_nodes = (nlon * nlat + 1):(2 * nlon * nlat)
        for i in yin_nodes
            Test.@test all(j -> j in yin_nodes, FG.Connectivity.neighbors(yyc, i))
        end
        for i in yang_nodes
            Test.@test all(j -> j in yang_nodes, FG.Connectivity.neighbors(yyc, i))
        end

        # HEALPix: documented RING neighbors for nside=4, pix=1 (0-based)
        nbr0 = FG.healpix_neighbors(4, 1)
        Test.@test Set(nbr0) == Set([16, 6, 5, 0, 3, 2, 8, 7])
        hpc = FG.build_connectivity(FG.HEALPixSampling(2))
        Test.@test FG.nnodes(hpc) == 12 * 2 * 2
        Test.@test _symmetric(hpc)
        # nside=1 → every pixel has 6 neighbors
        hp1 = FG.build_connectivity(FG.HEALPixSampling(1))
        Test.@test all(i -> FG.nneighbors(hp1, i) == 6, 1:12)
        Test.@test _symmetric(hp1)

        # Icosahedral ν=1: 12 vertices, each degree 5; 30 undirected edges → 60 directed
        ic1 = FG.build_connectivity(FG.IcosahedralSampling(1))
        Test.@test FG.nnodes(ic1) == 12
        Test.@test all(i -> FG.nneighbors(ic1, i) == 5, 1:12)
        Test.@test FG.nedges(ic1) == 60
        Test.@test _symmetric(ic1)
        mesh2 = FG.icosahedral_mesh(2)
        Test.@test length(mesh2.λ) == FG.icosahedral_nvertices(2)
        ic2 = FG.build_connectivity(FG.IcosahedralSampling(2))
        Test.@test FG.nnodes(ic2) == length(mesh2.λ)
        Test.@test _symmetric(ic2)
        Test.@test _min_degree(ic2, 5)

        ug = FG.unstructured_grid(FG.CubedSphereSampling(), 3)
        Test.@test ug isa FG.UnstructuredGrid
        Test.@test length(FG.neighbors(ug, 1)) == FG.nneighbors(FG.build_connectivity(FG.CubedSphereSampling(), 3), 1)
    end

    Test.@testset "SparseArrays sparse_adjacency_matrix (optional)" begin
        using SparseArrays: SparseArrays as Sp
        geom = FG.CartesianGeometry(1.0, 1.0)
        grid = FG.StructuredGrid(geom, 0.0:1.0:1.0, 0.0:1.0:1.0, trues(2, 2))
        conn = FG.build_connectivity(grid)
        ne = FG.nedges(conn)
        I = Vector{Int}(undef, ne)
        J = Vector{Int}(undef, ne)
        Test.@test FG.sparse_adjacency_coo!(I, J, conn) == ne
        S = FG.sparse_adjacency_matrix(conn)
        Test.@test S isa Sp.SparseMatrixCSC
        Test.@test size(S) == (4, 4)
        Test.@test Sp.nnz(S) == ne
        Test.@test S[1, 2] && S[2, 1]
        Test.@test Matrix(S) == FG.adjacency_matrix(conn)
    end

    Test.@testset "StaticArrays extension" begin
        using StaticArrays: StaticArrays as SA
        geom = FG.SphericalGeometry(1.0)
        p1 = SA.SVector{2,Float64}(0.0, 0.0)
        p2 = SA.SVector{2,Float64}(0.1, 0.2)
        d = FG.distance(geom, p1, p2)
        Test.@test d ≈ FG.distance(geom, Tuple(p1), Tuple(p2))
        ê = FG.local_tangent_basis(geom, p1)
        Test.@test ê.λ isa NTuple
    end

end
