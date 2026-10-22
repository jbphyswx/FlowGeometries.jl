Test.@testset "No method ambiguities or unbound static parameters" begin
    Test.@test isempty(Test.detect_ambiguities(FG; recursive = true))
    Test.@test isempty(Test.detect_unbound_args(FG; recursive = true))
end

Test.@testset "An out-of-range index errors, and @inbounds still opts out" begin
    geom = FG.Geometry.CartesianGeometry()
    g = FG.Grids.StructuredGrid(geom, 0.0:1.0:4.0, 0.0:1.0:3.0, trues(5, 4))
    Test.@test_throws BoundsError FG.Grids.measure(g, 99, 99)
    Test.@test_throws BoundsError FG.Grids.area(g, 99, 99)
    Test.@test_throws BoundsError FG.Grids.isactive(g, 99, 99)
    Test.@test_throws BoundsError FG.Grids.coords(g, 99, 99)
    Test.@test_throws BoundsError FG.Grids.coords(NTuple{2,Float64}, g, 6, 1)
    Test.@test_throws BoundsError FG.Grids.coords!(zeros(2), g, 1, 5)
    # An index per direction: a third index on a 2-D grid, or one too few, names both counts.
    for idx in ((1, 1, 2), (1,))
        err = try
            FG.Grids.coords(g, idx...)
            nothing
        catch e
            e
        end
        Test.@test err isa ArgumentError
        Test.@test occursin("2 directions; got $(length(idx)) indices", sprint(showerror, err))
    end

    nx, ny = 4, 3
    xm = [Float64(i) for i in 1:nx, j in 1:ny]
    ym = [Float64(j) for i in 1:nx, j in 1:ny]
    cv = FG.Grids.CurvilinearGrid(geom, xm, ym, trues(nx, ny); keep_corners = true)
    Test.@test_throws BoundsError FG.Grids.coords(cv, nx + 1, 1)
    Test.@test_throws BoundsError FG.Grids.corner_coords(cv, nx + 3, 1)
    Test.@test_throws BoundsError FG.Grids.measure(cv, 1, ny + 1)

    un = FG.Grids.UnstructuredGrid(geom, [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], trues(2))
    Test.@test_throws BoundsError FG.Grids.coords(un, 3)
    Test.@test_throws BoundsError FG.Grids.measure(un, 3)
end

Test.@testset "Public names the suite had not been calling" begin
    GE = FG.Geometry
    C = FG.Connectivity
    S = FG.SphericalSampling

    # cartesian_to_spherical: the documented inverse of spherical_to_cartesian.
    sph = GE.SphericalGeometry(6.371e6)
    for p in ((0.0, 0.0), (1.2, -0.4), (5.9, 1.1))
        xyz = GE.spherical_to_cartesian(sph, p)
        back = GE.cartesian_to_spherical(sph, xyz)
        Test.@test mod(back.λ, 2π) ≈ mod(p[1], 2π) atol = 1e-12
        Test.@test back.φ ≈ p[2] atol = 1e-12
    end
    # A 3-D point carries its own radius through the round trip.
    r3 = GE.cartesian_to_spherical(sph, GE.spherical_to_cartesian(sph, (0.3, 0.4, 7.0e6)))
    Test.@test r3.r ≈ 7.0e6

    # cartesian_index is linear_index's inverse on every architecture that has one.
    geo = GE.CartesianGeometry()
    g = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:2.0, 0.0:1.0:1.0)
    Test.@test all(C.linear_index(g, Tuple(C.cartesian_index(g, k))...) == k
                   for k in 1:length(FG.Grids.mask(g)))
    Test.@test C.cartesian_index(g, 1) == CartesianIndex(1, 1, 1)

    # empty_csr / csr_connectivity: the storage type's own constructors.
    e = C.empty_csr(5)
    Test.@test C.nnodes(e) == 5 && C.nedges(e) == 0
    Test.@test all(isempty(e.nbrs[e.ptr[i]:(e.ptr[i + 1] - 1)]) for i in 1:5)
    Test.@test eltype(C.empty_csr(4, Int32).ptr) === Int32
    csr = C.csr_connectivity([2, 1, 3, 2], [1, 2, 4, 5])
    Test.@test C.nnodes(csr) == 3 && C.nedges(csr) == 4
    Test.@test collect(csr.nbrs[csr.ptr[2]:(csr.ptr[3] - 1)]) == [1, 3]
    Test.@test C.is_symmetric_adjacency(csr)
    # …and it validates the buffers it is handed.
    Test.@test_throws ArgumentError C.csr_connectivity([1], [2, 2])        # ptr[1] != 1
    Test.@test_throws ArgumentError C.csr_connectivity([1, 2], [1, 2])     # length mismatch

    # geographic_latitude / colatitude are each other's inverse.
    Test.@test S.geographic_latitude(0.0) ≈ π / 2
    Test.@test S.geographic_latitude(π) ≈ -π / 2
    Test.@test S.geographic_latitude(S.colatitude(0.37)) ≈ 0.37
    Test.@test S.colatitude(S.geographic_latitude(1.1)) ≈ 1.1

    # ring_info: rings tile the map contiguously, and each one's latitude is pix2ang's.
    for nside in (1, 2, 4, 8)
        nrings = 4 * nside - 1
        infos = [S.ring_info(nside, r) for r in 1:nrings]
        Test.@test sum(i -> i.ringpix, infos) == 12 * nside^2
        Test.@test infos[1].startpix == 0
        Test.@test all(infos[r].startpix + infos[r].ringpix == infos[r + 1].startpix
                       for r in 1:(nrings - 1))
        Test.@test all(S.pix2ang(nside, infos[r].startpix)[1] ≈ infos[r].colatitude
                       for r in 1:nrings)
        Test.@test all(i -> i.latitude ≈ π / 2 - i.colatitude, infos)
        Test.@test all(infos[r].ringpix == 4r for r in 1:(nside - 1))          # polar cap
        Test.@test all(infos[r].ringpix == 4nside for r in nside:(3nside))     # equatorial belt
        Test.@test all(infos[r].ringpix == infos[nrings + 1 - r].ringpix for r in 1:nrings)
        Test.@test all(infos[r].latitude ≈ -infos[nrings + 1 - r].latitude for r in 1:nrings)
    end
    Test.@test abs(S.ring_info(8, 16).latitude) < 1e-15      # ring 2·nside is the equator
    Test.@test_throws ArgumentError S.ring_info(4, 0)
    Test.@test_throws ArgumentError S.ring_info(4, 16)
    Test.@test_throws ArgumentError S.ring_info(0, 1)
end

Test.@testset "Asking for an element type gives back that element type, knowably" begin
    SS = FG.SphericalSampling
    gl = SS.GaussLegendreSampling()
    hp = SS.HEALPixSampling(4)

    for W in (Float64, Float32)
        sg = FG.Connectivity.structured_grid(W, gl, 12)
        Test.@test eltype(FG.Grids.axis(sg, 1)) === W
        Test.@test FG.Grids.coords(sg, 1, 1).λ isa W
    end

    # An explicit geometry is carried to the requested width, keeping its radius.
    gsm = FG.Connectivity.structured_grid(Float32, gl, 12;
                                          geometry = FG.Geometry.SphericalGeometry(3.0e6))
    Test.@test FG.Grids.grid_geometry(gsm) === FG.Geometry.SphericalGeometry{Float32}(3.0f6)
    Test.@test eltype(FG.Grids.axis(gsm, 2)) === Float32

    # With no element type named, the geometry's own is the one meant.
    g32 = FG.Connectivity.structured_grid(gl, 12;
                                          geometry = FG.Geometry.SphericalGeometry(6.371f6))
    Test.@test eltype(FG.Grids.axis(g32, 1)) === Float32
    for W in (Float64, Float32)
        Test.@test FG.Geometry.float_type(FG.Geometry.SphericalGeometry{W}(6.371e6)) === W
        Test.@test FG.Geometry.float_type(FG.Geometry.CartesianGeometry{W}()) === W
    end

    # A geometry defined outside the package, already at the width asked for, comes back as itself.
    Test.@test FG.Geometry.similar_geometry(Float64, OneSphere{Float64}()) isa OneSphere{Float64}
    let og = FG.Connectivity.structured_grid(gl, 8; geometry = OneSphere{Float64}())
        Test.@test FG.Grids.grid_geometry(og) isa OneSphere{Float64}
        Test.@test eltype(FG.Grids.axis(og, 1)) === Float64
    end

    for W in (Float64, Float32)
        Test.@test FG.Geometry.similar_geometry(W, FG.Geometry.CartesianGeometry()) ===
                   FG.Geometry.CartesianGeometry{W}()
        sp = FG.Geometry.similar_geometry(W, FG.Geometry.SpheroidGeometry())
        Test.@test sp isa FG.Geometry.SpheroidGeometry{W}
        Test.@test FG.Geometry.flattening(sp) ≈ FG.Geometry.flattening(FG.Geometry.SpheroidGeometry())
    end

    # The width asked for is the width returned.
    Test.@test eltype(SS.spherical_axes(Float32, gl, 12).φ) === Float32
    Test.@test eltype(SS.spherical_points(Float32, hp).λ) === Float32
    Test.@test eltype(SS.icosahedral_vertices(Float32, 2).λ) === Float32
    Test.@test SS.ring_info(Float32, 4, 3).colatitude isa Float32
    Test.@test SS.pix2ang(Float32, 4, 10) isa NTuple{2,Float32}
    Test.@test SS.pix2vec(Float32, 4, 10) isa NTuple{3,Float32}

    # A unit vector really is the direction its angles name, at either width.
    for W in (Float64, Float32)
        θ, ϕ = SS.pix2ang(W, 8, 100)
        v = SS.pix2vec(W, 8, 100)
        Test.@test all(v .≈ (sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)))
    end
end
