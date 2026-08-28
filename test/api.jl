Test.@testset "No method ambiguities or unbound static parameters" begin
    Test.@test isempty(Test.detect_ambiguities(FG; recursive = true))
    Test.@test isempty(Test.detect_unbound_args(FG; recursive = true))
end

Test.@testset "An out-of-range index errors, and @inbounds still opts out" begin
    # An out-of-range index must error rather than read past the end of the array. `@boundscheck`
    # is elided at an `@inbounds` call site, so a hot loop still pays nothing.
    geom = FG.Geometry.CartesianGeometry()
    g = FG.Grids.StructuredGrid(geom, 0.0:1.0:4.0, 0.0:1.0:3.0, trues(5, 4))
    Test.@test_throws BoundsError FG.Grids.measure(g, 99, 99)
    Test.@test_throws BoundsError FG.Grids.area(g, 99, 99)
    Test.@test_throws BoundsError FG.Grids.isactive(g, 99, 99)
    Test.@test_throws BoundsError FG.Grids.coords(g, 99, 99)
    Test.@test_throws BoundsError FG.Grids.coords(NTuple{2,Float64}, g, 6, 1)
    Test.@test_throws BoundsError FG.Grids.coords!(zeros(2), g, 1, 5)
    Test.@test_throws BoundsError FG.Grids.measure(g, 0, 1)

    nx, ny = 4, 3
    xm = [Float64(i) for i in 1:nx, j in 1:ny]
    ym = [Float64(j) for i in 1:nx, j in 1:ny]
    cv = FG.Grids.CurvilinearGrid(geom, xm, ym, trues(nx, ny))
    Test.@test_throws BoundsError FG.Grids.coords(cv, nx + 1, 1)
    Test.@test_throws BoundsError FG.Grids.corner_coords(cv, nx + 3, 1)
    Test.@test_throws BoundsError FG.Grids.measure(cv, 1, ny + 1)

    un = FG.Grids.UnstructuredGrid(geom, [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], trues(2))
    Test.@test_throws BoundsError FG.Grids.coords(un, 3)
    Test.@test_throws BoundsError FG.Grids.measure(un, 3)

    # In-range access is unaffected, and a caller that opts out still pays nothing.
    Test.@test FG.Grids.measure(g, 2, 2) ≈ 1.0
    Test.@test FG.Grids.coords(g, 2, 3) == (x = 1.0, y = 2.0)
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
    # …and it validates, rather than trusting the buffers.
    Test.@test_throws ArgumentError C.csr_connectivity([1], [2, 2])        # ptr[1] != 1
    Test.@test_throws ArgumentError C.csr_connectivity([1, 2], [1, 2])     # length mismatch

    # StencilNeighbors is the lazy per-cell sequence; iterating it must equal neighbors!.
    gm = FG.Grids.StructuredGrid(geo, 0.0:1.0:4.0, 0.0:1.0:3.0)
    it = FG.Grids.neighbors(gm, 2, 2)
    Test.@test it isa C.StencilNeighbors
    buf = Vector{Int}(undef, 8)
    n = C.neighbors!(buf, gm, 2, 2)
    Test.@test sort(collect(it)) == sort(buf[1:n])
    Test.@test length(collect(FG.Grids.neighbors(gm, 1, 1))) == C.nneighbors(gm, 1, 1)

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
    Test.@test S.ring_info(Float32, 4, 3).colatitude isa Float32
    Test.@test_throws ArgumentError S.ring_info(4, 0)
    Test.@test_throws ArgumentError S.ring_info(4, 16)
    Test.@test_throws ArgumentError S.ring_info(0, 1)
end

Test.@testset "Asking for an element type gives back that element type, knowably" begin
    SS = FG.SphericalSampling
    gl = SS.GaussLegendreSampling()
    hp = SS.HEALPixSampling(4)
    fb = SS.FibonacciSampling(200)
    cb = SS.CubedSphereSampling()
    yy = SS.YinYangSampling()
    ic = SS.IcosahedralSampling(2)
    rg = SS.OctahedralGaussianSampling(8)

    # Every entry point that builds values of a chosen element type, inferred with the OTHER
    # arguments left non-constant. Passing the type as a keyword instead of positionally makes
    # each of these come back abstract, which then propagates into the caller's own inference —
    # the reason the whole set takes it positionally.
    for W in (Float64, Float32)
        for (name, f, rest) in (
            ("_gauss_legendre_μ",   t_gl,    Tuple{Int}),
            ("spherical_axes",      t_axes,  Tuple{typeof(gl),Int}),
            ("spherical_quadrature", t_quad, Tuple{typeof(gl),Int}),
            ("latitude_weights",    t_wts,   Tuple{typeof(gl),Int}),
            ("latitude_weights/rg", t_wts1,  Tuple{typeof(rg)}),
            ("spherical_points/tp", t_pts,   Tuple{typeof(gl),Int}),
            ("spherical_points/hp", t_pts1,  Tuple{typeof(hp)}),
            ("spherical_points/fb", t_pts1,  Tuple{typeof(fb)}),
            ("spherical_points/rg", t_pts1,  Tuple{typeof(rg)}),
            ("spherical_points/ic", t_pts1,  Tuple{typeof(ic)}),
            ("spherical_points/cb", t_pts,   Tuple{typeof(cb),Int}),
            ("spherical_points/yy", t_pts2,  Tuple{typeof(yy),Int,Int}),
            ("ring_latitudes",      t_rlat,  Tuple{typeof(rg)}),
            ("cubed_sphere_points", t_cube,  Tuple{Int}),
            ("icosahedral_vertices", t_icov, Tuple{Int}),
            ("icosahedral_mesh",    t_icom,  Tuple{Int}),
            ("yin_yang_panels",     t_yy,    Tuple{Int,Int}),
            ("ring_info",           t_ring,  Tuple{Int,Int}),
            ("pix2ang",             t_p2a,   Tuple{Int,Int}),
            ("pix2vec",             t_p2v,   Tuple{Int,Int}),
            ("stencil_scratch",     t_scr,   Tuple{Int,Int}),
        )
            ok, ty = concrete_return(f, Tuple{Type{W},rest.parameters...})
            ok || println("    ", name, " (", W, ") -> ", ty)
            Test.@test ok
        end
    end

    # The two grid constructors are deliberately not in that list. A grid's type records whether
    # each direction is periodic and whether each axis is uniform, and both are DETECTED from the
    # axis values, so the type cannot be known before the axes exist. What must still hold is
    # that the width asked for is the width built — which is a different claim, checked below.
    for W in (Float64, Float32)
        sg = FG.Connectivity.structured_grid(W, gl, 12)
        Test.@test eltype(FG.Grids.axis(sg, 1)) === W
        Test.@test FG.Grids.grid_geometry(sg) isa FG.Geometry.AbstractGeometry{W}
        Test.@test FG.Grids.coords(sg, 1, 1).λ isa W

        # A layout takes its width from its geometry alone: it stores no coordinates to carry one.
        hg = FG.Grids.HEALPixGrid(FG.Geometry.SphericalGeometry(W(6.371e6)), 4)
        Test.@test FG.Grids.grid_geometry(hg) isa FG.Geometry.AbstractGeometry{W}
        Test.@test FG.Grids.coords(hg, 1).λ isa W
        Test.@test FG.Grids.measure(hg, 1) isa W
        Test.@test eltype(first(FG.Grids.materialize(hg))) === W
    end

    # An explicit geometry is carried to the requested width too, keeping its shape: asking for a
    # Float32 grid around a 3000 km sphere must not hand back a Float64 one.
    gsm = FG.Connectivity.structured_grid(Float32, gl, 12;
                                          geometry = FG.Geometry.SphericalGeometry(3.0e6))
    Test.@test FG.Grids.grid_geometry(gsm) === FG.Geometry.SphericalGeometry{Float32}(3.0f6)
    Test.@test eltype(FG.Grids.axis(gsm, 2)) === Float32

    # With no element type named, the GEOMETRY's is the one meant — defaulting to `Float64` here
    # would quietly rebuild a caller's Float32 geometry at double the width.
    g32 = FG.Connectivity.structured_grid(gl, 12;
                                          geometry = FG.Geometry.SphericalGeometry(6.371f6))
    Test.@test eltype(FG.Grids.axis(g32, 1)) === Float32
    Test.@test FG.Grids.grid_geometry(g32) isa FG.Geometry.AbstractGeometry{Float32}
    r32 = FG.Grids.RingGrid(FG.Geometry.SphericalGeometry(6.371f6),
                            FG.SphericalSampling.OctahedralGaussianSampling(8))
    Test.@test FG.Grids.grid_geometry(r32) isa FG.Geometry.AbstractGeometry{Float32}
    Test.@test FG.Grids.coords(r32, 1).λ isa Float32
    for W in (Float64, Float32)
        Test.@test FG.Geometry.float_type(FG.Geometry.SphericalGeometry{W}(6.371e6)) === W
        Test.@test FG.Geometry.float_type(FG.Geometry.CartesianGeometry{W}()) === W
    end

    # A geometry defined outside the package must still inherit the whole stack, so a width it is
    # already at asks nothing of it.
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

    # And the width asked for is the width returned.
    Test.@test eltype(SS.spherical_axes(Float32, gl, 12).φ) === Float32
    Test.@test eltype(SS.spherical_points(Float32, hp).λ) === Float32
    Test.@test eltype(SS.latitude_weights(Float32, gl, 12)) === Float32
    Test.@test eltype(SS.icosahedral_vertices(Float32, 2).λ) === Float32
    Test.@test SS.ring_info(Float32, 4, 3).colatitude isa Float32
    Test.@test SS.pix2ang(Float32, 4, 10) isa NTuple{2,Float32}
    Test.@test SS.pix2vec(Float32, 4, 10) isa NTuple{3,Float32}
    Test.@test SS.pix2vec(Float64, 4, 10) isa NTuple{3,Float64}

    # The default stays Float64 and stays knowable.
    Test.@test Test.@inferred(SS.pix2vec(4, 10)) isa NTuple{3,Float64}
    Test.@test Test.@inferred(SS.pix2ang(4, 10)) isa NTuple{2,Float64}
    Test.@test eltype(SS.spherical_points(hp).λ) === Float64

    # A unit vector really is the direction its angles name, at either width.
    for W in (Float64, Float32)
        θ, ϕ = SS.pix2ang(W, 8, 100)
        v = SS.pix2vec(W, 8, 100)
        Test.@test all(v .≈ (sin(θ) * cos(ϕ), sin(θ) * sin(ϕ), cos(θ)))
    end
end

Test.@testset "Every public name is allocation-checked or has a stated reason not to be" begin
    public_of(m) = Set(s for s in (Symbol(b.var) for b in keys(Base.Docs.meta(m)))
                       if !startswith(String(s), "_"))

    ALLOCATION_CHECKED = Set([
        :coords, :measure, :isactive, :displacement, :neighbors, :neighbors!, :nneighbors,
        :nneighbors_within, :neighbors_within!, :fold_within, :metric_window, :k_nearest!,
        :MetricTopology, :minimum_spacing, :maximum_spacing, :axis_stats, :AxisStats,
        :area, :coords!, :axis, :spacing, :origin, :extent, :bounds, :isperiodic, :isuniform,
        :period, :size_tuple, :mask, :coordinate_names, :periodic_flags, :topology, :coordinates,
        :apply_stencil!, :foreach_within, :mapreduce_within, :embedded_radius, :fold_candidates,
        :fold_candidates_at, :locate, :embed_point, :fold_at,
        :fd_weights!, :nearest_index, :interpolation_weights, :scale_factors, :jacobian,
        :local_spacing, :cell_width, :metric_floor, :metric_band, :gradient!,
        :stencil_scratch, :interpolate!, :healpix_neighbor_ids,
    ])

    # Geometry, Axes and Stencils are per-point kernels almost throughout, so each name below is
    # measured at zero allocation rather than declared to be.
    GEOMETRY_CHECKED = Set([
        :radius, :semimajor_axis, :semiminor_axis, :flattening, :eccentricity²,
        :meridional_radius, :prime_vertical_radius, :as_ntuple, :as_tensor6, :point_names,
        :named_point, :area_element, :volume_element, :spherical_to_cartesian,
        :cartesian_to_spherical, :geodetic_to_cartesian, :unit_vector, :local_tangent_basis,
        :project_to_tangent_plane, :nonuniform_first_derivative, :vector_to_cartesian,
        :vector_from_cartesian, :tensor_to_local, :tensor_from_local, :spherical_excess,
        :triangle_area, :triangle_area_from_unit_vectors, :rotate!, :unrotate,
        :similar_geometry, :float_type,
        :spacing_trait, :similar_axis, :uniform_axis, :wrap_sign,
        :offsets, :nstencil, :reach, :foreach_offset, :fold_offsets,
        :bandlimit, :nlat_for_bandlimit, :nlon_for_nlat, :npoints, :nrings, :axes_lengths,
        :colatitude, :geographic_latitude, :ang2pix, :pix2ang, :pix2vec, :vec2pix, :ring2nest,
        :ring_info, :admits_exact_bandlimited_quadrature, :nlon_in_ring, :ring_range,
    ])

    # Names whose whole job is to build something, or that name a thing rather than compute one.
    GEOMETRY_NOT_CHECKED = Set(Iterators.flatten((
        # geometry, axis, stencil and sampling TYPES
        [:AbstractGeometry, :AbstractCartesianGeometry, :AbstractSphericalGeometry,
         :AbstractEllipsoidalGeometry, :CartesianGeometry, :SphericalGeometry, :SpheroidGeometry,
         :AbstractUniformAxis, :UniformAxis, :ConstantVector, :UniformSpacing, :NonuniformSpacing,
         :AbstractStencil, :Axial, :VonNeumann, :Moore, :Diagonal, :Anisotropic, :Custom,
         :Vertex, :CellRadius, :MetricBall,
         :AbstractSphericalSampling, :AbstractTensorProductSphericalSampling,
         :AbstractLatLonSampling, :AbstractReducedGaussianSampling, :AbstractRingSampling,
         :AbstractSpectralQuadratureSampling, :ClenshawCurtisSampling, :CubedSphereSampling,
         :DriscollHealySampling, :DriscollHealyEqualSampling, :FibonacciSampling,
         :GaussLegendreSampling, :HEALPixSampling, :IcosahedralSampling, :LatLonSampling,
         :McEwenWiauxSampling, :OctahedralGaussianSampling, :ReducedGaussianSampling,
         :ScatteredSphericalSampling, :YinYangSampling, :RingScheme, :Ring, :Nested, :OpenNodes,
         :AbstractEquiangularAlgorithm, :Recurrence, :Transform],
        # allocating by contract: each returns a fresh array, which is the request
        [:nlon_per_ring, :ring_latitudes, :spherical_axes, :spherical_points,
         :spherical_quadrature, :latitude_weights, :icosahedral_vertices, :icosahedral_mesh,
         :cubed_sphere_points, :yin_yang_panels, :chunk_ranges],
        # the `!` forms of those, whose buffers are the caller's — covered by the `!`-form testset
        [:spherical_axes!, :spherical_points!, :spherical_quadrature!, :latitude_weights!,
         :icosahedral_vertices!, :cubed_sphere_points!, :yin_yang_panels!],
        # take a function and run it; cost is the body's, not theirs
        [:run_chunks, :run_indices, :map_chunks],
        # builds a NamedTuple from names given at runtime, so the caller chooses the cost
        [:build_point],
    )))

    # Adding a public name without putting it in one of the two sets fails this test. That is the
    # point: what is covered is derived from what the module documents, not from a list someone
    # remembered to update.
    NOT_CHECKED_BECAUSE = Set(Iterators.flatten((
        # types and traits
        [:AbstractGrid, :AbstractStructuredGrid, :AbstractCurvilinearGrid, :AbstractUnstructuredGrid,
         :StructuredGrid, :CurvilinearGrid, :UnstructuredGrid, :AbstractTopology, :Bounded, :Periodic,
         :AllActive, :SeparableMeasure, :PoleRotation, :AbstractImageConvention, :AbstractReach,
         :NearestImage, :AllImages, :Unrestricted, :Connected, :CSRConnectivity, :IndexTopology,
         :StencilNeighbors, :AbstractEmbedding, :CartesianEmbedding, :ChordEmbedding,
         :ArcEmbedding, :CellListIndex, :AbstractMaskPolicy, :BlankMasked, :ShiftWithinRun,
         :ReduceInRun, :AbstractLocation, :Center, :Face, :GradientPlan, :StencilScratch,
         :MeshNeighbors, :AbstractCellAddress, :CartesianCells, :FlatCells,
         :AbstractAdjacency, :IndexStencilNeighbors, :FormulaNeighbors, :StoredMeshNeighbors,
         :AbstractCandidateSource, :SeparableWindow, :IndexedCandidates],
        # bulk or one-off operations, not per-cell hot paths
        [:build_connectivity, :build_connectivity_within, :foreach_within, :mapreduce_within,
         :adjacency_matrix, :adjacency_matrix!, :sparse_adjacency_matrix, :sparse_adjacency_matrix!,
         :sparse_adjacency_coo!, :sparse_adjacency_csc!, :sort_neighbors!, :is_symmetric_adjacency,
         :connected_components, :count_holes, :interior, :boundary_cells, :csr_connectivity,
         :empty_csr, :healpix_neighbors!, :indexed, :ball_scratch, :structured_grid,
         :rotate, :measure_array, :measure_factors, :derivative!, :gradient_plan,
         :interpolate,
         :corners, :corner_coords, :neighbor_nbrs, :distance, :embedded_points, :cell_list,
         :incident_nodes, :cell_address, :adjacency_source, :candidate_source,
         :sampling, :rebuild, :ncoordinates, :materialize, :cells, :cell_at, :embedded_at,
         :embedding_of, :max_neighbors, :formula_neighbors, :nside, :scheme, :npixels,
         :HEALPixGrid, :RingGrid, :RingwiseVector, :FormulaNeighborSeq, :ring_of,
         :CubedSphereGrid, :YinYangGrid, :GridMeasure,
         :npanels, :panel_size, :panel_shape, :panel_cell, :cell_id],
    # allocating forms, whose whole job is to return a fresh array
    [:neighbors_within, :k_nearest, :fd_weights, :lagrange_weights, :axis_stencils,
     :centers, :faces, :nodes, :cell_widths],
    # grid constructors
    [:structured_grid, :unstructured_grid],
        # extension hooks, which throw until their trigger package is loaded
        [:spatial_index, :index_within!, :has_spatial_index],
        # returns the axis itself, so its type varies by direction: a runtime `d` cannot be stable
        [:axis],
    )))

    for m in (FG.Grids, FG.Connectivity, FG.Discretization, FG.Operators, FG.Geometry, FG.Axes,
              FG.Stencils, FG.SphericalSampling, FG.Execution)
        unclassified = setdiff(public_of(m), ALLOCATION_CHECKED, NOT_CHECKED_BECAUSE,
                               GEOMETRY_CHECKED, GEOMETRY_NOT_CHECKED)
        isempty(unclassified) ||
            println("  unclassified in ", nameof(m), ": ", join(sort!(collect(unclassified)), ", "))
        Test.@test isempty(unclassified)
    end
end
