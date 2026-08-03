Test.@testset "Abstract hierarchy" begin
    Test.@test FG.Geometry.AbstractCartesianGeometry <: FG.Geometry.AbstractGeometry
    Test.@test FG.Geometry.AbstractSphericalGeometry <: FG.Geometry.AbstractGeometry
    Test.@test FG.Geometry.CartesianGeometry <: FG.Geometry.AbstractCartesianGeometry
    Test.@test FG.Geometry.SphericalGeometry <: FG.Geometry.AbstractSphericalGeometry
    Test.@test FG.Grids.AbstractStructuredGrid <: FG.Grids.AbstractGrid
    Test.@test FG.Grids.AbstractCurvilinearGrid <: FG.Grids.AbstractGrid
    Test.@test FG.Grids.AbstractUnstructuredGrid <: FG.Grids.AbstractGrid
    Test.@test FG.Grids.StructuredGrid <: FG.Grids.AbstractStructuredGrid
    Test.@test FG.Grids.CurvilinearGrid <: FG.Grids.AbstractCurvilinearGrid
    Test.@test FG.Grids.UnstructuredGrid <: FG.Grids.AbstractUnstructuredGrid
end

Test.@testset "User geometry subtype participates" begin
    struct StretchedCartesian{T} <: FG.Geometry.AbstractCartesianGeometry{T} end
    g = StretchedCartesian{Float64}()
    Test.@test g isa FG.Geometry.AbstractGeometry
    Test.@test FG.Geometry.area_element(g, 2.0, 3.0) ≈ 6.0
    p1 = (0.0, 0.0)
    p2 = (3.0, 4.0)
    Test.@test FG.Geometry.distance(g, p1, p2) ≈ 5.0

    # A spherical geometry supplies `radius` and nothing else. The field is named unlike
    # `SphericalGeometry`'s so that any method reaching for `.R` fails here.
    struct ScaledSphere{T} <: FG.Geometry.AbstractSphericalGeometry{T}
        r0::T
    end
    FG.Geometry.radius(g::ScaledSphere) = g.r0

    s = ScaledSphere(6.371e6)
    ref = FG.Geometry.SphericalGeometry(6.371e6)
    Test.@test FG.Geometry.radius(s) == FG.Geometry.radius(ref)
    for (a, b) in ((( 0.0, 0.0), (0.0, π / 2)), ((0.3, -0.4), (2.9, 1.1)))
        Test.@test FG.Geometry.distance(s, a, b) == FG.Geometry.distance(ref, a, b)
    end
    Test.@test FG.Geometry.spherical_to_cartesian(s, (0.3, 0.4)) ==
               FG.Geometry.spherical_to_cartesian(ref, (0.3, 0.4))
    Test.@test FG.Geometry.area_element(s, 0.4, 0.01, 0.02) ==
               FG.Geometry.area_element(ref, 0.4, 0.01, 0.02)
    Test.@test FG.Geometry.scale_factors(s, (0.3, 0.4)) ==
               FG.Geometry.scale_factors(ref, (0.3, 0.4))
    # …and the whole grid stack downstream of it, including the cell measure.
    λ = range(0.0; step = 2π / 8, length = 8)
    φ = range(-1.2; step = 0.3, length = 9)
    gs, gr = FG.Grids.StructuredGrid(s, λ, φ), FG.Grids.StructuredGrid(ref, λ, φ)
    Test.@test FG.Grids.measure_array(gs) == FG.Grids.measure_array(gr)
    Test.@test sum(FG.Grids.measure(gs)) == sum(FG.Grids.measure(gr))
    Test.@test FG.Grids.area(gs, 3, 4) == FG.Grids.area(gr, 3, 4)
    Test.@test FG.Grids.coords(gs, 3, 4) == FG.Grids.coords(gr, 3, 4)

    # An ellipsoid supplies the two shape parameters; everything else is derived.
    struct MySpheroid{T} <: FG.Geometry.AbstractEllipsoidalGeometry{T}
        eq::T
        flat::T
    end
    FG.Geometry.semimajor_axis(g::MySpheroid) = g.eq
    FG.Geometry.flattening(g::MySpheroid) = g.flat

    e = MySpheroid(6378137.0, inv(298.257223563))
    eref = FG.Geometry.SpheroidGeometry()
    for f in (FG.Geometry.semimajor_axis, FG.Geometry.flattening, FG.Geometry.semiminor_axis,
              FG.Geometry.eccentricity²)
        Test.@test f(e) == f(eref)
    end
    Test.@test FG.Geometry.prime_vertical_radius(e, 0.7) ==
               FG.Geometry.prime_vertical_radius(eref, 0.7)
    Test.@test FG.Geometry.meridional_radius(e, 0.7) ==
               FG.Geometry.meridional_radius(eref, 0.7)
    Test.@test FG.Geometry.distance(e, (0.0, 0.0), (0.5, 0.6)) ==
               FG.Geometry.distance(eref, (0.0, 0.0), (0.5, 0.6))
    Test.@test FG.Geometry.area_element(e, 0.4, 0.01, 0.02) ==
               FG.Geometry.area_element(eref, 0.4, 0.01, 0.02)
    Test.@test FG.Geometry.scale_factors(e, (0.3, 0.4)) ==
               FG.Geometry.scale_factors(eref, (0.3, 0.4))
end

Test.@testset "Cartesian geometry" begin
    geom = FG.Geometry.CartesianGeometry()
    p1 = (0.0, 0.0)
    p2 = (3000.0, 4000.0)
    Test.@test FG.Geometry.distance(geom, p1, p2) ≈ 5000.0
    # The geometry carries no spacing: a cell's extents are the caller's (in practice, the grid's).
    Test.@test FG.Geometry.area_element(geom, 1000.0, 1000.0) ≈ 1.0e6
    Test.@test FG.Geometry.volume_element(geom, 1.0, 2.0, 3.0) ≈ 6.0
    Test.@test FG.Geometry.CartesianGeometry(Float32) === FG.Geometry.CartesianGeometry{Float32}()
    Test.@test FG.Geometry.CartesianGeometry{Float32}() isa FG.Geometry.AbstractCartesianGeometry{Float32}
end

Test.@testset "Spherical geometry + Cartesian" begin
    geom = FG.Geometry.SphericalGeometry(6.371e6)
    london = (deg2rad(-0.1276), deg2rad(51.5074))
    paris = (deg2rad(2.3522), deg2rad(48.8566))
    d_km = FG.Geometry.distance(geom, london, paris) / 1000.0
    Test.@test 300 < d_km < 400

    λ, φ = 0.3, 0.4
    uλ, uφ = 1.0, -0.5
    p = FG.Geometry.vector_to_cartesian(geom, uλ, uφ, λ, φ)
    back = FG.Geometry.vector_from_cartesian(geom, p, λ, φ)
    Test.@test back.λ ≈ uλ
    Test.@test back.φ ≈ uφ
    Test.@test abs(back.r) < 1e-12
end

Test.@testset "nonuniform_first_derivative" begin
    # Uniform: recovers (f_p - f_m)/(2h)
    Test.@test FG.Geometry.nonuniform_first_derivative(1.0, 2.0, 3.0, 1.0, 1.0) ≈ 1.0
    # Exact for quadratic on nonuniform stencil
    # f(x)=x^2 at x=0 with neighbors -1 and 2: f'=2x=0 at center... use x=-1,0,2
    f_m, f_0, f_p = 1.0, 0.0, 4.0  # (-1)^2, 0, 2^2
    Test.@test FG.Geometry.nonuniform_first_derivative(f_m, f_0, f_p, 1.0, 2.0) ≈ 0.0 atol = 1e-12
end

Test.@testset "Rank-2 tensors rotate between the ambient and local frames" begin
    GE = FG.Geometry
    geo = GE.SphericalGeometry(6.371e6)
    τ = (1.3, -0.7, 2.1, 0.4, -1.1, 0.9)              # xx yy zz xy xz yz
    pts = ((0.0, 0.0), (0.7, -0.4), (3.1, 1.2), (5.0, -1.5))
    sym6(t) = [t[1] t[4] t[5]; t[4] t[2] t[6]; t[5] t[6] t[3]]
    # `R` formed explicitly from the basis the package already defines, so the contraction under
    # test — which never forms `R` — is checked against the thing it is supposed to be.
    function rmat(λ, φ)
        b = GE.local_tangent_basis(geo, (λ, φ))
        return vcat(collect(b.λ)', collect(b.φ)', collect(GE.unit_vector(Float64, (λ, φ)))')
    end

    for (λ, φ) in pts
        R = rmat(λ, φ)
        got = GE.tensor_to_local(geo, τ..., λ, φ)
        Test.@test maximum(abs.(sym6(Tuple(got)) .- R * sym6(τ) * R')) < 1e-12
        back = GE.tensor_from_local(geo, got, λ, φ)
        Test.@test maximum(abs.(collect(Tuple(back)) .- collect(τ))) < 1e-12
        # Consistent with the rank-1 rotation: u⊗u must rotate to (Ru)⊗(Ru).
        u = (0.3, -1.2, 0.8)
        t = (u[1]^2, u[2]^2, u[3]^2, u[1] * u[2], u[1] * u[3], u[2] * u[3])
        v = GE.vector_from_cartesian(geo, u..., λ, φ)
        w = (v.λ^2, v.φ^2, v.r^2, v.λ * v.φ, v.λ * v.r, v.φ * v.r)
        Test.@test maximum(abs.(collect(Tuple(GE.tensor_to_local(geo, t..., λ, φ))) .-
                                collect(w))) < 1e-12
    end

    # Invariants a rotation cannot change. The determinant is written out rather than pulling in
    # LinearAlgebra for one 3×3.
    det3(m) = m[1, 1] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) -
              m[1, 2] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) +
              m[1, 3] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1])
    let l = Tuple(GE.tensor_to_local(geo, τ..., 1.1, 0.3))
        Test.@test l[1] + l[2] + l[3] ≈ τ[1] + τ[2] + τ[3]
        Test.@test det3(sym6(l)) ≈ det3(sym6(τ))
    end

    # Every representation, as elsewhere, and a requested one on the way out.
    let nt = (xx = τ[1], yy = τ[2], zz = τ[3], xy = τ[4], xz = τ[5], yz = τ[6])
        a = Tuple(GE.tensor_to_local(geo, τ, 0.7, -0.4))
        Test.@test a == Tuple(GE.tensor_to_local(geo, collect(τ), 0.7, -0.4))
        Test.@test a == Tuple(GE.tensor_to_local(geo, nt, 0.7, -0.4))
        s = GE.tensor_to_local(StaticArrays.SVector{6,Float64}, geo, τ..., 0.7, -0.4)
        Test.@test s isa StaticArrays.SVector{6,Float64} && all(s .≈ collect(a))
    end
    Test.@test_throws ArgumentError GE.as_tensor6([1.0, 2.0, 3.0])

    # Meant for a hot loop, so it must not allocate — the reason a caller would hand-roll it.
    # Through `_alloc`, not a local closure: a closure over a testset local boxes what it captures
    # and would measure the harness rather than the function.
    Test.@test _alloc(q_tensor_local, geo, τ, (0.7, -0.4)) == 0

    # Float32 in, Float32 out.
    let g32 = GE.SphericalGeometry(Float32(6.371e6))
        t32 = GE.tensor_to_local(g32, 1.3f0, -0.7f0, 2.1f0, 0.4f0, -1.1f0, 0.9f0, 0.7f0, -0.4f0)
        Test.@test all(x -> x isa Float32, Tuple(t32))
    end
end

Test.@testset "local_tangent_basis / project_to_tangent_plane" begin
    cgeom = FG.Geometry.CartesianGeometry()
    c = (0.0, 0.0)
    n = (1.0, 2.0)
    Test.@test FG.Geometry.project_to_tangent_plane(cgeom, c, n) == (; x = 1.0, y = 2.0)
    Test.@test FG.Geometry.distance(cgeom, (x = 0.0, y = 0.0), (x = 3.0, y = 4.0)) ≈ 5.0

    sgeom = FG.Geometry.SphericalGeometry(1.0)
    ê = FG.Geometry.local_tangent_basis(sgeom, (0.0, 0.0))
    Test.@test abs(sqrt(sum(abs2, ê.λ)) - 1) < 1e-14
    Test.@test abs(sqrt(sum(abs2, ê.φ)) - 1) < 1e-14
    Test.@test FG.Geometry.distance(sgeom, (λ = 0.0, φ = 0.0), (λ = 0.1, φ = 0.0)) ≈ sgeom.R * 0.1 atol = 1e-10
end

Test.@testset "Point functions accept every representation and element type" begin
    using StaticArrays: StaticArrays as SA
    cgeo = FG.Geometry.CartesianGeometry()     # Float64 geometry …
    sgeo = FG.Geometry.SphericalGeometry(6.371e6)

    R = 6.371e6

    # … fed points whose element type is NOT the geometry's: they are converted on entry, and
    # every representation reaches the same kernel rather than recursing through normalization.
    Test.@test FG.Geometry.distance(cgeo, (0, 0), (3, 4)) ≈ 5.0
    Test.@test FG.Geometry.distance(cgeo, (0.0f0, 0.0f0), (3.0f0, 4.0f0)) ≈ 5.0
    Test.@test FG.Geometry.distance(cgeo, (x = 0.0, y = 0.0), SA.SVector(3.0, 4.0)) ≈ 5.0
    Test.@test FG.Geometry.distance(cgeo, [0.0, 0.0], (3.0, 4.0)) ≈ 5.0
    Test.@test FG.Geometry.distance(sgeo, (0.1f0, 0.2f0), (0.3f0, 0.4f0)) ≈
               FG.Geometry.distance(sgeo, (Float64(0.1f0), Float64(0.2f0)), (Float64(0.3f0), Float64(0.4f0)))

    # 3-component spherical points carry an absolute radius; on the reference sphere the chord
    # distance agrees with the 2-component great-circle distance to within chord-vs-arc.
    Test.@test FG.Geometry.distance(sgeo, (0.1, 0.2, R), (0.1, 0.2, R)) ≈ 0.0 atol = 1e-9
    Test.@test FG.Geometry.distance(sgeo, (0.0, 0.0, R), (0.0, 0.0, 2R)) ≈ R

    # An unsupported point length is an error, not unbounded recursion.
    Test.@test_throws MethodError FG.Geometry.distance(sgeo, (0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4))
    Test.@test_throws ArgumentError FG.Geometry.distance(cgeo, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])

    # `spherical_to_cartesian` — (λ, φ) = (0, 0) is the +x axis at radius R.
    P0 = FG.Geometry.spherical_to_cartesian(sgeo, (0.0, 0.0))
    Test.@test all(((P0.x, P0.y, P0.z) .- (R, 0.0, 0.0)) .< 1e-6)
    Pref = FG.Geometry.spherical_to_cartesian(sgeo, (0.1, 0.2))
    for pt in ((0.1, 0.2), (λ = 0.1, φ = 0.2), SA.SVector(0.1, 0.2), [0.1, 0.2])
        Test.@test FG.Geometry.spherical_to_cartesian(sgeo, pt) === Pref
    end

    # The spherical tangent-plane projection is built on it, and was unreachable before.
    Δ0 = FG.Geometry.project_to_tangent_plane(sgeo, (λ = 0.1, φ = 0.2), (λ = 0.1, φ = 0.2))
    Test.@test Δ0.λ ≈ 0.0 atol = 1e-9
    Test.@test Δ0.φ ≈ 0.0 atol = 1e-9
    # A small eastward step lies along ê_λ only, with arc length R·cos(φ)·Δλ.
    λ0, φ0, Δλ = 0.7, 0.2, 1e-6
    Δe = FG.Geometry.project_to_tangent_plane(sgeo, (λ0, φ0), (λ0 + Δλ, φ0))
    Test.@test Δe.λ ≈ R * cos(φ0) * Δλ rtol = 1e-6
    Test.@test abs(Δe.φ) < 1e-3 * abs(Δe.λ)
end

Test.@testset "Requested point representation is honored exactly" begin
    using StaticArrays: StaticArrays as SA
    sgeo = FG.Geometry.SphericalGeometry(6.371e6)
    cgeo = FG.Geometry.CartesianGeometry()
    xs = collect(0.0:1.0:4.0)
    cgrid = FG.Grids.StructuredGrid(cgeo, xs, xs, trues(5, 5))
    sgrid = FG.Grids.StructuredGrid(sgeo, deg2rad.(xs), deg2rad.(xs), trues(5, 5))

    Test.@test Test.@inferred(FG.Grids.coords(cgrid, 2, 3)) === (x = 1.0, y = 2.0)
    Test.@test Test.@inferred(FG.Grids.coords(NamedTuple, sgrid, 2, 3)) === FG.Grids.coords(sgrid, 2, 3)
    Test.@test Test.@inferred(FG.Grids.coords(Tuple, cgrid, 2, 3)) === (1.0, 2.0)
    Test.@test Test.@inferred(FG.Grids.coords(NTuple{2,Float32}, cgrid, 2, 3)) === (1.0f0, 2.0f0)
    Test.@test Test.@inferred(FG.Grids.coords(SA.SVector{2,Float64}, cgrid, 2, 3)) === SA.SVector(1.0, 2.0)
    Test.@test Test.@inferred(FG.Grids.coords(SA.SVector, cgrid, 2, 3)) === SA.SVector(1.0, 2.0)
    Test.@test Test.@inferred(FG.Grids.coords(SA.MVector{2,Float64}, cgrid, 2, 3)) == SA.MVector(1.0, 2.0)
    Test.@test Test.@inferred(FG.Grids.coords(Vector{Float64}, cgrid, 2, 3)) == [1.0, 2.0]

    # The vector-returning geometry functions take the same leading-type escape.
    p, q = SA.SVector(0.1, 0.2), SA.SVector(0.11, 0.21)
    Test.@test Test.@inferred(FG.Geometry.project_to_tangent_plane(SA.SVector{2,Float64}, sgeo, p, q)) ==
               SA.SVector(Tuple(FG.Geometry.project_to_tangent_plane(sgeo, p, q)))
    Test.@test Test.@inferred(FG.Geometry.spherical_to_cartesian(SA.SVector{3,Float64}, sgeo, p)) ==
               SA.SVector(Tuple(FG.Geometry.spherical_to_cartesian(sgeo, p)))
    Test.@test Test.@inferred(FG.Geometry.vector_to_cartesian(SA.SVector{3,Float64}, sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)) ==
               SA.SVector(Tuple(FG.Geometry.vector_to_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)))
    Test.@test Test.@inferred(FG.Geometry.vector_from_cartesian(SA.MVector{3,Float64}, sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)) ==
               SA.MVector(Tuple(FG.Geometry.vector_from_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)))
    Test.@test Test.@inferred(FG.Geometry.local_tangent_basis(SA.SVector{3,Float64}, sgeo, p)).λ isa SA.SVector{3,Float64}
    # A velocity may itself be given in any representation.
    Test.@test FG.Geometry.vector_from_cartesian(sgeo, SA.SVector(1.0, 2.0, 3.0), 0.1, 0.2) ===
               FG.Geometry.vector_from_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)
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
end

Test.@testset "A pole rotation applies to a point set and to a grid" begin
    GE = FG.Geometry
    GR = FG.Grids
    rot = GE.PoleRotation(0.7, 0.3)
    Test.@test GE.rotate(rot, 0.7, 0.3)[2] ≈ π / 2      # the rotation's own pole

    λ = [0.1, 1.2, 3.0, 5.5]
    φ = [0.0, -0.4, 0.9, 0.2]
    Λ, Φ = GE.rotate(rot, λ, φ)
    Test.@test all((Λ[i], Φ[i]) == GE.rotate(rot, λ[i], φ[i]) for i in eachindex(λ))
    λ2, φ2 = copy(λ), copy(φ)
    GE.rotate!(λ2, φ2, rot)
    Test.@test λ2 == Λ && φ2 == Φ
    GE.unrotate!(λ2, φ2, rot)
    Test.@test all(isapprox.(λ2, λ; atol = 1e-12)) && all(isapprox.(φ2, φ; atol = 1e-12))
    rr() = FG.Geometry.rotate!(λ2, φ2, rot)
    rr()
    Test.@test @allocated(rr()) == 0
    # 2-D fields (a grid's coordinate arrays) go through the same method.
    Λm = [0.1i for i in 1:3, _ in 1:4]
    Test.@test size(GE.rotate!(Λm, [0.1j for _ in 1:3, j in 1:4], rot)[1]) == (3, 4)
    Test.@test_throws DimensionMismatch GE.rotate!(zeros(3), zeros(4), rot)

    # Rotating a rectilinear spherical grid gives a curvilinear one — which is what it is — and
    # the cell measure carries over EXACTLY, because a rotation is an isometry of the sphere.
    sph = FG.Geometry.SphericalGeometry(6.371e6)
    λa = range(0, 2π; length = 25)[1:24]
    φa = range(-1.2, 1.2; length = 13)
    gs = GR.StructuredGrid(sph, λa, φa)
    gr = GR.unrotate(gs, rot)
    Test.@test gr isa GR.CurvilinearGrid && size(gr) == (24, 13)
    Test.@test all(GR.measure(gr, i, j) == GR.measure(gs, i, j) for i in 1:24, j in 1:13)
    Test.@test sum(GR.measure(gr)) ≈ sum(GR.measure(gs))
    Test.@test GR._raw_coords(gr, 3, 4) == GE.unrotate(rot, λa[3], φa[4])
    Test.@test GR._raw_coords(GR.rotate(gs, rot), 3, 4) == GE.rotate(rot, λa[3], φa[4])
    # The mesh is unchanged, so its index topology and wrap length are too (λ is an angle in
    # either frame).
    Test.@test GR.isperiodic(gr, 1) && !GR.isperiodic(gr, 2)
    Test.@test GR.period(gr, 1) ≈ 2π
    Test.@test size(GR.corners(gr, 1)) == (25, 14)
    Test.@test GR.coordinate_names(gr) == (:λ, :φ)
    Test.@test FG.Connectivity.nneighbors(gr, 1, 5) == 4     # wraps, like the original
end

Test.@testset "Metric scale factors and the Jacobian" begin
    G = FG.Geometry
    sg = G.SphericalGeometry(2.0)
    Test.@test G.scale_factors(sg, (0.3, 0.0)) == (2.0, 2.0)
    Test.@test all(isapprox.(G.scale_factors(sg, (0.0, π / 3)), (1.0, 2.0); atol = 1e-15))
    Test.@test all(isapprox.(G.scale_factors(sg, (0.0, π / 3, 5.0)), (2.5, 5.0, 1.0); atol = 1e-15))
    Test.@test G.scale_factors(FG.Geometry.CartesianGeometry(), (1.0, 2.0)) == (1.0, 1.0)
    Test.@test G.scale_factors(FG.Geometry.CartesianGeometry(), (1.0, 2.0, 3.0)) == (1.0, 1.0, 1.0)
    Test.@test G.jacobian(sg, (0.0, 0.5)) ≈ 4 * cos(0.5)
    Test.@test G.jacobian(FG.Geometry.CartesianGeometry(), (1.0, 2.0)) == 1.0
    # The Jacobian is the area element per unit coordinate area, which is what the grid's own
    # measure is built from.
    gg = FG.Grids.StructuredGrid(sg, collect(range(0, 2π; length = 40)),
                           collect(range(-1.2, 1.2; length = 30)))
    λ, φ = FG.Grids.coordinate_names(gg)
    i, j = 7, 11
    p = FG.Grids.coords(gg, i, j)
    Δλ = FG.Discretization.cell_width(FG.Grids.coordinates(gg, 1), i, FG.Grids.period(gg, 1))
    Δφ = FG.Discretization.cell_width(FG.Grids.coordinates(gg, 2), j)
    Test.@test FG.Grids.measure(gg, i, j) ≈ G.jacobian(sg, (p.λ, p.φ)) * Δλ * Δφ rtol = 1e-12
    # Any point representation is accepted.
    Test.@test G.scale_factors(sg, (λ = 0.0, φ = π / 3)) == G.scale_factors(sg, (0.0, π / 3))
    Test.@test G.scale_factors(sg, [0.0, π / 3]) == G.scale_factors(sg, (0.0, π / 3))
end

Test.@testset "Oblate spheroid geometry" begin
    G = FG.Geometry
    g = G.SpheroidGeometry()                       # WGS 84
    Test.@test g.a == 6378137.0
    Test.@test G.semiminor_axis(g) ≈ 6356752.314245 atol = 1e-6
    Test.@test G.eccentricity²(g) ≈ 0.00669437999014 atol = 1e-12
    # The quarter meridian is a published WGS 84 constant, and it exercises the whole geodesic.
    Test.@test G.distance(g, (0.0, 0.0), (0.0, π / 2)) ≈ 10001965.729 atol = 1e-2
    # The equator is a circle of radius a, so a quarter of it is exactly π·a/2.
    Test.@test G.distance(g, (0.0, 0.0), (π / 2, 0.0)) ≈ π * g.a / 2 atol = 1e-3
    # A meridian geodesic IS the meridian arc, so it can be checked against ∫M(φ)dφ directly.
    q = FG.SphericalSampling.spherical_quadrature(FG.SphericalSampling.GaussLegendreSampling(), 400)
    function meridian_arc(geo, φa, φb)
        mid = (φa + φb) / 2
        half = (φb - φa) / 2
        return half * sum(q.w[j] * G.meridional_radius(geo, mid + half * sin(q.φ[j]))
                          for j in eachindex(q.φ))
    end
    for (φa, φb) in ((0.0, 0.3), (-0.9, 0.2), (0.5, 1.4), (-1.5, 1.5))
        Test.@test G.distance(g, (0.4, φa), (0.4, φb)) ≈ meridian_arc(g, φa, φb) atol = 1e-3
    end
    # Symmetry, the zero case, and a finite answer for a near-antipodal pair.
    p1, p2 = (0.3, -0.4), (1.9, 0.55)
    Test.@test G.distance(g, p1, p2) ≈ G.distance(g, p2, p1)
    Test.@test G.distance(g, p1, p1) == 0.0
    d = G.distance(g, (0.0, 0.0), (π - 1e-9, 1e-9))
    Test.@test isfinite(d) && 1.9e7 < d < 2.1e7
    # Zero flattening must reduce exactly to the sphere.
    Test.@test G.distance(G.SpheroidGeometry(6371000.0, 0.0), (0.1, 0.2), (0.3, 0.4)) ≈
               G.distance(G.SphericalGeometry(6371000.0), (0.1, 0.2), (0.3, 0.4)) rtol = 1e-9
    # The area element integrates to the closed-form spheroid surface area.
    e = sqrt(G.eccentricity²(g))
    exact = 2π * g.a^2 * (1 + (1 - e^2) / e * atanh(e))
    total = 2π * sum(q.w[j] * G.meridional_radius(g, q.φ[j]) * G.prime_vertical_radius(g, q.φ[j])
                     for j in eachindex(q.φ))
    Test.@test total ≈ exact rtol = 1e-9
    # Curvature radii bracket a, and the scale factors are built from them.
    Test.@test G.prime_vertical_radius(g, 0.0) ≈ g.a
    Test.@test G.meridional_radius(g, 0.0) < g.a < G.meridional_radius(g, π / 2)
    Test.@test G.scale_factors(g, (0.0, 0.0)) == (g.a, G.meridional_radius(g, 0.0))
    Test.@test G.area_element(g, 0.0, 1.0, 1.0) ≈
               G.meridional_radius(g, 0.0) * G.prime_vertical_radius(g, 0.0)
    # Coordinate names follow the geodetic convention.
    Test.@test G.point_names(g, Val(2)) == (:λ, :φ)
    Test.@test G.point_names(g, Val(3)) == (:λ, :φ, :h)
    Test.@test G.point_names(g, Val(5)) == (:λ, :φ, :h, :q4, :q5)
    Test.@test_throws ArgumentError G.SpheroidGeometry(-1.0, 0.1)
    Test.@test_throws ArgumentError G.SpheroidGeometry(1.0, 1.5)

    # The geodetic volume element offsets BOTH curvature radii by h, so it does not factor.
    Test.@test G.volume_element(g, 0.4, 0.0, 1.0, 1.0, 1.0) ≈
               G.area_element(g, 0.4, 1.0, 1.0)
    Test.@test G.volume_element(g, 0.4, 100.0, 1e-3, 1e-3, 2.0) ≈
               (G.prime_vertical_radius(g, 0.4) + 100.0) * cos(0.4) *
               (G.meridional_radius(g, 0.4) + 100.0) * 1e-3 * 1e-3 * 2.0
end

Test.@testset "A spheroid drives the grid stack in every dimension" begin
    GE = FG.Geometry
    geo = GE.SpheroidGeometry()
    a, e² = GE.semimajor_axis(geo), GE.eccentricity²(geo)
    λ8 = collect(range(0, 2π; length = 9)[1:8])
    φ7 = collect(range(-π / 2, π / 2; length = 7))
    h4 = collect(range(0.0, 3000.0; length = 4))
    g1 = FG.Grids.StructuredGrid(geo, λ8)
    g2 = FG.Grids.StructuredGrid(geo, λ8, φ7)
    g3 = FG.Grids.StructuredGrid(geo, λ8, φ7, h4)
    g4 = FG.Grids.StructuredGrid(geo, λ8, φ7, h4, [0.0, 2.0])
    Test.@test size.((g1, g2, g3, g4)) == ((8,), (8, 7), (8, 7, 4), (8, 7, 4, 2))
    # Longitude closes the circle, so it is periodic without being asked; latitude is not.
    Test.@test FG.Grids.isperiodic(g2, 1) && !FG.Grids.isperiodic(g2, 2)
    Test.@test FG.Grids.coordinate_names(g3) == (:λ, :φ, :h)
    Test.@test FG.Grids.coords(g3, 2, 3, 2) == (λ = λ8[2], φ = φ7[3], h = h4[2])

    # 1-D: with no latitude the parallel is the equator, so the circumference is exactly 2πa.
    Test.@test sum(FG.Grids.measure(g1)) ≈ 2π * a rtol = 1e-14

    # 2-D factors; 3-D cannot, so it is stored dense.
    Test.@test FG.Grids.measure_factors(g2) !== nothing
    Test.@test FG.Grids.measure_factors(g3) === nothing

    # Every cell equals the geometry's own element, exactly — not to a tolerance.
    wλ = FG.Discretization.cell_widths(FG.Grids.coordinates(g2, 1), 2π)
    wφ = FG.Discretization.cell_widths(FG.Grids.coordinates(g2, 2), nothing)
    wh = FG.Discretization.cell_widths(FG.Grids.coordinates(g3, 3), nothing)
    Test.@test all(
        FG.Grids.measure(g2, i, j) ≈ GE.area_element(geo, φ7[j], wλ[i], wφ[j])
        for j in eachindex(φ7), i in eachindex(λ8)
    )
    Test.@test all(
        FG.Grids.measure(g3, i, j, k) ≈
        GE.volume_element(geo, φ7[j], h4[k], wλ[i], wφ[j], wh[k])
        for k in eachindex(h4), j in eachindex(φ7), i in eachindex(λ8)
    )
    # A further direction enters as a plain width.
    Test.@test sum(FG.Grids.measure(g4)) ≈
               sum(FG.Discretization.cell_widths([0.0, 2.0], nothing)) * sum(FG.Grids.measure(g3)) rtol = 1e-13

    # Independent check: the closed-form ellipsoid area, approached at second order.
    e = sqrt(e²)
    A_exact = 2π * a^2 * (1 + (1 - e²) / e * atanh(e))
    errs = map((24, 96, 384)) do n
        gn = FG.Grids.StructuredGrid(geo, collect(range(0, 2π; length = n + 1)[1:n]),
                                     collect(range(-π / 2, π / 2; length = n ÷ 2 + 1)))
        abs(sum(FG.Grids.measure(gn)) - A_exact) / A_exact
    end
    Test.@test errs[3] < 3e-5
    Test.@test errs[1] / errs[2] > 8 && errs[2] / errs[3] > 8   # 4× refinement ⇒ ~16×
    # A sphere is the f = 0 spheroid, and must agree with SphericalGeometry cell for cell.
    gsph = FG.Grids.StructuredGrid(GE.SphericalGeometry(a), λ8, φ7)
    gflat = FG.Grids.StructuredGrid(GE.SpheroidGeometry(a, 0.0), λ8, φ7)
    Test.@test all(FG.Grids.measure(gflat, i, j) ≈ FG.Grids.measure(gsph, i, j)
                   for j in eachindex(φ7), i in eachindex(λ8))

    Test.@test FG.Connectivity.nneighbors(g2, 1, 3) == 4       # wraps in λ
    Test.@test FG.Connectivity.nedges(FG.Connectivity.build_connectivity(g2)) > 0

    # A degenerate angular direction drops the differential that no longer exists, so a transect
    # measures arc length rather than an area with a placeholder in it.
    # Zonal at the equator: the parallel radius is N(0)·cos0 = a, so the circle closes at 2πa.
    gzon = FG.Grids.StructuredGrid(geo, λ8, [0.0])
    Test.@test sum(FG.Grids.measure(gzon)) ≈ 2π * a rtol = 1e-14
    # Zonal at latitude φ: radius N(φ)·cosφ.
    gzon2 = FG.Grids.StructuredGrid(geo, λ8, [0.7])
    Test.@test sum(FG.Grids.measure(gzon2)) ≈
               2π * GE.prime_vertical_radius(geo, 0.7) * cos(0.7) rtol = 1e-14
    # Meridional: each cell is exactly M(φ)·Δφ, the meridian arc element.
    φm = collect(range(-π / 2, π / 2; length = 33))
    gmer = FG.Grids.StructuredGrid(geo, [0.0], φm)
    wm = FG.Discretization.cell_widths(φm, nothing)
    Test.@test all(FG.Grids.measure(gmer, 1, j) ≈ GE.meridional_radius(geo, φm[j]) * wm[j]
                   for j in eachindex(φm))
    # The total approaches ∫M dφ — twice the published quarter meridian — at FIRST order, because
    # `n` cell-centres own `n` widths and so span π(1 + 1/n). Unlike the area case there is no cosφ
    # factor to annihilate that end-cell excess, and the error is exactly the 1/n it predicts.
    mer = map((48, 192, 768)) do n
        gm = FG.Grids.StructuredGrid(geo, [0.0], collect(range(-π / 2, π / 2; length = n + 1)))
        abs(sum(FG.Grids.measure(gm)) - 2 * 10001965.729) / (2 * 10001965.729)
    end
    Test.@test all(isapprox(mer[i], 1 / (48 * 4^(i - 1)); rtol = 0.02) for i in 1:3)
    # A single point has no extent in either direction.
    Test.@test sum(FG.Grids.measure(FG.Grids.StructuredGrid(geo, [0.3], [0.4]))) == 1.0

    # The dense measure is a second storage form, so the accessors around it get their own checks.
    Test.@test size(FG.Grids.measure_array(g3)) == size(g3)
    Test.@test sum(FG.Grids.measure(g3)) == sum(FG.Grids.measure_array(g3))
    Test.@test_throws BoundsError FG.Grids.measure(g3, 99, 1, 1)
    Test.@test_throws BoundsError FG.Grids.coords(g3, 1, 99, 1)
    Test.@test occursin("h ", sprint(show, MIME"text/plain"(), g3))
    mk = trues(size(g3)...)
    mk[1, 1, 1] = false
    gm = FG.Grids.StructuredGrid(geo, λ8, φ7, h4, mk)
    Test.@test !FG.Grids.isactive(gm, 1, 1, 1) && FG.Grids.isactive(gm, 2, 2, 2)
    Test.@test FG.Connectivity.nneighbors(gm, 2, 2, 2) == 6
end

Test.@testset "Pole rotation" begin
    G = FG.Geometry
    for (λp, φp) in ((0.0, π / 2), (0.7, 0.3), (2.0, -0.9), (4.5, 0.0))
        rot = G.PoleRotation(λp, φp)
        # The rotation's own pole becomes the new north pole.
        Test.@test G.rotate(rot, λp, φp)[2] ≈ π / 2 atol = 1e-12
        # `unrotate` inverts `rotate` everywhere.
        for λ in range(0, 2π; length = 13), φ in range(-1.4, 1.4; length = 9)
            a, b = G.rotate(rot, λ, φ)
            c, d = G.unrotate(rot, a, b)
            Test.@test abs(mod(c - λ + π, 2π) - π) < 1e-12
            Test.@test abs(d - φ) < 1e-12
            Test.@test -π/2 - 1e-12 ≤ b ≤ π/2 + 1e-12
            Test.@test 0 ≤ a < 2π + 1e-12
        end
    end
    # A rotation is an isometry: it preserves angular distance.
    rot = G.PoleRotation(0.7, 0.3)
    u = G.SphericalGeometry(1.0)
    for (p, q) in (((0.2, 0.1), (1.1, -0.4)), ((3.0, 0.9), (5.5, -1.2)))
        Test.@test G.distance(u, p, q) ≈
                   G.distance(u, G.rotate(rot, p...), G.rotate(rot, q...)) atol = 1e-12
    end
    # The identity rotation leaves the frame alone.
    idr = G.PoleRotation(0.0, π / 2)
    Test.@test all(isapprox.(G.rotate(idr, 1.2, 0.4), (1.2, 0.4); atol = 1e-12))
end
