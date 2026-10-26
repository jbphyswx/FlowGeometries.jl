Test.@testset "StructuredGrid preserves uniform spacing" begin
    geom = FG.Geometry.CartesianGeometry()
    x = 0.0:2000.0:20_000.0
    y = 0.0:2000.0:10_000.0
    mask = trues(length(x), length(y))
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)
    Test.@test FG.Grids.isuniform(grid, 1) && FG.Grids.isuniform(grid, 2)
    Test.@test FG.Grids.spacing(grid, 1) === 2000.0
    Test.@test FG.Grids.spacing(grid, 2) === 2000.0
    Test.@test collect(grid.x) == collect(x)
    Test.@test collect(grid.y) == collect(y)
    Test.@test FG.Grids.bounds(grid, 1) == (0.0, 20_000.0)
    Test.@test FG.Grids.extent(grid, 1) == 20_000.0
    Test.@test FG.Grids.origin(grid, 1) == 0.0
    # A uniform axis has one gap, so both ends of its range of gaps are that gap.
    Test.@test FG.Grids.minimum_spacing(grid, 1) == FG.Grids.maximum_spacing(grid, 1) == 2000.0
    Test.@test FG.Grids.size_tuple(grid) == (length(x), length(y))
    Test.@test FG.Grids.area(grid, 2, 2) ≈ 2000.0 * 2000.0
    Test.@test FG.Grids.coords(grid, 2, 3) == (x = 2000.0, y = 4000.0)
    out = zeros(2)
    FG.Grids.coords!(out, grid, 2, 3)
    Test.@test out == [2000.0, 4000.0]
    Test.@test FG.Grids.isactive(grid, 1, 1)
    Test.@test !FG.Grids.isperiodic(grid, 1)  # a plain axis is bounded unless asked
end

Test.@testset "A periodic longitude axis wraps, a plain longitude axis is bounded" begin
    R = 6.371e6
    geom = FG.Geometry.SphericalGeometry(R)
    nλ, nφ = 8, 5
    dλ = 2π / nλ
    lon = FG.Axes.PeriodicAxis(0.0, 2π, nλ)
    lat = range(-π / 4, π / 4; length = nφ)
    grid = FG.Grids.StructuredGrid(geom, lon, lat, trues(nλ, nφ))
    Test.@test FG.Grids.isperiodic(grid, 1)
    Test.@test !FG.Grids.isperiodic(grid, 2)
    Test.@test FG.Grids.period(grid, 1) == 2π
    Test.@test FG.Grids.isuniform(grid, 1)
    Test.@test FG.Grids.spacing(grid, 1) == dλ
    # The same samples in a range are bounded unless asked.
    plain = range(0.0; step = dλ, length = nλ)
    Test.@test !FG.Grids.isperiodic(FG.Grids.StructuredGrid(geom, plain, lat), 1)
    Test.@test FG.Grids.isperiodic(FG.Grids.StructuredGrid(geom, plain, lat; topology = FG.Grids.Periodic()), 1)
    p = FG.Grids.coords(grid, 1, 1)
    Test.@test p.λ ≈ 0.0
    Test.@test p.φ ≈ -π / 4
end

Test.@testset "Coordinate names follow the geometry, never stand in for each other" begin
    cgeom = FG.Geometry.CartesianGeometry()
    sgeom = FG.Geometry.SphericalGeometry(6.371e6)
    xs = collect(0.0:1.0:4.0)
    cgrid = FG.Grids.StructuredGrid(cgeom, xs, xs, trues(5, 5))
    sgrid = FG.Grids.StructuredGrid(sgeom, deg2rad.(xs), deg2rad.(xs), trues(5, 5))

    Test.@test FG.Grids.coordinate_names(cgrid) == (:x, :y)
    Test.@test FG.Grids.coordinate_names(sgrid) == (:λ, :φ)
    Test.@test cgrid.x === FG.Grids.coordinates(cgrid, 1)
    Test.@test sgrid.λ === FG.Grids.coordinates(sgrid, 1)

    nx, ny = 4, 3
    xm = [Float64(i) for i in 1:nx, j in 1:ny]
    ym = [Float64(j) for i in 1:nx, j in 1:ny]
    cv = FG.Grids.CurvilinearGrid(sgeom, deg2rad.(xm), deg2rad.(ym), trues(nx, ny);
                                  keep_corners = true)
    Test.@test FG.Grids.coordinate_names(cv) == (:λ, :φ)
    Test.@test cv.λ === FG.Grids.coordinates(cv, 1)
    Test.@test size(FG.Grids.corners(cv, 1)) == (nx + 1, ny + 1)
    Test.@test keys(FG.Grids.corner_coords(cv, 1, 1)) == (:λ, :φ)

    un = FG.Grids.UnstructuredGrid(sgeom, [0.0, 0.1, 0.2], [0.0, 0.1, 0.0], [1.0, 1.0, 1.0], trues(3))
    Test.@test FG.Grids.coordinate_names(un) == (:λ, :φ)
    Test.@test un.λ == [0.0, 0.1, 0.2]
end

Test.@testset "Grids implement the Base collection surface" begin
    geom = FG.Geometry.CartesianGeometry()
    xs = collect(0.0:1.0:4.0)
    ys = collect(0.0:1.0:3.0)
    grid = FG.Grids.StructuredGrid(geom, xs, ys, trues(5, 4))
    Test.@test size(grid) == (5, 4)
    Test.@test size(grid, 2) == 4
    Test.@test length(grid) == 20
    Test.@test ndims(grid) == 2
    Test.@test eltype(grid) === Float64
    Test.@test axes(grid) == (Base.OneTo(5), Base.OneTo(4))
end

Test.@testset "Cell measure is the exact integral of the metric over each cell" begin
    cw = FG.Discretization.cell_width
    sgeo = FG.Geometry.SphericalGeometry(6.371e6)
    cgeo = FG.Geometry.CartesianGeometry()
    cgeo3 = FG.Geometry.CartesianGeometry()
    R = sgeo.R

    # The reference in BigFloat: faces midway between centres and one gap out at each end, latitude
    # faces clamped to the poles, and the band area `|sin φ₊ − sin φ₋|` between them.
    function bfaces(x)
        b = big.(x)
        n = length(b)
        return vcat(b[1] - (b[2] - b[1]) / 2, (b[1:(n - 1)] .+ b[2:n]) ./ 2, b[n] + (b[n] - b[n - 1]) / 2)
    end
    pole(x) = clamp(x, -big(π) / 2, big(π) / 2)
    band(f, j) = abs(sin(pole(f[j + 1])) - sin(pole(f[j])))

    # A spherical cell is `R²·Δλ·(sin φ₊ − sin φ₋)` between its faces, on a nonuniform grid.
    λ = collect(range(0.0; step = 2π / 12, length = 12))
    φ = cumsum([-1.0, 0.2, 0.5, 0.15, 0.4, 0.3])
    g = FG.Grids.StructuredGrid(sgeo, λ, φ, trues(length(λ), length(φ)); topology = FG.Grids.Periodic())
    λper = 2π
    fφ = bfaces(φ)
    ref = [Float64(big(R)^2 * cw(λ, i, λper) * band(fφ, j)) for i in eachindex(λ), j in eachindex(φ)]
    Test.@test FG.Grids.measure(g) ≈ ref rtol = 1e-14

    # A shell cell is `Δλ·(sin φ₊ − sin φ₋)·(r₊³ − r₋³)/3`.
    r = collect(6.30e6:1.0e4:6.34e6)
    g3 = FG.Grids.StructuredGrid(sgeo, λ, φ, r, trues(length(λ), length(φ), length(r));
                                 topology = FG.Grids.Periodic())
    fr = bfaces(r)
    ref3 = [Float64(cw(λ, i, λper) * band(fφ, j) * (fr[k + 1]^3 - fr[k]^3) / 3)
            for i in eachindex(λ), j in eachindex(φ), k in eachindex(r)]
    Test.@test FG.Grids.measure(g3) ≈ ref3 rtol = 1e-14

    # A degenerate angular axis contributes no differential, so the measure is the arc element.
    zonal = FG.Grids.StructuredGrid(sgeo, λ, [0.4], trues(length(λ), 1); topology = FG.Grids.Periodic())
    Test.@test FG.Grids.measure(zonal) ≈ [sgeo.R * cos(0.4) * cw(λ, i, 2π) for i in eachindex(λ), _ in 1:1]
    merid = FG.Grids.StructuredGrid(sgeo, [0.3], φ, trues(1, length(φ)))
    Test.@test FG.Grids.measure(merid) ≈ [sgeo.R * cw(φ, j) for _ in 1:1, j in eachindex(φ)]

    # A whole sphere's cells sum to 4πR², whether the outer faces fall on the poles or beyond them:
    # a node on a pole owns its polar cap.
    λ16 = FG.Axes.PeriodicAxis(0.0, 2π, 16)
    for φs in (collect(range(-π / 2 + π / 64, π / 2 - π / 64; length = 32)),
               collect(range(-π / 2, π / 2; length = 33)),
               collect(range(π / 2, -π / 2; length = 33)))
        Test.@test sum(FG.Grids.measure(FG.Grids.StructuredGrid(sgeo, λ16, φs))) ≈ 4π * R^2 rtol = 1e-13
    end
    let φs = collect(range(-π / 2, π / 2; length = 33)),
        m = FG.Grids.measure(FG.Grids.StructuredGrid(sgeo, λ16, φs))
        Test.@test sum(m[:, end]) ≈ 2π * R^2 * (1 - cos(π / 64)) rtol = 1e-13
    end

    # Float32 cells are exact to a few ulps against the same centres read in BigFloat, including the
    # narrow polar bands.
    let φ32 = Float32.(collect(range(-π / 2 + π / 4096, π / 2 - π / 4096; length = 2048))),
        R32 = Float32(R),
        w32 = FG.Grids.measure_factors(
            FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(R32), Float32.(λ16), φ32))[2]
        f = bfaces(φ32)
        Test.@test maximum(abs(big(w32[j]) / (big(R32)^2 * band(f, j)) - 1) for j in eachindex(φ32)) <
                   4 * eps(Float32)
    end
    Test.@test FG.Grids._half_pi_lo(Float64) == Float64(big(π) / 2 - big(Float64(π)) / 2)
    Test.@test FG.Grids._half_pi_lo(Float32) == Float32(big(π) / 2 - big(Float32(π)) / 2)

    # A periodic axis wraps its boundary cell in 3-D exactly as it already did in 2-D.
    xnu = cumsum([0.0, 10.0, 40.0, 15.0, 60.0, 25.0])
    yy = collect(0.0:50.0:100.0)
    zz = collect(0.0:10.0:20.0)
    # A nonuniform periodic Cartesian direction has no period to infer, its samples leaving the seam
    # gap undetermined, so the period is required at construction.
    xper = FG.Grids._cartesian_period(xnu)
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(
        cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz)); topology = FG.Grids.Periodic())
    gp = FG.Grids.StructuredGrid(cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz));
                           topology = FG.Grids.Periodic(), period = xper)
    Test.@test FG.Grids.measure(gp) ≈ [cw(xnu, i, xper) * cw(yy, j) * cw(zz, k)
                                 for i in eachindex(xnu), j in eachindex(yy), k in eachindex(zz)]
    # …and 2-D and 3-D agree on that wrapped width.
    g2p = FG.Grids.StructuredGrid(cgeo, xnu, yy, trues(length(xnu), length(yy));
                            topology = FG.Grids.Periodic(), period = xper)
    Test.@test FG.Grids.measure(g2p)[end, 1] / cw(yy, 1) ≈ FG.Grids.measure(gp)[end, 1, 1] / (cw(yy, 1) * cw(zz, 1))
end

Test.@testset "A grid on a quadrature sampling is measured by its weights" begin
    SS = FG.SphericalSampling
    R = 6.371e6
    geo = FG.Geometry.SphericalGeometry(R)
    for (s, n, degree) in ((SS.GaussLegendreSampling(), 24, 47), (SS.Fejer1Sampling(), 24, 23),
                           (SS.DriscollHealySampling(), 24, 23), (SS.DriscollHealyEqualSampling(), 24, 23))
        g = FG.Connectivity.structured_grid(s, n; geometry = geo)
        q = SS.spherical_quadrature(s, n)
        m = FG.Grids.measure(g)
        Test.@test all(isapprox(m[i, j], R^2 * (2π / length(q.λ)) * q.w[j]; rtol = 1e-13)
                       for i in axes(m, 1), j in axes(m, 2))
        # `Σ measure·f` is the quadrature, so it integrates `sin^d φ` over the sphere exactly up to the
        # rule's degree: `4πR²/(d+1)` for even `d`, zero for odd.
        μ = sin.(FG.Grids.coordinates(g, 2))
        Test.@test all(0:degree) do d
            got = sum(m[i, j] * μ[j]^d for i in axes(m, 1), j in axes(m, 2))
            abs(got - (iseven(d) ? 4π * R^2 / (d + 1) : 0.0)) < 1e-12 * 4π * R^2
        end
    end
    # A sampling with no weights is measured geometrically, and still covers the sphere.
    gm = FG.Connectivity.structured_grid(SS.McEwenWiauxSampling(), 24; geometry = geo)
    Test.@test sum(FG.Grids.measure(gm)) ≈ 4π * R^2 rtol = 1e-13
    # A staggered location of such a grid is a mesh of faces, with no quadrature of its own.
    let gl = FG.Connectivity.structured_grid(SS.GaussLegendreSampling(), 12; geometry = geo),
        gf = FG.Grids.grid_at(FG.Grids.StaggeredGrid(gl), (FG.Discretization.Center(), FG.Discretization.Face()))
        Test.@test FG.Grids.sampling(gf) === nothing
        Test.@test size(gf) == (23, 13)
    end
end

Test.@testset "UnstructuredGrid carries a caller-chosen index type" begin
    geom = FG.Geometry.CartesianGeometry()
    x = [0.0, 1.0, 0.0]
    y = [0.0, 0.0, 1.0]
    areas = [0.5, 0.5, 0.5]
    # CSR: node 1 → [2,3], node 2 → [1], node 3 → [1]
    nbrs = Int32[2, 3, 1, 1]
    ptr = Int32[1, 3, 4, 5]
    grid = FG.Grids.UnstructuredGrid(geom, x, y, areas, trues(3), nbrs, ptr)
    Test.@test collect(FG.Grids.neighbors(grid, 1)) == Int32[2, 3]
    Test.@test FG.Grids.measure(grid, 2) ≈ 0.5
    Test.@test FG.Grids.coords(grid, 2) == (x = 1.0, y = 0.0)
    # Mismatched adjacency length is caught at construction.
    Test.@test_throws ArgumentError FG.Grids.UnstructuredGrid(geom, x, y, areas, trues(3), nbrs, Int32[1, 3])
    # With no adjacency given, a node has no neighbours.
    Test.@test isempty(FG.Grids.neighbors(FG.Grids.UnstructuredGrid(geom, x, y, areas, trues(3)), 1))
end

Test.@testset "Curvilinear areas hold two corner rows, not the whole field" begin
    geo = FG.Geometry.SphericalGeometry()
    for n in (32,)
        λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
        φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
        g = FG.Grids.CurvilinearGrid(geo, λ, φ, trues(n, n))
        a = FG.Grids.measure(g)
        # Reference: the same two triangles per cell in BigFloat, from the corners the kernel reads.
        xc, yc = FG.Grids._centers_to_corners(λ), FG.Grids._centers_to_corners(φ)
        dirs = [(cos(big(yc[i, j])) * cos(big(xc[i, j])), cos(big(yc[i, j])) * sin(big(xc[i, j])),
                 sin(big(yc[i, j]))) for i in 1:(n + 1), j in 1:(n + 1)]
        ref = [big(geo.R)^2 * (FG.Geometry.spherical_excess(dirs[i, j], dirs[i + 1, j], dirs[i + 1, j + 1]) +
                               FG.Geometry.spherical_excess(dirs[i, j], dirs[i + 1, j + 1], dirs[i, j + 1]))
               for i in 1:n, j in 1:n]
        Test.@test maximum(abs.(a ./ ref .- 1)) < 16 * eps(Float64)
    end
end

Test.@testset "A curvilinear spherical area holds below a cell of R·sqrt(eps(T))" begin
    # A cell whose solid angle is under `eps(T)` is where the excess is most exposed to cancellation:
    # the triple product behind it is of order the area while the vertex components are of order one.
    # 391x264 over 4°x4° gives a solid angle of 3.9e-8 against `eps(Float32) = 1.2e-7`.
    nlon, nlat = 391, 264
    for T in (Float32, Float64)
        geo = FG.Geometry.SphericalGeometry{T}(T(6.371e6))
        lon = range(T(deg2rad(-125)), T(deg2rad(-121)); length = nlon)
        lat = range(T(deg2rad(34)), T(deg2rad(38)); length = nlat)
        g = FG.Grids.CurvilinearGrid(geo, T[lon[i] for i in 1:nlon, j in 1:nlat],
                                     T[lat[j] for i in 1:nlon, j in 1:nlat], trues(nlon, nlat))
        a = FG.Grids.measure(g)
        # Ground truth per cell: `R²·Δλ·(sin φ₂ − sin φ₁)`, with no excess in it.
        dλ = Float64(lon[2]) - Float64(lon[1])
        dφ = Float64(lat[2]) - Float64(lat[1])
        worst = 0.0
        for j in 2:(nlat - 1), i in 2:(nlon - 1)
            φ = Float64(lat[j])
            truth = Float64(6.371e6)^2 * dλ * (sin(φ + dφ / 2) - sin(φ - dφ / 2))
            worst = max(worst, abs(Float64(a[i, j]) - truth) / truth)
        end
        Test.@test worst < (T === Float32 ? 5e-3 : 1e-6)
        # The spread across 4° of latitude is `cos φ`, and nothing wider. Rounding noise shows up
        # here as a spread many times the real one while the total area still agrees.
        spread = maximum(a) / minimum(a)
        Test.@test spread ≈ cos(Float64(lat[1])) / cos(Float64(lat[end])) rtol = 0.02
        # The area of the quadrilateral the stored corners span, to a few ulps of `T` at a cell of
        # 1e-4 of the radius: the reference is BigFloat from those same corners.
        let gk = FG.Grids.CurvilinearGrid(geo, T[lon[i] for i in 1:nlon, j in 1:nlat],
                                          T[lat[j] for i in 1:nlon, j in 1:nlat], trues(nlon, nlat);
                                          keep_corners = true),
            xc = FG.Grids.corners(gk, 1), yc = FG.Grids.corners(gk, 2)
            dir(i, j) = (cos(big(yc[i, j])) * cos(big(xc[i, j])), cos(big(yc[i, j])) * sin(big(xc[i, j])),
                         sin(big(yc[i, j])))
            rel = maximum(((i, j),) -> abs(FG.Grids.measure(gk, i, j) /
                (big(geo.R)^2 * (FG.Geometry.spherical_excess(dir(i, j), dir(i + 1, j), dir(i + 1, j + 1)) +
                                 FG.Geometry.spherical_excess(dir(i, j), dir(i + 1, j + 1), dir(i, j + 1)))) - 1),
                ((i, j) for i in (1, 100, 200, nlon), j in (1, 77, 150, nlat)))
            Test.@test rel < 16 * eps(T)
        end
    end
end

Test.@testset "Cell measure is stored factored, not materialized" begin
    geo = FG.Geometry.SphericalGeometry()
    for (nx, ny) in ((16, 9),)
        x = collect(range(0, 2π; length = nx))
        y = collect(range(-1.3, 1.3; length = ny))
        g = FG.Grids.StructuredGrid(geo, x, y, FG.Grids.AllActive((nx, ny)))
        m = FG.Grids.measure(g)
        # Indexing agrees with the dense outer product to the bit.
        wx, wy = FG.Grids.measure_factors(g)
        dense = wx .* transpose(wy)
        Test.@test all(m[i, j] === dense[i, j] for i in 1:nx, j in 1:ny)
        Test.@test FG.Grids.measure_array(g) == dense
        Test.@test FG.Grids.measure(g, 3, 4) === dense[3, 4]
        # sum is ∏ᵈ ∑ᵢ, i.e. O(Nx+Ny), and agrees with the dense sum to roundoff.
        Test.@test sum(m) ≈ sum(dense) rtol = 1e-14
        Test.@test_throws BoundsError m[nx + 1, 1]
    end
    # A 1-D grid is separable too, its single factor serving as the measure.
    g1 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), collect(0.0:0.5:5.0), FG.Grids.AllActive((11,)))
    Test.@test collect(only(FG.Grids.measure_factors(g1))) == collect(FG.Grids.measure(g1))
    Test.@test FG.Grids.measure(g1, 3) ≈ 0.5
end

Test.@testset "Every architecture can be built without a mask" begin
    GD = FG.Grids
    C = FG.Connectivity
    D = FG.Discretization
    O = FG.Operators
    cart = FG.Geometry.CartesianGeometry{Float64}()
    n = 8
    x = [t for t in range(0.0, 7.0; length = n), _ in 1:n]
    y = [u for _ in 1:n, u in range(0.0, 7.0; length = n)]

    # A curvilinear or node grid built without a mask is all-active.
    g0 = GD.CurvilinearGrid(cart, x, y; measure = fill(1.0, n, n))
    gm = GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
    Test.@test all(GD.isactive(g0, i, j) == GD.isactive(gm, i, j) for i in 1:n, j in 1:n)

    # The mask is recognised by its element type, so the optional positional measure parses either
    # way, and equals the keyword form.
    let gp = GD.CurvilinearGrid(cart, x, y, fill(2.0, n, n)),
        gpm = GD.CurvilinearGrid(cart, x, y, fill(2.0, n, n), trues(n, n))
        Test.@test GD.measure(gp, 2, 2) == 2.0
        Test.@test GD.measure(gpm) == GD.measure(GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(2.0, n, n)))
    end

    let nn = 40, xs = 10.0 .* rand(nn), ys = 6.0 .* rand(nn)
        u0 = GD.UnstructuredGrid(cart, (xs, ys); k = 4, areas = ones(nn))
        um = GD.UnstructuredGrid(cart, (xs, ys), trues(nn); k = 4, areas = ones(nn))
        Test.@test all(GD.isactive(u0, i) == GD.isactive(um, i) for i in 1:nn)
    end

    # Everything downstream is unchanged by which representation the mask has.
    let f = 2.0 .* x .- 3.0 .* y, g1 = zeros(n, n), g2 = zeros(n, n)
        O.gradient!(g1, g2, f, O.gradient_plan(g0))
        Test.@test maximum(abs.(g1 .- 2.0)) < 1e-10 && maximum(abs.(g2 .+ 3.0)) < 1e-10
        Test.@test C.nneighbors_within(g0, 4, 4; ball = 2.0) ==
                   C.nneighbors_within(gm, 4, 4; ball = 2.0)
        Test.@test abs(O.interpolate(f, g0, (3.3, 2.2); k = 8) - (2 * 3.3 - 3 * 2.2)) < 1e-9
        Test.@test sum(GD.measure(g0)) == sum(GD.measure(gm))
        Test.@test length(C.build_connectivity(g0).nbrs) == length(C.build_connectivity(gm).nbrs)
    end
end

Test.@testset "An all-active mask carries no per-cell storage" begin
    m = FG.Grids.AllActive((7, 5))
    Test.@test size(m) == (7, 5) && length(m) == 35
    Test.@test all(m[i, j] for i in 1:7, j in 1:5)
    Test.@test count(m) == 35 && all(m) && any(m)
    Test.@test collect(m) == trues(7, 5)
    Test.@test_throws BoundsError m[8, 1]
    g = FG.Connectivity.structured_grid(FG.SphericalSampling.Fejer1Sampling(), 33)
    Test.@test all(FG.Grids.isactive(g, i, j) for i in 1:size(g, 1), j in 1:size(g, 2))
    # It is smaller than a BitArray.
    Test.@test Base.summarysize(FG.Grids.AllActive((4000, 2000))) < Base.summarysize(trues(4000, 2000))
    # An explicit mask still overrides it and still masks.
    mm = trues(FG.SphericalSampling.axes_lengths(FG.SphericalSampling.Fejer1Sampling(), 9).nlon, 9)
    mm[2, 3] = false
    gm = FG.Connectivity.structured_grid(FG.SphericalSampling.Fejer1Sampling(), 9; mask = mm)
    Test.@test !FG.Grids.isactive(gm, 2, 3)
end

Test.@testset "A periodic axis wraps whichever way it is stored, and states its own topology" begin
    # A circle is a circle whichever way it is traversed, and a descending axis is routine.
    sg = FG.Geometry.SphericalGeometry()
    n = 8
    asc = FG.Axes.PeriodicAxis(0.0, 2π, n)
    desc = FG.Axes.PeriodicAxis(2π - 2π / n, -2π, n)
    Test.@test collect(desc) ≈ reverse(collect(asc))
    Test.@test FG.Axes.period(desc) == 2π && step(desc) == -step(asc)
    for x in (asc, desc)
        g = FG.Grids.StructuredGrid(sg, x, [0.0, 0.1], trues(n, 2))
        Test.@test FG.Grids.isperiodic(g, 1) && FG.Grids.period(g, 1) == 2π
    end
    # The same samples in a vector are bounded in either order until the caller says otherwise.
    for x in (collect(asc), reverse(collect(asc)))
        Test.@test !FG.Grids.isperiodic(FG.Grids.StructuredGrid(sg, x, [0.0, 0.1], trues(n, 2)), 1)
        Test.@test FG.Grids.isperiodic(FG.Grids.StructuredGrid(sg, x, [0.0, 0.1], trues(n, 2);
                                                               topology = FG.Grids.Periodic()), 1)
    end
    # The measure follows the periodicity, so a descending circle tiles as an ascending one does,
    # seam cell included, and a vector declared periodic tiles it as the periodic axis does.
    φ = collect(range(-1.0, 1.0; length = 5))
    ma = FG.Grids.measure(FG.Grids.StructuredGrid(sg, asc, φ, FG.Grids.AllActive((n, 5))))
    md = FG.Grids.measure(FG.Grids.StructuredGrid(sg, desc, φ, FG.Grids.AllActive((n, 5))))
    mv = FG.Grids.measure(FG.Grids.StructuredGrid(sg, reverse(collect(asc)), φ,
                                                  FG.Grids.AllActive((n, 5)); topology = FG.Grids.Periodic()))
    Test.@test sum(ma) ≈ sum(md) rtol = 1e-14
    Test.@test sum(ma) ≈ sum(mv) rtol = 1e-14
    # A periodic axis declared bounded, a longitude axis of another period, or a periodic axis given
    # another period raises.
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(sg, asc, φ; topology = FG.Grids.Bounded())
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(sg, FG.Axes.PeriodicAxis(0.0, π, n), φ)
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(sg, asc, φ; period = π)
    # A Cartesian periodic axis wraps at its own period.
    cg = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry{Float32}(), FG.Axes.PeriodicAxis(0.0, 5.0, 10),
                                 0.0f0:1.0f0:3.0f0)
    Test.@test FG.Grids.isperiodic(cg, 1) && FG.Grids.period(cg, 1) == 5.0f0
end

Test.@testset "A distance across a periodic seam keeps the precision of the gap" begin
    G = FG.Grids
    GE = FG.Geometry
    for T in (Float32, Float64)
        # A sphere at 1°: the two stored seam longitudes differ by nearly a turn, the cells by one step.
        R = T(6.371e6)
        gs = G.StructuredGrid(GE.SphericalGeometry(R), FG.Axes.PeriodicAxis{T}(zero(T), T(2π), 360),
                              collect(T, range(-1.2, 1.2; length = 9)))
        for j in (1, 5, 9)
            a, b = G.coords(Tuple, gs, 360, j), G.coords(Tuple, gs, 1, j)
            ref = GE.distance(GE.SphericalGeometry(big(R)), big.(a), big.(b))   # the stored values, exactly
            Test.@test abs(GE.distance(gs, (360, j), (1, j)) - ref) / ref < 16eps(T)
            gap = big(b[1]) + 2big(π) - big(a[1])
            Test.@test abs(G.displacement(gs, (360, j), (1, j))[1] - gap) / gap < 4eps(T)
        end
        # A Cartesian periodic axis off the origin, whose two seam coordinates are not within a factor
        # of two.
        L = T(10)
        gc = G.StructuredGrid(GE.CartesianGeometry(T), FG.Axes.PeriodicAxis{T}(T(0.003), L, 1000), T[0, 1])
        a, b = G.coords(Tuple, gc, 1000, 1), G.coords(Tuple, gc, 1, 1)
        gap = big(b[1]) + big(L) - big(a[1])
        Test.@test abs(G.displacement(gc, (1000, 1), (1, 1))[1] - gap) / gap < 2eps(T)
        Test.@test abs(GE.distance(gc, (1000, 1), (1, 1)) - gap) / gap < 2eps(T)
    end
end

Test.@testset "A separable measure stays separable under the operations that preserve it" begin
    GR = FG.Grids
    geo = FG.Geometry.CartesianGeometry()
    n = 400
    g = GR.StructuredGrid(geo, range(0.0; step = 0.5, length = n),
                          range(0.0; step = 0.25, length = n))
    m = GR.measure(g)
    dense = collect(m)
    probes = ((1, 1), (7, 13), (n, n))

    # Scaling, a multiplicative map, and a factor-wise product all keep the product form, so a unit
    # conversion stays `O(∑Nᵈ)` in storage.
    for got in (2.0 .* m, m .* 2.0, m ./ 4.0, abs.(m), abs2.(m), sqrt.(m), inv.(m),
                m .* m, m ./ m)
        Test.@test Base.summarysize(got) < 200
    end
    Test.@test all(≈((2.0 .* m)[I...], 2 * dense[I...]) for I in probes)
    Test.@test all(≈(abs2.(m)[I...], abs2(dense[I...])) for I in probes)
    Test.@test all(≈((m .* m)[I...], dense[I...]^2) for I in probes)

    # The closed-form reductions still apply to the result.
    s = 3.0 .* m
    Test.@test sum(s) ≈ 3 * sum(m)
    Test.@test maximum(s) ≈ 3 * maximum(m)
    Test.@test all(extrema(s) .≈ 3 .* extrema(m))
    Test.@test findmax(s)[2] == findmax(m)[2]

    # Every other map materializes. `findmax`'s per-axis argmax holds for non-negative factors alone,
    # so a negative scale is one of them.
    Test.@test exp.(m)[3, 4] ≈ exp(dense[3, 4])
    Test.@test (-2.0 .* m)[3, 4] ≈ -2 * dense[3, 4]
    Test.@test findmax(-2.0 .* m)[1] ≈ maximum(-2 .* dense)

    d = Array{Float64}(undef, size(m))
    copyto!(d, m)
    Test.@test all(d[I...] == dense[I...] for I in probes)
    Test.@test_throws DimensionMismatch m .* GR.measure(
        GR.StructuredGrid(geo, range(0.0; step = 1.0, length = 3),
                          range(0.0; step = 1.0, length = 3)))
end

Test.@testset "Curvilinear and node grids work in any number of dimensions" begin
    geo = FG.Geometry.CartesianGeometry()
    C = FG.Connectivity
    GR = FG.Grids

    # 2-D still derives its corner areas: a uniform 1.0 × 0.5 mesh has exactly that cell area.
    X2 = [x for x in 0.0:1.0:4.0, _ in 0.0:0.5:2.0]
    Y2 = [y for _ in 0.0:1.0:4.0, y in 0.0:0.5:2.0]
    g2 = GR.CurvilinearGrid(geo, X2, Y2, trues(5, 5); keep_corners = true)
    Test.@test all(≈(0.5), GR.measure(g2))
    Test.@test GR.coords(g2, 2, 3) == (x = 1.0, y = 1.0)
    # The reconstructed corners sit exactly a half-cell outside the outermost centres.
    Test.@test GR.corner_coords(g2, 1, 1) == (x = -0.5, y = -0.25)

    X3 = [x for x in 0.0:1.0:3.0, _ in 1:3, _ in 1:2]
    Y3 = [y for _ in 1:4, y in 0.0:2.0:4.0, _ in 1:2]
    Z3 = [z for _ in 1:4, _ in 1:3, z in 0.0:0.5:0.5]
    # Past 2-D the corner-area kernel does not apply, and the constructor says so.
    Test.@test_throws ArgumentError GR.CurvilinearGrid(geo, X3, Y3, Z3, trues(4, 3, 2))
    vol = fill(1.0 * 2.0 * 0.5, 4, 3, 2)
    g3 = GR.CurvilinearGrid(geo, X3, Y3, Z3, vol, trues(4, 3, 2); keep_corners = true)
    Test.@test GR.coordinate_names(g3) == (:x, :y, :z)
    Test.@test GR.coords(g3, 2, 3, 2) == (x = 1.0, y = 4.0, z = 0.5)
    Test.@test GR.measure(g3, 2, 2, 1) == 1.0
    Test.@test size(GR.corners(g3, 1)) == (5, 4, 3)
    Test.@test GR.corner_coords(g3, 1, 1, 1) == (x = -0.5, y = -1.0, z = -0.25)

    W = ntuple(d -> [Tuple(ci)[d] * 1.0 for ci in CartesianIndices((3, 3, 2, 2))], 4)
    g4 = GR.CurvilinearGrid(geo, W..., ones(3, 3, 2, 2), trues(3, 3, 2, 2))
    Test.@test GR.coords(g4, 2, 3, 1, 2) == (x1 = 2.0, x2 = 3.0, x3 = 1.0, x4 = 2.0)

    # The ghost ring is exact for a field linear in each direction, corners included, in 2-D and
    # beyond.
    L3 = [1.5i - 2.0j + 0.5k for i in 1:4, j in 1:5, k in 1:3]
    Test.@test all(FG.Grids._ghosted(L3, (i, j, k)) ≈ 1.5i - 2.0j + 0.5k
                   for i in 0:5, j in 0:6, k in 0:4)
    # Connectivity follows into N-D, and the dense adjacency agrees with the CSR.
    Test.@test C.nneighbors(g3, 2, 2, 1) == 5          # 6 face neighbours minus one wall
    Test.@test C.linear_index(g3, 2, 3, 2) == 2 + 2 * 4 + 1 * 12
    Test.@test C.neighbors!(Vector{Int}(undef, 8), g3, 2, 2, 1) == 5
    csr3 = C.build_connectivity(g3)
    Test.@test C.is_symmetric_adjacency(csr3)
    Test.@test C.adjacency_matrix!(Matrix{Bool}(undef, 24, 24), g3) == C.adjacency_matrix(csr3)
    Test.@test C.adjacency_matrix!(Matrix{Bool}(undef, 25, 25), g2) ==
               C.adjacency_matrix(C.build_connectivity(g2))
    ball3 = C.build_connectivity_within(g3; ball = 1.01)
    k3 = C.linear_index(g3, 2, 2, 1)
    Test.@test sort(ball3.nbrs[ball3.ptr[k3]:(ball3.ptr[k3 + 1] - 1)]) ==
               sort(C.neighbors_within(g3, 2, 2, 1; ball = 1.01))

    # Periodicity is per-direction in N-D too.
    gp = GR.CurvilinearGrid(geo, X3, Y3, Z3, vol, trues(4, 3, 2);
                            topology = GR.Periodic(), period = (4.0, 0.0, 0.0))
    Test.@test GR.isperiodic(gp, 1) && !GR.isperiodic(gp, 2)
    Test.@test GR.period(gp, 1) == 4.0
    Test.@test C.nneighbors(gp, 1, 2, 1) == 5

    # Node grids: coordinates as a tuple.
    x, y, z = rand(6), rand(6), rand(6)
    gu3 = GR.UnstructuredGrid(geo, (x, y, z), ones(6), trues(6))
    Test.@test GR.coordinate_names(gu3) == (:x, :y, :z)
    Test.@test GR.coords(gu3, 4) == (x = x[4], y = y[4], z = z[4])
    Test.@test isempty(GR.neighbors(gu3, 4))          # no CSR given ⇒ no adjacency
    gu3a = GR.UnstructuredGrid(geo, (x, y, z), ones(6), trues(6), [2, 1], [1, 2, 3, 3, 3, 3, 3])
    Test.@test collect(GR.neighbors(gu3a, 1)) == [2]
    gu1 = GR.UnstructuredGrid(geo, ([0.0, 1.0, 5.0],), ones(3), trues(3))
    Test.@test GR.coordinate_names(gu1) == (:x,)
    Test.@test sort(C.neighbors_within(gu1, 1; ball = 1.5)) == [2]
    # The two-direction positional form is unchanged.
    Test.@test GR.coordinate_names(
        GR.UnstructuredGrid(geo, rand(8), rand(8), ones(8), trues(8))) == (:x, :y)

    # The k-d-tree neighbour search works in any number of dimensions.
    nn = 40
    px, py, pz, pw = rand(nn), rand(nn), rand(nn), rand(nn)
    bruteknn(pts, i, kk) = sort([t[2] for t in sort(
        [(sqrt(sum((pts[q][i] - pts[q][j])^2 for q in eachindex(pts))), j)
         for j in 1:nn if j != i])[1:kk]])
    nbrs3, ptr3 = GR._build_kdtree_neighbors(Int32, geo, (px, py, pz); k = 4)
    Test.@test eltype(nbrs3) === Int32 && eltype(ptr3) === Int
    Test.@test all(sort(nbrs3[ptr3[i]:(ptr3[i + 1] - 1)]) == bruteknn((px, py, pz), i, 4)
                   for i in 1:nn)
    nbrs4, ptr4 = GR._build_kdtree_neighbors(Int, geo, (px, py, pz, pw); k = 3)
    Test.@test eltype(nbrs4) === Int
    Test.@test all(sort(nbrs4[ptr4[i]:(ptr4[i + 1] - 1)]) == bruteknn((px, py, pz, pw), i, 3)
                   for i in 1:nn)
    # Periodic ghost images are placed per direction, so a face node finds the opposite face.
    fx, fy, fz = [0.02, 0.98, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    nbw, ptw = GR._build_kdtree_neighbors(Int32, geo, (fx, fy, fz); k = 1,
                                          periodic = (true, false, false),
                                          period = (1.0, 0.0, 0.0))
    Test.@test nbw[ptw[1]] == 2
    nbb, ptb = GR._build_kdtree_neighbors(Int32, geo, (fx, fy, fz); k = 1)
    Test.@test nbb[ptb[1]] == 3
    # `(λ, φ, r)` embeds at its own radius, so a radius query there is a true chord.
    sph1 = FG.Geometry.SphericalGeometry(1.0)
    sλ, sφ, sr = [0.0, 0.0], [0.0, 0.0], [1.0, 1.5]
    Test.@test GR._build_kdtree_neighbors(Int32, sph1, (sλ, sφ, sr); k = 1, radius = 0.6)[2][2] == 2
    Test.@test GR._build_kdtree_neighbors(Int32, sph1, (sλ, sφ, sr); k = 1, radius = 0.4)[2][2] == 1
    Test.@test_throws ArgumentError GR._build_kdtree_neighbors(Int32, sph1, (sλ, sφ, sr, sr); k = 1)
    # Ids beyond the index type raise.
    Test.@test_throws ArgumentError GR._build_kdtree_neighbors(Int8, geo, (rand(200), rand(200)); k = 2)
    Test.@test_throws ArgumentError GR.UnstructuredGrid(geo, (px, py, pz), trues(nn); k = 4)
end

Test.@testset "Grids and connectivity work in any number of dimensions" begin
    geo = FG.Geometry.CartesianGeometry()
    # A 4-D and a 5-D rectilinear grid, built through the one varargs constructor.
    g4 = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:3.0)
    Test.@test FG.Grids.measure(g4, 2, 2, 2, 2) ≈ 1.0
    Test.@test sum(FG.Grids.measure(g4)) ≈ 4.0^4 rtol = 1e-12
    Test.@test FG.Grids.coordinate_names(g4) == (:x1, :x2, :x3, :x4)
    Test.@test FG.Grids.coords(g4, 2, 3, 1, 4) == (x1 = 1.0, x2 = 2.0, x3 = 0.0, x4 = 3.0)
    Test.@test FG.Connectivity.nneighbors(g4, 2, 2, 2, 2) == 8      # 2N faces
    Test.@test FG.Connectivity.nneighbors(g4, 2, 2, 2, 2; stencil = FG.Stencils.Moore(1)) == 80  # 3^4-1

    g5 = FG.Grids.StructuredGrid(geo, ntuple(_ -> 0.0:1.0:2.0, 5)...)
    Test.@test FG.Connectivity.nneighbors(g5, 2, 2, 2, 2, 2) == 10

    names7() = FG.Geometry.point_names(FG.Geometry.SphericalGeometry(1.0), Val(7))
    Test.@test names7() == (:λ, :φ, :r, :q4, :q5, :q6, :q7)

    # A spherical grid in 1-D measures arc length, and in 4-D keeps its metric directions.
    sgeo = FG.Geometry.SphericalGeometry(2.0)
    g1 = FG.Grids.StructuredGrid(sgeo, FG.Axes.PeriodicAxis(0.0, 2π, 9))
    Test.@test sum(FG.Grids.measure(g1)) ≈ 2.0 * 2π rtol = 1e-14   # R·Δλ over the full circle
    g4s = FG.Grids.StructuredGrid(sgeo, collect(range(0, 2π; length = 8)),
                            collect(range(-1.0, 1.0; length = 6)),
                            collect(range(1.0, 2.0; length = 4)),
                            collect(range(0.0, 1.0; length = 3)))
    Test.@test FG.Grids.coordinate_names(g4s) == (:λ, :φ, :r, :q4)

    # Topology lives in the type, and the mask defaults to all-active.
    Test.@test FG.Grids.topology(g4) == ntuple(_ -> FG.Grids.Bounded(), 4)
    gp = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0;
                           topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
    Test.@test FG.Grids.topology(gp, 1) === FG.Grids.Periodic()
    Test.@test FG.Grids.isperiodic(gp, 1) && !FG.Grids.isperiodic(gp, 2)
    Test.@test FG.Grids.periodic_flags(gp) == (true, false)
    Test.@test FG.Grids.period(gp, 1) ≈ 4.0            # uniform closure: n·|Δ|
    Test.@test FG.Grids.period(gp, 2) == 0.0           # bounded: no period
    # A lone tag is direction 1's, and a short tuple names the leading directions.
    gq = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0; topology = FG.Grids.Periodic())
    Test.@test FG.Grids.topology(gq) === (FG.Grids.Periodic(), FG.Grids.Bounded())
    g3q = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:2.0;
                                  topology = (FG.Grids.Bounded(), FG.Grids.Periodic()))
    Test.@test FG.Grids.topology(g3q) === (FG.Grids.Bounded(), FG.Grids.Periodic(), FG.Grids.Bounded())
    # Anything but tags, or more tags than directions, raises.
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0; topology = true)
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0;
                                                            topology = (true, false))
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0;
                                                            topology = ntuple(_ -> FG.Grids.Periodic(), 3))
    # A mask may be given positionally or by keyword, but not both.
    m = trues(4, 4); m[2, 2] = false
    Test.@test !FG.Grids.isactive(FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, m), 2, 2)
    Test.@test !FG.Grids.isactive(FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0; mask = m), 2, 2)
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, m; mask = m)
    Test.@test_throws DimensionMismatch FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, trues(3, 3))
end

Test.@testset "Mask topology: interior, boundary, components and holes" begin
    C = FG.Connectivity
    S = FG.Stencils
    geo = FG.Geometry.CartesianGeometry()
    m = trues(9, 9)
    m[4:6, 4:6] .= false      # an enclosed block
    m[1, 1] = false            # touches the edge, so not enclosed
    g = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0, m)
    Test.@test C.count_holes(g) == 1
    labels, ncomp = C.connected_components(g)
    Test.@test ncomp == 1                       # the active region is one piece
    Test.@test all(labels[ci] == 0 for ci in CartesianIndices(m) if !m[ci])
    Test.@test all(labels[ci] == 1 for ci in CartesianIndices(m) if m[ci])
    # Interior and boundary partition the active cells, on every stencil and under any backend.
    int = C.interior(g)
    bnd = C.boundary_cells(g)
    Test.@test count(int) + count(bnd) == count(m)
    Test.@test !any(int .& bnd)
    for st in (S.Axial(1), S.Moore(1), S.VonNeumann(2))
        i2 = C.interior(g; stencil = st)
        b2 = C.boundary_cells(g; stencil = st)
        Test.@test count(i2) + count(b2) == count(m)
        Test.@test !any(i2 .& b2)
        # Every active cell is one or the other, and no inactive cell is either.
        Test.@test all(m[ci] == (i2[ci] | b2[ci]) for ci in CartesianIndices(m))
        # `boundary_cells` is exactly the active non-interior cells.
        Test.@test b2 == [m[ci] & !i2[ci] for ci in CartesianIndices(m)]
    end
    Test.@test all(!int[ci] for ci in CartesianIndices(m) if !m[ci])
    Test.@test !int[1, 5]                       # a domain edge is not interior
    Test.@test int[3, 3]

    # Wrapping removes the edges, so a region that only touched one becomes enclosed.
    gp = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0, m;
                           topology = (FG.Grids.Periodic(), FG.Grids.Periodic()), period = (9.0, 9.0))
    Test.@test C.count_holes(gp) == 2
    # Two separated blocks are two holes.
    m2 = trues(11, 11); m2[3:4, 3:4] .= false; m2[8:9, 8:9] .= false
    Test.@test C.count_holes(FG.Grids.StructuredGrid(geo, 0.0:1.0:10.0, 0.0:1.0:10.0, m2)) == 2
    # A fully active grid has no holes and one component.
    gfull = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0)
    Test.@test C.count_holes(gfull) == 0
    Test.@test C.connected_components(gfull)[2] == 1
    # Two disconnected active regions are two components.
    m3 = falses(9, 3); m3[1:3, :] .= true; m3[7:9, :] .= true
    Test.@test C.connected_components(
        FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:2.0, m3))[2] == 2
    # A 3-D shell encloses its cavity.
    m4 = trues(7, 7, 7); m4[3:5, 3:5, 3:5] .= false
    Test.@test C.count_holes(
        FG.Grids.StructuredGrid(geo, 0.0:1.0:6.0, 0.0:1.0:6.0, 0.0:1.0:6.0, m4)) == 1
end

Test.@testset "A formula layout stores its arithmetic's parameters, not its results" begin
    G = FG.Grids
    C = FG.Connectivity
    SS = FG.SphericalSampling
    geo = FG.Geometry.SphericalGeometry()
    R = FG.Geometry.radius(geo)

    Test.@testset "HEALPixGrid storage is independent of nside" begin
        sizes = [Base.summarysize(G.HEALPixGrid(geo, ns)) for ns in (1, 1024)]
        Test.@test allequal(sizes)
        Test.@test length(G.HEALPixGrid(geo, 1024)) == 12 * 1024^2
    end

    Test.@testset "HEALPix coords, measure and neighbours are the pixelization's" begin
        for ns in (1, 8)
            g = G.HEALPixGrid(geo, ns)
            pts = SS.spherical_points(SS.HEALPixSampling(ns))
            npix = SS.healpix_npix(ns)
            Test.@test length(g) == npix
            buf = Vector{Int}(undef, 8)
            for i in 1:npix
                c = G.coords(g, i)
                Test.@test c.λ ≈ pts.λ[i] atol = 1e-13
                Test.@test c.φ ≈ pts.φ[i] atol = 1e-13
                # Every pixel carries the same area, which the pixelization is named for.
                Test.@test G.measure(g, i) ≈ 4π * R^2 / npix rtol = 1e-14
                # The face-table walk speaks 0-based pixels; a cell here is 1-based.
                n = C.healpix_neighbors!(buf, ns, i - 1)
                Test.@test sort(collect(G.neighbors(g, i))) == sort(buf[1:n] .+ 1)
                Test.@test C.nneighbors(g, i) == n
            end
            Test.@test sum(G.measure(g)) ≈ 4π * R^2 rtol = 1e-12
            # `materialize` is the ring walk; the per-cell accessor is the pixel formula. Same numbers.
            λ, φ = G.materialize(g)
            Test.@test λ == pts.λ
            Test.@test φ == pts.φ

            # The nested ids are the ring order permuted. Every `nside` here is a power of two, which
            # the nested quadtree needs.
            gn = G.HEALPixGrid(geo, ns; scheme = SS.Nested())
            nλ, nφ = G.materialize(gn)
            Test.@test all(nλ[SS.ring2nest(ns, r) + 1] === λ[r + 1] for r in 0:(npix - 1))
            Test.@test all(nφ[SS.ring2nest(ns, r) + 1] === φ[r + 1] for r in 0:(npix - 1))
            Test.@test maximum(abs(nλ[i] - G.coords(gn, i).λ) for i in 1:npix) < 1e-14
            Test.@test all(nφ[i] === G.coords(gn, i).φ for i in 1:npix)
        end
    end

    Test.@testset "RingGrid storage is O(nrings), not O(npoints)" begin
        a = G.RingGrid(geo, SS.OctahedralGaussianSampling(32))
        b = G.RingGrid(geo, SS.OctahedralGaussianSampling(64))
        Test.@test length(b) > 3 * length(a)          # points grow with N²
        Test.@test Base.summarysize(b) < 2.5 * Base.summarysize(a)   # storage grows with N
        Test.@test length(b) == SS.npoints(SS.OctahedralGaussianSampling(64))
        Test.@test G.nrings(b) == SS.nrings(SS.OctahedralGaussianSampling(64))
        # Tables of another element type are converted to the geometry's.
        let g32 = G.RingGrid(FG.Geometry.SphericalGeometry(6.371f6), [0.5, -0.5], Int32[4, 4], [1.0, 2.0])
            Test.@test G.measure(g32, 5) == 2.0f0
        end
    end

    Test.@testset "RingGrid coords and ring bookkeeping are the sampling's" begin
        for N in (8,)
            oct = SS.OctahedralGaussianSampling(N)
            g = G.RingGrid(geo, oct)
            pts = SS.spherical_points(oct)
            for i in eachindex(pts.λ)
                c = G.coords(g, i)
                Test.@test c.λ ≈ pts.λ[i] atol = 1e-13
                Test.@test c.φ ≈ pts.φ[i] atol = 1e-13
            end
            for r in 1:SS.nrings(oct)
                Test.@test G.ring_range(g, r) == SS.ring_range(oct, r)
                Test.@test G.nlon_in_ring(g, r) == SS.nlon_in_ring(oct, r)
                Test.@test all(G.ring_of(g, i) == r for i in G.ring_range(g, r))
            end
            λ, φ = G.materialize(g)
            Test.@test λ == pts.λ
            Test.@test φ == pts.φ
        end
    end

    Test.@testset "A ringwise measure is exact and carries one value per ring" begin
        ratios = Float64[]
        for N in (4, 64)
            g = G.RingGrid(geo, SS.OctahedralGaussianSampling(N))
            m = G.measure(g)
            # `Σ nlon_r · w_r · R² · 2π/nlon_r = 2π R² Σ w_r = 4π R²`, the weights summing to 2.
            Test.@test sum(m) ≈ 4π * R^2 rtol = 1e-12
            Test.@test sum(m) ≈ sum(collect(m)) rtol = 1e-12
            for r in (1, 2, G.nrings(g) ÷ 2, G.nrings(g))
                Test.@test allequal(m[i] for i in G.ring_range(g, r))
            end
            push!(ratios, Base.summarysize(collect(m)) / Base.summarysize(m))
        end
        # Storage tracks rings, so the saving over the dense form grows with resolution.
        Test.@test issorted(ratios)
        Test.@test first(ratios) > 1
    end

    Test.@testset "A ring grid's adjacency is in-ring plus the straddling pairs" begin
        g = G.RingGrid(geo, SS.OctahedralGaussianSampling(8))
        for i in 1:length(g)
            nb = collect(G.neighbors(g, i))
            Test.@test allunique(nb)
            Test.@test !(i in nb)
            Test.@test length(nb) ≤ G.max_neighbors(g)
            r = G.ring_of(g, i)
            # Both in-ring neighbours are distinct wherever the ring holds three or more points.
            G.nlon_in_ring(g, r) ≥ 3 &&
                Test.@test count(j -> G.ring_of(g, j) == r, nb) == 2
        end
        # Adjacent rings of unequal width make the relation directed: a wide ring's point can straddle
        # a narrow ring's point that does not straddle it back. The materialized CSR reproduces the
        # per-cell query exactly.
        conn = C.build_connectivity(g)
        for i in 1:length(g)
            Test.@test sort(collect(G.neighbors(conn, i))) == sort(collect(G.neighbors(g, i)))
        end

        # The cross-ring neighbours are the brute-force straddle, in exact integer arithmetic: point `j`
        # of an `m`-point ring sits `(j−1)·mm` units round an `mm`-point ring whose point `k` sits at
        # `(k−1)·m`, and its neighbours there are the last point at or before it and the first at or
        # after it.
        function straddle(j, m, mm)
            at = (j - 1) * mm
            lo = maximum(k for k in 1:mm if (k - 1) * m ≤ at)
            return sort(unique([lo, (lo - 1) * m == at ? lo : mod1(lo + 1, mm)]))
        end
        for gs in (g, G.RingGrid(geo, SS.ReducedGaussianSampling([12, 12, 16, 16, 12, 12])))
            ok = true
            for r in 1:G.nrings(gs), (j, i) in enumerate(G.ring_range(gs, r)), rr in (r - 1, r + 1)
                1 ≤ rr ≤ G.nrings(gs) || continue
                base = first(G.ring_range(gs, rr)) - 1
                got = sort([k - base for k in G.neighbors(gs, i) if G.ring_of(gs, k) == rr])
                ok &= got == straddle(j, G.nlon_in_ring(gs, r), G.nlon_in_ring(gs, rr))
            end
            Test.@test ok
        end
        # Where adjacent rings hold equal counts the points line up one to one, so the graph is
        # symmetric and an interior point has four neighbours.
        let ge = G.RingGrid(geo, SS.ReducedGaussianSampling(fill(12, 5)))
            Test.@test C.is_symmetric_adjacency(C.build_connectivity(ge))
            Test.@test all(C.nneighbors(ge, i) == (G.ring_of(ge, i) in (1, 5) ? 3 : 4) for i in 1:length(ge))
        end
    end

end

Test.@testset "Every formula and panel layout answers the grid interface, and a ball query as a scan does" begin
    G = FG.Grids
    C = FG.Connectivity
    GE = FG.Geometry
    SS = FG.SphericalSampling
    layouts(geo) = (G.HEALPixGrid(geo, 4), G.RingGrid(geo, SS.OctahedralGaussianSampling(8)), G.CubedSphereGrid(geo, 4),
                    G.YinYangGrid(geo, 8, 6), G.IcosahedralGrid(geo, 4))
    for (g, g32) in zip(layouts(GE.SphericalGeometry()), layouts(GE.SphericalGeometry(6.371f6)))
        n = length(g)
        # Two coordinates held as arithmetic: no coordinate arrays and no index space for a stencil.
        Test.@test G.ncoordinates(g) == 2
        Test.@test_throws ArgumentError G.coordinates(g)
        Test.@test_throws ArgumentError C.IndexTopology(g)
        Test.@test eltype(g32) === Float32 && G.coords(g32, 7).λ isa Float32 && G.measure(g32, 7) isa Float32
        # A mask set by a rebuild takes the cell out of the graph and leaves coordinates and measures alone.
        m = trues(n); m[5] = false
        gm = G.rebuild(g, (mask = m,))
        j = first(G.neighbors(g, 5))
        Test.@test !G.isactive(gm, 5) && C.nneighbors(gm, 5) == 0
        Test.@test !(5 in collect(G.neighbors(gm, j))) && 5 in collect(G.neighbors(gm, j; active_only = false))
        Test.@test G.coords(gm, 7) == G.coords(g, 7) && G.measure(gm, 7) == G.measure(g, 7)
        # A ball, scanned and through a cell list, against every pairwise distance.
        r = 1.5e6
        for i in (1, n ÷ 2, n)
            ref = sort([k for k in 1:n if k != i && GE.distance(g, i, k) ≤ r])
            Test.@test sort(C.neighbors_within(g, i; ball = r)) == ref
            Test.@test sort(C.neighbors_within(g, i; ball = r, topology = C.MetricTopology(g; index = G.cell_list(g; ball = r)))) == ref
        end
    end
    Test.@test_throws ArgumentError G.rebuild(G.HEALPixGrid(GE.SphericalGeometry(), 4), (nlon = trues(3),))
end

Test.@testset "The panel layouts store their resolution, not their panels" begin
    G = FG.Grids
    C = FG.Connectivity
    GE = FG.Geometry
    SS = FG.SphericalSampling
    geo = GE.SphericalGeometry()
    R = GE.radius(geo)

    # An independent statement of a gnomonic cell's area: the spherical excess of the two triangles
    # through its four corner directions. The layout answers with the closed-form solid angle, so
    # two exact formulas of different form have to agree.
    function cube_area_by_excess(n, f, i, j, R)
        Δ = π / 2 / n
        edge(k) = -π / 4 + (k - 1) * Δ
        dir(a, b) = let p = SS._cubed_face_to_xyz(f, tan(edge(a)), tan(edge(b)), Float64)
            s = sqrt(p.x^2 + p.y^2 + p.z^2)
            (p.x / s, p.y / s, p.z / s)
        end
        c1 = dir(i, j); c2 = dir(i + 1, j); c3 = dir(i + 1, j + 1); c4 = dir(i, j + 1)
        return R^2 * (GE.spherical_excess(c1, c2, c3) + GE.spherical_excess(c1, c3, c4))
    end

    Test.@testset "CubedSphereGrid stores one quadrant of a panel's areas" begin
        sizes = [Base.summarysize(G.CubedSphereGrid(geo, n)) - sizeof(Float64) * cld(n, 2)^2 for n in (1, 512)]
        Test.@test allequal(sizes)
        Test.@test length(G.CubedSphereGrid(geo, 512)) == 6 * 512^2
    end

    Test.@testset "Cubed-sphere coords and numbering are the sampling's" begin
        for n in (1, 3)
            g = G.CubedSphereGrid(geo, n)
            p = SS.cubed_sphere_points(n)
            Test.@test length(g) == length(p.λ)
            for k in eachindex(p.λ)
                c = G.coords(g, k)
                Test.@test c.λ ≈ p.λ[k] atol = 1e-13
                Test.@test c.φ ≈ p.φ[k] atol = 1e-13
                f, i, j = G.panel_cell(g, k)
                Test.@test f == p.panel[k]
                Test.@test G.cell_id(g, f, i, j) == k
                Test.@test 1 ≤ i ≤ n && 1 ≤ j ≤ n
            end
            λ, φ = G.materialize(g)
            Test.@test λ ≈ p.λ atol = 1e-13
            Test.@test φ ≈ p.φ atol = 1e-13
        end
    end

    Test.@testset "A gnomonic cell's area is its exact solid angle" begin
        for n in (3, 8)
            g = G.CubedSphereGrid(geo, n)
            m = G.measure(g)
            for k in 1:length(g)
                f, i, j = G.panel_cell(g, k)
                Test.@test m[k] ≈ cube_area_by_excess(n, f, i, j, R) rtol = 1e-12
                # The six panels are rigid images of one another and a panel is mirror-symmetric, so the
                # congruent cells hold one number.
                Test.@test m[k] === m[G.cell_id(g, 1, i, j)] === m[G.cell_id(g, 2, n + 1 - i, n + 1 - j)]
            end
            # `sum` is `4πR²` from the fact that the panels tile the sphere; adding the cells up has to
            # give the same number.
            Test.@test sum(m) ≈ 4π * R^2 rtol = 1e-14
            Test.@test sum(m) ≈ sum(collect(m)) rtol = 1e-12
        end
    end

    Test.@testset "A seam neighbour is symmetric across all twelve seams" begin
        for n in (1, 3)
            g = G.CubedSphereGrid(geo, n)
            conn = C.build_connectivity(g)
            Test.@test C.is_symmetric_adjacency(conn)
            nseam = 0
            for k in 1:length(g)
                nb = collect(G.neighbors(g, k))
                Test.@test allunique(nb)
                Test.@test !(k in nb)
                Test.@test length(nb) ≤ G.max_neighbors(g)
                f, _, _ = G.panel_cell(g, k)
                for j in nb
                    G.panel_cell(g, j)[1] == f || (nseam += 1)
                end
                # The CSR agrees with the per-cell query.
                Test.@test sort(collect(G.neighbors(conn, k))) == sort(nb)
            end
            Test.@test nseam > 0                     # seams are actually being crossed
        end
        # Only a panel corner loses a neighbour: its diagonal exit belongs to no single panel.
        g = G.CubedSphereGrid(geo, 4)
        short = [k for k in 1:length(g) if C.nneighbors(g, k) < 4]
        Test.@test isempty(short)
    end

    Test.@testset "YinYangGrid stores one panel's measures, which both panels share" begin
        g = G.YinYangGrid(geo, 192, 128)
        Test.@test length(g) == 2 * 192 * 128
        Test.@test Base.summarysize(g) < sizeof(Float64) * 192 * 128 + 1024
    end

    Test.@testset "Yin–Yang coords are the panel formula and its rotation" begin
        for (nlon, nlat) in ((8, 6),)
            g = G.YinYangGrid(geo, nlon, nlat)
            p = SS.spherical_points(SS.YinYangSampling(), nlon, nlat)
            np = nlon * nlat
            Test.@test length(g) == 2np
            for k in eachindex(p.λ)
                c = G.coords(g, k)
                Test.@test c.λ ≈ p.λ[k] atol = 1e-13
                Test.@test c.φ ≈ p.φ[k] atol = 1e-13
                pn, i, j = G.panel_cell(g, k)
                Test.@test pn == (k ≤ np ? 1 : 2)
                Test.@test G.cell_id(g, pn, i, j) == k
            end
            # Yin's own frame is the global one, so its cells are a plain lat–lon patch, each longitude
            # reported in [0, 2π).
            Δλ = (3π / 2) / nlon
            Δφ = (π / 2) / nlat
            for j in 1:nlat, i in 1:nlon
                c = G.coords(g, G.cell_id(g, 1, i, j))
                Test.@test 0 ≤ c.λ < 2π
                Test.@test abs(rem2pi(c.λ - (-3π / 4 + (i - 0.5) * Δλ), RoundNearest)) < 1e-14
                Test.@test c.φ ≈ -π / 4 + (j - 0.5) * Δφ atol = 1e-14
            end
        end
    end

    Test.@testset "A Yin–Yang cell's measure is its area less half its overlap" begin
        # Whether a point given in one panel's frame lies in the other panel: the Kageyama–Sato map
        # `(x, y, z) ↦ (−x, z, y)` to the other frame, and that panel's box there.
        function in_other(λ, φ)
            x, y, z = cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ)
            return abs(atan(z, -x)) ≤ 3π / 4 && abs(asin(y)) ≤ π / 4
        end
        # The overlap of `[λa, λb] × [φa, φb]` on the unit sphere by brute force: the midpoint rule over
        # longitude, and at each longitude the exact `∫cos φ dφ` over the stretches of latitude inside
        # the other panel, their ends found by bisection.
        function overlap_by_scan(λa, λb, φa, φb; nλ = 2000, nφ = 256)
            edge(λ, lo, hi) = (for _ in 1:60
                                   mid = (lo + hi) / 2
                                   in_other(λ, mid) == in_other(λ, lo) ? (lo = mid) : (hi = mid)
                               end; (lo + hi) / 2)
            acc = 0.0
            hλ = (λb - λa) / nλ
            for a in 1:nλ
                λ = λa + (a - 0.5) * hλ
                for b in 1:nφ
                    lo, hi = φa + (b - 1) * (φb - φa) / nφ, φa + b * (φb - φa) / nφ
                    il, ih = in_other(λ, lo), in_other(λ, hi)
                    il && ih && (acc += hλ * (sin(hi) - sin(lo)))
                    il != ih && (c = edge(λ, lo, hi); acc += hλ * (il ? sin(c) - sin(lo) : sin(hi) - sin(c)))
                end
            end
            return acc
        end

        for (nlon, nlat) in ((16, 12), (191, 127))
            g = G.YinYangGrid(geo, nlon, nlat)
            m = G.measure(g)
            # The panels' union is the sphere, counted once, and the sum agrees with the cells added up.
            Test.@test sum(m) ≈ 4π * R^2 rtol = 1e-14
            Test.@test sum(collect(m)) ≈ 4π * R^2 rtol = 1e-13
            Δλ, Δφ = (3π / 2) / nlon, (π / 2) / nlat
            # The band area as `2 cos φ̄ sin(Δφ/2)`, which holds its relative precision where a difference
            # of sines does not.
            area(j) = R^2 * Δλ * 2 * cos(-π / 4 + (j - 0.5) * Δφ) * sin(Δφ / 2)
            Test.@test all(m[G.cell_id(g, 1, i, j)] === m[G.cell_id(g, 2, i, j)] for i in 1:nlon, j in 1:nlat)
            Test.@test all(0 < m[G.cell_id(g, 1, i, j)] ≤ area(j) * (1 + 8eps()) for i in 1:nlon, j in 1:nlat)
        end

        # Cell by cell: a panel corner, the middle of an edge, two cells the overlap's edge crosses, one
        # in the top row, and one in the middle the other panel does not reach. Against the scan, which
        # resolves a sliver of overlap to about `1e-6` of the cell at this width. The closed form, a
        # difference of antiderivatives of size one, against itself at 256 bits on the same cell edges: a
        # few `eps` on the unit sphere at either width. A `Float32` grid's measures against the cells' own at
        # 256 bits: `Float32`'s precision per cell.
        let g16 = G.YinYangGrid(geo, 16, 12), dλ = (3π / 2) / 16, dφ = (π / 2) / 12,
            g32 = G.YinYangGrid(GE.SphericalGeometry(Float32(R)), 16, 12)
            for (i, j) in ((1, 1), (1, 6), (2, 1), (1, 3), (11, 12), (8, 6))
                λa, φa = -3π / 4 + (i - 1) * dλ, -π / 4 + (j - 1) * dφ
                cell = (λa, λa + dλ, φa, φa + dφ)
                a = dλ * (sin(φa + dφ) - sin(φa))
                got = G.measure(g16, G.cell_id(g16, 1, i, j)) / R^2
                Test.@test abs(got - (a - overlap_by_scan(cell...; nλ = 4000) / 2)) < 1e-5 * a
                for T in (Float64, Float32)
                    Test.@test abs(G._yin_yang_overlap(T.(cell)...) - G._yin_yang_overlap(big.(T.(cell))...)) <
                               8eps(T)
                end
                bλ, bφ = (3 * big(π) / 2) / 16, (big(π) / 2) / 12
                bλa, bφa = -3 * big(π) / 4 + (i - 1) * bλ, -big(π) / 4 + (j - 1) * bφ
                want = big(R)^2 * (bλ * (sin(bφa + bφ) - sin(bφa)) -
                                   G._yin_yang_overlap(bλa, bλa + bλ, bφa, bφa + bφ) / 2)
                Test.@test abs(G.measure(g32, G.cell_id(g32, 1, i, j)) - want) ≤ 4eps(Float32) * want
            end
            Test.@test G.measure(g16, G.cell_id(g16, 1, 8, 6)) ≈ R^2 * dλ * sin(dφ) rtol = 1e-14
        end

        # A Float32 grid's measures sum to the sphere too.
        let g32 = G.YinYangGrid(GE.SphericalGeometry(Float32(R)), 48, 32)
            Test.@test sum(collect(G.measure(g32))) ≈ 4π * Float32(R)^2 rtol = 64eps(Float32)
        end

        # `integrate` counts every point of the sphere once: exact for 1, and converging at second order
        # on `∫ (1 + x + z² + xy) dA = 4πR²(1 + 1/3)` in the global frame.
        f(λ, φ) = 1 + cos(φ) * cos(λ) + sin(φ)^2 + cos(φ)^2 * cos(λ) * sin(λ)
        errs = map(((24, 16), (48, 32), (96, 64))) do (nl, nt)
            gg = G.YinYangGrid(geo, nl, nt)
            Test.@test G.integrate(ones(length(gg)), gg) ≈ 4π * R^2 rtol = 1e-13
            fld = [f(G.coords(gg, k)...) for k in 1:length(gg)]
            abs(G.integrate(fld, gg) / (4π * R^2 * 4 / 3) - 1)
        end
        Test.@test errs[1] / errs[2] > 3 && errs[2] / errs[3] > 3
    end

    Test.@testset "Yin–Yang panels are not cross-linked" begin
        g = G.YinYangGrid(geo, 8, 6)
        np = 8 * 6
        conn = C.build_connectivity(g)
        Test.@test C.is_symmetric_adjacency(conn)
        for k in 1:length(g)
            nb = collect(G.neighbors(g, k))
            Test.@test allunique(nb)
            Test.@test !(k in nb)
            # Every neighbour is on this cell's own panel.
            Test.@test all(((k ≤ np) == (j ≤ np)) for j in nb)
            Test.@test 2 ≤ length(nb) ≤ 4           # a panel corner has two, an interior cell four
            Test.@test sort(collect(G.neighbors(conn, k))) == sort(nb)
        end
        # The panel-local graph is the same one the sampling-level builder makes.
        ref = C.build_connectivity(SS.YinYangSampling(), 8, 6)
        for k in 1:length(g)
            Test.@test sort(collect(G.neighbors(ref, k))) == sort(collect(G.neighbors(g, k)))
        end
    end

end

Test.@testset "Cell vertices are kept when asked for, and the measure is the same either way" begin
    GR = FG.Grids
    geo = FG.Geometry.CartesianGeometry()
    n = 60
    X = [1.0i + 0.1sin(0.3j) for i in 1:n, j in 1:n]
    Y = [1.0j + 0.1cos(0.2i) for i in 1:n, j in 1:n]
    m = trues(n, n)

    lean = GR.CurvilinearGrid(geo, X, Y, m)
    full = GR.CurvilinearGrid(geo, X, Y, m; keep_corners = true)

    # The measure is identical whether or not the vertices are kept.
    Test.@test GR.measure(lean) == GR.measure(full)

    Test.@test !GR.has_corners(lean)
    Test.@test GR.has_corners(full)
    Test.@test_throws ArgumentError GR.corners(lean)
    Test.@test size(GR.corners(full, 1)) == (n + 1, n + 1)

    # Supplying them keeps them without asking: they are the caller's own arrays.
    kc = (GR._centers_to_corners(X), GR._centers_to_corners(Y))
    given = GR.CurvilinearGrid(geo, X, Y, m; corners = kc)
    Test.@test GR.has_corners(given)
    Test.@test GR.corners(given, 1) == kc[1]
    Test.@test GR.measure(given) == GR.measure(lean)

    # A grid given its own measure derives no vertices at all, so a single cell is constructible.
    one_cell = GR.CurvilinearGrid(geo, fill(0.0, 1, 1), fill(0.0, 1, 1), fill(2.5, 1, 1),
                                  trues(1, 1))
    Test.@test GR.measure(one_cell, 1, 1) == 2.5
    Test.@test !GR.has_corners(one_cell)
    # …and asking for them on a grid too small to reconstruct them says so.
    Test.@test_throws ArgumentError GR.CurvilinearGrid(geo, fill(0.0, 1, 1), fill(0.0, 1, 1),
                                                       fill(2.5, 1, 1), trues(1, 1);
                                                       keep_corners = true)

    # A node set with no adjacency has `n+1` offsets and no edges.
    un = GR.UnstructuredGrid(geo, (collect(1.0:1.0:500.0), collect(1.0:1.0:500.0)),
                             ones(500), trues(500))
    Test.@test length(GR.neighbor_ptr(un)) == 501
    Test.@test all(isempty(GR.incident_nodes(un, i)) for i in 1:500)
    Test.@test all(FG.Connectivity.nneighbors(un, i) == 0 for i in 1:500)
    Test.@test FG.Connectivity.nedges(FG.Connectivity.build_connectivity(un)) == 0
end

Test.@testset "The icosahedral geodesic is a layout, read out of its own numbering" begin
    G = FG.Grids
    C = FG.Connectivity
    GE = FG.Geometry
    SS = FG.SphericalSampling
    geo = GE.SphericalGeometry()
    R = GE.radius(geo)

    # Vertex `k`'s dual area in BigFloat from its neighbours alone: two mutually adjacent neighbours
    # close a triangle around `k`, and `k`'s share of it is the quadrilateral from `k` through the two
    # edge midpoints and the circumcentre.
    function dual_area_ref(g, k)
        ν = G.frequency(g)
        dir(id) = SS._ico_vertex_dir(BigFloat, id, ν)
        nrm(p) = p ./ sqrt(sum(abs2, p))
        V = dir(k)
        nb = collect(G.neighbors(g, k))
        a = big(0.0)
        for x in nb, y in nb
            (x < y && y in collect(G.neighbors(g, x))) || continue
            A, B = dir(x), dir(y)
            u, w = A .- V, B .- V
            O = nrm((u[2] * w[3] - u[3] * w[2], u[3] * w[1] - u[1] * w[3], u[1] * w[2] - u[2] * w[1]))
            sum(O .* V) < 0 && (O = .-O)
            a += GE.spherical_excess(V, nrm(V .+ A), O) + GE.spherical_excess(V, O, nrm(V .+ B))
        end
        return big(GE.radius(G.grid_geometry(g)))^2 * a
    end

    Test.@testset "storage is one area per class beside the frequency" begin
        # The classes are the partitions of `ν` into three parts, counted by smallest part `c`; the offsets
        # hold one more than there are `c`.
        nclass(ν) = sum((ν - c) ÷ 2 - c + 1 for c in 0:(ν ÷ 3))
        sizes = [Base.summarysize(G.IcosahedralGrid(geo, ν)) - sizeof(Float64) * nclass(ν) -
                 sizeof(Int) * (ν ÷ 3 + 2) for ν in (1, 512)]
        Test.@test allequal(sizes)
        Test.@test length(G.IcosahedralGrid(geo, 512)) == 10 * 512^2 + 2
        Test.@test_throws ArgumentError G.IcosahedralGrid(geo, 0)
    end

    # The mesh builder is the reference for all three: it walks the entities in id order, where the
    # layout reads a single id backwards.
    for ν in (1, 4)
        g = G.IcosahedralGrid(geo, ν)
        mesh = SS.icosahedral_mesh(ν)
        n = length(g)
        Test.@test n == length(mesh.λ)

        # Each check runs over every vertex and reports the vertices that fail it.
        Test.@testset "ν = $ν: coords, adjacency and dual areas" begin
            # Bit-identical: the same expressions on the same entity.
            Test.@test filter(k -> Tuple(G.coords(g, k)) != (mesh.λ[k], mesh.φ[k]), 1:n) == []
            λ, φ = G.materialize(g)
            Test.@test λ == mesh.λ
            Test.@test φ == mesh.φ

            # Adjacency is the mesh's own undirected edge list.
            ref = [Int[] for _ in 1:n]
            for (a, b) in mesh.edges
                push!(ref[a], b)
                push!(ref[b], a)
            end
            nbs = [sort(collect(G.neighbors(g, k))) for k in 1:n]
            Test.@test filter(k -> nbs[k] != sort(unique(ref[k])), 1:n) == []
            Test.@test filter(k -> k in nbs[k] || length(nbs[k]) > G.max_neighbors(g), 1:n) == []
            conn = C.build_connectivity(g)
            Test.@test C.is_symmetric_adjacency(conn)
            Test.@test filter(k -> sort(collect(G.neighbors(conn, k))) != nbs[k], 1:n) == []

            # Exactly twelve pentagons — the base icosahedron's corners — and hexagons elsewhere.
            deg = [C.nneighbors(g, k) for k in 1:n]
            Test.@test count(==(5), deg) == 12
            Test.@test count(==(6), deg) == n - 12
            Test.@test all(k -> C.nneighbors(g, k) == 5, 1:12)

            m = G.measure(g)
            Test.@test filter(k -> !isapprox(m[k], dual_area_ref(g, k); rtol = 16 * eps(Float64)), 1:n) == []
            Test.@test sum(m) ≈ 4π * R^2 rtol = 1e-14
            Test.@test sum(m) ≈ sum(collect(m)) rtol = 1e-12
            # A pentagon is the smallest cell on the mesh.
            ν ≥ 4 && Test.@test maximum(m[k] for k in 1:12) < minimum(m[k] for k in 13:n)

            # Vertices of one symmetry class report one number.
            cls = [G._ico_class(k, ν) for k in 1:n]
            first_of = Dict{eltype(cls),Int}()
            foreach(k -> get!(first_of, cls[k], k), 1:n)
            Test.@test filter(k -> m[k] !== m[first_of[cls[k]]], 1:n) == []
            Test.@test G.measure_array(g) == collect(m)
        end
    end

    Test.@testset "a dual cell keeps the precision of its element type at any frequency" begin
        for W in (Float64, Float32), ν in (1024,)
            g = G.IcosahedralGrid(GE.SphericalGeometry(W(6.371e6)), ν)
            n = length(g)
            # a pentagon, a macro-edge vertex beside it, and two face interiors
            for k in (1, 13, n ÷ 2, n)
                Test.@test G.measure(g, k) ≈ dual_area_ref(g, k) rtol = 16 * eps(W)
            end
        end
    end

end

Test.@testset "A staggered grid is one mesh read at any Arakawa C location" begin
    GD = FG.Grids
    GE = FG.Geometry
    D = FG.Discretization
    C, F = D.Center(), D.Face()
    cart = GE.CartesianGeometry{Float64}()

    # A bounded direction of N cells carries N+1 faces; a wrapping one carries N, its last face
    # being its first, so the seam holds one degree of freedom.
    sg = GD.StaggeredGrid(cart, collect(0.0:1.0:5.0), collect(0.0:2.0:6.0))
    Test.@test GD.size_tuple(GD.center_grid(sg)) == (6, 4)
    Test.@test GD.size_tuple(GD.grid_at(sg, (C, C))) == (6, 4)
    Test.@test GD.size_tuple(GD.grid_at(sg, (F, C))) == (7, 4)
    Test.@test GD.size_tuple(GD.grid_at(sg, (C, F))) == (6, 5)
    Test.@test GD.size_tuple(GD.grid_at(sg, (F, F))) == (7, 5)
    Test.@test GD.size_tuple(GD.grid_at(sg, 1)) == (7, 4)
    Test.@test GD.locations(sg) == ((F, C), (C, F))
    Test.@test GD.center_location(sg) == (C, C)

    let sph = GE.SphericalGeometry(6.371e6),
        λ = collect(range(0, 2π; length = 13)[1:12]),
        φ = collect(range(-1.2, 1.2; length = 9))
        sp = GD.StaggeredGrid(sph, λ, φ; topology = GD.Periodic())
        Test.@test GD.size_tuple(GD.grid_at(sp, (F, C))) == (12, 9)   # wraps: 12 faces, not 13
        Test.@test GD.size_tuple(GD.grid_at(sp, (C, F))) == (12, 10)
        # The staggered grids keep the mesh's topology, so a query on one wraps as the mesh does.
        Test.@test GD.isperiodic(GD.grid_at(sp, (F, C)), 1)
        Test.@test GD.period(GD.grid_at(sp, (F, C)), 1) ≈ 2π
        # A periodic axis's faces are another half a step back, and wrap at the same period.
        sr = GD.StaggeredGrid(sph, FG.Axes.PeriodicAxis(0.0, 2π, 12), φ)
        fr = GD.axis_at(sr, 1, F)
        Test.@test FG.Axes.period(fr) == 2π
        Test.@test collect(fr) ≈ collect(GD.axis_at(sp, 1, F))
        Test.@test GD.isperiodic(GD.grid_at(sr, (F, C)), 1)
    end

    # The face samples sit on the cell boundaries, on a stretched mesh as much as a uniform one.
    let x = [0.0, 1.0, 3.0, 6.0, 10.0], y = collect(0.0:2.0:6.0)
        s = GD.StaggeredGrid(cart, x, y)
        fx = GD.axis_at(s, 1, F)
        Test.@test collect(fx) == collect(D.faces(x))
        Test.@test all(fx[i] == (x[i - 1] + x[i]) / 2 for i in 2:length(x))
        Test.@test fx[1] == x[1] - (x[2] - x[1]) / 2
        Test.@test fx[end] == x[end] + (x[end] - x[end - 1]) / 2
        Test.@test all(GD.coords(GD.grid_at(s, (F, C)), i, j) == (x = fx[i], y = y[j])
                       for i in 1:6, j in 1:4)
    end

    # A uniform mesh stays uniform once staggered.
    for xs in (0.0:0.5:5.0, FG.Axes.UniformAxis(0.0, 0.5, 11), range(0.0, 5.0; length = 11))
        s = GD.StaggeredGrid(cart, xs, 0.0:1.0:3.0)
        Test.@test FG.Axes.isuniform(GD.axis_at(s, 1, F))
        Test.@test FG.Axes.spacing(GD.axis_at(s, 1, F)) ≈ 0.5
        Test.@test GD.isuniform(GD.grid_at(s, (F, C)), 1)
    end

    # A mask is given over the centres, and a staggered point is active where every centre it is
    # built from is — the finite-volume rule, under which a face between an active and an inactive
    # cell is a boundary.
    let mk = trues(5, 4)
        mk[3, 2] = false
        s = GD.StaggeredGrid(GD.StructuredGrid(cart, collect(0.0:1.0:4.0), collect(0.0:1.0:3.0), mk))
        gu, gv, gz = GD.grid_at(s, (F, C)), GD.grid_at(s, (C, F)), GD.grid_at(s, (F, F))
        Test.@test !GD.isactive(gu, 3, 2) && !GD.isactive(gu, 4, 2)   # the two u-faces of that cell
        Test.@test GD.isactive(gu, 2, 2) && GD.isactive(gu, 5, 2)
        Test.@test !GD.isactive(gv, 3, 2) && !GD.isactive(gv, 3, 3)   # its two v-faces
        Test.@test count(!, GD.mask(gu)) == 2 && count(!, GD.mask(gv)) == 2
        Test.@test count(!, GD.mask(gz)) == 4                          # its four corners
    end

    # Three directions, where the velocity locations are U, V and W.
    let s = GD.StaggeredGrid(cart, collect(0.0:1.0:3.0), collect(0.0:1.0:2.0), collect(0.0:1.0:1.0))
        Test.@test GD.size_tuple(GD.grid_at(s, 1)) == (5, 3, 2)
        Test.@test GD.size_tuple(GD.grid_at(s, 2)) == (4, 4, 2)
        Test.@test GD.size_tuple(GD.grid_at(s, 3)) == (4, 3, 3)
        Test.@test GD.locations(s) == ((F, C, C), (C, F, C), (C, C, F))
    end
end

Test.@testset "Span and spacing agree with the axis, and cost one pass to summarise" begin
    GD = FG.Grids
    GE = FG.Geometry
    C = FG.Connectivity
    cart = GE.CartesianGeometry{Float64}()

    # The direct O(n) forms, as a reference for the accessors under test.
    ref_mingap(x) = length(x) < 2 ? Inf : minimum(abs(x[i] - x[i - 1]) for i in 2:length(x))
    ref_maxgap(x) = length(x) < 2 ? 0.0 : maximum(abs(x[i] - x[i - 1]) for i in 2:length(x))

    for a in (collect(0.0:1.0:9.0), 0.0:0.5:5.0, FG.Axes.UniformAxis(0.0, 0.25, 12),
              [0.0, 1.0, 3.0, 6.0, 10.0, 15.0],          # stretched, ascending
              collect(range(6.0, 0.0; length = 9)),      # descending: bounds still order lo ≤ hi
              FG.Axes.PowerAxis(0.0, 100.0, 1.7, 11))    # a formula axis
        g = GD.StructuredGrid(cart, a, collect(0.0:1.0:4.0))
        x = collect(a)
        Test.@test GD.bounds(g, 1) == extrema(x)
        Test.@test GD.extent(g, 1) == extrema(x)[2] - extrema(x)[1]
        Test.@test GD.origin(g, 1) == first(x)
        Test.@test GD.minimum_spacing(g, 1) ≈ ref_mingap(x)
        Test.@test GD.maximum_spacing(g, 1) ≈ ref_maxgap(x)
        Test.@test GD.isuniform(g, 1) == FG.Axes.isuniform(a)
    end

    let n = 512
        c1 = CountingAxis(cumsum(1.0 .+ 0.5 .* sin.(range(0, 3π; length = n))))
        c2 = CountingAxis(cumsum(1.0 .+ 0.5 .* sin.(range(0, 3π; length = 4n))))
        r1 = reads(() -> GD.StructuredGrid(cart, c1, 0.0:1.0:4.0), c1)
        r2 = reads(() -> GD.StructuredGrid(cart, c2, 0.0:1.0:4.0), c2)
        Test.@test r1 ≤ 4n                       # a few passes, never a scan per cell
        Test.@test r2 ≤ 4.5 * r1                 # linear in the axis, so 4× the axis is ~4× the reads
    end

    # Off a rectilinear grid the coordinates are per-cell fields with no order; their extremes and the
    # total measure are taken at construction, so the span and the default topology read none of them.
    let λ = CountingAxis([0.3i for i in 0:99]), φ = CountingAxis([0.2 * sin(3.0i) for i in 0:99]),
        sph = GE.SphericalGeometry(6.371e6), w = 1.0 .+ 0.01 .* (1:100)
        gu = GD.UnstructuredGrid(sph, (λ, φ), w, trues(100))
        Test.@test GD.bounds(gu, 1) == extrema(λ.data) && GD.bounds(gu, 2) == extrema(φ.data)
        Test.@test GD._total_measure(gu) == sum(w)
        Test.@test reads(() -> GD.bounds(gu, 2), φ) == 0
        Test.@test reads(() -> C.MetricTopology(gu), λ) == 0
        mt = C.MetricTopology(gu)
        Test.@test all(mt.span[d] == GD.extent(gu, d) for d in 1:2)
        # Reordering renumbers the nodes, and the extremes and the total are the set's own.
        gr = GD.reorder(gu, reverse(1:100))
        Test.@test GD.bounds(gr, 1) == GD.bounds(gu, 1) && GD._total_measure(gr) ≈ sum(w)
    end
    let n = 6, xm = [i + 0.1j for i in 1:n, j in 1:n], ym = [j - 0.05i for i in 1:n, j in 1:n]
        cg = GD.CurvilinearGrid(cart, xm, ym, trues(n, n); measure = fill(2.0, n, n))
        Test.@test GD.bounds(cg, 1) == extrema(xm)
        Test.@test GD.bounds(cg, 2) == extrema(ym)
        Test.@test GD._total_measure(cg) == 2.0 * n^2
        # A rebuild moves storage and carries both over.
        cr = GD.rebuild(cg, (; mask = GD.AllActive((n, n))))
        Test.@test GD.bounds(cr, 2) == GD.bounds(cg, 2) && GD._total_measure(cr) == GD._total_measure(cg)
    end
end

Test.@testset "The domain length is what the cells cover" begin
    GD = FG.Grids
    cart = FG.Geometry.CartesianGeometry{Float64}()

    # A bounded axis spans from its first face to its last, and the measure integrates over exactly those
    # cells: on a Cartesian grid its total is the product of the domain lengths.
    for a in (0.0:0.5:5.0, [0.0, 1.0, 3.0, 6.0, 10.0, 15.0], collect(range(6.0, 0.0; length = 9)))
        f = FG.Discretization.faces(collect(Float64, a))
        g = GD.StructuredGrid(cart, a, 0.0:1.0:4.0; topology = (GD.Bounded(), GD.Periodic()), period = (0.0, 7.5))
        Test.@test GD.domain_length(g, 1) ≈ abs(f[end] - f[1]) rtol = 1e-14
        Test.@test GD.domain_length(g, 2) == 7.5
        Test.@test sum(GD.measure(g)) ≈ GD.domain_length(g, 1) * GD.domain_length(g, 2) rtol = 1e-14
    end
    Test.@test GD.domain_length(GD.StructuredGrid(cart, 0.0:0.25:2.75), 1) ≈ 12 * 0.25
    Test.@test GD.domain_length(GD.StructuredGrid(cart, [3.0], [0.0, 1.0]), 1) == 1.0

    # A curvilinear grid's cells span its vertices, held or reconstructed.
    n = 8
    X = [0.5 * (i - 0.5) for i in 1:n, j in 1:n]
    Y = [0.25 * (j - 0.5) for i in 1:n, j in 1:n]
    cg = GD.CurvilinearGrid(cart, X, Y, trues(n, n))
    Test.@test GD.domain_length(cg, 1) ≈ 0.5n rtol = 1e-14
    Test.@test GD.domain_length(cg, 2) ≈ 0.25n rtol = 1e-14
    Xc = [0.3 * (i - 1) + 0.1 * (j - 1) for i in 1:(n + 1), j in 1:(n + 1)]
    Yc = [0.2 * (j - 1) for i in 1:(n + 1), j in 1:(n + 1)]
    ck = GD.CurvilinearGrid(cart, (Xc[1:n, 1:n] .+ Xc[2:end, 2:end]) ./ 2, (Yc[1:n, 1:n] .+ Yc[2:end, 2:end]) ./ 2,
                            trues(n, n); corners = (Xc, Yc))
    Test.@test GD.domain_length(ck, 1) ≈ 0.3n + 0.1n rtol = 1e-14
    Test.@test GD.domain_length(ck, 2) ≈ 0.2n rtol = 1e-14

    # A node set on a lattice counts the lattice, in any order.
    xs = [0.5 * i for i in 0:5 for j in 0:3]
    ys = [0.25 * j for i in 0:5 for j in 0:3]
    p = [7, 1, 24, 13, 2, 19, 5, 11, 3, 22, 8, 16, 4, 20, 9, 14, 6, 23, 10, 17, 12, 21, 15, 18]
    lat = GD.UnstructuredGrid(cart, (xs[p], ys[p]), ones(24))
    Test.@test GD.node_counts(lat) == (6, 4)
    Test.@test GD.domain_length(lat, 1) ≈ 6 * 0.5 rtol = 1e-14
    Test.@test GD.domain_length(lat, 2) ≈ 4 * 0.25 rtol = 1e-14
    # Nodes on a line are one row of a lattice.
    row = GD.UnstructuredGrid(cart, (collect(0.0:0.5:2.5), zeros(6)), ones(6))
    Test.@test GD.node_counts(row) == (6, 1)
    Test.@test GD.domain_length(row, 2) == 1.0

    # Off a lattice, the count a uniform density of the nodes gives over each span.
    xr = [4 * ((i * 0.618034) % 1) for i in 1:200]
    yr = [(i * 0.4142136) % 1 for i in 1:200]
    cl = GD.UnstructuredGrid(cart, (xr, yr), ones(200))
    ex, ey = GD.extent(cl, 1), GD.extent(cl, 2)
    ρ = sqrt(200 / (ex * ey))
    Test.@test GD.node_counts(cl) == (round(Int, ex * ρ), round(Int, ey * ρ))
    Test.@test GD.domain_length(cl, 1) ≈ ex * GD.node_counts(cl)[1] / (GD.node_counts(cl)[1] - 1)
    # A wrapping direction takes its period, also in the density.
    cp = GD.UnstructuredGrid(cart, (xr, yr), ones(200); topology = GD.Periodic(), period = (5.0, 0.0))
    Test.@test GD.domain_length(cp, 1) == 5.0
    Test.@test GD.node_counts(cp)[2] == round(Int, ey * sqrt(200 / (5.0 * ey)))

    # A layout with no faces along a bounded direction says so.
    Test.@test_throws ArgumentError GD.domain_length(GD.HEALPixGrid(FG.Geometry.SphericalGeometry(1.0), 2), 2)
end

Test.@testset "Rings, coverage and integrals read the grid's own structure" begin
    GD = FG.Grids
    GE = FG.Geometry
    SS = FG.SphericalSampling
    R = 6.371e6
    geo = GE.SphericalGeometry(R)

    # A structured sphere: a ring per latitude row, longitude fastest.
    g = FG.Connectivity.structured_grid(SS.GaussLegendreSampling(), 12; geometry = geo)
    nlon, nlat = size(g)
    φ = GD.coordinates(g, 2)
    Test.@test GD.nrings(g) == nlat
    Test.@test all(GD.ring_range(g, r) == ((r - 1) * nlon + 1):(r * nlon) for r in 1:nlat)
    Test.@test all(GD.ring_latitude(g, r) == φ[r] for r in 1:nlat)
    Test.@test all(GD.ring_of(g, i) == (i - 1) ÷ nlon + 1 for i in 1:length(g))
    Test.@test reduce(vcat, collect.(GD.eachring(g))) == 1:length(g)
    Test.@test_throws ArgumentError GD.ring_range(g, nlat + 1)
    # In three directions, a ring per latitude and level.
    g3 = GD.StructuredGrid(geo, FG.Axes.PeriodicAxis(0.0, 2π, 8), collect(range(-1.0, 1.0; length = 5)),
                           [R, R + 1e3, R + 2e3])
    Test.@test GD.nrings(g3) == 15
    Test.@test GD.ring_latitude(g3, 7) == GD.coordinates(g3, 2)[2]
    Test.@test GD.ring_range(g3, 15) == 113:120

    # A tabulated ring grid of very uneven rings: its one-comparison lookup is the bisection of the
    # cumulative counts at every point, and so is its measure's.
    let counts = [1, 7, 2, 30, 3, 3, 11, 1, 19], lat = collect(range(1.2, -1.2; length = 9))
        rg = GD.RingGrid(geo, lat, counts, collect(1.0:9.0))
        off = vcat(0, cumsum(counts))
        Test.@test all(GD.ring_of(rg, i) == searchsortedlast(off, i - 1) for i in 1:sum(counts))
        Test.@test collect(GD.measure(rg)) == [Float64(searchsortedlast(off, i - 1)) for i in 1:sum(counts)]
    end

    # HEALPix in RING order: the sampling's ring table, read off the grid.
    hp = GD.HEALPixGrid(geo, 4)
    s = SS.HEALPixSampling(4)
    Test.@test GD.nrings(hp) == 15
    Test.@test all(GD.ring_range(hp, r) == SS.ring_range(s, r) for r in 1:15)
    Test.@test all(GD.ring_latitude(hp, r) ≈ GD.coords(hp, first(GD.ring_range(hp, r))).φ for r in 1:15)
    Test.@test all(GD.ring_of(hp, i) == r for r in 1:15 for i in GD.ring_range(hp, r))
    Test.@test sum(GD.nlon_in_ring(hp, r) for r in 1:15) == GD.npixels(hp)

    # A ring grid: its own tables, and the rule it was built from.
    s8 = SS.OctahedralGaussianSampling(4)
    rg = GD.RingGrid(geo, s8)
    Test.@test all(GD.ring_range(rg, r) == SS.ring_range(s8, r) for r in 1:GD.nrings(rg))
    Test.@test GD.sampling(GD.RingGrid(geo, [0.5, -0.5], [4, 4], [1.0, 1.0])) === nothing

    # Layouts without rings say so.
    cs = GD.CubedSphereGrid(geo, 4)
    Test.@test_throws ArgumentError GD.nrings(cs)
    Test.@test_throws ArgumentError GD.nrings(GD.StructuredGrid(GE.CartesianGeometry(), 0.0:1.0:3.0,
                                                                0.0:1.0:2.0))

    # A pixelization is its recipe.
    Test.@test GD.sampling(hp) === s
    Test.@test GD.sampling(GD.IcosahedralGrid(geo, 3)) === SS.IcosahedralSampling(3)
    Test.@test GD.sampling(cs) === SS.CubedSphereSampling()
    Test.@test GD.sampling(GD.YinYangGrid(geo, 8, 4)) === SS.YinYangSampling()

    # Gauss–Legendre integrates μ² = sin²φ exactly: 4πR²/3 over the sphere. A sum of `n` positive terms
    # is within `n·eps` of its value.
    n = length(g)
    f = [sin(b)^2 for _ in 1:nlon, b in φ]
    Test.@test GD.integrate(f, g) ≈ 4π * R^2 / 3 rtol = n * eps()
    Test.@test GD.integrate(ones(nlon, nlat), g) ≈ 4π * R^2 rtol = n * eps()
    fb = cat(f, 2f, ones(nlon, nlat); dims = 3)
    Test.@test GD.integrate(fb, g) ≈ [4π * R^2 / 3, 8π * R^2 / 3, 4π * R^2] rtol = n * eps()
    Test.@test GD.integrate(f, g; execution = CB.ThreadedBackend()) ≈ GD.integrate(f, g) rtol = n * eps()
    Test.@test GD.sky_fraction(g) == 1

    # Every measure container — separable, per ring, constant, formula, dense — against its own dense
    # form summed at 256 bits, masked.
    masked = (
        FG.Connectivity.structured_grid(SS.GaussLegendreSampling(), 12; geometry = geo,
                                        mask = [isodd(i + j) || j == 1 for i in 1:nlon, j in 1:nlat]),
        GD.HEALPixGrid(geo, 4; mask = [isodd(i) for i in 1:192]),
        GD.RingGrid(geo, s8; mask = [i % 3 != 0 for i in 1:length(rg)]),
        GD.IcosahedralGrid(geo, 3; mask = [i % 4 != 0 for i in 1:92]),
        FG.Connectivity.unstructured_grid(SS.IcosahedralSampling(3); geometry = geo,
                                          mask = [i % 4 != 0 for i in 1:92]),
    )
    for gg in masked
        m = collect(GD.measure(gg))
        act = collect(GD.mask(gg))
        fld = reshape([1.5 + sin(0.37i) for i in 1:length(gg)], size(gg))
        cov = sum(big(m[i]) for i in eachindex(m) if act[i])
        tol = length(gg) * eps()
        Test.@test GD.covered_area(gg) ≈ cov rtol = tol
        Test.@test GD.sky_fraction(gg) ≈ cov / sum(big.(m)) rtol = 2tol
        Test.@test GD.integrate(fld, gg) ≈ sum(big(fld[i]) * big(m[i]) for i in eachindex(m) if act[i]) rtol = tol
        Test.@test GD.integrate(fld, gg; active_only = false) ≈ sum(big.(vec(fld)) .* big.(vec(m))) rtol = tol
    end

    # A spheroid point's direction from the centre, against the embedded point at 256 bits.
    sp = GE.SpheroidGeometry()
    spb = GE.SpheroidGeometry(big(sp.a), big(sp.f))
    for φ0 in (-1.4, -0.3, 0.0, 0.7, π / 2)
        p = GE.geodetic_to_cartesian(spb, (big(0.3), big(φ0)))
        Test.@test GE.geocentric_latitude(sp, φ0) ≈ atan(p.z, hypot(p.x, p.y)) rtol = 2eps() atol = 0
    end
    Test.@test GE.geocentric_latitude(geo, 0.7) == 0.7

    # Per-call accessors and serial reductions allocate nothing.
    for gg in (g, hp, rg, masked[1])
        Test.@test _alloc(GD.ring_structure, gg) == 0
        Test.@test _alloc(GD.nrings, gg) == 0
        Test.@test _alloc(GD.ring_range, gg, 2) == 0
        Test.@test _alloc(GD.ring_latitude, gg, 2) == 0
        Test.@test _alloc(GD.ring_of, gg, 5) == 0
        Test.@test _alloc(GD.eachring, gg) == 0
        Test.@test _alloc(GD.covered_area, gg) == 0
        Test.@test _alloc(GD.sky_fraction, gg) == 0
    end
    Test.@test _alloc(GD.integrate, f, g) == 0
    Test.@test _alloc(GD.integrate!, zeros(3), fb, g) == 0
    Test.@test _alloc(GE.geocentric_latitude, sp, 0.7) == 0
end
