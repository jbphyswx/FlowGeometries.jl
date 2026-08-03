Test.@testset "StructuredGrid preserves uniform spacing" begin
    geom = FG.Geometry.CartesianGeometry()
    x = 0.0:2000.0:20_000.0
    y = 0.0:2000.0:10_000.0
    mask = trues(length(x), length(y))
    grid = FG.Grids.StructuredGrid(geom, x, y, mask)
    # What a range axis is FOR is the guarantee of constant spacing, so that is what is asserted:
    # the property survives, the spacing is exactly the one asked for, and the samples are equal
    # to the input. The concrete container is an implementation detail (`Axes.UniformAxis`, which
    # unlike `StepRangeLen` keeps its arithmetic in the grid's own element type).
    Test.@test FG.Grids.isuniform(grid)
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
    Test.@test FG.Grids.coords(NTuple{2,Float64}, grid, 2, 3) == (2000.0, 4000.0)
    out = zeros(2)
    FG.Grids.coords!(out, grid, 2, 3)
    Test.@test out == [2000.0, 4000.0]
    Test.@test FG.Grids.isactive(grid, 1, 1)
    Test.@test FG.Grids.grid_geometry(grid) === geom
    Test.@test !FG.Grids.isperiodic(grid, 1)  # Cartesian not auto-periodic
end

Test.@testset "StructuredGrid with Vector axes is still structured" begin
    geom = FG.Geometry.CartesianGeometry()
    x = collect(0.0:1.0:4.0)
    y = collect(0.0:1.0:3.0)
    grid = FG.Grids.StructuredGrid(geom, x, y, trues(length(x), length(y)))
    Test.@test grid isa FG.Grids.StructuredGrid
    Test.@test grid.x isa Vector
    Test.@test grid.y isa Vector
    Test.@test FG.Grids.size_tuple(grid) == (5, 4)
end

Test.@testset "Spherical StructuredGrid auto-periodic longitude" begin
    R = 6.371e6
    geom = FG.Geometry.SphericalGeometry(R)
    # Full-circle longitude (closed to within one cell)
    nλ, nφ = 8, 5
    dλ = 2π / nλ
    lon = range(0.0; step = dλ, length = nλ)
    lat = range(-π / 4, π / 4; length = nφ)
    grid = FG.Grids.StructuredGrid(geom, lon, lat, trues(nλ, nφ))
    Test.@test FG.Grids.isperiodic(grid, 1)
    Test.@test !FG.Grids.isperiodic(grid, 2)
    Test.@test FG.Grids.isuniform(grid, 1)
    Test.@test FG.Grids.spacing(grid, 1) ≈ dλ
    Test.@test all(FG.Grids.area(grid, i, j) > 0 for i in 1:nλ, j in 1:nφ)
    p = FG.Grids.coords(grid, 1, 1)
    Test.@test keys(p) == (:λ, :φ)
    Test.@test p.λ ≈ 0.0
    Test.@test p.φ ≈ -π / 4
end

Test.@testset "Coordinate names follow the geometry, never stand in for each other" begin
    cgeom = FG.Geometry.CartesianGeometry()
    sgeom = FG.Geometry.SphericalGeometry(6.371e6)
    xs = collect(0.0:1.0:4.0)
    cgrid = FG.Grids.StructuredGrid(cgeom, xs, xs, trues(5, 5))
    sgrid = FG.Grids.StructuredGrid(sgeom, deg2rad.(xs), deg2rad.(xs), trues(5, 5))

    # A spherical grid holds longitude/latitude and says so; asking it for `x` is an error
    # rather than silently handing back λ.
    Test.@test FG.Grids.coordinate_names(cgrid) == (:x, :y)
    Test.@test FG.Grids.coordinate_names(sgrid) == (:λ, :φ)
    Test.@test cgrid.x === FG.Grids.coordinates(cgrid, 1)
    Test.@test sgrid.λ === FG.Grids.coordinates(sgrid, 1)
    Test.@test sgrid.φ === FG.Grids.coordinates(sgrid, 2)
    Test.@test_throws FieldError sgrid.x
    Test.@test_throws FieldError cgrid.λ
    Test.@test :λ in propertynames(sgrid)
    Test.@test :x in propertynames(cgrid)

    # `axis` is the rectilinear spelling of the same thing; `coordinates` works on every
    # architecture, including ones with no axes at all.
    Test.@test FG.Grids.axis(sgrid, 2) === FG.Grids.coordinates(sgrid, 2)
    Test.@test FG.Grids.coordinates(cgrid) === (FG.Grids.coordinates(cgrid, 1), FG.Grids.coordinates(cgrid, 2))

    nx, ny = 4, 3
    xm = [Float64(i) for i in 1:nx, j in 1:ny]
    ym = [Float64(j) for i in 1:nx, j in 1:ny]
    cv = FG.Grids.CurvilinearGrid(sgeom, deg2rad.(xm), deg2rad.(ym), trues(nx, ny))
    Test.@test FG.Grids.coordinate_names(cv) == (:λ, :φ)
    Test.@test cv.λ === FG.Grids.coordinates(cv, 1)
    Test.@test_throws FieldError cv.x
    Test.@test size(FG.Grids.corners(cv, 1)) == (nx + 1, ny + 1)
    Test.@test keys(FG.Grids.corner_coords(cv, 1, 1)) == (:λ, :φ)

    un = FG.Grids.UnstructuredGrid(sgeom, [0.0, 0.1, 0.2], [0.0, 0.1, 0.0], [1.0, 1.0, 1.0], trues(3))
    Test.@test FG.Grids.coordinate_names(un) == (:λ, :φ)
    Test.@test un.λ == [0.0, 0.1, 0.2]
    Test.@test_throws FieldError un.y
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
    Test.@test FG.Grids.size_tuple(grid) == size(grid)
    # `show` summarizes rather than dumping every coordinate array.
    s = sprint(show, MIME"text/plain"(), grid)
    Test.@test occursin("StructuredGrid{Float64} 5×4", s)
    Test.@test occursin("20 active", s)
    Test.@test count(==('\n'), s) < 10
    Test.@test sprint(show, grid) == "StructuredGrid{Float64}(5×4)"
end

Test.@testset "Cell measure is a separable outer product matching the metric formulas" begin
    cw = FG.Discretization.cell_width
    sgeo = FG.Geometry.SphericalGeometry(6.371e6)
    cgeo = FG.Geometry.CartesianGeometry()
    cgeo3 = FG.Geometry.CartesianGeometry()

    # Spherical area must equal R²cosφ·Δλ·Δφ cell by cell, on a nonuniform grid.
    λ = collect(range(0.0; step = 2π / 12, length = 12))
    φ = cumsum([-1.0, 0.2, 0.5, 0.15, 0.4, 0.3])
    g = FG.Grids.StructuredGrid(sgeo, λ, φ, trues(length(λ), length(φ)))
    λper = FG.Grids.isperiodic(g, 1) ? 2π : nothing
    ref = [FG.Geometry.area_element(sgeo, φ[j], cw(λ, i, λper), cw(φ, j))
           for i in eachindex(λ), j in eachindex(φ)]
    Test.@test FG.Grids.measure(g) ≈ ref rtol = 1e-14

    # Spherical volume must equal r²cosφ·Δλ·Δφ·Δr.
    r = collect(6.30e6:1.0e4:6.34e6)
    g3 = FG.Grids.StructuredGrid(sgeo, λ, φ, r, trues(length(λ), length(φ), length(r)))
    ref3 = [FG.Geometry.volume_element(sgeo, r[k], φ[j], cw(λ, i, λper), cw(φ, j), cw(r, k))
            for i in eachindex(λ), j in eachindex(φ), k in eachindex(r)]
    Test.@test FG.Grids.measure(g3) ≈ ref3 rtol = 1e-14

    # A degenerate angular axis drops the differential it no longer has, rather than substituting
    # a placeholder into the 2-D area formula.
    zonal = FG.Grids.StructuredGrid(sgeo, λ, [0.4], trues(length(λ), 1))
    Test.@test FG.Grids.measure(zonal) ≈ [sgeo.R * cos(0.4) * cw(λ, i, 2π) for i in eachindex(λ), _ in 1:1]
    merid = FG.Grids.StructuredGrid(sgeo, [0.3], φ, trues(1, length(φ)))
    Test.@test FG.Grids.measure(merid) ≈ [sgeo.R * cw(φ, j) for _ in 1:1, j in eachindex(φ)]

    # A whole sphere's cells sum to 4πR².
    n = 200
    λf = collect(range(0.0; step = 2π / n, length = n))
    φf = collect(range(-π / 2 + π / (2n), π / 2 - π / (2n); length = n))
    gf = FG.Grids.StructuredGrid(sgeo, λf, φf, trues(n, n))
    Test.@test sum(FG.Grids.measure(gf)) ≈ 4π * sgeo.R^2 rtol = 1e-4

    # A periodic axis wraps its boundary cell in 3-D exactly as it already did in 2-D.
    xnu = cumsum([0.0, 10.0, 40.0, 15.0, 60.0, 25.0])
    yy = collect(0.0:50.0:100.0)
    zz = collect(0.0:10.0:20.0)
    # A NONUNIFORM periodic Cartesian direction has no period to infer — its samples do not
    # determine the seam gap — so it must be stated rather than taken from whichever gap is first.
    xper = FG.Grids._cartesian_period(xnu)
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(
        cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz)); periodic = true)
    gp = FG.Grids.StructuredGrid(cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz));
                           periodic = true, period = xper)
    gn = FG.Grids.StructuredGrid(cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz)); periodic = false)
    Test.@test FG.Grids.measure(gp) ≈ [cw(xnu, i, xper) * cw(yy, j) * cw(zz, k)
                                 for i in eachindex(xnu), j in eachindex(yy), k in eachindex(zz)]
    Test.@test FG.Grids.measure(gp) != FG.Grids.measure(gn)
    # …and 2-D and 3-D agree on that wrapped width.
    g2p = FG.Grids.StructuredGrid(cgeo, xnu, yy, trues(length(xnu), length(yy));
                            periodic = true, period = xper)
    Test.@test FG.Grids.measure(g2p)[end, 1] / cw(yy, 1) ≈ FG.Grids.measure(gp)[end, 1, 1] / (cw(yy, 1) * cw(zz, 1))
end

Test.@testset "measure is the dimension-agnostic name for area/volume" begin
    geom3 = FG.Geometry.CartesianGeometry()
    x = 0.0:1.0:3.0
    z = 0.0:1.0:2.0
    g3 = FG.Grids.StructuredGrid(geom3, x, x, z, trues(4, 4, 3))
    Test.@test FG.Grids.measure(g3, 2, 2, 2) ≈ 1.0
    Test.@test FG.Grids.measure(g3, 2, 2, 2) == FG.Grids.area(g3, 2, 2, 2)
    Test.@test size(FG.Grids.measure(g3)) == (4, 4, 3)
end

Test.@testset "UnstructuredGrid carries a caller-chosen index type" begin
    geom = FG.Geometry.CartesianGeometry()
    x = [0.0, 1.0, 0.0]
    y = [0.0, 0.0, 1.0]
    areas = [0.5, 0.5, 0.5]
    # Int32 CSR indices: half the memory of Int64 on a large mesh, and the width GPU kernels use.
    nbrs = Int32[2, 3, 1, 1]
    ptr = Int32[1, 3, 4, 5]
    grid = FG.Grids.UnstructuredGrid(geom, x, y, areas, trues(3), nbrs, ptr)
    Test.@test eltype(grid.neighbor_nbrs) === Int32
    Test.@test collect(FG.Grids.neighbors(grid, 1)) == Int32[2, 3]
    Test.@test FG.Grids.measure(grid, 2) ≈ 0.5
    # Mismatched adjacency length is caught at construction.
    Test.@test_throws ArgumentError FG.Grids.UnstructuredGrid(geom, x, y, areas, trues(3), nbrs, Int32[1, 3])
end

Test.@testset "1D and 3D StructuredGrid" begin
    geom = FG.Geometry.CartesianGeometry()
    x = 0.0:1.0:9.0
    g1 = FG.Grids.StructuredGrid(geom, x, trues(length(x)))
    Test.@test FG.Grids.size_tuple(g1) == (10,)
    Test.@test FG.Grids.area(g1, 2) ≈ 1.0

    geom3 = FG.Geometry.CartesianGeometry()
    z = 0.0:1.0:4.0
    g3 = FG.Grids.StructuredGrid(geom3, x, x, z, trues(length(x), length(x), length(z)))
    Test.@test FG.Grids.size_tuple(g3) == (10, 10, 5)
    Test.@test FG.Grids.area(g3, 2, 2, 2) ≈ 1.0
end

Test.@testset "CurvilinearGrid" begin
    geom = FG.Geometry.CartesianGeometry()
    nx, ny = 4, 3
    x = [Float64(i) for i in 1:nx, j in 1:ny]
    y = [Float64(j) for i in 1:nx, j in 1:ny]
    grid = FG.Grids.CurvilinearGrid(geom, x, y, trues(nx, ny))
    Test.@test FG.Grids.size_tuple(grid) == (nx, ny)
    Test.@test FG.Grids.coords(grid, 2, 2) == (x = 2.0, y = 2.0)
    Test.@test FG.Grids.area(grid, 2, 2) > 0
end

Test.@testset "UnstructuredGrid explicit adjacency" begin
    geom = FG.Geometry.CartesianGeometry()
    x = [0.0, 1.0, 0.0]
    y = [0.0, 0.0, 1.0]
    areas = [0.5, 0.5, 0.5]
    mask = trues(3)
    # CSR: node 1 → [2,3], node 2 → [1], node 3 → [1]
    nbrs = [2, 3, 1, 1]
    ptr = [1, 3, 4, 5]
    grid = FG.Grids.UnstructuredGrid(geom, x, y, areas, mask, nbrs, ptr)
    Test.@test FG.Grids.size_tuple(grid) == (3,)
    Test.@test collect(FG.Grids.neighbors(grid, 1)) == [2, 3]
    Test.@test FG.Grids.coords(grid, 2) == (x = 1.0, y = 0.0)

    # Convenience no-neighbor constructor
    g0 = FG.Grids.UnstructuredGrid(geom, x, y, areas, mask)
    Test.@test isempty(FG.Grids.neighbors(g0, 1))
end

Test.@testset "UnstructuredGrid auto-build works with the extensions loaded" begin
    # With the extensions loaded, the auto-build must actually work, not merely throw without them.
    geom = FG.Geometry.CartesianGeometry()
    n = 24
    x = [0.5 + 0.4cos(2π * k / n) for k in 1:n]
    y = [0.5 + 0.4sin(2π * k / n) for k in 1:n]
    g = FG.Grids.UnstructuredGrid(geom, x, y, trues(n); k = 4)
    Test.@test FG.Grids.size_tuple(g) == (n,)
    Test.@test all(1 ≤ length(FG.Grids.neighbors(g, i)) ≤ 4 for i in 1:n)
    Test.@test all(>(0), FG.Grids.measure(g))          # Voronoi areas, via DelaunayTriangulation
    Test.@test all(isfinite, FG.Grids.measure(g))
end

Test.@testset "Curvilinear areas hold two corner rows, not the whole field" begin
    geo = FG.Geometry.SphericalGeometry()
    for n in (5, 32)
        λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
        φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
        g = FG.Grids.CurvilinearGrid(geo, λ, φ, trues(n, n))
        a = FG.Grids.measure(g)
        Test.@test size(a) == (n, n)
        Test.@test all(>(0), a)
        # Reference: the same two triangles per cell, from a full materialized corner field.
        xc, yc = FG.Grids._centers_to_corners(λ), FG.Grids._centers_to_corners(φ)
        dirs = [(cos(yc[i, j]) * cos(xc[i, j]), cos(yc[i, j]) * sin(xc[i, j]), sin(yc[i, j]))
                for i in 1:(n + 1), j in 1:(n + 1)]
        ref = [geo.R^2 * (FG.Geometry.spherical_excess(dirs[i, j], dirs[i + 1, j], dirs[i + 1, j + 1]) +
                          FG.Geometry.spherical_excess(dirs[i, j], dirs[i + 1, j + 1], dirs[i, j + 1]))
               for i in 1:n, j in 1:n]
        Test.@test a == ref     # same arithmetic, only the buffering differs
    end
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
end

Test.@testset "Cell measure is stored factored, not materialized" begin
    geo = FG.Geometry.SphericalGeometry()
    for (nx, ny) in ((16, 9), (64, 40))
        x = collect(range(0, 2π; length = nx))
        y = collect(range(-1.3, 1.3; length = ny))
        g = FG.Grids.StructuredGrid(geo, x, y, FG.Grids.AllActive((nx, ny)))
        m = FG.Grids.measure(g)
        Test.@test m isa FG.Grids.SeparableMeasure
        Test.@test size(m) == (nx, ny) == size(g)
        # Indexes exactly like the dense outer product it replaces — bit-identical, not close.
        wx, wy = FG.Grids.measure_factors(g)
        dense = wx .* transpose(wy)
        Test.@test all(m[i, j] === dense[i, j] for i in 1:nx, j in 1:ny)
        Test.@test FG.Grids.measure_array(g) == dense
        Test.@test collect(m) == dense
        Test.@test FG.Grids.measure(g, 3, 4) === dense[3, 4]
        # sum is ∏ᵈ ∑ᵢ, i.e. O(Nx+Ny); it must still agree with the dense sum to roundoff.
        Test.@test sum(m) ≈ sum(dense) rtol = 1e-14
        Test.@test_throws BoundsError m[nx + 1, 1]
        # Storage is the factors, not the cells.
        Test.@test sizeof(wx) + sizeof(wy) < sizeof(dense)
    end
    # A 1-D grid is separable too: its single factor IS the measure.
    g1 = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), collect(0.0:0.5:5.0), FG.Grids.AllActive((11,)))
    Test.@test FG.Grids.measure(g1) isa FG.Grids.SeparableMeasure
    f1 = FG.Grids.measure_factors(g1)
    Test.@test f1 !== nothing && length(f1) == 1
    Test.@test collect(only(f1)) == collect(FG.Grids.measure(g1))
    Test.@test FG.Grids.measure(g1, 3) ≈ 0.5

    # `show` must not sum every cell to print one line.
    big = FG.Grids.StructuredGrid(geo, collect(range(0, 2π; length = 2000)),
                            collect(range(-1.4, 1.4; length = 1000)), FG.Grids.AllActive((2000, 1000)))
    Test.@test occursin("2000×1000", sprint(show, MIME"text/plain"(), big))
end

Test.@testset "Every architecture can be built without a mask" begin
    GD = FG.Grids
    C = FG.Connectivity
    D = FG.Discretization
    cart = FG.Geometry.CartesianGeometry{Float64}()
    n = 8
    x = [t for t in range(0.0, 7.0; length = n), _ in 1:n]
    y = [u for _ in 1:n, u in range(0.0, 7.0; length = n)]

    # `AllActive` used to be reachable only through `StructuredGrid`, so a curvilinear or node
    # grid with nothing masked still paid for a dense all-true array — storage, plus a load and a
    # branch per cell where `isactive` should fold to a constant.
    g0 = GD.CurvilinearGrid(cart, x, y; measure = fill(1.0, n, n))
    gm = GD.CurvilinearGrid(cart, x, y, trues(n, n); measure = fill(1.0, n, n))
    Test.@test GD.mask(g0) isa GD.AllActive
    Test.@test !(GD.mask(gm) isa GD.AllActive)      # an explicit mask is still stored as given
    Test.@test all(GD.isactive(g0, i, j) == GD.isactive(gm, i, j) for i in 1:n, j in 1:n)

    # The mask is recognised by element type, not position, so the optional positional measure
    # still parses either way.
    let gp = GD.CurvilinearGrid(cart, x, y, fill(2.0, n, n)),
        gpm = GD.CurvilinearGrid(cart, x, y, fill(2.0, n, n), trues(n, n))
        Test.@test GD.mask(gp) isa GD.AllActive && GD.measure(gp, 2, 2) == 2.0
        Test.@test !(GD.mask(gpm) isa GD.AllActive) && GD.measure(gpm, 2, 2) == 2.0
    end

    let nn = 40, xs = 10.0 .* rand(nn), ys = 6.0 .* rand(nn)
        u0 = GD.UnstructuredGrid(cart, (xs, ys); k = 4, areas = ones(nn))
        um = GD.UnstructuredGrid(cart, (xs, ys), trues(nn); k = 4, areas = ones(nn))
        Test.@test GD.mask(u0) isa GD.AllActive
        Test.@test GD.mask(GD.UnstructuredGrid(cart, xs, ys; k = 4, areas = ones(nn))) isa GD.AllActive
        Test.@test all(GD.isactive(u0, i) == GD.isactive(um, i) for i in 1:nn)
    end

    # Everything downstream must be unchanged by which representation the grid holds.
    let f = 2.0 .* x .- 3.0 .* y, g1 = zeros(n, n), g2 = zeros(n, n)
        D.gradient!(g1, g2, f, C.gradient_plan(g0))
        Test.@test maximum(abs.(g1 .- 2.0)) < 1e-10 && maximum(abs.(g2 .+ 3.0)) < 1e-10
        Test.@test C.nneighbors_within(g0, 4, 4; ball = 2.0) ==
                   C.nneighbors_within(gm, 4, 4; ball = 2.0)
        Test.@test abs(D.interpolate(f, g0, (3.3, 2.2); k = 8) - (2 * 3.3 - 3 * 2.2)) < 1e-9
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
    # It is what the sampling constructors reach for, and it is smaller than a BitArray.
    g = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 33)
    Test.@test FG.Grids.mask(g) isa FG.Grids.AllActive
    Test.@test all(FG.Grids.isactive(g, i, j) for i in 1:size(g, 1), j in 1:size(g, 2))
    Test.@test Base.summarysize(FG.Grids.AllActive((4000, 2000))) < Base.summarysize(trues(4000, 2000))
    # An explicit mask still overrides it and still masks.
    mm = trues(FG.SphericalSampling.axes_lengths(FG.SphericalSampling.ClenshawCurtisSampling(), 9).nlon, 9)
    mm[2, 3] = false
    gm = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 9; mask = mm)
    Test.@test !FG.Grids.isactive(gm, 2, 3)
end

Test.@testset "Auto-periodicity does not depend on which way an axis is stored" begin
    # A circle is a circle whichever way it is traversed, and a descending axis is routine.
    sg = FG.Geometry.SphericalGeometry()
    n = 8
    asc = collect(range(0.0; step = 2π / n, length = n))
    for x in (asc, reverse(asc))
        Test.@test FG.Grids.isperiodic(FG.Grids.StructuredGrid(sg, x, [0.0, 0.1], trues(n, 2)), 1)
    end
    # A regional span is not periodic in either orientation.
    reg = collect(range(0.0; step = 0.05, length = n))
    for x in (reg, reverse(reg))
        Test.@test !FG.Grids.isperiodic(FG.Grids.StructuredGrid(sg, x, [0.0, 0.1], trues(n, 2)), 1)
    end
    # The measure follows the periodicity, so a descending full circle must tile the same as an
    # ascending one rather than losing its seam cell.
    φ = collect(range(-1.0, 1.0; length = 5))
    ma = FG.Grids.measure(FG.Grids.StructuredGrid(sg, asc, φ, FG.Grids.AllActive((n, 5))))
    md = FG.Grids.measure(FG.Grids.StructuredGrid(sg, reverse(asc), φ, FG.Grids.AllActive((n, 5))))
    Test.@test sum(ma) ≈ sum(md) rtol = 1e-14
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

    # Scaling, a multiplicative map, and a factor-wise product all keep the product form — so a
    # unit conversion costs bytes rather than the ∏Nᵈ values a dense result would.
    for got in (2.0 .* m, m .* 2.0, m ./ 4.0, abs.(m), abs2.(m), sqrt.(m), inv.(m),
                m .* m, m ./ m)
        Test.@test got isa GR.SeparableMeasure
        Test.@test Base.summarysize(got) < 200
    end
    Test.@test all(≈((2.0 .* m)[I...], 2 * dense[I...]) for I in probes)
    Test.@test all(≈(abs2.(m)[I...], abs2(dense[I...])) for I in probes)
    Test.@test all(≈((m .* m)[I...], dense[I...]^2) for I in probes)
    Test.@test Base.summarysize(dense) > 1_000_000     # what the lazy form avoids

    # The closed-form reductions still apply to the result, so the saving compounds.
    s = 3.0 .* m
    Test.@test sum(s) ≈ 3 * sum(m)
    Test.@test maximum(s) ≈ 3 * maximum(m)
    Test.@test all(extrema(s) .≈ 3 .* extrema(m))
    Test.@test findmax(s)[2] == findmax(m)[2]
    Test.@test all(f -> f isa FG.Axes.ConstantVector, GR.measure_factors(s))

    # Everything else materializes — correct, just dense. A negative scale is deliberately in this
    # group: `findmax`'s per-axis argmax is only valid for non-negative factors.
    for got in (exp.(m), log.(m), m .+ m, m .+ 1.0, -1.0 .* m, -2.0 .* m)
        Test.@test !(got isa GR.SeparableMeasure)
    end
    Test.@test exp.(m)[3, 4] ≈ exp(dense[3, 4])
    Test.@test (-2.0 .* m)[3, 4] ≈ -2 * dense[3, 4]
    Test.@test findmax(-2.0 .* m)[1] ≈ maximum(-2 .* dense)

    Test.@test typeof(similar(m)) == Matrix{Float64}
    d = Array{Float64}(undef, size(m))
    copyto!(d, m)
    Test.@test all(d[I...] == dense[I...] for I in probes)
    Test.@test GR.measure_factors(collect(m)) === nothing
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
    g2 = GR.CurvilinearGrid(geo, X2, Y2, trues(5, 5))
    Test.@test all(≈(0.5), GR.measure(g2))
    Test.@test GR.coords(g2, 2, 3) == (x = 1.0, y = 1.0)
    # The reconstructed corners sit exactly a half-cell outside the outermost centres.
    Test.@test GR.corner_coords(g2, 1, 1) == (x = -0.5, y = -0.25)

    X3 = [x for x in 0.0:1.0:3.0, _ in 1:3, _ in 1:2]
    Y3 = [y for _ in 1:4, y in 0.0:2.0:4.0, _ in 1:2]
    Z3 = [z for _ in 1:4, _ in 1:3, z in 0.0:0.5:0.5]
    # Past 2-D the corner-area kernel does not apply, and the error has to say so rather than
    # quietly producing a number from a 2-D formula.
    Test.@test_throws ArgumentError GR.CurvilinearGrid(geo, X3, Y3, Z3, trues(4, 3, 2))
    vol = fill(1.0 * 2.0 * 0.5, 4, 3, 2)
    g3 = GR.CurvilinearGrid(geo, X3, Y3, Z3, vol, trues(4, 3, 2))
    Test.@test size(g3) == (4, 3, 2) && ndims(g3) == 3
    Test.@test GR.coordinate_names(g3) == (:x, :y, :z)
    Test.@test GR.coords(g3, 2, 3, 2) == (x = 1.0, y = 4.0, z = 0.5)
    Test.@test GR.measure(g3, 2, 2, 1) == 1.0
    Test.@test size(GR.corners(g3, 1)) == (5, 4, 3)
    Test.@test GR.corner_coords(g3, 1, 1, 1) == (x = -0.5, y = -1.0, z = -0.25)

    W = ntuple(d -> [Tuple(ci)[d] * 1.0 for ci in CartesianIndices((3, 3, 2, 2))], 4)
    g4 = GR.CurvilinearGrid(geo, W..., ones(3, 3, 2, 2), trues(3, 3, 2, 2))
    Test.@test size(g4) == (3, 3, 2, 2)
    Test.@test GR.coords(g4, 2, 3, 1, 2) == (x1 = 2.0, x2 = 3.0, x3 = 1.0, x4 = 2.0)

    # The ghost ring is exact for a field linear in each direction, corners included — the
    # property the reconstruction exists to have, in 2-D and beyond.
    L3 = [1.5i - 2.0j + 0.5k for i in 1:4, j in 1:5, k in 1:3]
    Test.@test all(FG.Grids._ghosted(L3, (i, j, k)) ≈ 1.5i - 2.0j + 0.5k
                   for i in 0:5, j in 0:6, k in 0:4)
    # …and reconstruction costs only its own output. Minimum of several samples: a single
    # `@allocated` can land on a collection or on the tail of compilation, and the true cost is a
    # floor, so the minimum converges to it.
    cc(A) = (FG.Grids._centers_to_corners(A);
             minimum(@allocated(FG.Grids._centers_to_corners(A)) for _ in 1:3))
    big = [1.0i + 2.0j for i in 1:120, j in 1:120]
    Test.@test cc(big) < 1.2 * 8 * 121^2

    # Connectivity follows into N-D, and the dense adjacency agrees with the CSR.
    Test.@test C.nneighbors(g3, 2, 2, 1) == 5          # 6 face neighbours minus one wall
    Test.@test C.linear_index(g3, 2, 3, 2) == 2 + 2 * 4 + 1 * 12
    Test.@test length(collect(GR.neighbors(g3, 2, 2, 1))) == 5
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
                            periodic = (true, false, false), period = (4.0, 0.0, 0.0))
    Test.@test GR.isperiodic(gp, 1) && !GR.isperiodic(gp, 2)
    Test.@test GR.period(gp, 1) == 4.0
    Test.@test C.nneighbors(gp, 1, 2, 1) == 5

    # Node grids: coordinates as a tuple, since a run of vectors cannot say how many are
    # coordinates (a CurvilinearGrid counts them from `ndims(mask)`; a node set has no such handle).
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

    # A k-d tree is dimension-agnostic, so the neighbour search generalizes even though the
    # Voronoi control volumes behind the 2-D measure do not.
    nn = 40
    px, py, pz, pw = rand(nn), rand(nn), rand(nn), rand(nn)
    bruteknn(pts, i, kk) = sort([t[2] for t in sort(
        [(sqrt(sum((pts[q][i] - pts[q][j])^2 for q in eachindex(pts))), j)
         for j in 1:nn if j != i])[1:kk]])
    nbrs3, ptr3 = GR._build_kdtree_neighbors(geo, (px, py, pz); k = 4)
    Test.@test all(sort(nbrs3[ptr3[i]:(ptr3[i + 1] - 1)]) == bruteknn((px, py, pz), i, 4)
                   for i in 1:nn)
    nbrs4, ptr4 = GR._build_kdtree_neighbors(geo, (px, py, pz, pw); k = 3)
    Test.@test all(sort(nbrs4[ptr4[i]:(ptr4[i + 1] - 1)]) == bruteknn((px, py, pz, pw), i, 3)
                   for i in 1:nn)
    # Periodic ghost images are placed per direction, so a face node finds the opposite face.
    fx, fy, fz = [0.02, 0.98, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    nbw, ptw = GR._build_kdtree_neighbors(geo, (fx, fy, fz); k = 1,
                                          periodic = (true, false, false),
                                          period = (1.0, 0.0, 0.0))
    Test.@test nbw[ptw[1]] == 2
    nbb, ptb = GR._build_kdtree_neighbors(geo, (fx, fy, fz); k = 1)
    Test.@test nbb[ptb[1]] == 3
    # `(λ, φ, r)` embeds at its own radius, so a radius query there is a true chord.
    sph1 = FG.Geometry.SphericalGeometry(1.0)
    sλ, sφ, sr = [0.0, 0.0], [0.0, 0.0], [1.0, 1.5]
    Test.@test GR._build_kdtree_neighbors(sph1, (sλ, sφ, sr); k = 1, radius = 0.6)[2][2] == 2
    Test.@test GR._build_kdtree_neighbors(sph1, (sλ, sφ, sr); k = 1, radius = 0.4)[2][2] == 1
    Test.@test_throws ArgumentError GR._build_kdtree_neighbors(sph1, (sλ, sφ, sr, sr); k = 1)
    g3kd = GR.UnstructuredGrid(geo, (px, py, pz), trues(nn); k = 4, areas = ones(nn))
    Test.@test length(collect(GR.neighbors(g3kd, 1))) == 4
    Test.@test_throws ArgumentError GR.UnstructuredGrid(geo, (px, py, pz), trues(nn); k = 4)
end

Test.@testset "Grids and connectivity work in any number of dimensions" begin
    geo = FG.Geometry.CartesianGeometry()
    # A 4-D and a 5-D rectilinear grid, built through the one varargs constructor.
    g4 = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:3.0, 0.0:1.0:3.0)
    Test.@test size(g4) == (4, 4, 4, 4)
    Test.@test FG.Grids.mask(g4) isa FG.Grids.AllActive          # no mask argument needed
    Test.@test ndims(g4) == 4
    Test.@test FG.Grids.measure(g4, 2, 2, 2, 2) ≈ 1.0
    Test.@test sum(FG.Grids.measure(g4)) ≈ 4.0^4 rtol = 1e-12
    Test.@test FG.Grids.coordinate_names(g4) == (:x1, :x2, :x3, :x4)
    Test.@test FG.Grids.coords(g4, 2, 3, 1, 4) == (x1 = 1.0, x2 = 2.0, x3 = 0.0, x4 = 3.0)
    Test.@test FG.Connectivity.nneighbors(g4, 2, 2, 2, 2) == 8      # 2N faces
    Test.@test FG.Connectivity.nneighbors(g4, 2, 2, 2, 2; stencil = FG.Stencils.Moore(1)) == 80  # 3^4-1
    Test.@test FG.Connectivity.nedges(FG.Connectivity.build_connectivity(g4)) > 0

    g5 = FG.Grids.StructuredGrid(geo, ntuple(_ -> 0.0:1.0:2.0, 5)...)
    Test.@test size(g5) == (3, 3, 3, 3, 3)
    Test.@test FG.Connectivity.nneighbors(g5, 2, 2, 2, 2, 2) == 10

    # A traversal must stay allocation-free past N = 3 too. Coordinate names are numbered rather
    # than lettered from N = 4 on, and a per-cell query reaches them; building those symbols at run
    # time would allocate on every call, so this covers more than the loop itself.
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
    Test.@test names7() == (:λ, :φ, :r, :q4, :q5, :q6, :q7)

    # A spherical grid in 1-D measures arc length, and in 4-D keeps its metric directions.
    sgeo = FG.Geometry.SphericalGeometry(2.0)
    g1 = FG.Grids.StructuredGrid(sgeo, collect(range(0, 2π; length = 9)))
    Test.@test size(g1) == (9,)
    Test.@test FG.Grids.measure_factors(g1) !== nothing
    Test.@test sum(FG.Grids.measure(g1)) ≈ 2.0 * 2π rtol = 1e-2   # R·Δλ over the full circle
    g4s = FG.Grids.StructuredGrid(sgeo, collect(range(0, 2π; length = 8)),
                            collect(range(-1.0, 1.0; length = 6)),
                            collect(range(1.0, 2.0; length = 4)),
                            collect(range(0.0, 1.0; length = 3)))
    Test.@test FG.Grids.coordinate_names(g4s) == (:λ, :φ, :r, :q4)
    Test.@test all(>(0), FG.Grids.measure(g4s))

    # Topology lives in the type, and the mask defaults to all-active.
    Test.@test FG.Grids.topology(g4) == ntuple(_ -> FG.Grids.Bounded(), 4)
    gp = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0;
                           topology = (FG.Grids.Periodic(), FG.Grids.Bounded()))
    Test.@test FG.Grids.topology(gp, 1) === FG.Grids.Periodic()
    Test.@test FG.Grids.isperiodic(gp, 1) && !FG.Grids.isperiodic(gp, 2)
    Test.@test FG.Grids.periodic_flags(gp) == (true, false)
    Test.@test FG.Grids.period(gp, 1) ≈ 4.0            # uniform closure: n·|Δ|
    Test.@test FG.Grids.period(gp, 2) == 0.0           # bounded: no period
    # `periodic = true` still means "direction 1 wraps".
    gq = FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0; periodic = true)
    Test.@test FG.Grids.isperiodic(gq, 1) && !FG.Grids.isperiodic(gq, 2)
    # A mask may be given positionally or by keyword, but not both.
    m = trues(4, 4); m[2, 2] = false
    Test.@test !FG.Grids.isactive(FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, m), 2, 2)
    Test.@test !FG.Grids.isactive(FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0; mask = m), 2, 2)
    Test.@test_throws ArgumentError FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, m; mask = m)
    Test.@test_throws DimensionMismatch FG.Grids.StructuredGrid(geo, 0.0:1.0:3.0, 0.0:1.0:3.0, trues(3, 3))
end

Test.@testset "Mask topology: interior, boundary, components and holes" begin
    C = FG.Connectivity
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
    # Interior and boundary partition the active cells.
    int = C.interior(g)
    bnd = C.boundary_cells(g)
    Test.@test count(int) + count(bnd) == count(m)
    Test.@test !any(int .& bnd)
    Test.@test all(!int[ci] for ci in CartesianIndices(m) if !m[ci])
    Test.@test !int[1, 5]                       # a domain edge is not interior
    Test.@test int[3, 3]

    # Wrapping removes the edges, so a region that only touched one becomes enclosed.
    gp = FG.Grids.StructuredGrid(geo, 0.0:1.0:8.0, 0.0:1.0:8.0, m;
                           periodic = (true, true), period = (9.0, 9.0))
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

Test.@testset "A curvilinear measure may be given positionally or by keyword" begin
    # The docstring offers both forms, so both must build the same grid.
    geo = FG.Geometry.CartesianGeometry{Float64}()
    xs = [x for x in range(0.0, 3.0; length = 4), _ in 1:4]
    ys = [y for _ in 1:4, y in range(0.0, 3.0; length = 4)]
    a = fill(2.5, 4, 4)
    bykw = FG.Grids.CurvilinearGrid(geo, xs, ys, trues(4, 4); measure = a)
    bypos = FG.Grids.CurvilinearGrid(geo, xs, ys, a, trues(4, 4))
    Test.@test FG.Grids.measure(bykw) == FG.Grids.measure(bypos) == a
end
