using FlowGeometries: FlowGeometries
using Test: Test

# Load every weak dependency up front so the extensions are exercised by the WHOLE suite. They were
# previously only loaded inside individual testsets — or not at all — which is exactly why the
# Quickhull and DelaunayTriangulation extensions could both be dead on arrival unnoticed.
using DelaunayTriangulation: DelaunayTriangulation
using NearestNeighbors: NearestNeighbors
using Quickhull: Quickhull
using SparseArrays: SparseArrays
using StaticArrays: StaticArrays

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

    Test.@testset "Spherical geometry + Cartesian" begin
        geom = FG.SphericalGeometry(6.371e6)
        london = (deg2rad(-0.1276), deg2rad(51.5074))
        paris = (deg2rad(2.3522), deg2rad(48.8566))
        d_km = FG.distance(geom, london, paris) / 1000.0
        Test.@test 300 < d_km < 400

        λ, φ = 0.3, 0.4
        uλ, uφ = 1.0, -0.5
        p = FG.vector_to_cartesian(geom, uλ, uφ, λ, φ)
        back = FG.vector_from_cartesian(geom, p, λ, φ)
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
        Test.@test grid.λ isa AbstractRange
        Test.@test all(FG.area(grid, i, j) > 0 for i in 1:nλ, j in 1:nφ)
        p = FG.coords(grid, 1, 1)
        Test.@test keys(p) == (:λ, :φ)
        Test.@test p.λ ≈ 0.0
        Test.@test p.φ ≈ -π / 4
    end

    Test.@testset "Coordinate names follow the geometry, never stand in for each other" begin
        cgeom = FG.CartesianGeometry(1.0, 1.0)
        sgeom = FG.SphericalGeometry(6.371e6)
        xs = collect(0.0:1.0:4.0)
        cgrid = FG.StructuredGrid(cgeom, xs, xs, trues(5, 5))
        sgrid = FG.StructuredGrid(sgeom, deg2rad.(xs), deg2rad.(xs), trues(5, 5))

        # A spherical grid holds longitude/latitude and says so; asking it for `x` is an error
        # rather than silently handing back λ.
        Test.@test FG.coordinate_names(cgrid) == (:x, :y)
        Test.@test FG.coordinate_names(sgrid) == (:λ, :φ)
        Test.@test cgrid.x === FG.coordinates(cgrid, 1)
        Test.@test sgrid.λ === FG.coordinates(sgrid, 1)
        Test.@test sgrid.φ === FG.coordinates(sgrid, 2)
        Test.@test_throws FieldError sgrid.x
        Test.@test_throws FieldError cgrid.λ
        Test.@test :λ in propertynames(sgrid)
        Test.@test :x in propertynames(cgrid)

        # `axis` is the rectilinear spelling of the same thing; `coordinates` works on every
        # architecture, including ones with no axes at all.
        Test.@test FG.axis(sgrid, 2) === FG.coordinates(sgrid, 2)
        Test.@test FG.coordinates(cgrid) === (FG.coordinates(cgrid, 1), FG.coordinates(cgrid, 2))

        nx, ny = 4, 3
        xm = [Float64(i) for i in 1:nx, j in 1:ny]
        ym = [Float64(j) for i in 1:nx, j in 1:ny]
        cv = FG.CurvilinearGrid(sgeom, deg2rad.(xm), deg2rad.(ym), trues(nx, ny))
        Test.@test FG.coordinate_names(cv) == (:λ, :φ)
        Test.@test cv.λ === FG.coordinates(cv, 1)
        Test.@test_throws FieldError cv.x
        Test.@test size(FG.corners(cv, 1)) == (nx + 1, ny + 1)
        Test.@test keys(FG.corner_coords(cv, 1, 1)) == (:λ, :φ)

        un = FG.UnstructuredGrid(sgeom, [0.0, 0.1, 0.2], [0.0, 0.1, 0.0], [1.0, 1.0, 1.0], trues(3))
        Test.@test FG.coordinate_names(un) == (:λ, :φ)
        Test.@test un.λ == [0.0, 0.1, 0.2]
        Test.@test_throws FieldError un.y
    end

    Test.@testset "Grids implement the Base collection surface" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        xs = collect(0.0:1.0:4.0)
        ys = collect(0.0:1.0:3.0)
        grid = FG.StructuredGrid(geom, xs, ys, trues(5, 4))
        Test.@test size(grid) == (5, 4)
        Test.@test size(grid, 2) == 4
        Test.@test length(grid) == 20
        Test.@test ndims(grid) == 2
        Test.@test eltype(grid) === Float64
        Test.@test axes(grid) == (Base.OneTo(5), Base.OneTo(4))
        Test.@test FG.size_tuple(grid) == size(grid)
        # `show` summarizes rather than dumping every coordinate array.
        s = sprint(show, MIME"text/plain"(), grid)
        Test.@test occursin("StructuredGrid{Float64} 5×4", s)
        Test.@test occursin("20 active", s)
        Test.@test count(==('\n'), s) < 10
        Test.@test sprint(show, grid) == "StructuredGrid{Float64}(5×4)"
    end

    Test.@testset "Cell measure is a separable outer product matching the metric formulas" begin
        cw = FG.Grids._cell_width
        sgeo = FG.SphericalGeometry(6.371e6)
        cgeo = FG.CartesianGeometry(1.0, 1.0)
        cgeo3 = FG.CartesianGeometry(1.0, 1.0, 1.0)

        # Spherical area must equal R²cosφ·Δλ·Δφ cell by cell, on a nonuniform grid.
        λ = collect(range(0.0; step = 2π / 12, length = 12))
        φ = cumsum([-1.0, 0.2, 0.5, 0.15, 0.4, 0.3])
        g = FG.StructuredGrid(sgeo, λ, φ, trues(length(λ), length(φ)))
        λper = FG.isperiodic(g, 1) ? 2π : nothing
        ref = [FG.area_element(sgeo, φ[j], cw(λ, i, λper), cw(φ, j))
               for i in eachindex(λ), j in eachindex(φ)]
        Test.@test FG.measure(g) ≈ ref rtol = 1e-14

        # Spherical volume must equal r²cosφ·Δλ·Δφ·Δr.
        r = collect(6.30e6:1.0e4:6.34e6)
        g3 = FG.StructuredGrid(sgeo, λ, φ, r, trues(length(λ), length(φ), length(r)))
        ref3 = [FG.volume_element(sgeo, r[k], φ[j], cw(λ, i, λper), cw(φ, j), cw(r, k))
                for i in eachindex(λ), j in eachindex(φ), k in eachindex(r)]
        Test.@test FG.measure(g3) ≈ ref3 rtol = 1e-14

        # Degenerate angular axes drop the differential that no longer exists rather than
        # substituting a placeholder into the 2-D area formula.
        zonal = FG.StructuredGrid(sgeo, λ, [0.4], trues(length(λ), 1))
        Test.@test FG.measure(zonal) ≈ [sgeo.R * cos(0.4) * cw(λ, i, 2π) for i in eachindex(λ), _ in 1:1]
        merid = FG.StructuredGrid(sgeo, [0.3], φ, trues(1, length(φ)))
        Test.@test FG.measure(merid) ≈ [sgeo.R * cw(φ, j) for _ in 1:1, j in eachindex(φ)]

        # A whole sphere's cells sum to 4πR².
        n = 200
        λf = collect(range(0.0; step = 2π / n, length = n))
        φf = collect(range(-π / 2 + π / (2n), π / 2 - π / (2n); length = n))
        gf = FG.StructuredGrid(sgeo, λf, φf, trues(n, n))
        Test.@test sum(FG.measure(gf)) ≈ 4π * sgeo.R^2 rtol = 1e-4

        # A periodic axis wraps its boundary cell in 3-D exactly as it already did in 2-D.
        xnu = cumsum([0.0, 10.0, 40.0, 15.0, 60.0, 25.0])
        yy = collect(0.0:50.0:100.0)
        zz = collect(0.0:10.0:20.0)
        gp = FG.StructuredGrid(cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz)); periodic = true)
        gn = FG.StructuredGrid(cgeo3, xnu, yy, zz, trues(length(xnu), length(yy), length(zz)); periodic = false)
        xper = FG.Grids._cartesian_period(xnu)
        Test.@test FG.measure(gp) ≈ [cw(xnu, i, xper) * cw(yy, j) * cw(zz, k)
                                     for i in eachindex(xnu), j in eachindex(yy), k in eachindex(zz)]
        Test.@test FG.measure(gp) != FG.measure(gn)
        # …and 2-D and 3-D agree on that wrapped width.
        g2p = FG.StructuredGrid(cgeo, xnu, yy, trues(length(xnu), length(yy)); periodic = true)
        Test.@test FG.measure(g2p)[end, 1] / cw(yy, 1) ≈ FG.measure(gp)[end, 1, 1] / (cw(yy, 1) * cw(zz, 1))
    end

    Test.@testset "measure is the dimension-agnostic name for area/volume" begin
        geom3 = FG.CartesianGeometry(1.0, 1.0, 1.0)
        x = 0.0:1.0:3.0
        z = 0.0:1.0:2.0
        g3 = FG.StructuredGrid(geom3, x, x, z, trues(4, 4, 3))
        Test.@test FG.measure(g3, 2, 2, 2) ≈ 1.0
        Test.@test FG.measure(g3, 2, 2, 2) == FG.area(g3, 2, 2, 2)
        Test.@test size(FG.measure(g3)) == (4, 4, 3)
    end

    Test.@testset "Axis eltype conversion preserves the container type" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        # A Float32 Vector into a Float64 geometry must copy — but into the same kind of container,
        # not unconditionally into a plain `Vector`.
        x32 = Float32[0, 1, 2, 3]
        grid = FG.StructuredGrid(geom, x32, x32, trues(4, 4))
        Test.@test FG.coordinates(grid, 1) isa Vector{Float64}
        # A Range keeps its Range-ness (its type is the proof of uniform spacing) across conversion.
        r32 = Float32(0):Float32(1):Float32(3)
        gridr = FG.StructuredGrid(geom, r32, r32, trues(4, 4))
        Test.@test FG.coordinates(gridr, 1) isa AbstractRange
        Test.@test eltype(FG.coordinates(gridr, 1)) === Float64
    end

    Test.@testset "Axis widths use bulk operations, not per-element scalar reads" begin
        # An array type that counts scalar `getindex` calls. Device arrays make scalar indexing an
        # error precisely because it is a per-element round trip; the property that matters is that
        # the O(n) work is done in bulk, so the scalar count must not grow with n.
        mutable struct CountingVector{T} <: AbstractVector{T}
            data::Vector{T}
            scalar_reads::Int
        end
        CountingVector(v::Vector{T}) where {T} = CountingVector{T}(v, 0)
        Base.size(c::CountingVector) = size(c.data)
        Base.IndexStyle(::Type{<:CountingVector}) = IndexLinear()
        function Base.getindex(c::CountingVector, i::Int)
            c.scalar_reads += 1
            return c.data[i]
        end
        Base.setindex!(c::CountingVector, v, i::Int) = (c.data[i] = v)
        Base.similar(c::CountingVector, ::Type{S}, dims::Dims) where {S} =
            CountingVector{S}(Vector{S}(undef, dims...), 0)
        # Views of a CountingVector go through the parent's getindex, so bulk broadcasts over
        # `@view` still register — count them via the parent after the fact.

        for n in (50, 500, 5000)
            c = CountingVector(collect(range(0.0; step = 1.5, length = n)))
            w = FG.Grids._axis_widths(c)
            Test.@test collect(w) ≈ [FG.Grids._cell_width(c.data, i) for i in 1:n]
            # Bulk broadcasts do touch elements, but the count must be O(n) with a small constant
            # and never O(n) per output element.
            Test.@test c.scalar_reads <= 4n + 16
        end

        # The endpoint handling is the only genuinely scalar part, and it is O(1).
        n = 4000
        c1 = CountingVector(collect(range(0.0; step = 1.0, length = n)))
        c2 = CountingVector(collect(range(0.0; step = 1.0, length = 2n)))
        FG.Grids._axis_widths(c1)
        FG.Grids._axis_widths(c2)
        Test.@test c2.scalar_reads - c1.scalar_reads <= 4n + 16   # grows linearly, not quadratically
    end

    Test.@testset "UnstructuredGrid carries a caller-chosen index type" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        x = [0.0, 1.0, 0.0]
        y = [0.0, 0.0, 1.0]
        areas = [0.5, 0.5, 0.5]
        # Int32 CSR indices: half the memory of Int64 on a large mesh, and the width GPU kernels use.
        nbrs = Int32[2, 3, 1, 1]
        ptr = Int32[1, 3, 4, 5]
        grid = FG.UnstructuredGrid(geom, x, y, areas, trues(3), nbrs, ptr)
        Test.@test eltype(grid.neighbor_nbrs) === Int32
        Test.@test collect(FG.neighbors(grid, 1)) == Int32[2, 3]
        Test.@test FG.measure(grid, 2) ≈ 0.5
        # Mismatched adjacency length is caught at construction.
        Test.@test_throws ArgumentError FG.UnstructuredGrid(geom, x, y, areas, trues(3), nbrs, Int32[1, 3])
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

    Test.@testset "UnstructuredGrid auto-build works with the extensions loaded" begin
        # This previously asserted only that the auto-build THROWS when the extensions are absent,
        # which is why nobody noticed the k-d-tree path was dead on arrival even when they were
        # present. With every weakdep loaded at the top of this file, assert it actually works.
        geom = FG.CartesianGeometry(1.0, 1.0)
        n = 24
        x = [0.5 + 0.4cos(2π * k / n) for k in 1:n]
        y = [0.5 + 0.4sin(2π * k / n) for k in 1:n]
        g = FG.UnstructuredGrid(geom, x, y, trues(n); k = 4)
        Test.@test FG.size_tuple(g) == (n,)
        Test.@test all(1 ≤ length(FG.neighbors(g, i)) ≤ 4 for i in 1:n)
        Test.@test all(>(0), FG.measure(g))          # Voronoi areas, via DelaunayTriangulation
        Test.@test all(isfinite, FG.measure(g))
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

    Test.@testset "Gauss–Legendre solves once for axes and weights, and is exact to 2n-1" begin
        for n in (1, 2, 3, 7, 8, 64, 257)
            r = FG.SphericalSampling._gauss_legendre_μ(n)
            Test.@test issorted(r.μ)                       # ascending, as the eigen route was
            Test.@test r.μ ≈ -reverse(r.μ) atol = 1e-15    # roots come in ± pairs
            Test.@test all(>(0), r.w)
            Test.@test sum(r.w) ≈ 2 rtol = 1e-14
            # An n-point rule is exact for every polynomial up to degree 2n-1.
            for d in (0, 1, 2n - 2, 2n - 1)
                exact = isodd(d) ? 0.0 : 2 / (d + 1)
                Test.@test sum(r.w .* r.μ .^ d) ≈ exact atol = 1e-11
            end
        end
        # Odd n has its central node exactly at zero (and not at -0.0).
        Test.@test FG.SphericalSampling._gauss_legendre_μ(9).μ[5] === 0.0

        # Two regimes with a crossover at n = 60: the asymptotic expansion is a fixed-order Float64
        # coefficient set, poor for small n and unable to exceed Float64 precision; Newton is exact
        # arithmetic but O(n²). Each is used only where it is the better answer.
        SS = FG.SphericalSampling
        for n in (60, 61, 62)
            r = SS._gauss_legendre_μ(n)
            Test.@test issorted(r.μ)
            Test.@test sum(r.w) ≈ 2 rtol = 1e-14
            d = 2n - 2
            Test.@test sum(r.w .* r.μ .^ d) ≈ 2 / (d + 1) rtol = 1e-11
        end
        # Both regimes solve the same problem, so above the crossover they must still agree — to the
        # Newton's accuracy, which is the looser of the two there.
        for n in (64, 200, 1024)
            m = (n + 1) ÷ 2
            asy = SS._gauss_legendre_μ(n)                      # takes the asymptotic path
            mμ = Vector{Float64}(undef, n); mw = Vector{Float64}(undef, n)
            SS._gauss_legendre_newton!(Float64, Float64, mμ, mw, n, m)
            Test.@test maximum(abs.(asy.μ .- mμ)) < 1e-13
            Test.@test maximum(abs.(asy.w .- mw) ./ mw) < 1e-9
            Test.@test maximum(abs.(asy.μ .+ reverse(asy.μ))) < 1e-15   # ± symmetric
            Test.@test sum(mw) ≈ 2 rtol = 1e-13
        end
        # Each output stays independently optional on the asymptotic path too.
        Test.@test SS._gauss_legendre_asy!(Float64, nothing, nothing, 128, 64) === nothing
        # A wider element type must NOT take the Float64 coefficient path — it would cap precision.
        μb, wb = setprecision(BigFloat, 192) do
            μ = Vector{BigFloat}(undef, 128); w = Vector{BigFloat}(undef, 128)
            SS._gauss_legendre_μ!(μ, w); (μ, w)
        end
        Test.@test abs(sum(wb) - 2) < 1e-40        # far past what Float64 coefficients could give
        Test.@test eltype(wb) === BigFloat
        # …and the Float64 result agrees with that reference to Float64 precision.
        r64 = SS._gauss_legendre_μ(128)
        Test.@test maximum(abs.(r64.w .- Float64.(wb)) ./ Float64.(wb)) < 1e-14
        Test.@test maximum(abs.(r64.μ .- Float64.(μb))) < 1e-15

        # The Bonnet recurrence sums n terms, so Float32 cannot resolve the roots on its own; the
        # solve runs at Float64 and rounds once, keeping the result correctly rounded.
        r32 = FG.SphericalSampling._gauss_legendre_μ(64; T = Float32)
        r64 = FG.SphericalSampling._gauss_legendre_μ(64)
        Test.@test eltype(r32.μ) === Float32
        Test.@test maximum(abs.(Float64.(r32.w) .- r64.w) ./ r64.w) < 4 * eps(Float32)
        Test.@test maximum(abs.(Float64.(r32.μ) .- r64.μ)) < 4 * eps(Float32)

        # One solve serves both, and agrees with the two separate entry points.
        for n in (5, 32)
            q = FG.spherical_quadrature(FG.GaussLegendreSampling(), n)
            a = FG.spherical_axes(FG.GaussLegendreSampling(), n)
            Test.@test q.λ == a.λ
            Test.@test q.φ == a.φ
            Test.@test q.w == FG.latitude_weights(FG.GaussLegendreSampling(), n)
        end
        # Non-GL samplings have independent closed forms and take the generic method.
        for s in (FG.DriscollHealySampling(), FG.DriscollHealyEqualSampling(), FG.ClenshawCurtisSampling())
            q = FG.spherical_quadrature(s, 12)
            a = FG.spherical_axes(s, 12)
            Test.@test q.φ == a.φ && q.λ == a.λ
            Test.@test q.w == FG.latitude_weights(s, 12)
        end
        # McEwen–Wiaux has nodes but deliberately no weights, so the combined form must refuse too
        # rather than inventing a rule that is not exact even at l = 0.
        Test.@test_throws ArgumentError FG.spherical_quadrature(FG.McEwenWiauxSampling(), 12)
        # The in-place form writes into caller buffers and allocates no scratch of its own.
        n = 64
        sz = FG.axes_lengths(FG.GaussLegendreSampling(), n)
        λb = Vector{Float64}(undef, sz.nlon)
        φb = Vector{Float64}(undef, sz.nlat)
        wb = Vector{Float64}(undef, sz.nlat)
        FG.spherical_quadrature!(λb, φb, wb, FG.GaussLegendreSampling(), n)
        Test.@test φb == FG.spherical_quadrature(FG.GaussLegendreSampling(), n).φ
        # Only the returned NamedTuple; the solve itself needs O(1) scratch, not the O(n²)
        # eigenvector matrix a Golub–Welsch decomposition would.
        nalloc(f) = (f(); Base.gc_alloc_count((@timed f()).gcstats))
        Test.@test nalloc(() -> FG.spherical_quadrature!(λb, φb, wb, FG.GaussLegendreSampling(), n)) <= 1
        Test.@test nalloc(() -> FG.SphericalSampling._gauss_legendre_μ!(φb, wb)) <= 1
        # O(1) scratch means the count cannot grow with n.
        big = Vector{Float64}(undef, 4n)
        bigw = similar(big)
        Test.@test nalloc(() -> FG.SphericalSampling._gauss_legendre_μ!(big, bigw)) <= 1
        Test.@test_throws DimensionMismatch FG.spherical_quadrature!(
            λb, φb, Vector{Float64}(undef, n + 1), FG.GaussLegendreSampling(), n)
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

        yy = FG.yin_yang_panels(6, 4)
        Test.@test length(yy.yin.λ) == 6
        Test.@test length(yy.yin.φ) == 4
        Test.@test size(yy.yang.λ) == (6, 4)
        Test.@test size(yy.yang.φ) == (6, 4)

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
        Test.@test Set(FG.neighbors(conn, FG.linear_index(grid, 2, 2))) == Set(nbr)
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
        Test.@test Set(FG.neighbors(uc, 1)) == Set([2, 3])

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
            Test.@test all(j -> j in yin_nodes, FG.neighbors(yyc, i))
        end
        for i in yang_nodes
            Test.@test all(j -> j in yang_nodes, FG.neighbors(yyc, i))
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
        Test.@test Test.@inferred(FG.Geometry.as_ntuple(p2)) === (0.1, 0.2)
    end

    Test.@testset "Connectivity is built into contiguous CSR, not per-node vectors" begin
        # The allocation count must not scale with the node count: everything lands in one neighbor
        # block plus one offset array, however many nodes there are.
        allocs(f) = (f(); minimum(Base.gc_alloc_count((@timed f()).gcstats) for _ in 1:5))
        # nside 8 → 32 is a 16× jump in node count (768 → 12288). Both sizes are past the point
        # where the count settles: the very smallest grids take one or two fewer allocations, so
        # anchoring on nside = 4 would measure that step rather than any scaling with n.
        small = allocs(() -> FG.build_connectivity(FG.HEALPixSampling(8)))
        large = allocs(() -> FG.build_connectivity(FG.HEALPixSampling(32)))
        Test.@test large <= small
        Test.@test large < 16
        Test.@test allocs(() -> FG.build_connectivity(FG.CubedSphereSampling(), 16)) < 16

        # Degrees and reciprocity are unaffected by the storage change.
        conn = FG.build_connectivity(FG.HEALPixSampling(4))
        n = FG.nnodes(conn)
        Test.@test n == FG.healpix_npix(4)
        Test.@test all(6 ≤ FG.nneighbors(conn, i) ≤ 8 for i in 1:n)
        Test.@test all(!in(i, FG.neighbors(conn, i)) for i in 1:n)          # no self-loops
        Test.@test all(issorted(FG.neighbors(conn, i)) for i in 1:n)        # sorted, deduped rows
        Test.@test all(allunique(FG.neighbors(conn, i)) for i in 1:n)
        Test.@test all(i in FG.neighbors(conn, j) for i in 1:n for j in FG.neighbors(conn, i))

        # A duplicated edge in an edge list collapses rather than being stored twice.
        mesh = FG.icosahedral_mesh(1)
        c1 = FG.build_connectivity(FG.IcosahedralSampling(1))
        Test.@test FG.nedges(c1) == 2 * length(unique(map(e -> minmax(e[1], e[2]), mesh.edges)))
    end

    Test.@testset "Spectral samplings integrate band-limited fields exactly" begin
        # The property that DEFINES a spectral quadrature sampling: Σ_j w_j P_l(sin φ_j) vanishes
        # for every 1 ≤ l ≤ lmax. A wrong node set or weight set fails this even when the point
        # count is right.
        function legendre(l, x)
            l == 0 && return one(x)
            p0, p1 = one(x), x
            for k in 1:(l - 1)
                p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
            end
            return p1
        end
        for s in (FG.GaussLegendreSampling(), FG.DriscollHealySampling(),
                  FG.DriscollHealyEqualSampling(), FG.ClenshawCurtisSampling())
            for nlat in (8, 16, 24)
                ax = FG.spherical_axes(s, nlat)
                w = FG.latitude_weights(s, nlat)
                # One normalization for every sampling: Σ w = ∫₀^π sinθ dθ = 2, carrying the sinθ
                # Jacobian and nothing else, so the longitude factor is always the caller's.
                Test.@test sum(w) ≈ 2 rtol = 1e-13
                Test.@test all(>(0), w) || s isa FG.AbstractDriscollHealySampling  # DH's polar node has zero weight
                # Exact for every single P_l the node count can resolve.
                for l in 1:(nlat - 1)
                    Test.@test abs(sum(w[j] * legendre(l, sin(ax.φ[j])) for j in eachindex(ax.φ))) < 1e-11
                end
                # Gauss–Legendre goes further: exact to 2N-1, which is what makes it the sampling
                # whose quadrature is exact at its own stated band limit.
                if s isa FG.AbstractGaussLegendreSampling
                    for l in nlat:(2nlat - 1)
                        Test.@test abs(sum(w[j] * legendre(l, sin(ax.φ[j])) for j in eachindex(ax.φ))) < 1e-11
                    end
                end
            end
        end

        # Float32 all the way through.
        w32 = FG.latitude_weights(FG.ClenshawCurtisSampling(), 12; T = Float32)
        Test.@test eltype(w32) === Float32
        Test.@test sum(w32) ≈ 2 rtol = 1e-5

        # Samplings without a quadrature say so, without pointing at some other package.
        for s in (FG.McEwenWiauxSampling(), FG.LatLonSampling())
            err = try
                FG.latitude_weights(s, 8)
                nothing
            catch e
                sprint(showerror, e)
            end
            Test.@test err !== nothing
            Test.@test !occursin("FastTransforms", err)
            Test.@test !occursin("SSHT /", err)
        end
    end

    Test.@testset "Sampling axes are well formed" begin
        for (s, nlat, kw) in (
            (FG.GaussLegendreSampling(), 33, (;)),
            (FG.DriscollHealySampling(), 32, (;)),
            (FG.ClenshawCurtisSampling(), 33, (;)),
            (FG.McEwenWiauxSampling(), 33, (;)),
            (FG.LatLonSampling(), 33, (; nlon = 66)),
        )
            ax = FG.spherical_axes(s, nlat; kw...)
            sz = FG.axes_lengths(s, nlat; kw...)
            Test.@test length(ax.λ) == sz.nlon
            Test.@test length(ax.φ) == sz.nlat
            Test.@test issorted(ax.φ) || issorted(ax.φ; rev = true)
            Test.@test all(φ -> -π / 2 - 1e-12 ≤ φ ≤ π / 2 + 1e-12, ax.φ)
            Test.@test all(λ -> -1e-12 ≤ λ < 2π + 1e-12, ax.λ)
            Test.@test allunique(round.(ax.φ; digits = 12))
            p = FG.spherical_points(s, nlat; kw...)
            Test.@test length(p.λ) == sz.nlon * sz.nlat == FG.npoints(s, nlat; kw...)
        end
    end

    Test.@testset "Cubed-sphere points are cell centres, so panels do not share nodes" begin
        # Endpoint-inclusive panel coordinates put nodes ON the seams, so adjacent panels emit
        # coincident points — 12(n-2)+16 of them — while `_cubed_neighbor` treats those same edges
        # as folding onto a *different* panel's cells. Cell centres keep points and connectivity
        # consistent and give 6n² genuinely distinct nodes.
        for n in (1, 2, 4, 8, 16)
            p = FG.cubed_sphere_points(n)
            Test.@test length(p.λ) == 6n^2
            Test.@test length(unique(collect(zip(p.λ, p.φ)))) == 6n^2
            Test.@test length(unique(p.panel)) == 6
            Test.@test all(v -> abs(hypot(cos(v[2]) * cos(v[1]), cos(v[2]) * sin(v[1]), sin(v[2])) - 1) < 1e-12,
                           zip(p.λ, p.φ))
            # No point may sit exactly on a panel boundary (|ξ| = π/4 maps to the cube edges).
            Test.@test all(abs(abs(ξ) - π / 4) > 1e-12 for ξ in
                           range(-π / 4 + (π / 2 / n) / 2; step = π / 2 / n, length = n))
        end
    end

    Test.@testset "Default node areas follow the sampling's equal-area trait" begin
        using Quickhull: Quickhull
        R = 6.371e6
        geo = FG.SphericalGeometry(R)
        tot = 4π * R^2

        # Equal-area by construction: uniform is exact.
        gh = FG.unstructured_grid(FG.HEALPixSampling(4); geometry = geo)
        ah = FG.measure(gh)
        Test.@test all(≈(tot / length(ah)), ah)
        Test.@test sum(ah) ≈ tot rtol = 1e-12

        # NOT equal-area: a uniform default would be silently wrong, so real dual areas are used.
        for g in (FG.unstructured_grid(FG.IcosahedralSampling(4); geometry = geo),
                  FG.unstructured_grid(FG.CubedSphereSampling(), 8; geometry = geo))
            a = FG.measure(g)
            Test.@test sum(a) ≈ tot rtol = 1e-8      # still tiles the sphere exactly
            Test.@test all(>(0), a)                   # no degenerate zero-area cells
            Test.@test minimum(a) / maximum(a) < 0.95 # genuinely non-uniform
        end

        # An explicit `areas` always wins over any default.
        gx = FG.unstructured_grid(FG.IcosahedralSampling(2); geometry = geo, areas = fill(1.0, 42))
        Test.@test all(==(1.0), FG.measure(gx))
    end

    Test.@testset "Yin–Yang cells tile each panel exactly; the overlap is resolution-independent" begin
        geo = FG.SphericalGeometry()
        R2 = geo.R^2
        box = sqrt(2.0) * (3π / 2) * R2   # one panel's exact [-3π/4,3π/4] × [-π/4,π/4] area
        for (nlon, nlat) in ((8, 6), (16, 12), (48, 32))
            g = FG.unstructured_grid(FG.YinYangSampling(), nlon, nlat; geometry = geo)
            a = FG.measure(g)
            np = nlon * nlat
            Test.@test length(a) == 2np
            # Cell centres, not panel edges: the nlon×nlat cells tile the panel box exactly. Sampling
            # the endpoints instead inflates this by nlon/(nlon-1) × nlat/(nlat-1).
            Test.@test sum(@view a[1:np]) ≈ box rtol = 1e-14
            # Yang is a rigid rotation of yin, so the two blocks are elementwise identical.
            Test.@test @view(a[1:np]) == @view(a[(np + 1):(2np)])
            # The panels overlap by construction: the excess over the sphere is exactly 3√2π/4π at
            # every resolution. A resolution-*dependent* excess would mean a discretisation bug.
            Test.@test sum(a) / (4π * R2) ≈ 3 * sqrt(2) / 4 rtol = 1e-14
            # Areas vary as cos φ across the panel, and nowhere degenerate.
            Test.@test all(>(0), a)
            pts = collect(zip(round.(g.λ; digits = 10), round.(g.φ; digits = 10)))
            Test.@test length(unique(pts)) == 2np
        end
        Test.@test minimum(FG.measure(FG.unstructured_grid(FG.YinYangSampling(), 192, 128))) /
                   maximum(FG.measure(FG.unstructured_grid(FG.YinYangSampling(), 192, 128))) ≈
                   cos(π / 4) rtol = 1e-2
    end

    Test.@testset "Icosahedral dual areas are exact and need no tessellation dependency" begin
        geo = FG.SphericalGeometry()
        tot = 4π * geo.R^2
        for ν in (1, 2, 4, 8, 16)
            mesh = FG.icosahedral_mesh(ν)
            Test.@test length(mesh.triangles) == 20ν^2
            Test.@test length(mesh.verts) == length(mesh.λ) == 10ν^2 + 2
            # Euler: V - E + F = 2.
            Test.@test (10ν^2 + 2) - length(mesh.edges) + length(mesh.triangles) == 2
            # Every triangle references three distinct in-range vertices.
            Test.@test all(t -> length(unique(t)) == 3 && all(1 .<= t .<= 10ν^2 + 2), mesh.triangles)

            g = FG.unstructured_grid(FG.IcosahedralSampling(ν); geometry = geo)
            a = FG.measure(g)
            # The per-triangle Voronoi shares tile each triangle, so the cells tile the sphere.
            Test.@test sum(a) ≈ tot rtol = 1e-14
            Test.@test all(>(0), a)
            # It is the true spherical Voronoi dual — cross-checked against the convex-hull route.
            vor = FG.Grids._voronoi_areas(geo, g.λ, g.φ)
            Test.@test maximum(abs.(a .- vor) ./ vor) < 1e-12
        end
        # A geodesic sphere has exactly 12 pentagons (the icosahedron's corners); they are the
        # smallest cells, and all the rest are hexagons.
        a8 = FG.measure(FG.unstructured_grid(FG.IcosahedralSampling(8); geometry = geo))
        Test.@test count(x -> x < minimum(a8) * (1 + 1e-9), a8) == 12
        # A uniform 4πR²/N default would be ~±25% wrong here, which is why it is not used.
        Test.@test minimum(a8) / maximum(a8) < 0.6
    end

    Test.@testset "Coarsest cubed sphere (one node per face) is constructible" begin
        # `range(a, b; length = 1)` is an error, so n = 1 needs its own handling — and
        # `build_connectivity` already accepted n ≥ 1, so this was reachable.
        p = FG.cubed_sphere_points(1)
        Test.@test length(p.λ) == 6
        Test.@test length(unique(p.panel)) == 6
        # Each face center is a distinct point on the sphere.
        pts = [(cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ)) for (λ, φ) in zip(p.λ, p.φ)]
        Test.@test length(unique(x -> round.(x; digits = 9), pts)) == 6
        Test.@test all(v -> abs(hypot(v...) - 1) < 1e-12, pts)
        conn = FG.build_connectivity(FG.CubedSphereSampling(), 1)
        Test.@test FG.nnodes(conn) == 6
        Test.@test_throws ArgumentError FG.cubed_sphere_points(0)
        for n in (1, 2, 5)
            Test.@test length(FG.cubed_sphere_points(n).λ) == 6n^2
        end
    end

    Test.@testset "Icosahedral mesh is indexed topologically, not by hashing coordinates" begin
        for ν in (1, 2, 3, 5, 8)
            mesh = FG.icosahedral_mesh(ν)
            nv = length(mesh.λ)
            Test.@test nv == FG.icosahedral_nvertices(ν) == 10ν^2 + 2
            # A geodesic sphere has exactly 30ν² edges, no duplicates and no self-loops.
            Test.@test length(mesh.edges) == 30ν^2
            Test.@test allunique(mesh.edges)
            Test.@test all(e -> e[1] < e[2], mesh.edges)
            # …and exactly 12 degree-5 vertices (the icosahedron corners); every other is degree 6.
            deg = zeros(Int, nv)
            for (a, b) in mesh.edges
                deg[a] += 1
                deg[b] += 1
            end
            Test.@test count(==(5), deg) == 12
            Test.@test count(==(6), deg) == nv - 12
            # Vertices are distinct points on the unit sphere.
            pts = [(cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ)) for (λ, φ) in zip(mesh.λ, mesh.φ)]
            Test.@test all(p -> abs(hypot(p...) - 1) < 1e-12, pts)
            Test.@test length(unique(p -> round.(p; digits = 9), pts)) == nv
            # Edge lengths are quasi-uniform, as a geodesic subdivision must be.
            elen = [hypot((pts[a] .- pts[b])...) for (a, b) in mesh.edges]
            Test.@test maximum(elen) / minimum(elen) < 1.5
        end

        # Construction cost must not scale with a per-vertex hash table.
        function allocs(f)
            f()
            r = @timed f()
            return Base.gc_alloc_count(r.gcstats)
        end
        Test.@test allocs(() -> FG.icosahedral_mesh(4)) < 200
        Test.@test allocs(() -> FG.icosahedral_mesh(32)) < 200
    end

    Test.@testset "Neighbor traversal allocates nothing" begin
        geom = FG.CartesianGeometry(1.0, 1.0)
        n = 40
        xs = collect(0.0:1.0:(n - 1))
        grid = FG.StructuredGrid(geom, xs, xs, trues(n, n))

        function sweep(g, n)
            c = 0
            for j in 1:n, i in 1:n
                for v in FG.neighbors(g, i, j)
                    c += v
                end
            end
            return c
        end
        sweep(grid, 3)
        Test.@test @allocated(sweep(grid, n)) == 0

        # The lazy sequence agrees with the buffer-filling form, element for element.
        buf = Vector{Int}(undef, 8)
        for (i, j) in ((1, 1), (1, 5), (n, n), (7, 9))
            k = FG.neighbors!(buf, grid, i, j)
            Test.@test collect(FG.neighbors(grid, i, j)) == buf[1:k]
            Test.@test length(FG.neighbors(grid, i, j)) == k
        end

        # A masked-out cell reports no neighbors, matching nneighbors.
        m = trues(n, n)
        m[5, 5] = false
        masked = FG.StructuredGrid(geom, xs, xs, m)
        Test.@test isempty(collect(FG.neighbors(masked, 5, 5)))
        Test.@test length(FG.neighbors(masked, 5, 5)) == 0
        Test.@test !(FG.linear_index(masked, 5, 5) in collect(FG.neighbors(masked, 5, 6)))
        # …but is still reachable when the caller does not filter on the mask.
        Test.@test FG.linear_index(masked, 5, 5) in collect(FG.neighbors(masked, 5, 6; active_only = false))

        # Curvilinear grids take the same lazy path.
        nx, ny = 6, 5
        xm = [Float64(i) for i in 1:nx, j in 1:ny]
        ym = [Float64(j) for i in 1:nx, j in 1:ny]
        cv = FG.CurvilinearGrid(geom, xm, ym, trues(nx, ny))
        Test.@test length(FG.neighbors(cv, 3, 3)) == 4
        Test.@test collect(FG.neighbors(cv, 3, 3)) == let b = Vector{Int}(undef, 8)
            k = FG.neighbors!(b, cv, 3, 3); b[1:k]
        end
    end

    Test.@testset "k-d-tree adjacency, open and wrapping" begin
        using NearestNeighbors: NearestNeighbors
        cgeo = FG.CartesianGeometry(1.0, 1.0)
        n, L = 4, 4.0
        xs = Float64[i for i in 0:(n - 1), j in 0:(n - 1)][:]
        ys = Float64[j for i in 0:(n - 1), j in 0:(n - 1)][:]
        N = length(xs)
        areas = ones(N)

        # The non-periodic build must work at all — the point matrix handed to the tree used to be
        # constructed in a form `KDTree` does not accept, so this path threw for every input.
        g_open = FG.UnstructuredGrid(cgeo, xs, ys, trues(N); k = 4, areas = areas)
        Test.@test FG.nnodes(FG.build_connectivity(g_open)) == N
        Test.@test all(1 ≤ length(FG.neighbors(g_open, i)) ≤ 4 for i in 1:N)
        Test.@test !FG.isperiodic(g_open, 1)

        g_per = FG.UnstructuredGrid(cgeo, xs, ys, trues(N); k = 4, areas = areas,
                                    periodic = (true, true), period = (L, L))
        Test.@test FG.isperiodic(g_per, 1) && FG.isperiodic(g_per, 2)
        Test.@test FG.period(g_per, 1) == L
        # On a wrapped lattice every node is interior: exactly four neighbours, each one cell away.
        Test.@test all(length(FG.neighbors(g_per, i)) == 4 for i in 1:N)
        Test.@test sort(collect(FG.neighbors(g_per, 1))) == [2, 4, 5, 13]
        minsep(a, b, L) = min(abs(a - b), L - abs(a - b))
        Test.@test all(minsep(xs[i], xs[j], L)^2 + minsep(ys[i], ys[j], L)^2 ≈ 1.0
                       for i in 1:N for j in FG.neighbors(g_per, i))
        Test.@test all(i in FG.neighbors(g_per, j) for i in 1:N for j in FG.neighbors(g_per, i))
        # Wrapping genuinely changes the graph rather than being recorded and ignored.
        Test.@test any(sort(collect(FG.neighbors(g_open, i))) != sort(collect(FG.neighbors(g_per, i)))
                       for i in 1:N)
        # Radius queries honor it too.
        g_rad = FG.UnstructuredGrid(cgeo, xs, ys, trues(N); radius = 1.01, areas = areas,
                                    periodic = (true, true), period = (L, L))
        Test.@test all(length(FG.neighbors(g_rad, i)) == 4 for i in 1:N)

        # A Cartesian box has no period to infer, so wrapping without one is an error.
        Test.@test_throws ArgumentError FG.UnstructuredGrid(cgeo, xs, ys, trues(N);
                                                            k = 4, areas = areas, periodic = true)

        # Spherical longitude wraps with no ghosting: the embedding identifies λ with λ+2π.
        sgeo = FG.SphericalGeometry(1.0)
        gs = FG.UnstructuredGrid(sgeo, [0.01, 6.27, 3.14], [0.0, 0.0, 0.0], trues(3);
                                 k = 1, areas = ones(3))
        Test.@test FG.isperiodic(gs, 1)
        Test.@test only(FG.neighbors(gs, 1)) == 2      # across the seam, not the far-away node 3
    end

    Test.@testset "HEALPix RING neighbours are emitted in ascending order" begin
        C = FG.Connectivity
        # The dedup pass is an insertion sort, whose cost is the inversion count. Walking the compass
        # offsets in RING order (`_HP_NB_ORDER`) makes the emission almost always already sorted, so
        # the sort has nothing to move. It still runs — it is what GUARANTEES the order — so this
        # only has to show the input got better, never that correctness moved into the emission.
        out = Vector{Int}(undef, 8)
        for nside in (1, 2, 4, 16, 64)
            npix = 12 * nside^2
            inversions = 0
            presorted = 0
            valid = true
            for p in 0:(npix - 1)
                m = C.healpix_neighbors!(out, nside, p)
                v = view(out, 1:m)
                k = 0
                for a in 1:m, b in (a + 1):m
                    v[a] > v[b] && (k += 1)
                end
                inversions += k
                k == 0 && (presorted += 1)
                # Aggregate rather than assert per pixel: the emitted set must be distinct in-range
                # pixel ids that exclude the node itself, whatever the order.
                valid &= m ≥ 6 && all(0 .≤ v .< npix) && length(unique(v)) == m && p ∉ v
            end
            Test.@test valid
            # Only the ring seam is left out of order, and the seam is ~4·nside of 12·nside² pixels,
            # so the residual disorder falls like 1/nside rather than to a constant.
            Test.@test inversions / npix < 6 / nside
            Test.@test presorted / npix > 1 - 8 / nside
        end
        # The CSR is unchanged by the reordering: rows sorted, adjacency reciprocal, degrees 7 or 8.
        for nside in (1, 2, 8, 32)
            c = FG.build_connectivity(FG.HEALPixSampling(nside))
            n = FG.nnodes(c)
            Test.@test all(issorted(FG.neighbors(c, i)) for i in 1:n)
            Test.@test FG.is_symmetric_adjacency(c)
            Test.@test sort(unique(length(FG.neighbors(c, i)) for i in 1:n)) ⊆ [6, 7, 8]
        end
    end

    Test.@testset "Threading is opt-in and changes no result" begin
        using ComputationalBackends: ComputationalBackends as CB
        Test.@test Base.get_extension(FG, :FlowGeometriesComputationalBackendsExt) !== nothing

        # Chunking must partition exactly — no gaps, no overlap, contiguous, whatever the remainder.
        for (n, k) in ((10, 3), (10, 1), (3, 10), (1, 1), (100, 7))
            rs = FG.Execution.chunk_ranges(n, k)
            Test.@test reduce(vcat, collect.(rs)) == collect(1:n)
            Test.@test all(!isempty, rs) && length(rs) ≤ max(1, min(k, n))
        end
        # `nothing` hands the body the whole range in one call: the serial path adds no partitioning.
        seen = UnitRange{Int}[]
        FG.Execution.run_chunks(17, nothing) do r; push!(seen, r); end
        Test.@test seen == [1:17]

        # Every threaded kernel must be bit-identical to serial, not merely close.
        geo = FG.SphericalGeometry()
        thr = CB.ThreadedBackend()
        for n in (7, 64)
            a = FG.cubed_sphere_points(n)
            b = FG.cubed_sphere_points(n; backend = thr)
            c = FG.cubed_sphere_points(n; backend = CB.SerialBackend())
            Test.@test a.λ == b.λ == c.λ
            Test.@test a.φ == b.φ == c.φ
            Test.@test a.panel == b.panel == c.panel
            sp = FG.spherical_points(FG.CubedSphereSampling(), n; backend = thr)
            Test.@test sp.λ == a.λ && sp.φ == a.φ
        end
        for (g, lbl) in ((geo, "spherical"), (FG.CartesianGeometry(1.0, 1.0), "cartesian"))
            for n in (3, 40)   # n = 3 gives fewer rows than threads, exercising the short case
                λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
                φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
                m = fill(true, n, n)
                Test.@test FG.measure(FG.CurvilinearGrid(g, λ, φ, m)) ==
                           FG.measure(FG.CurvilinearGrid(g, λ, φ, m; backend = thr))
            end
        end

        # Connectivity: both cell passes write only slots their own cell owns. Masked and periodic
        # cases matter most — they make the per-cell degree vary, so a chunk boundary landing
        # mid-row would show up as a wrong offset rather than a wrong count.
        for topo in (FG.IndexTopology((37, 21), (true, false), nothing),
                     FG.IndexTopology((37, 21), (true, true), nothing),
                     FG.IndexTopology((5, 4), (false, false), nothing),
                     FG.IndexTopology((31, 29), (true, false),
                                      [isodd(i * 7 + j * 3) for i in 1:31, j in 1:29]))
            for st in (:face, :vertex), ao in (true, false)
                a = FG.build_connectivity(topo; stencil = st, active_only = ao)
                b = FG.build_connectivity(topo; stencil = st, active_only = ao, backend = thr)
                Test.@test a.ptr == b.ptr
                Test.@test a.nbrs == b.nbrs
            end
        end
        g = FG.structured_grid(FG.ClenshawCurtisSampling(), 33)
        Test.@test FG.build_connectivity(g).nbrs ==
                   FG.build_connectivity(g; backend = thr).nbrs

        # The candidate builder emits and dedups concurrently, so each sampling's `emit!` has to be
        # free of state shared between nodes. HEALPix used to keep one scratch vector across calls;
        # nside = 1 and 2 cover the singular pixels where a node has 7 neighbours, not 8.
        for s in (FG.HEALPixSampling(1), FG.HEALPixSampling(2), FG.HEALPixSampling(8))
            a = FG.build_connectivity(s)
            b = FG.build_connectivity(s; backend = thr)
            Test.@test a.ptr == b.ptr
            Test.@test a.nbrs == b.nbrs
        end
        for n in (1, 2, 9), st in (:face, :vertex)
            a = FG.build_connectivity(FG.CubedSphereSampling(), n; stencil = st)
            b = FG.build_connectivity(FG.CubedSphereSampling(), n; stencil = st, backend = thr)
            Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
        end
        for (nlon, nlat) in ((1, 1), (7, 5)), st in (:face, :vertex)
            a = FG.build_connectivity(FG.YinYangSampling(), nlon, nlat; stencil = st)
            b = FG.build_connectivity(FG.YinYangSampling(), nlon, nlat; stencil = st, backend = thr)
            Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
        end
    end

    Test.@testset "Grids can be moved to another storage backend" begin
        using Adapt: Adapt, adapt
        # A stand-in device array: a distinct type, so a successful adapt shows in the grid's type.
        struct FakeDev end
        struct DevArr{T,N} <: AbstractArray{T,N}
            a::Array{T,N}
        end
        Base.size(d::DevArr) = size(d.a)
        Base.getindex(d::DevArr, I...) = getindex(d.a, I...)
        Adapt.adapt_storage(::FakeDev, x::Array) = DevArr(x)

        geo = FG.SphericalGeometry()
        g = FG.structured_grid(FG.ClenshawCurtisSampling(), 9)
        d = adapt(FakeDev(), g)
        Test.@test FG.coordinates(d, 1) isa DevArr && FG.coordinates(d, 2) isa DevArr
        # A separable measure must adapt its FACTORS — materializing the outer product onto a device
        # is exactly what the factored form exists to avoid.
        Test.@test FG.measure(d) isa FG.SeparableMeasure
        Test.@test all(f -> f isa DevArr, FG.measure_factors(d))
        Test.@test all(FG.measure(d)[i, j] == FG.measure(g)[i, j]
                       for i in 1:size(g, 1), j in 1:size(g, 2))
        Test.@test FG.mask(d) isa FG.AllActive           # size only; nothing to move
        Test.@test size(d) == size(g) && FG.grid_geometry(d) === FG.grid_geometry(g)
        Test.@test FG.isperiodic(d, 1) == FG.isperiodic(g, 1)

        n = 12
        λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
        φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
        cg0 = FG.CurvilinearGrid(geo, λ, φ, fill(true, n, n))
        cg = adapt(FakeDev(), cg0)
        Test.@test FG.coordinates(cg, 1) isa DevArr && FG.measure(cg) isa DevArr
        Test.@test all(FG.measure(cg)[i, j] == FG.measure(cg0)[i, j] for i in 1:n, j in 1:n)

        ug0 = FG.unstructured_grid(FG.IcosahedralSampling(3))
        ug = adapt(FakeDev(), ug0)
        Test.@test FG.coordinates(ug, 1) isa DevArr
        Test.@test getfield(ug, :neighbor_nbrs) isa DevArr
        Test.@test getfield(ug, :neighbor_ptr) isa DevArr
        Test.@test all(FG.measure(ug)[i] == FG.measure(ug0)[i] for i in eachindex(FG.measure(ug0)))

        c = adapt(FakeDev(), FG.build_connectivity(FG.HEALPixSampling(2)))
        Test.@test c.nbrs isa DevArr && c.ptr isa DevArr

        t = FG.IndexTopology((4, 3), (true, false), nothing)
        Test.@test adapt(FakeDev(), t) === t              # no mask, nothing to move
        t2 = FG.IndexTopology((4, 3), (true, false), fill(true, 4, 3))
        Test.@test adapt(FakeDev(), t2).mask isa DevArr
        Test.@test adapt(FakeDev(), FG.AllActive((5, 5))) isa FG.AllActive
    end

    Test.@testset "A symmetric adjacency is read as CSC without transposing a second copy" begin
        using SparseArrays: SparseArrays
        for (g, lbl) in ((FG.structured_grid(FG.ClenshawCurtisSampling(), 17), "structured"),
                         (FG.CurvilinearGrid(FG.SphericalGeometry(),
                                             [2π * (i - 1) / 12 for i in 1:12, j in 1:9],
                                             [asin(2 * (j - 0.5) / 9 - 1) for i in 1:12, j in 1:9],
                                             trues(12, 9)), "curvilinear"))
            conn = FG.build_connectivity(g)
            n = FG.nnodes(conn)
            Test.@test FG.is_symmetric_adjacency(conn)          # what licenses the shortcut
            A = FG.sparse_adjacency_matrix(g)                   # shortcut route
            B = FG.sparse_adjacency_matrix(FG.build_connectivity(g))  # transpose route
            Test.@test A == B                                   # identical matrix, not merely similar
            Test.@test Matrix(A) == FG.adjacency_matrix(conn)
            Test.@test SparseArrays.nnz(A) == FG.nedges(conn)
            Test.@test all(issorted(@view SparseArrays.rowvals(A)[SparseArrays.nzrange(A, j)])
                           for j in 1:n)
            # A non-default index type cannot alias Int buffers, so it falls back and must still match.
            Test.@test FG.sparse_adjacency_matrix(g; Ti = Int32) == A
            Test.@test eltype(FG.sparse_adjacency_matrix(g; Ti = Int32).colptr) === Int32
        end
        # `sort_neighbors!` orders each block in place and changes nothing else.
        c = FG.build_connectivity(FG.HEALPixSampling(4))
        before = [sort(collect(FG.neighbors(c, i))) for i in 1:FG.nnodes(c)]
        FG.sort_neighbors!(c)
        Test.@test all(issorted(FG.neighbors(c, i)) for i in 1:FG.nnodes(c))
        Test.@test all(collect(FG.neighbors(c, i)) == before[i] for i in 1:FG.nnodes(c))
        # k-nearest adjacency is NOT symmetric in general, so it must not take the shortcut.
        λ = [2π * ((i * 0.6180339887498949) % 1) for i in 1:150]
        φ = [asin(2 * (i / 151) - 1) for i in 1:150]
        ug = FG.UnstructuredGrid(FG.SphericalGeometry(), λ, φ, trues(150); k = 4, areas = ones(150))
        uc = FG.build_connectivity(ug)
        Test.@test !FG.is_symmetric_adjacency(uc)
        Test.@test Matrix(FG.sparse_adjacency_matrix(uc)) == FG.adjacency_matrix(uc)
    end

    Test.@testset "Equiangular weights: FFT path and recurrence fallback agree with the definition" begin
        using FFTW: FFTW          # loads AbstractFFTs, which fires the extension
        Test.@test Base.get_extension(FG, :FlowGeometriesAbstractFFTsExt) !== nothing

        # Literal evaluation of the defining sum — no recurrence, no transform.
        function direct(family, nlat)
            nterm = (nlat + 1) ÷ 2
            θ(i) = family === :open ? π * (i - 0.5) / nlat : π * (i - 1) / nlat
            return [(4 / nlat) * sin(θ(i)) * sum(sin((2k + 1) * θ(i)) / (2k + 1) for k in 0:(nterm - 1))
                    for i in 1:nlat]
        end
        for nlat in (8, 64, 512)
            dh = FG.latitude_weights(FG.DriscollHealySampling(), nlat)
            cc = FG.latitude_weights(FG.ClenshawCurtisSampling(), nlat)
            Test.@test maximum(abs.(dh .- direct(:closed, nlat))) < 1e-13
            Test.@test maximum(abs.(cc .- direct(:open, nlat))) < 1e-13
            Test.@test sum(dh) ≈ 2 atol = 1e-13
            Test.@test sum(cc) ≈ 2 atol = 1e-13
        end
        # The transform is O(n log n), so the cost ratio over a 4× size step must be far below the
        # 16× a quadratic rule would show.
        best(f) = (f(); minimum(@elapsed f() for _ in 1:3))
        t1 = best(() -> FG.latitude_weights(FG.DriscollHealySampling(), 1024))
        t2 = best(() -> FG.latitude_weights(FG.DriscollHealySampling(), 4096))
        Test.@test t2 / t1 < 8

        # Without an FFT implementation loaded the recurrence must still produce the same weights.
        script = """
        using FlowGeometries
        w = FlowGeometries.latitude_weights(FlowGeometries.ClenshawCurtisSampling(), 64)
        v = FlowGeometries.latitude_weights(FlowGeometries.DriscollHealySampling(), 64)
        print(Base.get_extension(FlowGeometries, :FlowGeometriesAbstractFFTsExt) === nothing,
              " ", sum(w), " ", sum(v), " ", w[7], " ", v[7])
        """
        out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`, String)
        parts = split(out)
        Test.@test parts[1] == "true"                       # extension genuinely absent
        Test.@test parse(Float64, parts[2]) ≈ 2 atol = 1e-13
        Test.@test parse(Float64, parts[3]) ≈ 2 atol = 1e-13
        Test.@test parse(Float64, parts[4]) ≈ direct(:open, 64)[7] atol = 1e-13
        Test.@test parse(Float64, parts[5]) ≈ direct(:closed, 64)[7] atol = 1e-13
    end

    Test.@testset "Curvilinear areas hold two corner rows, not the whole field" begin
        geo = FG.SphericalGeometry()
        for n in (5, 32)
            λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
            φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
            g = FG.CurvilinearGrid(geo, λ, φ, trues(n, n))
            a = FG.measure(g)
            Test.@test size(a) == (n, n)
            Test.@test all(>(0), a)
            # Reference: the same two triangles per cell, from a full materialized corner field.
            xc, yc = FG.Grids._centers_to_corners(λ), FG.Grids._centers_to_corners(φ)
            dirs = [(cos(yc[i, j]) * cos(xc[i, j]), cos(yc[i, j]) * sin(xc[i, j]), sin(yc[i, j]))
                    for i in 1:(n + 1), j in 1:(n + 1)]
            ref = [geo.R^2 * (FG.Grids._tri_excess(dirs[i, j], dirs[i + 1, j], dirs[i + 1, j + 1]) +
                              FG.Grids._tri_excess(dirs[i, j], dirs[i + 1, j + 1], dirs[i, j + 1]))
                   for i in 1:n, j in 1:n]
            Test.@test a == ref     # same arithmetic, only the buffering differs
        end
        # Construction memory must scale with the grid's own stored content (corners + areas), not
        # carry an extra full-size unit-vector field on top of it.
        function mib(n)
            λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
            φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
            m = trues(n, n)
            f = () -> FG.CurvilinearGrid(geo, λ, φ, m)
            f()
            return (@allocated f()) / 2^20
        end
        n = 200
        stored = 3 * 8 * (n + 1)^2 / 2^20      # xc, yc, areas
        Test.@test mib(n) < 1.6 * stored
    end

    Test.@testset "The ! forms allocate nothing beyond their return value" begin
        # Minimum of several samples: a single `@timed` in a long suite can catch a collection
        # mid-call and report allocations the function did not make. The true count is a floor,
        # so the minimum is the estimator that converges to it — the bounds below stay tight.
        nalloc(f) = (f(); minimum(Base.gc_alloc_count((@timed f()).gcstats) for _ in 1:5))

        # Gauss–Legendre: the node and weight outputs are independently optional, so asking for one
        # does not force a scratch vector for the other.
        n = 128
        gl = FG.GaussLegendreSampling()
        w = Vector{Float64}(undef, n)
        φ = Vector{Float64}(undef, n)
        λ = Vector{Float64}(undef, FG.axes_lengths(gl, n).nlon)
        FG.latitude_weights!(w, gl, n)
        FG.spherical_axes!(λ, φ, gl, n)
        Test.@test nalloc(() -> FG.latitude_weights!(w, gl, n)) == 0
        Test.@test nalloc(() -> FG.spherical_axes!(λ, φ, gl, n)) <= 1
        # …and both still agree with the combined solve.
        q = FG.spherical_quadrature(gl, n)
        Test.@test w == q.w && φ == q.φ && λ == q.λ

        # Cubed sphere: the panel id is not part of `spherical_points!`'s result, so it is not built.
        m = 32
        N = 6m^2
        cλ = Vector{Float64}(undef, N); cφ = Vector{Float64}(undef, N)
        FG.spherical_points!(cλ, cφ, FG.CubedSphereSampling(), m)
        Test.@test nalloc(() -> FG.spherical_points!(cλ, cφ, FG.CubedSphereSampling(), m)) <= 1
        p = FG.cubed_sphere_points(m)
        Test.@test cλ == p.λ && cφ == p.φ          # identical to the panel-carrying form
        Test.@test length(unique(p.panel)) == 6     # which still reports panels

        # Icosahedral vertices need no edge or triangle list, and must not build one.
        for ν in (8, 16)
            full = FG.icosahedral_mesh(ν)
            vo = FG.icosahedral_mesh(ν; topology = false)
            Test.@test vo.λ == full.λ && vo.φ == full.φ && vo.verts == full.verts
            Test.@test isempty(vo.edges) && isempty(vo.triangles)
            iλ = Vector{Float64}(undef, FG.icosahedral_nvertices(ν))
            iφ = similar(iλ)
            FG.icosahedral_vertices!(iλ, iφ, ν)
            Test.@test iλ == full.λ && iφ == full.φ
            # Allocation count must not grow with ν once the topology is skipped.
        end
        a8 = nalloc(() -> FG.icosahedral_vertices(8))
        a32 = nalloc(() -> FG.icosahedral_vertices(32))
        Test.@test a32 <= a8 + 2
    end

    Test.@testset "The icosahedron's faces are a written-down constant, and the right one" begin
        # The table replaces a runtime rediscovery of a fixed combinatorial fact. Re-derive it here
        # from the vertex geometry so the table cannot drift from the vertex ordering.
        faces = FG.SphericalSampling._ICOSAHEDRON_FACES
        Test.@test length(faces) == 20
        φg = (1.0 + sqrt(5.0)) / 2
        raw = [(0.0, 1.0, φg), (0.0, -1.0, φg), (0.0, 1.0, -φg), (0.0, -1.0, -φg),
               (1.0, φg, 0.0), (-1.0, φg, 0.0), (1.0, -φg, 0.0), (-1.0, -φg, 0.0),
               (φg, 0.0, 1.0), (φg, 0.0, -1.0), (-φg, 0.0, 1.0), (-φg, 0.0, -1.0)]
        v = [p ./ sqrt(sum(abs2, p)) for p in raw]
        d(i, j) = sqrt(sum(abs2, v[i] .- v[j]))
        edge = minimum(d(i, j) for i in 1:12 for j in (i + 1):12)
        # Every listed face is an equilateral triangle of three mutually adjacent vertices.
        for f in faces
            Test.@test length(unique(f)) == 3 && all(1 .<= f .<= 12)
            Test.@test all(d(a, b) ≈ edge for (a, b) in ((f[1], f[2]), (f[2], f[3]), (f[1], f[3])))
        end
        # Exactly the 20 such triangles exist, each listed once.
        derived = Set(Tuple(sort(collect(f))) for f in faces)
        expected = Set(Tuple(sort([a, b, c])) for a in 1:12 for b in (a + 1):12 for c in (b + 1):12
                       if d(a, b) ≈ edge && d(b, c) ≈ edge && d(a, c) ≈ edge)
        Test.@test derived == expected && length(derived) == 20
        # Every vertex is in 5 faces, and V - E + F = 2.
        Test.@test all(count(f -> v in f, faces) == 5 for v in 1:12)
        Test.@test length(unique(Tuple(sort([f[i], f[j]])) for f in faces
                                 for (i, j) in ((1, 2), (2, 3), (1, 3)))) == 30
        Test.@test 12 - 30 + 20 == 2
    end

    Test.@testset "Cell measure is stored factored, not materialized" begin
        geo = FG.SphericalGeometry()
        for (nx, ny) in ((16, 9), (64, 40))
            x = collect(range(0, 2π; length = nx))
            y = collect(range(-1.3, 1.3; length = ny))
            g = FG.StructuredGrid(geo, x, y, FG.AllActive((nx, ny)))
            m = FG.measure(g)
            Test.@test m isa FG.SeparableMeasure
            Test.@test size(m) == (nx, ny) == size(g)
            # Indexes exactly like the dense outer product it replaces — bit-identical, not close.
            wx, wy = FG.measure_factors(g)
            dense = wx .* transpose(wy)
            Test.@test all(m[i, j] === dense[i, j] for i in 1:nx, j in 1:ny)
            Test.@test FG.measure_array(g) == dense
            Test.@test collect(m) == dense
            Test.@test FG.measure(g, 3, 4) === dense[3, 4]
            # sum is ∏ᵈ ∑ᵢ, i.e. O(Nx+Ny); it must still agree with the dense sum to roundoff.
            Test.@test sum(m) ≈ sum(dense) rtol = 1e-14
            Test.@test_throws BoundsError m[nx + 1, 1]
            # Storage is the factors, not the cells.
            Test.@test sizeof(wx) + sizeof(wy) < sizeof(dense)
        end
        # A 1-D grid's measure is already O(N); it stays a plain vector.
        g1 = FG.StructuredGrid(FG.CartesianGeometry(1.0, 1.0), collect(0.0:0.5:5.0), FG.AllActive((11,)))
        Test.@test FG.measure(g1) isa AbstractVector
        Test.@test FG.measure_factors(g1) === nothing

        # `show` must not sum every cell to print one line.
        big = FG.StructuredGrid(geo, collect(range(0, 2π; length = 2000)),
                                collect(range(-1.4, 1.4; length = 1000)), FG.AllActive((2000, 1000)))
        Test.@test occursin("2000×1000", sprint(show, MIME"text/plain"(), big))
    end

    Test.@testset "An all-active mask carries no per-cell storage" begin
        m = FG.AllActive((7, 5))
        Test.@test size(m) == (7, 5) && length(m) == 35
        Test.@test all(m[i, j] for i in 1:7, j in 1:5)
        Test.@test count(m) == 35 && all(m) && any(m)
        Test.@test collect(m) == trues(7, 5)
        Test.@test_throws BoundsError m[8, 1]
        # It is what the sampling constructors reach for, and it is smaller than a BitArray.
        g = FG.structured_grid(FG.ClenshawCurtisSampling(), 33)
        Test.@test FG.mask(g) isa FG.AllActive
        Test.@test all(FG.isactive(g, i, j) for i in 1:size(g, 1), j in 1:size(g, 2))
        Test.@test Base.summarysize(FG.AllActive((4000, 2000))) < Base.summarysize(trues(4000, 2000))
        # An explicit mask still overrides it and still masks.
        mm = trues(FG.axes_lengths(FG.ClenshawCurtisSampling(), 9).nlon, 9)
        mm[2, 3] = false
        gm = FG.structured_grid(FG.ClenshawCurtisSampling(), 9; mask = mm)
        Test.@test !FG.isactive(gm, 2, 3)
    end

    Test.@testset "Sampling connectivity is built from index topology, not a discarded grid" begin
        # Identical CSR to routing through `structured_grid`, without evaluating the axes (for
        # Gauss–Legendre that is the O(n²) root solve), the dense measure, or a `trues` mask.
        for (s, nlat, kw) in (
            (FG.GaussLegendreSampling(), 12, (;)),
            (FG.ClenshawCurtisSampling(), 9, (;)),
            (FG.DriscollHealySampling(), 8, (;)),
            (FG.LatLonSampling(), 7, (; nlon = 10)),
        )
            for st in (:face, :vertex)
                direct = FG.build_connectivity(s, nlat; stencil = st, kw...)
                viagrid = FG.build_connectivity(FG.structured_grid(s, nlat; kw...); stencil = st)
                Test.@test FG.nnodes(direct) == FG.nnodes(viagrid)
                Test.@test direct.nbrs == viagrid.nbrs
                Test.@test direct.ptr == viagrid.ptr
            end
        end
        # A mask still applies, and `nothing` means "all active" without materializing one.
        sz = FG.axes_lengths(FG.ClenshawCurtisSampling(), 9)
        m = trues(sz.nlon, sz.nlat)
        m[2, 2] = false
        masked = FG.build_connectivity(FG.ClenshawCurtisSampling(), 9; mask = m)
        Test.@test FG.nnodes(masked) == sz.nlon * sz.nlat
        Test.@test length(FG.neighbors(masked, sz.nlon + 2)) == 0   # (i,j) = (2,2), i fastest

        # The grid-free path must not scale its allocations with nlat the way building a grid does.
        nalloc(f) = (f(); Base.gc_alloc_count((@timed f()).gcstats))
        small = nalloc(() -> FG.build_connectivity(FG.GaussLegendreSampling(), 16))
        large = nalloc(() -> FG.build_connectivity(FG.GaussLegendreSampling(), 128))
        Test.@test large <= small + 2
    end

    Test.@testset "k-d-tree knn allocations do not scale with the node count" begin
        using NearestNeighbors: NearestNeighbors
        geo = FG.SphericalGeometry(1.0)
        nalloc(f) = (f(); Base.gc_alloc_count((@timed f()).gcstats))
        build(n) = begin
            λ = [2π * (i * 0.6180339887498949 % 1) for i in 1:n]
            φ = [asin(2 * (i / (n + 1)) - 1) for i in 1:n]
            () -> FG.Grids._build_kdtree_neighbors(geo, λ, φ; k = 6)
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

        # Correctness is unchanged: the k nearest by great-circle distance, nearest-first.
        n = 200
        λ = [2π * (i * 0.6180339887498949 % 1) for i in 1:n]
        φ = [asin(2 * (i / (n + 1)) - 1) for i in 1:n]
        nbrs, ptr = FG.Grids._build_kdtree_neighbors(geo, λ, φ; k = 5)
        v = [(cos(φ[i]) * cos(λ[i]), cos(φ[i]) * sin(λ[i]), sin(φ[i])) for i in 1:n]
        for i in 1:n
            d = [j == i ? Inf : sum(abs2, v[i] .- v[j]) for j in 1:n]
            Test.@test sort(nbrs[ptr[i]:(ptr[i + 1] - 1)]) == sort(partialsortperm(d, 1:5))
            # nearest-first within the node's block
            Test.@test issorted([d[j] for j in nbrs[ptr[i]:(ptr[i + 1] - 1)]])
        end
    end

    Test.@testset "Sparse adjacency assembles straight into CSC" begin
        using SparseArrays: SparseArrays
        conn = FG.build_connectivity(FG.HEALPixSampling(8))
        n, ne = FG.nnodes(conn), FG.nedges(conn)
        A = FG.sparse_adjacency_matrix(conn)
        Test.@test size(A) == (n, n)
        Test.@test SparseArrays.nnz(A) == ne
        # Same matrix the dense builder produces.
        Test.@test Matrix(A) == FG.adjacency_matrix(conn)
        # Row indices ascending within each column, as CSC requires — obtained without a sort.
        Test.@test all(issorted(@view SparseArrays.rowvals(A)[SparseArrays.nzrange(A, j)]) for j in 1:n)

        # A caller-supplied index type, and caller-owned buffers reused with no allocation of the
        # matrix's own storage.
        Test.@test eltype(FG.sparse_adjacency_matrix(conn; Ti = Int32).colptr) === Int32
        colptr = Vector{Int}(undef, n + 1)
        rowval = Vector{Int}(undef, ne)
        nzval = Vector{Bool}(undef, ne)
        B = FG.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)
        Test.@test B == A
        Test.@test B.colptr === colptr && B.rowval === rowval && B.nzval === nzval
        FG.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)
        Test.@test @allocated(FG.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)) < 200

        # The COO route still agrees with the direct one.
        I = Vector{Int}(undef, ne); J = Vector{Int}(undef, ne)
        Test.@test FG.sparse_adjacency_coo!(I, J, conn) == ne
        Test.@test SparseArrays.sparse(I, J, trues(ne), n, n) == A
    end

    Test.@testset "HEALPix pixel centers are distinct and correctly ringed" begin
        # These tile the sphere, so no two may coincide, and the ring structure is fully determined:
        # 4nside-1 rings holding 4, 8, … 4(nside-1), then 4nside for 2nside+1 rings, then back down.
        # The suite previously only checked `length`, which is why a broken equatorial ring index
        # (dividing by 2nside instead of 4nside) went unnoticed for every nside ≥ 4.
        for nside in (1, 2, 4, 8, 16)
            p = FG.spherical_points(FG.HEALPixSampling(nside))
            npix = FG.healpix_npix(nside)
            Test.@test length(p.λ) == npix
            Test.@test length(unique(collect(zip(p.λ, p.φ)))) == npix

            zs = sort(unique(round.(sin.(p.φ); digits = 11)); rev = true)
            Test.@test length(zs) == FG.healpix_nring(nside) == 4nside - 1
            counts = [count(φ -> isapprox(sin(φ), z; atol = 1e-10), p.φ) for z in zs]
            expected = vcat([4r for r in 1:(nside - 1)], fill(4nside, 2nside + 1),
                            [4r for r in (nside - 1):-1:1])
            Test.@test counts == expected
            Test.@test sum(counts) == npix

            # Nearest-neighbour separation scales as ~1/nside and is never zero.
            Test.@test all(-π / 2 ≤ φ ≤ π / 2 for φ in p.φ)
            Test.@test all(0 ≤ λ < 2π + 1e-12 for λ in p.λ)
        end
    end

    Test.@testset "Spherical Voronoi areas tile the sphere" begin
        using Quickhull: Quickhull
        R = 6.371e6
        geo = FG.SphericalGeometry(R)
        for nside in (2, 4)
            p = FG.spherical_points(FG.HEALPixSampling(nside))
            a = FG.Grids._voronoi_areas(geo, p.λ, p.φ)
            Test.@test length(a) == length(p.λ)
            Test.@test all(>(0), a)
            Test.@test sum(a) ≈ 4π * R^2 rtol = 1e-10   # a tessellation covers the sphere exactly
        end
        # Non-equal-area sampling: still tiles exactly, but cells genuinely differ — which is why a
        # uniform 4πR²/N default is wrong for it.
        m = FG.icosahedral_mesh(4)
        ai = FG.Grids._voronoi_areas(geo, m.λ, m.φ)
        Test.@test sum(ai) ≈ 4π * R^2 rtol = 1e-10
        Test.@test minimum(ai) / maximum(ai) < 0.8

        # Float32 all the way through, and a clear error rather than a degenerate hull.
        p32 = FG.spherical_points(FG.HEALPixSampling(2); T = Float32)
        a32 = FG.Grids._voronoi_areas(FG.SphericalGeometry(Float32(R)), p32.λ, p32.φ)
        Test.@test eltype(a32) === Float32
        Test.@test sum(a32) ≈ 4Float32(π) * Float32(R)^2 rtol = 1e-4
        Test.@test_throws ArgumentError FG.Grids._voronoi_areas(geo, [0.0, 1.0], [0.0, 0.5])

        # The restored helper accepts mixed point representations, as its caller passes.
        A = FG.Grids._sph_triangle_area(geo, (0.0, 0.0), (0.1, 0.0), (λ = 0.0, φ = 0.1))
        Test.@test A ≈ 0.5 * 0.1 * 0.1 * R^2 rtol = 2e-2
        Test.@test FG.Grids._sph_triangle_area(geo, (0.0, 0.0), (0.1, 0.0), (0.2, 0.0)) ≈ 0 atol = 1e-3
    end

    Test.@testset "Planar Voronoi areas are complete and degeneracy-safe" begin
        using DelaunayTriangulation: DelaunayTriangulation
        cgeo = FG.CartesianGeometry(1.0, 1.0)
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
        a32 = FG.Grids._voronoi_areas(FG.CartesianGeometry(1.0f0, 1.0f0), Float32.(xs), Float32.(ys))
        Test.@test eltype(a32) === Float32
        Test.@test sum(a32) ≈ Float32(sum(a)) rtol = 1e-5
    end

    Test.@testset "Dense and sparse adjacency agree; sparse is the scalable one" begin
        using SparseArrays: SparseArrays
        cgeo = FG.CartesianGeometry(1.0, 1.0)
        small = FG.StructuredGrid(cgeo, range(0.0, 1.0; length = 20), range(0.0, 1.0; length = 20),
                                  trues(20, 20))
        A = FG.adjacency_matrix(small)
        Test.@test size(A) == (400, 400)
        Test.@test A == A'
        Test.@test Matrix(FG.sparse_adjacency_matrix(small)) == A

        # Dense is n² bytes — quadratic in nodes, quartic in grid side — so a 1000² grid is ~10¹²
        # bytes and simply is not the right tool. The sparse route on that same grid is routine.
        big = FG.StructuredGrid(cgeo, range(0.0, 1.0; length = 1000), range(0.0, 1.0; length = 1000),
                                trues(1000, 1000))
        S = FG.sparse_adjacency_matrix(big)
        Test.@test size(S) == (10^6, 10^6)
        Test.@test SparseArrays.nnz(S) == FG.nedges(FG.build_connectivity(big))
    end

    Test.@testset "Quadrature-exactness trait matches measured exactness" begin
        # The trait is about integrating PRODUCTS of two degree-lmax functions, which is what
        # spectral analysis forms. Clenshaw–Curtis's grid represents to N-1 but its quadrature only
        # integrates a single P_l to N-1, so it cannot claim exactness at its own band limit.
        Test.@test FG.admits_exact_bandlimited_quadrature(FG.GaussLegendreSampling())
        Test.@test FG.admits_exact_bandlimited_quadrature(FG.DriscollHealySampling())
        Test.@test !FG.admits_exact_bandlimited_quadrature(FG.ClenshawCurtisSampling())
        Test.@test !FG.admits_exact_bandlimited_quadrature(FG.McEwenWiauxSampling())
        Test.@test !FG.admits_exact_bandlimited_quadrature(FG.LatLonSampling())

        # Back the claim numerically: GL integrates degree 2·lmax exactly; CC does not.
        function legendre(l, x)
            l == 0 && return one(x)
            p0, p1 = one(x), x
            for k in 1:(l - 1)
                p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
            end
            p1
        end
        n = 16
        for s in (FG.GaussLegendreSampling(), FG.ClenshawCurtisSampling())
            ax = FG.spherical_axes(s, n)
            w = FG.latitude_weights(s, n)
            lmax = FG.bandlimit(s, n)
            err = abs(sum(w[j] * legendre(2lmax, sin(ax.φ[j])) for j in eachindex(ax.φ)))
            if FG.admits_exact_bandlimited_quadrature(s)
                Test.@test err < 1e-10
            else
                Test.@test err > 1e-6      # genuinely inexact at 2·lmax, as the trait now says
            end
        end
    end

    Test.@testset "No method ambiguities or unbound static parameters" begin
        Test.@test isempty(Test.detect_ambiguities(FG; recursive = true))
        Test.@test isempty(Test.detect_unbound_args(FG; recursive = true))
    end

    Test.@testset "Point functions accept every representation and element type" begin
        using StaticArrays: StaticArrays as SA
        cgeo = FG.CartesianGeometry(1.0, 1.0)     # Float64 geometry …
        sgeo = FG.SphericalGeometry(6.371e6)

        R = 6.371e6

        # … fed points whose element type is NOT the geometry's: they are converted on entry, and
        # every representation reaches the same kernel rather than recursing through normalization.
        Test.@test FG.distance(cgeo, (0, 0), (3, 4)) ≈ 5.0
        Test.@test FG.distance(cgeo, (0.0f0, 0.0f0), (3.0f0, 4.0f0)) ≈ 5.0
        Test.@test FG.distance(cgeo, (x = 0.0, y = 0.0), SA.SVector(3.0, 4.0)) ≈ 5.0
        Test.@test FG.distance(cgeo, [0.0, 0.0], (3.0, 4.0)) ≈ 5.0
        Test.@test FG.distance(sgeo, (0.1f0, 0.2f0), (0.3f0, 0.4f0)) ≈
                   FG.distance(sgeo, (Float64(0.1f0), Float64(0.2f0)), (Float64(0.3f0), Float64(0.4f0)))

        # 3-component spherical points carry an absolute radius; on the reference sphere the chord
        # distance agrees with the 2-component great-circle distance to within chord-vs-arc.
        Test.@test FG.distance(sgeo, (0.1, 0.2, R), (0.1, 0.2, R)) ≈ 0.0 atol = 1e-9
        Test.@test FG.distance(sgeo, (0.0, 0.0, R), (0.0, 0.0, 2R)) ≈ R

        # An unsupported point length is an error, not unbounded recursion.
        Test.@test_throws MethodError FG.distance(sgeo, (0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4))
        Test.@test_throws ArgumentError FG.distance(cgeo, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])

        # `spherical_to_cartesian` — (λ, φ) = (0, 0) is the +x axis at radius R.
        P0 = FG.Geometry.spherical_to_cartesian(sgeo, (0.0, 0.0))
        Test.@test all(((P0.x, P0.y, P0.z) .- (R, 0.0, 0.0)) .< 1e-6)
        Pref = FG.Geometry.spherical_to_cartesian(sgeo, (0.1, 0.2))
        for pt in ((0.1, 0.2), (λ = 0.1, φ = 0.2), SA.SVector(0.1, 0.2), [0.1, 0.2])
            Test.@test FG.Geometry.spherical_to_cartesian(sgeo, pt) === Pref
        end

        # The spherical tangent-plane projection is built on it, and was unreachable before.
        Δ0 = FG.project_to_tangent_plane(sgeo, (λ = 0.1, φ = 0.2), (λ = 0.1, φ = 0.2))
        Test.@test Δ0.λ ≈ 0.0 atol = 1e-9
        Test.@test Δ0.φ ≈ 0.0 atol = 1e-9
        # A small eastward step lies along ê_λ only, with arc length R·cos(φ)·Δλ.
        λ0, φ0, Δλ = 0.7, 0.2, 1e-6
        Δe = FG.project_to_tangent_plane(sgeo, (λ0, φ0), (λ0 + Δλ, φ0))
        Test.@test Δe.λ ≈ R * cos(φ0) * Δλ rtol = 1e-6
        Test.@test abs(Δe.φ) < 1e-3 * abs(Δe.λ)
    end

    Test.@testset "Requested point representation is honored exactly" begin
        using StaticArrays: StaticArrays as SA
        sgeo = FG.SphericalGeometry(6.371e6)
        cgeo = FG.CartesianGeometry(1.0, 1.0)
        xs = collect(0.0:1.0:4.0)
        cgrid = FG.StructuredGrid(cgeo, xs, xs, trues(5, 5))
        sgrid = FG.StructuredGrid(sgeo, deg2rad.(xs), deg2rad.(xs), trues(5, 5))

        Test.@test Test.@inferred(FG.coords(cgrid, 2, 3)) === (x = 1.0, y = 2.0)
        Test.@test Test.@inferred(FG.coords(NamedTuple, sgrid, 2, 3)) === FG.coords(sgrid, 2, 3)
        Test.@test Test.@inferred(FG.coords(Tuple, cgrid, 2, 3)) === (1.0, 2.0)
        Test.@test Test.@inferred(FG.coords(NTuple{2,Float32}, cgrid, 2, 3)) === (1.0f0, 2.0f0)
        Test.@test Test.@inferred(FG.coords(SA.SVector{2,Float64}, cgrid, 2, 3)) === SA.SVector(1.0, 2.0)
        Test.@test Test.@inferred(FG.coords(SA.SVector, cgrid, 2, 3)) === SA.SVector(1.0, 2.0)
        Test.@test Test.@inferred(FG.coords(SA.MVector{2,Float64}, cgrid, 2, 3)) == SA.MVector(1.0, 2.0)
        Test.@test Test.@inferred(FG.coords(Vector{Float64}, cgrid, 2, 3)) == [1.0, 2.0]

        # The vector-returning geometry functions take the same leading-type escape.
        p, q = SA.SVector(0.1, 0.2), SA.SVector(0.11, 0.21)
        Test.@test Test.@inferred(FG.project_to_tangent_plane(SA.SVector{2,Float64}, sgeo, p, q)) ==
                   SA.SVector(Tuple(FG.project_to_tangent_plane(sgeo, p, q)))
        Test.@test Test.@inferred(FG.Geometry.spherical_to_cartesian(SA.SVector{3,Float64}, sgeo, p)) ==
                   SA.SVector(Tuple(FG.Geometry.spherical_to_cartesian(sgeo, p)))
        Test.@test Test.@inferred(FG.vector_to_cartesian(SA.SVector{3,Float64}, sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)) ==
                   SA.SVector(Tuple(FG.vector_to_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)))
        Test.@test Test.@inferred(FG.vector_from_cartesian(SA.MVector{3,Float64}, sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)) ==
                   SA.MVector(Tuple(FG.vector_from_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)))
        Test.@test Test.@inferred(FG.local_tangent_basis(SA.SVector{3,Float64}, sgeo, p)).λ isa SA.SVector{3,Float64}
        # A velocity may itself be given in any representation.
        Test.@test FG.vector_from_cartesian(sgeo, SA.SVector(1.0, 2.0, 3.0), 0.1, 0.2) ===
                   FG.vector_from_cartesian(sgeo, 1.0, 2.0, 3.0, 0.1, 0.2)
    end

    Test.@testset "Point handling allocates nothing" begin
        using StaticArrays: StaticArrays as SA
        sgeo = FG.SphericalGeometry(6.371e6)
        xs = deg2rad.(collect(0.0:1.0:9.0))
        sgrid = FG.StructuredGrid(sgeo, xs, xs, trues(10, 10))

        function svec_loop(grid, geo, n)
            s = 0.0
            c0 = FG.coords(SA.SVector{2,Float64}, grid, 1, 1)
            for j in 1:n, i in 1:n
                nb = FG.coords(SA.SVector{2,Float64}, grid, i, j)
                s += FG.distance(geo, c0, nb + SA.SVector{2,Float64}(1e-3, 1e-3))
            end
            return s
        end
        function namedtuple_loop(grid, geo, n)
            s = 0.0
            c0 = FG.coords(grid, 1, 1)
            for j in 1:n, i in 1:n
                Δ = FG.project_to_tangent_plane(geo, c0, FG.coords(grid, i, j))
                s += Δ[1]^2 + Δ[2]^2
            end
            return s
        end
        svec_loop(sgrid, sgeo, 2)
        namedtuple_loop(sgrid, sgeo, 2)
        Test.@test @allocated(svec_loop(sgrid, sgeo, 10)) == 0
        Test.@test @allocated(namedtuple_loop(sgrid, sgeo, 10)) == 0
    end

end
