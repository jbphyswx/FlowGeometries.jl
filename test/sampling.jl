Test.@testset "Spherical sampling hierarchy" begin
    Test.@test FG.SphericalSampling.GaussLegendreSampling <: FG.SphericalSampling.AbstractGaussLegendreSampling
    Test.@test FG.SphericalSampling.DriscollHealySampling <: FG.SphericalSampling.AbstractDriscollHealySampling
    Test.@test FG.SphericalSampling.ClenshawCurtisSampling <: FG.SphericalSampling.AbstractClenshawCurtisSampling
    Test.@test FG.SphericalSampling.McEwenWiauxSampling <: FG.SphericalSampling.AbstractMcEwenWiauxSampling
    Test.@test FG.SphericalSampling.LatLonSampling <: FG.SphericalSampling.AbstractLatLonSampling
    Test.@test FG.SphericalSampling.HEALPixSampling <: FG.SphericalSampling.AbstractHEALPixSampling
    Test.@test FG.SphericalSampling.CubedSphereSampling <: FG.SphericalSampling.AbstractCubedSphereSampling
    Test.@test FG.SphericalSampling.IcosahedralSampling <: FG.SphericalSampling.AbstractIcosahedralSampling
    Test.@test FG.SphericalSampling.YinYangSampling <: FG.SphericalSampling.AbstractYinYangSampling
    Test.@test FG.SphericalSampling.ScatteredSphericalSampling <: FG.SphericalSampling.AbstractScatteredSphericalSampling

    Test.@test FG.SphericalSampling.is_tensor_product(FG.SphericalSampling.ClenshawCurtisSampling())
    Test.@test FG.SphericalSampling.is_iso_latitude(FG.SphericalSampling.HEALPixSampling(2))
    Test.@test FG.SphericalSampling.is_equal_area(FG.SphericalSampling.HEALPixSampling(2))
    Test.@test FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.GaussLegendreSampling())
    Test.@test !FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.LatLonSampling())
    Test.@test !FG.SphericalSampling.is_tensor_product(FG.SphericalSampling.HEALPixSampling(1))
end

Test.@testset "Clenshaw–Curtis = FastSphericalHarmonics sph_points" begin
    nθ = 8
    (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.ClenshawCurtisSampling(), nθ)
    Test.@test length(φ) == nθ
    Test.@test length(λ) == 2nθ - 1
    Test.@test FG.SphericalSampling.bandlimit(FG.SphericalSampling.ClenshawCurtisSampling(), nθ) == nθ - 1
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
    (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.GaussLegendreSampling(), nθ)
    w = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.GaussLegendreSampling(), nθ)
    Test.@test length(λ) == 2nθ - 1
    Test.@test length(φ) == nθ
    Test.@test sum(w) ≈ 2 atol = 1e-12          # ∫_{-1}^{1} dμ
    Test.@test FG.SphericalSampling.bandlimit(FG.SphericalSampling.GaussLegendreSampling(), nθ) == nθ - 1
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
    r32 = FG.SphericalSampling._gauss_legendre_μ(Float32, 64)
    r64 = FG.SphericalSampling._gauss_legendre_μ(64)
    Test.@test eltype(r32.μ) === Float32
    Test.@test maximum(abs.(Float64.(r32.w) .- r64.w) ./ r64.w) < 4 * eps(Float32)
    Test.@test maximum(abs.(Float64.(r32.μ) .- r64.μ)) < 4 * eps(Float32)

    # One solve serves both, and agrees with the two separate entry points.
    for n in (5, 32)
        q = FG.SphericalSampling.spherical_quadrature(FG.SphericalSampling.GaussLegendreSampling(), n)
        a = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.GaussLegendreSampling(), n)
        Test.@test q.λ == a.λ
        Test.@test q.φ == a.φ
        Test.@test q.w == FG.SphericalSampling.latitude_weights(FG.SphericalSampling.GaussLegendreSampling(), n)
    end
    # Non-GL samplings have independent closed forms and take the generic method.
    for s in (FG.SphericalSampling.DriscollHealySampling(), FG.SphericalSampling.DriscollHealyEqualSampling(), FG.SphericalSampling.ClenshawCurtisSampling())
        q = FG.SphericalSampling.spherical_quadrature(s, 12)
        a = FG.SphericalSampling.spherical_axes(s, 12)
        Test.@test q.φ == a.φ && q.λ == a.λ
        Test.@test q.w == FG.SphericalSampling.latitude_weights(s, 12)
    end
    # McEwen–Wiaux has nodes but deliberately no weights, so the combined form must refuse too
    # rather than inventing a rule that is not exact even at l = 0.
    Test.@test_throws ArgumentError FG.SphericalSampling.spherical_quadrature(FG.SphericalSampling.McEwenWiauxSampling(), 12)
    # The in-place form writes into caller buffers and allocates no scratch of its own.
    n = 64
    sz = FG.SphericalSampling.axes_lengths(FG.SphericalSampling.GaussLegendreSampling(), n)
    λb = Vector{Float64}(undef, sz.nlon)
    φb = Vector{Float64}(undef, sz.nlat)
    wb = Vector{Float64}(undef, sz.nlat)
    FG.SphericalSampling.spherical_quadrature!(λb, φb, wb, FG.SphericalSampling.GaussLegendreSampling(), n)
    Test.@test φb == FG.SphericalSampling.spherical_quadrature(FG.SphericalSampling.GaussLegendreSampling(), n).φ
    # Only the returned NamedTuple; the solve itself needs O(1) scratch, not the O(n²)
    # eigenvector matrix a Golub–Welsch decomposition would.
    nalloc(f) = (f(); Base.gc_alloc_count((@timed f()).gcstats))
    Test.@test nalloc(() -> FG.SphericalSampling.spherical_quadrature!(λb, φb, wb, FG.SphericalSampling.GaussLegendreSampling(), n)) <= 1
    Test.@test nalloc(() -> FG.SphericalSampling._gauss_legendre_μ!(φb, wb)) <= 1
    # O(1) scratch means the count cannot grow with n.
    big = Vector{Float64}(undef, 4n)
    bigw = similar(big)
    Test.@test nalloc(() -> FG.SphericalSampling._gauss_legendre_μ!(big, bigw)) <= 1
    Test.@test_throws DimensionMismatch FG.SphericalSampling.spherical_quadrature!(
        λb, φb, Vector{Float64}(undef, n + 1), FG.SphericalSampling.GaussLegendreSampling(), n)
end

Test.@testset "Driscoll–Healy DH1 / DH2" begin
    lmax = 5
    nlat = FG.SphericalSampling.nlat_for_bandlimit(FG.SphericalSampling.DriscollHealySampling(), lmax)
    Test.@test nlat == 2 * (lmax + 1)
    ax = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.DriscollHealySampling(), nlat)
    λ2 = ax.λ
    φ = ax.φ
    ax = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.DriscollHealyEqualSampling(), nlat)
    λ1 = ax.λ
    φ1 = ax.φ
    Test.@test length(λ2) == 2nlat
    Test.@test length(λ1) == nlat
    Test.@test φ ≈ φ1
    Test.@test φ[1] ≈ π / 2 atol = 1e-14       # north pole
    Test.@test φ[end] > -π / 2 + 1e-10         # south excluded
    w = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), nlat)
    Test.@test w[1] ≈ 0 atol = 1e-14           # north pole weight vanishes
    Test.@test FG.SphericalSampling.bandlimit(FG.SphericalSampling.DriscollHealySampling(), nlat) == lmax
end

Test.@testset "McEwen–Wiaux axes" begin
    lmax = 7
    nlat = FG.SphericalSampling.nlat_for_bandlimit(FG.SphericalSampling.McEwenWiauxSampling(), lmax)
    (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.McEwenWiauxSampling(), nlat)
    L = lmax + 1
    Test.@test length(φ) == L
    Test.@test length(λ) == 2L - 1
    θ = FG.SphericalSampling.colatitude.(φ)
    Test.@test θ[1] ≈ π / (2L - 1)
    Test.@test θ[end] ≈ π
end

Test.@testset "LatLonSampling" begin
    (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.LatLonSampling(), 5; nlon = 8)
    Test.@test length(λ) == 8
    Test.@test length(φ) == 5
    Test.@test φ[1] ≈ -π / 2
    Test.@test φ[end] ≈ π / 2
end

Test.@testset "HEALPix" begin
    s = FG.SphericalSampling.HEALPixSampling(1)
    Test.@test FG.SphericalSampling.healpix_npix(s) == 12
    Test.@test FG.SphericalSampling.healpix_nring(s) == 3
    Test.@test FG.SphericalSampling.healpix_pixel_area(s) ≈ 4π / 12
    (; λ, φ) = FG.SphericalSampling.spherical_points(s)
    Test.@test length(λ) == 12
    Test.@test length(φ) == 12
    s4 = FG.SphericalSampling.HEALPixSampling(4)
    Test.@test FG.SphericalSampling.healpix_npix(s4) == 12 * 16
    ax = FG.SphericalSampling.spherical_points(s4)
    λ4 = ax.λ
    φ4 = ax.φ
    Test.@test length(λ4) == 192
    # Equal-area: all pixels same area by construction; centers on |φ| < π/2
    Test.@test all(abs.(φ4) .≤ π / 2 + 1e-12)
    # nside=32 first ring: 4 pixels at same latitude, φ = 45° + 90°k
    ax = FG.SphericalSampling.spherical_points(FG.SphericalSampling.HEALPixSampling(32))
    λ32 = ax.λ
    φ32 = ax.φ
    Test.@test length(unique(round.(φ32[1:4]; digits = 12))) == 1
    Test.@test rad2deg.(λ32[1:4]) ≈ [45.0, 135.0, 225.0, 315.0] atol = 1e-8
    Test.@test rad2deg(FG.SphericalSampling.colatitude(φ32[1])) ≈ 1.46197116 atol = 1e-5
end

Test.@testset "Cubed sphere / Yin–Yang / icosahedral" begin
    (; λ, φ, panel) = FG.SphericalSampling.cubed_sphere_points(4)
    Test.@test length(λ) == 6 * 16
    Test.@test extrema(panel) == (1, 6)
    Test.@test all(abs.(φ) .≤ π / 2 + 1e-10)

    yy = FG.SphericalSampling.yin_yang_panels(6, 4)
    Test.@test length(yy.yin.λ) == 6
    Test.@test length(yy.yin.φ) == 4
    Test.@test size(yy.yang.λ) == (6, 4)
    Test.@test size(yy.yang.φ) == (6, 4)

    ax = FG.SphericalSampling.icosahedral_vertices(1)

    λi = ax.λ

    φi = ax.φ
    Test.@test length(λi) == 12
    ax = FG.SphericalSampling.icosahedral_vertices(2)
    λ2 = ax.λ
    φ2 = ax.φ
    # ν=2 geodesic: 10ν²+2 = 42 vertices
    Test.@test length(λ2) == 42
end

Test.@testset "Sampling → StructuredGrid" begin
    geom = FG.Geometry.SphericalGeometry(1.0)
    (; λ, φ) = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.ClenshawCurtisSampling(), 6)
    grid = FG.Grids.StructuredGrid(geom, λ, φ, trues(length(λ), length(φ)))
    Test.@test FG.Grids.isperiodic(grid, 1)
    Test.@test FG.Grids.size_tuple(grid) == (length(λ), length(φ))
end

Test.@testset "bang spherical_axes! / points!" begin
    nθ = 8
    sz = FG.SphericalSampling.axes_lengths(FG.SphericalSampling.ClenshawCurtisSampling(), nθ)
    λ = Vector{Float64}(undef, sz.nlon)
    φ = Vector{Float64}(undef, sz.nlat)
    FG.SphericalSampling.spherical_axes!(λ, φ, FG.SphericalSampling.ClenshawCurtisSampling(), nθ)
    ax = FG.SphericalSampling.spherical_axes(FG.SphericalSampling.ClenshawCurtisSampling(), nθ)
    λ2 = ax.λ
    φ2 = ax.φ
    Test.@test λ == λ2
    Test.@test φ == φ2

    n = FG.SphericalSampling.npoints(FG.SphericalSampling.HEALPixSampling(2))
    Λ = Vector{Float64}(undef, n)
    Φ = Vector{Float64}(undef, n)
    FG.SphericalSampling.spherical_points!(Λ, Φ, FG.SphericalSampling.HEALPixSampling(2))
    ax = FG.SphericalSampling.spherical_points(FG.SphericalSampling.HEALPixSampling(2))
    Λ2 = ax.λ
    Φ2 = ax.φ
    Test.@test Λ == Λ2
    Test.@test Φ == Φ2

    w = Vector{Float64}(undef, 12)
    FG.SphericalSampling.latitude_weights!(w, FG.SphericalSampling.GaussLegendreSampling(), 12)
    Test.@test sum(w) ≈ 2 atol = 1e-12
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
    for s in (FG.SphericalSampling.GaussLegendreSampling(), FG.SphericalSampling.DriscollHealySampling(),
              FG.SphericalSampling.DriscollHealyEqualSampling(), FG.SphericalSampling.ClenshawCurtisSampling())
        for nlat in (8, 16, 24)
            ax = FG.SphericalSampling.spherical_axes(s, nlat)
            w = FG.SphericalSampling.latitude_weights(s, nlat)
            # One normalization for every sampling: Σ w = ∫₀^π sinθ dθ = 2, carrying the sinθ
            # Jacobian and nothing else, so the longitude factor is always the caller's.
            Test.@test sum(w) ≈ 2 rtol = 1e-13
            Test.@test all(>(0), w) || s isa FG.SphericalSampling.AbstractDriscollHealySampling  # DH's polar node has zero weight
            # Exact for every single P_l the node count can resolve.
            for l in 1:(nlat - 1)
                Test.@test abs(sum(w[j] * legendre(l, sin(ax.φ[j])) for j in eachindex(ax.φ))) < 1e-11
            end
            # Gauss–Legendre goes further: exact to 2N-1, which is what makes it the sampling
            # whose quadrature is exact at its own stated band limit.
            if s isa FG.SphericalSampling.AbstractGaussLegendreSampling
                for l in nlat:(2nlat - 1)
                    Test.@test abs(sum(w[j] * legendre(l, sin(ax.φ[j])) for j in eachindex(ax.φ))) < 1e-11
                end
            end
        end
    end

    # Float32 all the way through.
    w32 = FG.SphericalSampling.latitude_weights(Float32, FG.SphericalSampling.ClenshawCurtisSampling(), 12)
    Test.@test eltype(w32) === Float32
    Test.@test sum(w32) ≈ 2 rtol = 1e-5

    # Samplings without a quadrature say so, without pointing at some other package.
    for s in (FG.SphericalSampling.McEwenWiauxSampling(), FG.SphericalSampling.LatLonSampling())
        err = try
            FG.SphericalSampling.latitude_weights(s, 8)
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
        (FG.SphericalSampling.GaussLegendreSampling(), 33, (;)),
        (FG.SphericalSampling.DriscollHealySampling(), 32, (;)),
        (FG.SphericalSampling.ClenshawCurtisSampling(), 33, (;)),
        (FG.SphericalSampling.McEwenWiauxSampling(), 33, (;)),
        (FG.SphericalSampling.LatLonSampling(), 33, (; nlon = 66)),
    )
        ax = FG.SphericalSampling.spherical_axes(s, nlat; kw...)
        sz = FG.SphericalSampling.axes_lengths(s, nlat; kw...)
        Test.@test length(ax.λ) == sz.nlon
        Test.@test length(ax.φ) == sz.nlat
        Test.@test issorted(ax.φ) || issorted(ax.φ; rev = true)
        Test.@test all(φ -> -π / 2 - 1e-12 ≤ φ ≤ π / 2 + 1e-12, ax.φ)
        Test.@test all(λ -> -1e-12 ≤ λ < 2π + 1e-12, ax.λ)
        Test.@test allunique(round.(ax.φ; digits = 12))
        p = FG.SphericalSampling.spherical_points(s, nlat; kw...)
        Test.@test length(p.λ) == sz.nlon * sz.nlat == FG.SphericalSampling.npoints(s, nlat; kw...)
    end
end

Test.@testset "Cubed-sphere points are cell centres, so panels do not share nodes" begin
    # Endpoint-inclusive panel coordinates put nodes ON the seams, so adjacent panels emit
    # coincident points — 12(n-2)+16 of them — while `_cubed_neighbor` treats those same edges
    # as folding onto a *different* panel's cells. Cell centres keep points and connectivity
    # consistent and give 6n² genuinely distinct nodes.
    for n in (1, 2, 4, 8, 16)
        p = FG.SphericalSampling.cubed_sphere_points(n)
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

Test.@testset "Yin–Yang cells tile each panel exactly; the overlap is resolution-independent" begin
    geo = FG.Geometry.SphericalGeometry()
    R2 = geo.R^2
    box = sqrt(2.0) * (3π / 2) * R2   # one panel's exact [-3π/4,3π/4] × [-π/4,π/4] area
    for (nlon, nlat) in ((8, 6), (16, 12), (48, 32))
        g = FG.Connectivity.unstructured_grid(FG.SphericalSampling.YinYangSampling(), nlon, nlat; geometry = geo)
        a = FG.Grids.measure(g)
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
    Test.@test minimum(FG.Grids.measure(FG.Connectivity.unstructured_grid(FG.SphericalSampling.YinYangSampling(), 192, 128))) /
               maximum(FG.Grids.measure(FG.Connectivity.unstructured_grid(FG.SphericalSampling.YinYangSampling(), 192, 128))) ≈
               cos(π / 4) rtol = 1e-2
end

Test.@testset "Coarsest cubed sphere (one node per face) is constructible" begin
    # `range(a, b; length = 1)` is an error, so n = 1 needs its own handling — and
    # `build_connectivity` accepts n ≥ 1, so this size is reachable.
    p = FG.SphericalSampling.cubed_sphere_points(1)
    Test.@test length(p.λ) == 6
    Test.@test length(unique(p.panel)) == 6
    # Each face center is a distinct point on the sphere.
    pts = [(cos(φ) * cos(λ), cos(φ) * sin(λ), sin(φ)) for (λ, φ) in zip(p.λ, p.φ)]
    Test.@test length(unique(x -> round.(x; digits = 9), pts)) == 6
    Test.@test all(v -> abs(hypot(v...) - 1) < 1e-12, pts)
    conn = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), 1)
    Test.@test FG.Connectivity.nnodes(conn) == 6
    Test.@test_throws ArgumentError FG.SphericalSampling.cubed_sphere_points(0)
    for n in (1, 2, 5)
        Test.@test length(FG.SphericalSampling.cubed_sphere_points(n).λ) == 6n^2
    end
end

Test.@testset "Icosahedral mesh is indexed topologically, not by hashing coordinates" begin
    for ν in (1, 2, 3, 5, 8)
        mesh = FG.SphericalSampling.icosahedral_mesh(ν)
        nv = length(mesh.λ)
        Test.@test nv == FG.SphericalSampling.icosahedral_nvertices(ν) == 10ν^2 + 2
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
    Test.@test allocs(() -> FG.SphericalSampling.icosahedral_mesh(4)) < 200
    Test.@test allocs(() -> FG.SphericalSampling.icosahedral_mesh(32)) < 200
end

Test.@testset "HEALPix RING neighbours are emitted in ascending order" begin
    C = FG.Connectivity
    # The dedup pass is an insertion sort, so its cost is the inversion count. Walking the compass
    # offsets in RING order leaves the emission almost sorted. The sort still runs and still
    # guarantees the order; this measures only that its input is nearly ordered.
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
        c = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(nside))
        n = FG.Connectivity.nnodes(c)
        Test.@test all(issorted(FG.Grids.neighbors(c, i)) for i in 1:n)
        Test.@test FG.Connectivity.is_symmetric_adjacency(c)
        Test.@test sort(unique(length(FG.Grids.neighbors(c, i)) for i in 1:n)) ⊆ [6, 7, 8]
    end
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

Test.@testset "HEALPix pixel centers are distinct and correctly ringed" begin
    # These tile the sphere, so no two may coincide, and the ring structure is fully determined:
    # 4nside-1 rings holding 4, 8, … 4(nside-1), then 4nside for 2nside+1 rings, then back down.
    # The ring structure is fully determined, so it is asserted ring by ring, not just by count.
    for nside in (1, 2, 4, 8, 16)
        p = FG.SphericalSampling.spherical_points(FG.SphericalSampling.HEALPixSampling(nside))
        npix = FG.SphericalSampling.healpix_npix(nside)
        Test.@test length(p.λ) == npix
        Test.@test length(unique(collect(zip(p.λ, p.φ)))) == npix

        zs = sort(unique(round.(sin.(p.φ); digits = 11)); rev = true)
        Test.@test length(zs) == FG.SphericalSampling.healpix_nring(nside) == 4nside - 1
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

Test.@testset "Quadrature-exactness trait matches measured exactness" begin
    # The trait is about integrating PRODUCTS of two degree-lmax functions, which is what
    # spectral analysis forms. Clenshaw–Curtis's grid represents to N-1 but its quadrature only
    # integrates a single P_l to N-1, so it cannot claim exactness at its own band limit.
    Test.@test FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.GaussLegendreSampling())
    Test.@test FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.DriscollHealySampling())
    Test.@test !FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.ClenshawCurtisSampling())
    Test.@test !FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.McEwenWiauxSampling())
    Test.@test !FG.SphericalSampling.admits_exact_bandlimited_quadrature(FG.SphericalSampling.LatLonSampling())

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
    for s in (FG.SphericalSampling.GaussLegendreSampling(), FG.SphericalSampling.ClenshawCurtisSampling())
        ax = FG.SphericalSampling.spherical_axes(s, n)
        w = FG.SphericalSampling.latitude_weights(s, n)
        lmax = FG.SphericalSampling.bandlimit(s, n)
        err = abs(sum(w[j] * legendre(2lmax, sin(ax.φ[j])) for j in eachindex(ax.φ)))
        if FG.SphericalSampling.admits_exact_bandlimited_quadrature(s)
            Test.@test err < 1e-10
        else
            Test.@test err > 1e-6      # genuinely inexact at 2·lmax, as the trait now says
        end
    end
end

Test.@testset "Every ring-laid-out sampling answers the ring API" begin
    S = FG.SphericalSampling
    # `nrings`/`nlon_per_ring` are what a caller walking a map ring by ring loops over — a
    # per-ring longitude transform, a zonal reduction. They existed for two samplings, so that
    # caller had to branch on sampling type, which is what these accessors exist to prevent.
    cases = (("HEALPix", S.HEALPixSampling(4), ()),
             ("Octahedral", S.OctahedralGaussianSampling(8), ()),
             ("Reduced", S.ReducedGaussianSampling([8, 12, 16, 12, 8]), ()),
             ("GaussLegendre", S.GaussLegendreSampling(), (16,)),
             ("ClenshawCurtis", S.ClenshawCurtisSampling(), (16,)),
             ("DriscollHealy", S.DriscollHealySampling(), (16,)))
    for (_, s, args) in cases
        r = S.nrings(s, args...)
        v = S.nlon_per_ring(s, args...)
        Test.@test r ≥ 1
        Test.@test length(v) == r                 # one entry per ring
        Test.@test all(>(0), v)
        # The independent cross-check: the rings must account for every point.
        Test.@test sum(v) == S.npoints(s, args...)
    end
    # HEALPix ring widths are the ones `ring_info` reports, and they are symmetric about the
    # equator, widening 4, 8, … through the cap and flat across the belt.
    let ns = 4, v = S.nlon_per_ring(S.HEALPixSampling(ns))
        Test.@test v == [S.ring_info(ns, r).ringpix for r in 1:(4ns - 1)]
        Test.@test v == reverse(v)
        Test.@test v[1] == 4 && v[ns] == 4ns && v[2ns] == 4ns
    end
    # A tensor-product sampling is a rectangle, and honours an explicit `nlon`.
    Test.@test S.nlon_per_ring(S.GaussLegendreSampling(), 8; nlon = 20) == fill(20, 8)
end

Test.@testset "Octahedral and reduced Gaussian grids" begin
    SS = FG.SphericalSampling
    # The defining rule: 20 longitudes on the ring nearest the pole, +4 per ring, 4N(N+9) in all.
    for N in (1, 2, 5, 16, 80, 320)
        s = SS.OctahedralGaussianSampling(N)
        counts = SS.nlon_per_ring(s)
        Test.@test length(counts) == SS.nrings(s) == 2N
        Test.@test counts[1] == 20
        Test.@test N == 1 || all(diff(counts[1:N]) .== 4)
        Test.@test sum(counts) == 4 * N * (N + 9) == SS.npoints(s)
        Test.@test counts == reverse(counts)          # the hemispheres mirror
    end
    # Latitudes are the Gaussian ones, north to south, and the weights are Gauss–Legendre's.
    for N in (4, 12)
        s = SS.OctahedralGaussianSampling(N)
        lats = SS.ring_latitudes(s)
        Test.@test issorted(lats; rev = true)
        Test.@test sort(lats) ≈ FG.SphericalSampling.spherical_axes(FG.SphericalSampling.GaussLegendreSampling(), 2N).φ atol = 1e-12
        w = SS.latitude_weights(s)
        Test.@test sum(w) ≈ 2 rtol = 1e-13
        Test.@test all(>(0), w)
        # Exact for every P_l the ring count resolves.
        function legendre(l, x)
            l == 0 && return one(x)
            p0, p1 = one(x), x
            for k in 1:(l - 1)
                p0, p1 = p1, ((2k + 1) * x * p1 - k * p0) / (k + 1)
            end
            return p1
        end
        for l in 1:(2N - 1)
            Test.@test abs(sum(w[j] * legendre(l, sin(lats[j])) for j in eachindex(lats))) < 1e-11
        end
        # The full-sphere integral, with the per-ring longitude factor the varying counts require.
        counts = SS.nlon_per_ring(s)
        Test.@test sum(w[j] * (2π / counts[j]) * counts[j] for j in eachindex(counts)) ≈ 4π rtol = 1e-13
    end
    # Points are distinct, on the sphere, and ring-ordered.
    for N in (2, 8)
        s = SS.OctahedralGaussianSampling(N)
        p = SS.spherical_points(s)
        n = SS.npoints(s)
        Test.@test length(p.λ) == length(p.φ) == n
        Test.@test all(φ -> -π/2 ≤ φ ≤ π/2, p.φ)
        Test.@test all(λ -> 0 ≤ λ < 2π + 1e-12, p.λ)
        Test.@test length(unique([(round(a; digits = 12), round(b; digits = 12))
                                  for (a, b) in zip(p.λ, p.φ)])) == n
    end
    # A ring sampling is not a tensor product, but it is iso-latitude and has an exact quadrature.
    oct = SS.OctahedralGaussianSampling(4)
    Test.@test !SS.is_tensor_product(oct)
    Test.@test SS.is_iso_latitude(oct)
    Test.@test SS.admits_exact_bandlimited_quadrature(oct)
    Test.@test_throws ArgumentError SS.OctahedralGaussianSampling(0)

    # An explicit table is the other way a reduced grid is specified.
    rg = SS.ReducedGaussianSampling([20, 24, 24, 20])
    Test.@test SS.nrings(rg) == 4 && SS.npoints(rg) == 88
    Test.@test length(SS.spherical_points(rg).λ) == 88
    Test.@test issorted(SS.ring_latitudes(rg); rev = true)
    Test.@test_throws ArgumentError SS.ReducedGaussianSampling(Int[])
    Test.@test_throws ArgumentError SS.ReducedGaussianSampling([4, 0])
end

Test.@testset "Fibonacci lattice" begin
    SS = FG.SphericalSampling
    for n in (1, 7, 100, 3000)
        s = SS.FibonacciSampling(n)
        p = SS.spherical_points(s)
        Test.@test SS.npoints(s) == n == length(p.λ) == length(p.φ)
        Test.@test all(φ -> -π/2 - 1e-12 ≤ φ ≤ π/2 + 1e-12, p.φ)
        Test.@test all(λ -> 0 ≤ λ < 2π + 1e-12, p.λ)
        Test.@test length(unique([(round(a; digits = 10), round(b; digits = 10))
                                  for (a, b) in zip(p.λ, p.φ)])) == n
        # `z` advances in exactly equal steps, which is what makes the bands equal-area.
        n < 2 || Test.@test maximum(abs.(diff(sin.(p.φ)) .- 2 / n)) < 1e-12
    end
    # Quasi-uniform: nearest-neighbour separation barely varies, unlike a lat–lon grid's.
    p = SS.spherical_points(SS.FibonacciSampling(1500))
    v = [(cos(p.φ[i]) * cos(p.λ[i]), cos(p.φ[i]) * sin(p.λ[i]), sin(p.φ[i])) for i in 1:1500]
    nn = [minimum(sqrt(sum(abs2, v[i] .- v[j])) for j in 1:1500 if j != i) for i in 1:150]
    Test.@test minimum(nn) / maximum(nn) > 0.5
    Test.@test SS.is_equal_area(SS.FibonacciSampling(10))
    Test.@test_throws ArgumentError SS.FibonacciSampling(0)
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
    Test.@test_throws DimensionMismatch SS.spherical_points!(zeros(3), zeros(3),
                                                             SS.FibonacciSampling(4))
end

Test.@testset "HEALPix pixel indexing round-trips in both schemes" begin
    SS = FG.SphericalSampling
    for nside in (1, 2, 3, 4, 8, 16)
        npix = SS.healpix_npix(nside)
        # Locating a pixel's own centre must return that pixel, for every pixel.
        for p in 0:(npix - 1)
            θ, ϕ = SS.pix2ang(nside, p)
            Test.@test SS.ang2pix(nside, θ, ϕ) == p
            Test.@test SS.vec2pix(nside, SS.pix2vec(nside, p)) == p
        end
        Test.@test_throws ArgumentError SS.pix2ang(nside, npix)
        Test.@test_throws ArgumentError SS.pix2ang(nside, -1)
    end
    # RING <-> NESTED is a bijection, and both name the same point.
    for nside in (1, 2, 4, 8)
        npix = SS.healpix_npix(nside)
        seen = falses(npix)
        for p in 0:(npix - 1)
            n = SS.ring2nest(nside, p)
            Test.@test 0 ≤ n < npix
            Test.@test !seen[n + 1]
            seen[n + 1] = true
            Test.@test SS.nest2ring(nside, n) == p
            a = SS.pix2ang(nside, n; scheme = SS.Nested())
            b = SS.pix2ang(nside, p)
            Test.@test a[1] ≈ b[1] && a[2] ≈ b[2]
            Test.@test SS.ang2pix(nside, b[1], b[2]; scheme = SS.Nested()) == n
        end
        Test.@test all(seen)
    end
    # NESTED needs a power-of-two nside; RING does not.
    Test.@test_throws ArgumentError SS.ring2nest(3, 0)
    Test.@test_throws ArgumentError SS.nest2ring(3, 0)
    Test.@test_throws ArgumentError SS.ang2pix(3, 1.0, 1.0; scheme = SS.Nested())
    Test.@test SS.ang2pix(3, 1.0, 1.0) isa Int
    # `pix2ang` must name the same points `spherical_points` lists.
    for nside in (2, 8)
        p = SS.spherical_points(FG.SphericalSampling.HEALPixSampling(nside))
        for i in 0:(SS.healpix_npix(nside) - 1)
            θ, ϕ = SS.pix2ang(nside, i)
            Test.@test SS.geographic_latitude(θ) ≈ p.φ[i + 1]
            Test.@test ϕ ≈ p.λ[i + 1]
        end
    end
end

Test.@testset "A ring can be reached one at a time, in O(1), without building the table" begin
    SS = FG.SphericalSampling
    samplings = (
        ("octahedral",     SS.OctahedralGaussianSampling(6),        ()),
        ("octahedral N=1", SS.OctahedralGaussianSampling(1),        ()),
        ("reduced table",  SS.ReducedGaussianSampling([20, 24, 24, 20]), ()),
        ("healpix",        SS.HEALPixSampling(4),                   ()),
        ("healpix ns=1",   SS.HEALPixSampling(1),                   ()),
        ("gauss-legendre", SS.GaussLegendreSampling(),              (8,)),
        ("clenshaw",       SS.ClenshawCurtisSampling(),             (5,)),
    )
    for (name, s, args) in samplings
        table = SS.nlon_per_ring(s, args...)
        nr = SS.nrings(s, args...)
        Test.@test length(table) == nr

        # The scalar accessor and the table must agree everywhere, and the ranges must tile the
        # flattened point vector exactly: contiguous, in order, no gap and no overlap.
        next = 1
        ok = true
        for r in 1:nr
            SS.nlon_in_ring(s, args..., r) == table[r] || (ok = false)
            rng = SS.ring_range(s, args..., r)
            (first(rng) == next && length(rng) == table[r]) || (ok = false)
            next = last(rng) + 1
        end
        ok || println("    ring accessors disagree for ", name)
        Test.@test ok
        Test.@test next - 1 == sum(table)

        Test.@test_throws ArgumentError SS.nlon_in_ring(s, args..., 0)
        Test.@test_throws ArgumentError SS.ring_range(s, args..., nr + 1)
    end

    # And the ranges really are where `spherical_points` puts each ring: every point in ring r
    # must share that ring's latitude.
    for s in (SS.OctahedralGaussianSampling(5), SS.ReducedGaussianSampling([20, 24, 24, 20]))
        pts = SS.spherical_points(s)
        same = true
        for r in 1:SS.nrings(s)
            rng = SS.ring_range(s, r)
            allequal(view(pts.φ, rng)) || (same = false)
        end
        Test.@test same
    end
    for ns in (1, 4)
        s = SS.HEALPixSampling(ns)
        pts = SS.spherical_points(s)
        Test.@test all(allequal(view(pts.φ, SS.ring_range(s, r))) for r in 1:SS.nrings(s))
    end

    # For a tensor product the range also asserts an ORDERING — longitude fastest within a ring —
    # which the tiling check above cannot see, since a longitude-major layout tiles just as well.
    for (s, nlat) in ((SS.GaussLegendreSampling(), 8), (SS.ClenshawCurtisSampling(), 5),
                      (SS.DriscollHealySampling(), 6), (SS.DriscollHealyEqualSampling(), 6))
        pts = SS.spherical_points(s, nlat)
        ax = SS.spherical_axes(s, nlat)
        Test.@test all(allequal(view(pts.φ, SS.ring_range(s, nlat, r))) for r in 1:nlat)
        Test.@test all(pts.φ[first(SS.ring_range(s, nlat, r))] == ax.φ[r] for r in 1:nlat)
    end

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
        ref = SS.spherical_points(s)
        Test.@test λ == ref.λ && φ == ref.φ
    end
    Test.@test_throws DimensionMismatch SS.spherical_points!(
        Vector{Float64}(undef, SS.npoints(SS.OctahedralGaussianSampling(3))),
        Vector{Float64}(undef, SS.npoints(SS.OctahedralGaussianSampling(3))),
        SS.OctahedralGaussianSampling(3); scratch = Vector{Float64}(undef, 2),
    )
end

Test.@testset "The NESTED bit interleave is exact over its whole domain" begin
    SS = FG.SphericalSampling
    # The cascade replaced a per-bit loop; it must agree with the definition bit for bit, not
    # merely round-trip. Spreading places input bit b at output bit 2b.
    defn(v) = sum(((v >> b) & 1) << (2b) for b in 0:31; init = 0)
    Test.@test all(SS._spread_bits(v) == defn(v) for v in 0:4095)
    Test.@test all(SS._spread_bits(v) == defn(v) for v in (1 << 20, (1 << 21) - 1, 12345678))
    Test.@test all(SS._compress_bits(SS._spread_bits(v)) == v for v in 0:4095)
    Test.@test all(SS._compress_bits(SS._spread_bits(v)) == v for v in (1 << 20, 987654))
    # Interleaving two coordinates must not let one bleed into the other.
    Test.@test all(SS._compress_bits(SS._spread_bits(a) | (SS._spread_bits(b) << 1)) == a &&
                   SS._compress_bits((SS._spread_bits(a) | (SS._spread_bits(b) << 1)) >> 1) == b
                   for a in 0:63, b in 0:63)
    Test.@test _alloc(SS._spread_bits, 12345) == 0
    Test.@test _alloc(SS._compress_bits, 12345) == 0
end
