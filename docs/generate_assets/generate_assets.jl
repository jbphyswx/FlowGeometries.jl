"""
Generate static figure assets for FlowGeometries.jl docs and README.md.

Run from the repo root:
    julia --project=docs/generate_assets docs/generate_assets/generate_assets.jl

Figures are written to `docs/src/assets/`.
"""

using FlowGeometries: FlowGeometries as FG
using CairoMakie: CairoMakie as CM
using FFTW: FFTW   # an FFT backend, so the O(n log n) equiangular path is what gets timed
using Statistics: Statistics
using Printf: @sprintf

const ASSETS_DIR = joinpath(@__DIR__, "..", "src", "assets")
mkpath(ASSETS_DIR)

const R = FG.Geometry.SphericalGeometry().R
const VIEW = (0.35, 0.45)      # (λ₀, φ₀) of the orthographic viewpoint

# ─── Projection helpers ────────────────────────────────────────────────────

"""
    ortho(λ, φ, λ₀, φ₀) -> (x, y, visible)

Orthographic projection onto the disc seen from `(λ₀, φ₀)`.

`visible` is false on the far hemisphere. Those points must be DROPPED, not drawn: back-face points
project onto the front one and make a sampling look twice as dense as it is, which is exactly the
kind of figure that misleads.
"""
function ortho(λ, φ, λ₀, φ₀)
    sinφ₀, cosφ₀ = sincos(φ₀)
    sinφ, cosφ = sincos(φ)
    Δλ = λ - λ₀
    cosc = sinφ₀ * sinφ + cosφ₀ * cosφ * cos(Δλ)
    return cosφ * sin(Δλ), cosφ₀ * sinφ - sinφ₀ * cosφ * cos(Δλ), cosc ≥ 0
end

function disc_axis(fig, row, col, title)
    ax = CM.Axis(fig[row, col]; title = title, aspect = CM.DataAspect())
    CM.hidedecorations!(ax)
    CM.hidespines!(ax)
    CM.limits!(ax, -1.1, 1.1, -1.1, 1.1)
    θ = range(0, 2π; length = 400)
    CM.lines!(ax, cos.(θ), sin.(θ); color = (:gray, 0.5), linewidth = 1.0)
    return ax
end

"Scatter the visible hemisphere of a point set."
function sphere_scatter!(ax, λ, φ; color = :black, markersize = 3.0)
    xs = Float64[]
    ys = Float64[]
    for i in eachindex(λ)
        x, y, vis = ortho(λ[i], φ[i], VIEW...)
        vis && (push!(xs, x); push!(ys, y))
    end
    CM.scatter!(ax, xs, ys; color = color, markersize = markersize, strokewidth = 0)
    return length(xs)
end

"Flatten tensor-product axes into a point list."
function tp_points(s, nlat)
    a = FG.SphericalSampling.spherical_axes(s, nlat)
    λ = [a.λ[i] for j in eachindex(a.φ) for i in eachindex(a.λ)]
    φ = [a.φ[j] for j in eachindex(a.φ) for i in eachindex(a.λ)]
    return λ, φ
end

# ─── 1. The sampling families ──────────────────────────────────────────────

function fig_samplings()
    SS = FG.SphericalSampling
    cases = [
        ("Gauss–Legendre", tp_points(SS.GaussLegendreSampling(), 24)),
        ("Clenshaw–Curtis", tp_points(SS.ClenshawCurtisSampling(), 24)),
        # N chosen so the point count is comparable with the tensor-product panels beside it, which is
        # the whole point of the comparison: the same budget, spent differently.
        ("Octahedral Gaussian, N 14",
         let p = SS.spherical_points(SS.OctahedralGaussianSampling(14)); (p.λ, p.φ) end),
        ("HEALPix, nside 8",
         let p = SS.spherical_points(SS.HEALPixSampling(8)); (p.λ, p.φ) end),
        ("Cubed sphere, n 16", let p = SS.cubed_sphere_points(16); (p.λ, p.φ) end),
        ("Icosahedral, ν 12", let p = SS.icosahedral_vertices(12); (p.λ, p.φ) end),
        ("Fibonacci lattice, n 1400",
         let p = SS.spherical_points(SS.FibonacciSampling(1400)); (p.λ, p.φ) end),
        ("Yin–Yang, 32×22",
         let p = SS.spherical_points(SS.YinYangSampling(), 32, 22); (p.λ, p.φ) end),
    ]
    fig = CM.Figure(; size = (1300, 700))
    for (k, (name, (λ, φ))) in enumerate(cases)
        r, c = fldmod1(k, 4)
        ax = disc_axis(fig, r + 1, c, "$name\n$(length(λ)) points")
        sphere_scatter!(ax, λ, φ; markersize = 2.8)
    end
    CM.Label(fig[1, 1:4],
             "Where the points go: pole-crowding, ring-reduced, quasi-uniform, and overset";
             fontsize = 19, font = :bold)
    CM.save(joinpath(ASSETS_DIR, "samplings.png"), fig; px_per_unit = 2)
end

# ─── 2. Cell areas are not interchangeable ─────────────────────────────────

function fig_cell_areas()
    cases = [
        ("HEALPix", FG.Connectivity.unstructured_grid(FG.SphericalSampling.HEALPixSampling(16))),
        ("Cubed sphere", FG.Connectivity.unstructured_grid(FG.SphericalSampling.CubedSphereSampling(), 24)),
        ("Icosahedral", FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(16))),
    ]
    fig = CM.Figure(; size = (980, 420))
    for (k, (name, g)) in enumerate(cases)
        a = FG.Grids.measure(g)
        rel = a ./ Statistics.mean(a)
        ax = disc_axis(fig, 2, k, @sprintf("%s — min/max %.2f", name, minimum(a) / maximum(a)))
        xs = Float64[]; ys = Float64[]; cs = Float64[]
        for i in eachindex(g.λ)
            x, y, vis = ortho(g.λ[i], g.φ[i], VIEW...)
            vis && (push!(xs, x); push!(ys, y); push!(cs, rel[i]))
        end
        CM.scatter!(ax, xs, ys; color = cs, colormap = :viridis, colorrange = (0.5, 1.5),
                    markersize = 5.5, strokewidth = 0)
    end
    CM.Colorbar(fig[2, 4]; colormap = :viridis, limits = (0.5, 1.5), label = "cell area / mean")
    CM.Label(fig[1, 1:4],
             "Cell area relative to the mean — a uniform 4πR²/N default is exact only for HEALPix";
             fontsize = 17, font = :bold)
    CM.save(joinpath(ASSETS_DIR, "cell_areas.png"), fig; px_per_unit = 2)
end

# ─── 3. Quadrature: exactness and cost ─────────────────────────────────────

"Relative error of ∫₋₁¹ μᵈ dμ under a sampling's latitude rule."
function quad_error(s, nlat, d)
    q = FG.SphericalSampling.spherical_quadrature(s, nlat)
    got = sum(q.w .* sin.(q.φ) .^ d)
    want = isodd(d) ? 0.0 : 2 / (d + 1)
    return abs(got - want) / (isodd(d) ? 1.0 : abs(want))
end

function fig_quadrature()
    nlat = 32
    degrees = collect(0:2:(2nlat + 8))
    fig = CM.Figure(; size = (980, 400))

    ax1 = CM.Axis(fig[1, 1]; xlabel = "polynomial degree", ylabel = "relative error",
                  yscale = log10, title = "Exactness of the latitude rule (nlat = $nlat)")
    for (s, name, col) in ((FG.SphericalSampling.GaussLegendreSampling(), "Gauss–Legendre", :dodgerblue),
                           (FG.SphericalSampling.DriscollHealySampling(), "Driscoll–Healy", :seagreen),
                           (FG.SphericalSampling.ClenshawCurtisSampling(), "Clenshaw–Curtis", :darkorange))
        e = [max(quad_error(s, nlat, d), 1e-17) for d in degrees]
        CM.lines!(ax1, degrees, e; label = name, color = col, linewidth = 2)
    end
    CM.vlines!(ax1, [2nlat - 1]; color = (:gray, 0.7), linestyle = :dash)
    CM.text!(ax1, 2nlat - 1, 3e-17; text = "  2N−1", align = (:left, :bottom),
             fontsize = 11, color = :gray)
    CM.axislegend(ax1; position = :lt, framevisible = false)

    ax2 = CM.Axis(fig[1, 2]; xlabel = "nlat", ylabel = "time (ms)", xscale = log10,
                  yscale = log10,
                  title = "Cost with an FFT backend loaded")
    sizes = [64, 128, 256, 512, 1024, 2048]
    styles = (:solid, :dash, :dot)   # DH and CC cost the same; without this one hides the other
    for (k, (s, name, col)) in enumerate(((FG.SphericalSampling.GaussLegendreSampling(), "Gauss–Legendre", :dodgerblue),
                                          (FG.SphericalSampling.DriscollHealySampling(), "Driscoll–Healy", :seagreen),
                                          (FG.SphericalSampling.ClenshawCurtisSampling(), "Clenshaw–Curtis", :darkorange)))
        t = map(sizes) do n
            FG.SphericalSampling.latitude_weights(s, n)
            1e3 * minimum(@elapsed(FG.SphericalSampling.latitude_weights(s, n)) for _ in 1:5)
        end
        # Least-squares slope in log-log IS the scaling exponent; show it rather than assert it.
        lx = log.(Float64.(sizes)); ly = log.(t)
        p̂ = sum((lx .- Statistics.mean(lx)) .* (ly .- Statistics.mean(ly))) /
            sum((lx .- Statistics.mean(lx)) .^ 2)
        CM.scatterlines!(ax2, Float64.(sizes), t; color = col, linewidth = 2, markersize = 8,
                         linestyle = styles[k], label = @sprintf("%s  (p = %.2f)", name, p̂))
    end
    CM.axislegend(ax2; position = :lt, framevisible = false)

    CM.save(joinpath(ASSETS_DIR, "quadrature.png"), fig; px_per_unit = 2)
end

# ─── 4. Connectivity is topology ───────────────────────────────────────────

function fig_connectivity()
    fig = CM.Figure(; size = (940, 420))

    nside = 8
    g = FG.Connectivity.unstructured_grid(FG.SphericalSampling.HEALPixSampling(nside))
    conn = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(nside))
    ax = disc_axis(fig, 2, 1, "HEALPix RING adjacency (nside $nside)")
    for i in 1:FG.Connectivity.nnodes(conn)
        xi, yi, vi = ortho(g.λ[i], g.φ[i], VIEW...)
        vi || continue
        for j in FG.Grids.neighbors(conn, i)
            j > i || continue
            xj, yj, vj = ortho(g.λ[j], g.φ[j], VIEW...)
            vj || continue
            CM.lines!(ax, [xi, xj], [yi, yj]; color = (:steelblue, 0.45), linewidth = 0.6)
        end
    end
    sphere_scatter!(ax, g.λ, g.φ; markersize = 3.0)

    ν = 8
    mesh = FG.SphericalSampling.icosahedral_mesh(ν)
    ax2 = disc_axis(fig, 2, 2, "Icosahedral mesh (ν $ν) — $(length(mesh.triangles)) triangles")
    for (a, b) in mesh.edges
        xa, ya, va = ortho(mesh.λ[a], mesh.φ[a], VIEW...)
        xb, yb, vb = ortho(mesh.λ[b], mesh.φ[b], VIEW...)
        (va && vb) || continue
        CM.lines!(ax2, [xa, xb], [ya, yb]; color = (:seagreen, 0.5), linewidth = 0.6)
    end
    sphere_scatter!(ax2, mesh.λ, mesh.φ; markersize = 3.0)

    CM.Label(fig[1, 1:2], "Connectivity reads extent, wrapping and activity — never coordinates";
             fontsize = 17, font = :bold)
    CM.save(joinpath(ASSETS_DIR, "connectivity.png"), fig; px_per_unit = 2)
end

# ─── 5. Yin–Yang overlap is geometry, not error ────────────────────────────

function fig_yinyang()
    fig = CM.Figure(; size = (940, 420))
    nlon, nlat = 48, 32
    p = FG.SphericalSampling.spherical_points(FG.SphericalSampling.YinYangSampling(), nlon, nlat)
    np = nlon * nlat

    ax = disc_axis(fig, 2, 1, "Two panels, overlapping by construction")
    sphere_scatter!(ax, p.λ[1:np], p.φ[1:np]; color = (:crimson, 0.8), markersize = 3.4)
    sphere_scatter!(ax, p.λ[(np + 1):end], p.φ[(np + 1):end];
                    color = (:dodgerblue, 0.8), markersize = 3.4)

    ax2 = CM.Axis(fig[2, 2]; xlabel = "nlon  (nlat = 2·nlon/3)", ylabel = "Σ cell area / 4πR²",
                  title = "The excess does not shrink with resolution")
    ns = [12, 24, 48, 96, 192]
    ratio = map(ns) do n
        sum(FG.Grids.measure(FG.Connectivity.unstructured_grid(FG.SphericalSampling.YinYangSampling(), n, 2n ÷ 3))) / (4π * R^2)
    end
    CM.hlines!(ax2, [3 * sqrt(2) / 4]; color = (:gray, 0.8), linestyle = :dash)
    CM.scatterlines!(ax2, Float64.(ns), ratio; color = :purple, linewidth = 2, markersize = 9)
    CM.text!(ax2, 60, 3 * sqrt(2) / 4 - 0.012; text = "3√2/4 = 1.06066",
             fontsize = 12, color = :gray)
    CM.ylims!(ax2, 0.99, 1.11)

    CM.Label(fig[1, 1:2], "Yin–Yang: a 6.07% area excess at every resolution";
             fontsize = 17, font = :bold)
    CM.save(joinpath(ASSETS_DIR, "yinyang.png"), fig; px_per_unit = 2)
end


# ─── 6. Cartesian grids: uniform, stretched, and what the type records ─────

function fig_cartesian()
    geo = FG.Geometry.CartesianGeometry()
    fig = CM.Figure(; size = (1020, 470))

    # A uniform grid and a stretched one, drawn as their cell boundaries.
    nx = 24
    xu = range(0.0, 1.0; length = nx + 1)[1:nx]
    # stretched: cells cluster hard toward the lower edge
    t = range(0.0, 1.0; length = nx + 1)[1:nx]
    xs = @. (exp(4.0 * t) - 1) / (exp(4.0) - 1)

    for (k, (name, ax1, ax2)) in enumerate((
        ("uniform × uniform", xu, xu),
        ("uniform × stretched", xu, xs),
    ))
        g = FG.Grids.StructuredGrid(geo, ax1, ax2)
        fx = FG.Discretization.faces(FG.Grids.coordinates(g, 1))
        fy = FG.Discretization.faces(FG.Grids.coordinates(g, 2))
        uni1 = FG.Grids.isuniform(g, 1)
        uni2 = FG.Grids.isuniform(g, 2)
        ax = CM.Axis(fig[2, k];
                     title = @sprintf("%s\nisuniform = (%s, %s)", name, uni1, uni2),
                     aspect = CM.DataAspect(), xlabel = "x", ylabel = "y")
        # shade each cell by its own measure first, so the cell boundaries stay legible on top of it
        m = FG.Grids.measure(g)
        CM.heatmap!(ax, FG.Grids.coordinates(g, 1), FG.Grids.coordinates(g, 2),
                    [m[i, j] for i in 1:nx, j in 1:nx]; colormap = :viridis, alpha = 0.35)
        for x in fx
            CM.lines!(ax, [x, x], [first(fy), last(fy)]; color = (:black, 0.75), linewidth = 0.6)
        end
        for y in fy
            CM.lines!(ax, [first(fx), last(fx)], [y, y]; color = (:black, 0.75), linewidth = 0.6)
        end
        CM.xlims!(ax, first(fx), last(fx))
        CM.ylims!(ax, first(fy), last(fy))
    end

    # A masked domain, and the topology the mask has.
    m = trues(40, 40)
    for i in 1:40, j in 1:40
        (hypot(i - 14, j - 26) < 6 || hypot(i - 28, j - 13) < 5) && (m[i, j] = false)
    end
    gm = FG.Grids.StructuredGrid(geo, range(0.0, 1.0; length = 40), range(0.0, 1.0; length = 40), m)
    holes = FG.Connectivity.count_holes(gm)
    int = FG.Connectivity.interior(gm)
    bnd = FG.Connectivity.boundary_cells(gm)
    code = [!m[i, j] ? 0.0 : (int[i, j] ? 2.0 : 1.0) for i in 1:40, j in 1:40]
    ax3 = CM.Axis(fig[2, 3];
                  title = @sprintf("mask topology\n%d holes, %d boundary cells", holes, count(bnd)),
                  aspect = CM.DataAspect())
    CM.heatmap!(ax3, 1:40, 1:40, code; colormap = CM.cgrad([:white, :orangered, :steelblue], 3;
                                                           categorical = true))
    CM.hidedecorations!(ax3)
    CM.hidespines!(ax3)
    CM.Legend(fig[3, 3],
              [CM.PolyElement(; color = c, strokecolor = :gray40, strokewidth = 1)
               for c in (:white, :orangered, :steelblue)],
              ["masked out", "boundary", "interior"];
              orientation = :horizontal, framevisible = false, labelsize = 11)

    CM.Label(fig[1, 1:3],
             "Cartesian grids: each direction keeps its own spacing guarantee, and the mask has a topology";
             fontsize = 16, font = :bold)
    CM.save(joinpath(ASSETS_DIR, "cartesian.png"), fig; px_per_unit = 2)
end

# ─── Run ───────────────────────────────────────────────────────────────────

function main()
    CM.set_theme!(CM.theme_light())
    for (name, f) in (("samplings", fig_samplings), ("cell_areas", fig_cell_areas),
                      ("quadrature", fig_quadrature), ("connectivity", fig_connectivity),
                      ("yinyang", fig_yinyang), ("cartesian", fig_cartesian))
        print("  ", rpad(name, 14))
        f()
        println("ok")
    end
    println("figures written to ", normpath(ASSETS_DIR))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
