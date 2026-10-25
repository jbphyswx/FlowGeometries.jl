# Wall-clock measurements for FlowGeometries.jl, run on demand; the test suite asserts the same claims
# by counting operations.
#
#     julia --project=benchmark benchmark/benchmarks.jl
#     julia --project=benchmark benchmark/benchmarks.jl stencil     # one group
#
# Each measurement is the minimum over blocks of repeated calls, which rejects a garbage-collection
# pause.

using FlowGeometries: FlowGeometries as FG
using NearestNeighbors: NearestNeighbors
using Printf: Printf

const C = FG.Connectivity
const D = FG.Discretization
const O = FG.Operators
const GD = FG.Grids
const SS = FG.SphericalSampling

const CART = FG.Geometry.CartesianGeometry{Float64}()

"""
    best(f, reps; blocks = 3) -> seconds per call

Minimum over `blocks` blocks of `reps` calls; a block a collection lands in is discarded by the minimum.
"""
function best(f, reps::Int; blocks::Int = 3)
    f()
    t = Inf
    for _ in 1:blocks
        e = @elapsed for _ in 1:reps
            f()
        end
        t = min(t, e / reps)
    end
    return t
end

row(name, t) = Printf.@printf("  %-46s %10.2f µs\n", name, t * 1e6)
ratio_row(name, a, b) = Printf.@printf("  %-46s %10.2f×\n", name, b / a)
group(title) = (println(); println(title); println("  " * "-"^58))

# ---------------------------------------------------------------------------

curv(n) = GD.CurvilinearGrid(CART,
                             [t for t in range(0.0, 1.0 * (n - 1); length = n), _ in 1:n],
                             [t for _ in 1:n, t in range(0.0, 1.0 * (n - 1); length = n)],
                             trues(n, n); measure = fill(1.0, n, n))

stretched(n) = GD.StructuredGrid(CART, cumsum(1.0 .+ 0.5 .* sin.(range(0, 3π; length = n))),
                                 collect(0.0:3.0))

# `apply_stencil!`: the sweep, over both differenced directions and both sizes the issue reports.
function bench_stencil()
    group("apply_stencil! — the sweep")
    for n in (256, 1024)
        x = collect(range(0.0, 1.0; length = n))
        g = GD.StructuredGrid(CART, x, x)
        f = [sin(3xi) * cos(2yi) for xi in x, yi in x]
        out = similar(f)
        for dim in 1:2
            t = best(() -> O.apply_stencil!(out, f, g, dim; order = 1, nodes = 3), 5)
            row("N=$n dim=$dim  grid form (builds its table)", t)
        end
        # The table built once, as by a caller applying many fields.
        for dim in 1:2
            idx, w = D.axis_stencils(x, 1, 3)
            t = best(() -> O.apply_stencil!(out, f, idx, w, dim), 5)
            row("N=$n dim=$dim  prebuilt table", t)
        end
    end
end

# Point location: the claim is that neither walks the axis.
function bench_location()
    group("locate / nearest_index — search, not sweep")
    small = collect(range(0.0, 1.0; length = 64))
    big = collect(range(0.0, 1.0; length = 1 << 18))
    for (nm, g) in (("locate", D.locate), ("nearest_index", D.nearest_index))
        ts = best(() -> g(small, 0.37), 2000)
        tb = best(() -> g(big, 0.37), 2000)
        row("$nm  n=64", ts)
        row("$nm  n=262144", tb)
        ratio_row("$nm  growth over 4096× n", ts, tb)
    end
end

# Ball and k-nearest queries: cost per query against grid size, with and without an index.
function bench_queries()
    group("per-query cost against grid size")
    for (nm, build, small, big) in (("curvilinear", curv, 24, 96),
                                    ("stretched axis", stretched, 256, 4096))
        gs, gb = build(small), build(big)
        if nm == "curvilinear"
            ss = C.MetricTopology(gs; index = GD.cell_list(gs; ball = 2.5))
            sb = C.MetricTopology(gb; index = GD.cell_list(gb; ball = 2.5))
            is, ib = Vector{Int}(undef, 8), Vector{Int}(undef, 8)
            ds, db = Vector{Float64}(undef, 8), Vector{Float64}(undef, 8)
            a = best(() -> C.k_nearest!(is, ds, gs, 8, 8; k = 8, topology = ss), 200)
            b = best(() -> C.k_nearest!(ib, db, gb, 8, 8; k = 8, topology = sb), 200)
            row("$nm k_nearest!  n=$small", a); row("$nm k_nearest!  n=$big", b)
            ratio_row("$nm k_nearest!  growth over 16× cells", a, b)
            a = best(() -> C.nneighbors_within(gs, 8, 8; ball = 2.5, topology = ss), 200)
            b = best(() -> C.nneighbors_within(gb, 8, 8; ball = 2.5, topology = sb), 200)
            row("$nm indexed ball  n=$small", a); row("$nm indexed ball  n=$big", b)
            ratio_row("$nm indexed ball  growth over 16× cells", a, b)
        else
            a = best(() -> C.nneighbors_within(gs, 128, 2; ball = 3.0), 200)
            b = best(() -> C.nneighbors_within(gb, 128, 2; ball = 3.0), 200)
            row("$nm window  n=$small", a); row("$nm window  n=$big", b)
            ratio_row("$nm window  growth over 16× samples", a, b)
        end
    end

    group("the candidate buffer")
    g = curv(96)
    ix = C.indexed(g)
    buf = Vector{Int}(undef, 8192)
    s = C.ball_scratch()
    for (nm, r) in (("small ball", 2.5), ("310-candidate ball", 10.0))
        a = best(() -> C.neighbors_within!(buf, g, 48, 48; ball = r, topology = ix, scratch = s), 500)
        b = best(() -> C.neighbors_within!(buf, g, 48, 48; ball = r, topology = ix), 500)
        row("$nm  with scratch", a); row("$nm  without", b)
    end
end

# `searchsorted` on a uniform axis is closed-form because `UniformAxis <: AbstractRange`. The suite
# asserts the subtyping; the payoff is a number, and a number belongs here.
function bench_axis()
    group("searchsorted on a uniform axis")
    ts = best(() -> searchsortedfirst(FG.Axes.UniformAxis(0.0, 1e-1, 10), 0.5), 2000)
    tl = best(() -> searchsortedfirst(FG.Axes.UniformAxis(0.0, 1e-7, 10^7), 0.5), 2000)
    row("n=10", ts); row("n=10^7", tl)
    ratio_row("growth over 10^6× n", ts, tl)
end

# Driscoll–Healy latitude weights go through an FFT when one is loaded, at O(n log n).
function bench_weights()
    group("Driscoll–Healy latitude weights")
    t1 = best(() -> SS.latitude_weights(SS.DriscollHealySampling(), 1024), 3)
    t2 = best(() -> SS.latitude_weights(SS.DriscollHealySampling(), 4096), 3)
    row("n=1024", t1); row("n=4096", t2)
    ratio_row("growth over 4× n (quadratic would be 16×)", t1, t2)
end

# `derivative!` with its defaults on a 1440×720 lat–lon sphere, as a multiple of one copy of the field.
# A stencil and a copy each read the field once and write the output once.
function bench_derivative()
    group("derivative! on a 1440×720 sphere, in field copies")
    nλ, nφ = 1440, 720
    h = π / nφ
    g = GD.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6), FG.Axes.PeriodicAxis(0.0, 2π, nλ),
                          collect(range(-π / 2 + h / 2, π / 2 - h / 2; length = nφ)))
    f = [sin(3λ) * cos(φ) for λ in FG.Grids.coordinates(g, 1), φ in FG.Grids.coordinates(g, 2)]
    out = similar(f)
    tc = best(() -> copyto!(out, f), 5)
    row("copy", tc)
    for d in 1:2
        t = best(() -> O.derivative!(out, f, g, d), 5)
        row("along $d", t)
        ratio_row("along $d, in copies", tc, t)
    end
end

const GROUPS = ("stencil" => bench_stencil, "location" => bench_location,
                "queries" => bench_queries, "axis" => bench_axis, "weights" => bench_weights,
                "derivative" => bench_derivative)

function main(args)
    want = isempty(args) ? first.(GROUPS) : args
    for (name, f) in GROUPS
        name in want && f()
    end
    println()
end

main(ARGS)
