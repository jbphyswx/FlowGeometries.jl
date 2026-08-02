# Wall-clock measurements for FlowGeometries.jl.
#
# These live here rather than in `test/` deliberately. A benchmark repeats a call hundreds of times,
# so it makes `Pkg.test` slow, and it is decided by a clock, so it makes it flaky — one of these
# failed in the suite purely because a garbage collection landed in a measurement block. The suite
# now asserts the same claims by COUNTING operations, which is deterministic and instant; what is
# left here is the timing itself, run on demand:
#
#     julia --project=benchmark benchmark/benchmarks.jl
#     julia --project=benchmark benchmark/benchmarks.jl stencil     # one group
#
# No benchmarking package is used: the measurement is min-over-blocks, which rejects a GC pause
# without needing more samples, and adding a dependency to run three loops is not worth it.

using FlowGeometries: FlowGeometries as FG
using NearestNeighbors: NearestNeighbors
using Printf: @printf

const A = FG.Axes
const C = FG.Connectivity
const D = FG.Discretization
const GD = FG.Grids
const SS = FG.SphericalSampling

const CART = FG.Geometry.CartesianGeometry{Float64}()

"""
    best(f, reps; blocks = 3) -> seconds per call

Minimum over `blocks` blocks of `reps` calls. The minimum is what rejects a collection landing in one
block; more samples inside a block would only average one in rather than remove it.
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

row(name, t) = @printf("  %-46s %10.2f µs\n", name, t * 1e6)
ratio_row(name, a, b) = @printf("  %-46s %10.2f×\n", name, b / a)
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
            t = best(() -> D.apply_stencil!(out, f, g, dim; order = 1, nodes = 3), 5)
            row("N=$n dim=$dim  grid form (builds its table)", t)
        end
        # With the table built once, which is what a caller applying many fields pays.
        for dim in 1:2
            idx, w = D.axis_stencils(x, 1, 3)
            t = best(() -> D.apply_stencil!(out, f, idx, w, dim), 5)
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
    ts = best(() -> searchsortedfirst(A.UniformAxis(0.0, 1e-1, 10), 0.5), 2000)
    tl = best(() -> searchsortedfirst(A.UniformAxis(0.0, 1e-7, 10^7), 0.5), 2000)
    row("n=10", ts); row("n=10^7", tl)
    ratio_row("growth over 10^6× n", ts, tl)
end

# Driscoll–Healy latitude weights go through an FFT when one is loaded: O(n log n), not O(n²).
function bench_weights()
    group("Driscoll–Healy latitude weights")
    t1 = best(() -> SS.latitude_weights(SS.DriscollHealySampling(), 1024), 3)
    t2 = best(() -> SS.latitude_weights(SS.DriscollHealySampling(), 4096), 3)
    row("n=1024", t1); row("n=4096", t2)
    ratio_row("growth over 4× n (quadratic would be 16×)", t1, t2)
end

const GROUPS = ("stencil" => bench_stencil, "location" => bench_location,
                "queries" => bench_queries, "axis" => bench_axis, "weights" => bench_weights)

function main(args)
    want = isempty(args) ? first.(GROUPS) : args
    for (name, f) in GROUPS
        name in want && f()
    end
    println()
end

main(ARGS)
