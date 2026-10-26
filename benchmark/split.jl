# A split grid over MPI ranks: a halo exchange against the derivative it serves, and a transpose against
# PencilArrays' own on a held topology. Run under MPI:
#
#     mpiexec -n 4 julia --project=benchmark benchmark/split.jl
#
# Each time is the minimum over blocks of repeated calls, every rank calling; rank 0 prints.

using MPI: MPI
using PencilArrays: PencilArrays
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using Printf: Printf

MPI.Init()
const COMM = MPI.COMM_WORLD
const ROOT = MPI.Comm_rank(COMM) == 0

function best(f, reps::Int; blocks::Int = 3)
    f()
    t = Inf
    for _ in 1:blocks
        MPI.Barrier(COMM)
        e = @elapsed for _ in 1:reps
            f()
        end
        t = min(t, MPI.Allreduce(e, max, COMM) / reps)
    end
    return t
end

row(name, t) = ROOT && Printf.@printf("  %-50s %10.1f µs\n", name, t * 1e6)

function main()
    D, O, GD = FG.Decomposition, FG.Operators, FG.Grids
    nλ, nφ = 1440, 720
    h = π / nφ
    g = GD.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6), FG.Axes.PeriodicAxis(0.0, 2π, nλ),
                          collect(range(-π / 2 + h / 2, π / 2 - h / 2; length = nφ)))
    ROOT && println("julia ", VERSION, "  ranks: ", MPI.Comm_size(COMM), "  threads: ", Threads.nthreads(),
                    "  grid: $nλ×$nφ sphere")
    s = D.split_grid(g, CB.MPIBackend(); dims = (2,), halo = 3)
    f = D.distribute(s, [sin(3λ) * cos(φ) for λ in GD.coordinates(g, 1), φ in GD.coordinates(g, 2)])
    o = D.split_field(s, Float64)
    b = D.block(s)
    a, out = D.local_array(f), D.local_array(o)
    row("halo exchange along φ", best(() -> D.exchange_halo!(f, 2), 20))
    row("one block's derivative along φ", best(() -> O.derivative!(out, a, b.grid, 2; nodes = 3), 20))
    row("split derivative along φ (exchange + block)", best(() -> O.derivative!(o, f, 2; nodes = 3), 20))

    t = D.split_grid(s; dims = (1,))
    dst = D.split_field(t, Float64)
    row("redistribute! φ-split → λ-split", best(() -> D.redistribute!(dst, f), 5))
    topo = PencilArrays.MPITopology{1}(D.cartesian_comm(s))
    pa = PencilArrays.PencilArray{Float64}(undef, PencilArrays.Pencil(topo, size(g), (2,)))
    pb = PencilArrays.PencilArray{Float64}(undef, PencilArrays.Pencil(topo, size(g), (1,)))
    row("PencilArrays.transpose! on a held topology", best(() -> PencilArrays.transpose!(pb, pa), 5))
    row("MPITopology construction", best(() -> PencilArrays.MPITopology{1}(D.cartesian_comm(s)), 20))
end

main()
MPI.Finalize()
