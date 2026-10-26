# Run under `mpiexec -n 2` by the `mpi` topic. Every rank runs every check, as every rank calls every
# collective, and a failure on any rank fails the process.
using MPI: MPI
using PencilArrays: PencilArrays
using FlowGeometries: FlowGeometries as FG
using ComputationalBackends: ComputationalBackends as CB
using Test: Test

include(joinpath(@__DIR__, "..", "process_backends.jl"))

MPI.Init()
Test.@testset "rank $(MPI.Comm_rank(MPI.COMM_WORLD)) of $(MPI.Comm_size(MPI.COMM_WORLD))" begin
    check_process_backend(CB.MPIBackend())
    check_process_backend(CB.MPIBackend(CB.ThreadedBackend()))
    check_split(CB.MPIBackend(), MPI.Comm_size(MPI.COMM_WORLD))
    check_split(CB.MPIBackend(CB.ThreadedBackend()), MPI.Comm_size(MPI.COMM_WORLD))
end
MPI.Finalize()
