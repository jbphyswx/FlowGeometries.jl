using MPI: MPI

Test.@testset "An MPI backend runs a share per rank, and every rank ends with the whole result" begin
    script = joinpath(@__DIR__, "mpi", "execution.jl")
    cmd = `$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=$(Base.active_project()) $script`
    Test.@test success(pipeline(cmd; stdout = stdout, stderr = stderr))
end
