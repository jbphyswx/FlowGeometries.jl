using MPI: MPI

Test.@testset "A grid split over ranks gives each block the whole grid's values" begin
    script = joinpath(@__DIR__, "mpi", "execution.jl")
    cmd = `$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=$(Base.active_project()) $script`
    Test.@test success(pipeline(cmd; stdout = stdout, stderr = stderr))
end
