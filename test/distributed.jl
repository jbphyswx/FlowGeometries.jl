using Distributed: Distributed
include("process_backends.jl")

Test.@testset "A distributed backend runs a share per worker and gathers what each loop writes" begin
    added = Distributed.addprocs(2; exeflags = "--project=$(Base.active_project())")
    try
        Distributed.@everywhere added using FlowGeometries: FlowGeometries
        check_process_backend(CB.DistributedBackend())
        check_process_backend(CB.DistributedBackend(CB.ThreadedBackend()))
        # A loop that declares no output has no way to return what its workers write.
        Test.@test_throws ArgumentError FG.Execution.run_indices(i -> nothing, 4, CB.DistributedBackend())
    finally
        Distributed.rmprocs(added)
    end
end
