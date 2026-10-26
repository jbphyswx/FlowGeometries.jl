using Distributed: Distributed
include("process_backends.jl")

Test.@testset "A grid split over workers gives each block the whole grid's values" begin
    added = Distributed.addprocs(2; exeflags = "--project=$(Base.active_project())")
    try
        Distributed.@everywhere added using FlowGeometries: FlowGeometries
        check_split(CB.DistributedBackend(), 2)
        check_split(CB.DistributedBackend(CB.ThreadedBackend()), 2)
    finally
        Distributed.rmprocs(added)
    end
end
