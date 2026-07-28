module FlowGeometriesComputationalBackendsExt

using ComputationalBackends: AbstractSerialBackend, AbstractThreadedBackend
using FlowGeometries.Execution: Execution

Execution.run_chunks(f, n::Integer, ::AbstractSerialBackend) = Execution.run_chunks(f, n, nothing)

# One chunk per thread rather than `@threads for i in 1:n`: the loops this drives are short-bodied,
# so per-index scheduling would cost more than the body.
function Execution.run_chunks(f, n::Integer, ::AbstractThreadedBackend)
    n = Int(n)
    n > 0 || return nothing
    nt = Threads.nthreads()
    nt == 1 && return Execution.run_chunks(f, n, nothing)
    ranges = Execution.chunk_ranges(n, nt)
    Threads.@threads for c in eachindex(ranges)
        @inbounds f(ranges[c])
    end
    return nothing
end

end # module FlowGeometriesComputationalBackendsExt
