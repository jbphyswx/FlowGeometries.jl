module FlowGeometriesComputationalBackendsExt

using ComputationalBackends: AbstractSerialBackend, AbstractThreadedBackend
using FlowGeometries.Execution: Execution

Execution.run_chunks(f::F, n::Integer, ::AbstractSerialBackend) where {F} = Execution.run_chunks(f, n, nothing)

# One chunk per thread rather than `@threads for i in 1:n`: the loops this drives are short-bodied,
# so per-index scheduling would cost more than the body.
function Execution.run_chunks(f::F, n::Integer, ::AbstractThreadedBackend) where {F}
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

Execution.run_indices(f::F, n::Integer, ::AbstractSerialBackend) where {F} = Execution.run_indices(f, n, nothing)

# Per-index semantics, chunked granularity: these bodies are short, so scheduling each index would cost
# more than running it.
function Execution.run_indices(f::F, n::Integer, backend::AbstractThreadedBackend) where {F}
    return Execution.run_chunks(n, backend) do rng
        @inbounds for i in rng
            f(i)
        end
    end
end

Execution.map_chunks(f::F, n::Integer, ::AbstractSerialBackend) where {F} = Execution.map_chunks(f, n, nothing)

# Naming the serial backend explicitly reduces exactly as leaving `backend` unset does — one chunk, no
# vector of per-chunk results to build.
Execution._reduce_chunks(f::F, op::O, n::Integer, ::AbstractSerialBackend) where {F,O} =
    Execution._reduce_chunks(f, op, n, nothing)

# Results stay in chunk order, so the caller's combine sees the same sequence it would serially.
function Execution.map_chunks(f::F, n::Integer, ::AbstractThreadedBackend) where {F}
    n = Int(n)
    n > 0 || return Execution.map_chunks(f, 0, nothing)
    nt = Threads.nthreads()
    nt == 1 && return Execution.map_chunks(f, n, nothing)
    ranges = Execution.chunk_ranges(n, nt)
    out = Vector{Any}(undef, length(ranges))
    Threads.@threads for c in eachindex(ranges)
        @inbounds out[c] = f(ranges[c])
    end
    return identity.(out)
end

end # module FlowGeometriesComputationalBackendsExt
