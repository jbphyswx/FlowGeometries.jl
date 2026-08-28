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
#
# The output vector is typed from `f` on an empty range, which the contract already requires it to
# accept: a concrete vector holds the partials directly, where an `Any` one boxes each of them and then
# needs a second vector to narrow into — on the reduction path, whose serial form allocates nothing.
function Execution.map_chunks(f::F, n::Integer, ::AbstractThreadedBackend) where {F}
    n = Int(n)
    n > 0 || return Execution.map_chunks(f, 0, nothing)
    nt = Threads.nthreads()
    nt == 1 && return Execution.map_chunks(f, n, nothing)
    ranges = Execution.chunk_ranges(n, nt)
    out = Vector{typeof(f(1:0))}(undef, length(ranges))
    Threads.@threads for c in eachindex(ranges)
        @inbounds out[c] = f(ranges[c])
    end
    return out
end

# Per-index semantics, chunked granularity, and the partials combined in chunk order — so the answer does
# not depend on how the threads were scheduled.
function Execution.reduce_indices(
    f::F, op::O, init, n::Integer, backend::AbstractThreadedBackend,
) where {F,O}
    n = Int(n)
    n > 0 && return Execution._reduce_chunks(op, n, backend) do rng
        acc = init
        @inbounds for i in rng
            acc = op(acc, f(i))
        end
        return acc
    end
    return init
end

Execution.reduce_indices(f::F, op::O, init, n::Integer, ::AbstractSerialBackend) where {F,O} =
    Execution.reduce_indices(f, op, init, n, nothing)

Execution.exclusive_scan!(out::AbstractVector, counts::AbstractVector, ::AbstractSerialBackend;
                          init::Integer = 1) =
    Execution.exclusive_scan!(out, counts, nothing; init = init)

# Two passes over `counts` with a serial scan of one sum per chunk between them. Each chunk then writes
# only its own span of `out`, from a base the middle pass fixed.
function Execution.exclusive_scan!(
    out::AbstractVector, counts::AbstractVector, ::AbstractThreadedBackend; init::Integer = 1,
)
    n = length(counts)
    length(out) == n + 1 || throw(DimensionMismatch(
        "out must be one longer than counts: got $(length(out)) and $n",
    ))
    T = eltype(out)
    nt = Threads.nthreads()
    (n == 0 || nt == 1) && return Execution.exclusive_scan!(out, counts, nothing; init = init)
    ranges = Execution.chunk_ranges(n, nt)
    nc = length(ranges)
    sums = Vector{T}(undef, nc)
    Threads.@threads for c in 1:nc
        s = zero(T)
        @inbounds for i in ranges[c]
            s += counts[i]
        end
        @inbounds sums[c] = s
    end
    base = Vector{T}(undef, nc)
    acc = convert(T, init)
    @inbounds for c in 1:nc
        base[c] = acc
        acc += sums[c]
    end
    @inbounds out[n + 1] = acc
    Threads.@threads for c in 1:nc
        @inbounds a = base[c]
        @inbounds for i in ranges[c]
            out[i] = a
            a += counts[i]
        end
    end
    return out
end

end # module FlowGeometriesComputationalBackendsExt
