module Execution

# Where this package decides *how* a bulk loop runs. The core knows only "serial", which is the default
# everywhere. Loading ComputationalBackends adds the threaded tags, and loading KernelAbstractions adds
# a device path, each through an extension, so no execution dependency is imposed on callers who do not
# want one.
#
# Two shapes, because bulk loops here come in two: `run_chunks` hands a contiguous range to a body that
# writes across it, and `run_indices` applies a body to one index at a time. Only the second maps to a
# kernel — a chunk body carries loop-carried state a device launch cannot express.

"""
    chunk_ranges(n, k) -> Vector{UnitRange{Int}}

Partition `1:n` into at most `k` contiguous ranges of near-equal length. Contiguity matters: each chunk
then touches one span of every array the loop indexes.
"""
function chunk_ranges(n::Integer, k::Integer)
    n = Int(n); k = max(1, min(Int(k), max(n, 1)))
    base, rem = divrem(n, k)
    ranges = Vector{UnitRange{Int}}(undef, k)
    lo = 1
    @inbounds for c in 1:k
        len = base + (c ≤ rem ? 1 : 0)
        ranges[c] = lo:(lo + len - 1)
        lo += len
    end
    return ranges
end

"""
    run_chunks(f, n, backend)

Apply `f(range)` over a partition of `1:n`, under the execution policy `backend` names.

`f` must be safe to run on disjoint index ranges concurrently — every write it makes has to be
determined by the index, never accumulated across chunks. `nothing` means serial and hands `f` the
whole range in one call, so the serial path adds no partitioning at all.
"""
run_chunks(f::F, n::Integer, ::Nothing) where {F} = (n > 0 && f(1:Int(n)); nothing)

"""
    map_chunks(f, n, backend) -> Vector

Apply `f(range)` over a partition of `1:n` and collect one result per chunk, for a bulk operation that
reduces. The caller combines them, so `f` needs no lock and the combining order is the caller's to fix,
which keeps a threaded reduction independent of scheduling for an operation that is associative but not
commutative.

`f` is called at least once, on an empty range when `n == 0`, since a reduction has to produce a value;
[`run_chunks`](@ref) has nothing to write there and does not call `f`. Being callable on an empty range
is part of the contract, and it is how the result's element type is known before the chunks run.
"""
map_chunks(f::F, n::Integer, ::Nothing) where {F} = [f(1:Int(n))]

"""
    _reduce_chunks(f, op, n, backend)

`f(range)` over a partition of `1:n`, combined left to right with `op`.

Distinct from [`map_chunks`](@ref) because the serial path has nothing to collect: it hands `f` the whole
range and returns that value directly. `map_chunks` builds a one-element `Vector` there, which the
compiler usually erases; this path keeps a serial reduction allocation-free unconditionally.
"""
_reduce_chunks(f::F, ::O, n::Integer, ::Nothing) where {F,O} = f(1:Int(n))

_reduce_chunks(f::F, op::O, n::Integer, backend) where {F,O} =
    reduce(op, map_chunks(f, n, backend))

"""
    run_indices(f, n, backend)

Apply `f(i)` for each `i in 1:n`, under the execution policy `backend` names. `f` must write only what
`i` determines, with no accumulation across indices — the same contract as [`run_chunks`](@ref), stated
per index, the form a device launch expresses.

This is the form a kernel maps onto: with `KernelAbstractions` loaded and a device backend, `f` becomes
the body of a launch over `1:n`.
"""
function run_indices(f::F, n::Integer, ::Nothing) where {F}
    @inbounds for i in 1:Int(n)
        f(i)
    end
    return nothing
end

"""
    reduce_indices(f, op, init, n, backend)

Reduce `f(i)` over `i in 1:n` with `op`, under the execution policy `backend` names.

The per-index counterpart of [`_reduce_chunks`](@ref), and the reduction a device can run: `f` reads
index `i` and nothing else, so the work splits without a chunk body. Every integral, norm and count over
a grid goes through this.

`op` must be associative and `init` its identity. `init` seeds each partial as well as the whole, so
the partials combine in any grouping.

The grouping follows the partition, never the scheduling, so a threaded result is deterministic. It
matches the serial left fold wherever `op` is exactly associative; floating-point `+` is not, so a
threaded sum can differ from the serial one in its last bits.
"""
function reduce_indices(f::F, op::O, init, n::Integer, ::Nothing) where {F,O}
    acc = init
    @inbounds for i in 1:Int(n)
        acc = op(acc, f(i))
    end
    return acc
end

"""
    allocate(backend, T, n) -> AbstractVector{T}

An uninitialised length-`n` vector the loops running under `backend` can write.

A CSR build allocates its degree, offset and neighbour arrays through this, so the buffers land where
the passes that fill them run: an ordinary `Vector` under `nothing` and the host backends, device
memory under a backend that launches kernels.
"""
allocate(::Nothing, ::Type{T}, n::Integer) where {T} = Vector{T}(undef, Int(n))

"""
    scalar(v, i) -> v[i]

One element of a buffer that may not be host memory.
"""
@inline scalar(v::Array, i::Integer) = @inbounds v[Int(i)]
@inline scalar(v::AbstractVector, i::Integer) = @inbounds Array(view(v, Int(i):Int(i)))[1]

"""
    exclusive_scan!(out, counts, backend = nothing; init = 1) -> out

Write the exclusive prefix sums of `counts` into `out`, which is one element longer: `out[1] = init` and
`out[i+1] = out[i] + counts[i]`.

This is a CSR offset array — `out[k]` is where row `k` starts and `out[end] - init` is the total — and
every connectivity builder here calls it between its counting pass and its filling pass. Under a
threaded `backend` it is two passes over `counts` plus a serial scan of one sum per chunk.
"""
function exclusive_scan!(
    out::AbstractVector, counts::AbstractVector, ::Nothing = nothing; init::Integer = 1,
)
    length(out) == length(counts) + 1 || throw(DimensionMismatch(
        "out must be one longer than counts: got $(length(out)) and $(length(counts))",
    ))
    acc = convert(eltype(out), init)
    @inbounds out[1] = acc
    @inbounds for i in eachindex(counts)
        acc += counts[i]
        out[i + 1] = acc
    end
    return out
end

end # module Execution
