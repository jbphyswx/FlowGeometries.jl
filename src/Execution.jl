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

Partition `1:n` into at most `k` contiguous ranges of near-equal length. Contiguity matters: each
chunk then touches one span of every array the loop indexes, rather than striding across all of them.
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
reduces rather than writes. The caller combines them, so `f` needs no lock and the combining order is the
caller's to fix — which is what keeps a threaded reduction bit-identical to a serial one when the
operation is associative but not commutative.
"""
map_chunks(f::F, n::Integer, ::Nothing) where {F} = [f(1:Int(n))]

"""
    run_indices(f, n, backend)

Apply `f(i)` for each `i in 1:n`, under the execution policy `backend` names. `f` must write only what
`i` determines, with no accumulation across indices — the same contract as [`run_chunks`](@ref), stated
per index because that is what a device launch can express.

This is the form a kernel maps onto: with `KernelAbstractions` loaded and a device backend, `f` becomes
the body of a launch over `1:n` rather than a host loop.
"""
function run_indices(f::F, n::Integer, ::Nothing) where {F}
    @inbounds for i in 1:Int(n)
        f(i)
    end
    return nothing
end

end # module Execution
