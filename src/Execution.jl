module Execution

# Where this package decides *how* a bulk loop runs. The core knows only "serial", which is the
# default everywhere; loading ComputationalBackends adds the threaded and GPU tags through an
# extension, so no execution dependency is imposed on callers who do not want one.

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
run_chunks(f, n::Integer, ::Nothing) = (n > 0 && f(1:Int(n)); nothing)

end # module Execution
