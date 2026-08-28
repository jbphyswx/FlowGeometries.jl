module FlowGeometriesKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions
# `@kernel` rewrites the `@index` inside its body by name, so these two have to arrive unqualified.
using KernelAbstractions: @kernel, @index
using FlowGeometries.Execution: Execution

# One kernel for the whole package: every device-eligible loop here is "apply this body to index `i`",
# so there is nothing per-operation to write, and `f` is compiled for the device by the launch.
@kernel function _index_kernel(f)
    i = @index(Global, Linear)
    f(i)
end

"""
    Execution.run_indices(f, n, backend::KernelAbstractions.Backend)

Launch `f` over `1:n` on `backend` and wait. `f` has to be device-compatible: it may capture only isbits
values and device arrays, and it may not allocate. Both hold for the bodies this package passes, which is
what the allocation gate in the test suite exists to keep true.
"""
function Execution.run_indices(f::F, n::Integer, backend::KernelAbstractions.Backend) where {F}
    n = Int(n)
    n > 0 || return nothing
    kernel = _index_kernel(backend)
    kernel(f; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# A chunked body is a host loop by construction — it carries state across the indices in its range — so a
# device backend gets the one-index form or an error, never a silent host fallback.
function Execution.run_chunks(f::F, n::Integer, backend::KernelAbstractions.Backend) where {F}
    throw(ArgumentError(
        "a chunked loop cannot be launched on $(backend): its body accumulates across a range. Use the " *
        "index-parallel entry point, or run this operation on the host.",
    ))
end

function Execution.map_chunks(f::F, n::Integer, backend::KernelAbstractions.Backend) where {F}
    throw(ArgumentError(
        "a chunk-reducing loop cannot be launched on $(backend); reduce on the host, or use " *
        "`mapreduce_within` with a host backend.",
    ))
end

# Each work-item folds one contiguous span into its own slot, so what it writes is decided by its index
# and no barrier or local memory is needed.
@kernel function _partial_fold_kernel(partials, f, op, init, n, span)
    g = @index(Global, Linear)
    lo = (g - 1) * span + 1
    hi = min(g * span, n)
    acc = init
    for i in lo:hi
        acc = op(acc, f(i))
    end
    @inbounds partials[g] = acc
end

"""
    Execution.reduce_indices(f, op, init, n, backend::KernelAbstractions.Backend)

Reduce `f(i)` over `1:n` on `backend`: one partial per span on the device, then those combined in span
order on the host.

Two stages rather than one because a single launch has no way to combine across work-items. The span
count is bounded, so the host's second stage is `O(1)` in `n`. Spans are laid out so none is empty,
which keeps `init` from entering the combine more times than it seeds partials.
"""
function Execution.reduce_indices(
    f::F, op::O, init, n::Integer, backend::KernelAbstractions.Backend,
) where {F,O}
    n = Int(n)
    n > 0 || return init
    span = max(1, cld(n, 1024))
    ngroups = cld(n, span)
    partials = KernelAbstractions.allocate(backend, typeof(init), ngroups)
    kernel = _partial_fold_kernel(backend)
    kernel(partials, f, op, init, n, span; ndrange = ngroups)
    KernelAbstractions.synchronize(backend)
    host = Array(partials)
    acc = init
    @inbounds for g in 1:ngroups
        acc = op(acc, host[g])
    end
    return acc
end

# `out[k]` depends on every earlier count, so a scan is not an index-parallel loop. Both passes of the
# threaded form are, but the serial scan between them is what carries the offsets across chunks.
function Execution.exclusive_scan!(
    out::AbstractVector, counts::AbstractVector, backend::KernelAbstractions.Backend;
    init::Integer = 1,
)
    throw(ArgumentError(
        "a prefix scan cannot be launched on $(backend) through this entry point: each output depends " *
        "on every earlier count. Scan on the host, or pass a threaded backend.",
    ))
end

end # module FlowGeometriesKernelAbstractionsExt
