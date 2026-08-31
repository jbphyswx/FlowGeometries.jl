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
values and device arrays, and it may not allocate. Both hold for the bodies this package passes, and the
test suite gates every one of them at zero bytes.
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

Two stages, a single launch having no way to combine across work-items. The span count is bounded, so
the host's second stage is `O(1)` in `n`. Spans are laid out so none is empty, which keeps `init` from
entering the combine more times than it seeds partials.
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

# A buffer the launched passes can write, which on a device backend is device memory.
Execution.allocate(backend::KernelAbstractions.Backend, ::Type{T}, n::Integer) where {T} =
    KernelAbstractions.allocate(backend, T, Int(n))

# Span sums, one per work-item, so each writes only its own slot.
@kernel function _span_sum_kernel(sums, counts, n, span)
    g = @index(Global, Linear)
    lo = (g - 1) * span + 1
    hi = min(g * span, n)
    acc = zero(eltype(sums))
    for i in lo:hi
        @inbounds acc += counts[i]
    end
    @inbounds sums[g] = acc
end

# Each work-item walks its own span from the base the middle stage fixed, so the serial dependence is
# confined to one span and the spans are independent of each other.
@kernel function _span_scan_kernel(out, counts, bases, n, span)
    g = @index(Global, Linear)
    lo = (g - 1) * span + 1
    hi = min(g * span, n)
    @inbounds a = bases[g]
    for i in lo:hi
        @inbounds out[i] = a
        @inbounds a += counts[i]
    end
end

"""
    Execution.exclusive_scan!(out, counts, backend::KernelAbstractions.Backend; init = 1)

The CSR offset array on `backend`, in the three phases the threaded scan uses: span sums on the device,
those scanned into one base per span, then each span written from its base.

`out[k]` depends on every earlier count, so no single launch computes it. Splitting at the spans leaves
each work-item a serial walk of its own range with nothing shared, and the span count is bounded, so the
middle phase is `O(1)` in `n`.
"""
function Execution.exclusive_scan!(
    out::AbstractVector, counts::AbstractVector, backend::KernelAbstractions.Backend;
    init::Integer = 1,
)
    n = length(counts)
    length(out) == n + 1 || throw(DimensionMismatch(
        "out must be one longer than counts: got $(length(out)) and $n",
    ))
    T = eltype(out)
    acc = convert(T, init)
    if n > 0
        span = max(1, cld(n, 1024))
        ngroups = cld(n, span)
        sums = KernelAbstractions.allocate(backend, T, ngroups)
        _span_sum_kernel(backend)(sums, counts, n, span; ndrange = ngroups)
        KernelAbstractions.synchronize(backend)

        # One number per span, so this stage is bounded however long `counts` is.
        host = Array(sums)
        @inbounds for g in 1:ngroups
            s = host[g]
            host[g] = acc
            acc += s
        end
        bases = KernelAbstractions.allocate(backend, T, ngroups)
        copyto!(bases, host)

        _span_scan_kernel(backend)(out, counts, bases, n, span; ndrange = ngroups)
        KernelAbstractions.synchronize(backend)
    end
    # The total sits past the last count, where a CSR row bound reads it.
    total = acc
    Execution.run_indices(1, backend) do _
        @inbounds out[n + 1] = total
    end
    return out
end

end # module FlowGeometriesKernelAbstractionsExt
