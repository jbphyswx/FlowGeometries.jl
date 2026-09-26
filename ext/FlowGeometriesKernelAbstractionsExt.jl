module FlowGeometriesKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions
# `@kernel` rewrites the `@index` inside its body by name, so these two have to arrive unqualified.
using KernelAbstractions: @kernel, @index
using Adapt: Adapt
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries.Execution: Execution

# The device tag this extension launches on: a `GPUBackend` wrapping a KernelAbstractions backend.
const _KABackend = CB.GPUBackend{<:KernelAbstractions.Backend}

# One kernel for the whole package: every device-eligible loop here is "apply this body to index `i`",
# so there is nothing per-operation to write, and `f` is compiled for the device by the launch.
@kernel function _index_kernel(f)
    i = @index(Global, Linear)
    f(i)
end

"""
    Execution.run_indices(f, n, backend::GPUBackend{<:KernelAbstractions.Backend}, outputs...)

Launch `f` over `1:n` on `backend.backend` and wait. `f` has to be device-compatible: it may capture only
isbits values and device arrays, and it may not allocate. Both hold for the bodies this package passes,
and the test suite gates every one of them at zero bytes. The outputs are written in place.
"""
function Execution.run_indices(f::F, n::Integer, b::_KABackend, ::Execution.Written...) where {F}
    n = Int(n)
    n > 0 || return nothing
    kernel = _index_kernel(b.backend)
    kernel(f; ndrange = n)
    KernelAbstractions.synchronize(b.backend)
    return nothing
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
    Execution.reduce_indices(f, op, init, n, backend::GPUBackend{<:KernelAbstractions.Backend})

Reduce `f(i)` over `1:n` on the device: one partial per span, then those combined in span order on the
host.

Two stages, a single launch having no way to combine across work-items. The span count is bounded, so
the host's second stage is `O(1)` in `n`. Spans are laid out so none is empty, which keeps `init` from
entering the combine more times than it seeds partials.
"""
function Execution.reduce_indices(f::F, op::O, init, n::Integer, b::_KABackend) where {F,O}
    n = Int(n)
    n > 0 || return init
    span = max(1, cld(n, 1024))
    ngroups = cld(n, span)
    partials = KernelAbstractions.allocate(b.backend, typeof(init), ngroups)
    kernel = _partial_fold_kernel(b.backend)
    kernel(partials, f, op, init, n, span; ndrange = ngroups)
    KernelAbstractions.synchronize(b.backend)
    host = Array(partials)
    acc = init
    @inbounds for g in 1:ngroups
        acc = op(acc, host[g])
    end
    return acc
end

# A buffer the launched passes can write, which is device memory.
Execution.allocate(b::_KABackend, ::Type{T}, dims::Integer...) where {T} =
    KernelAbstractions.allocate(b.backend, T, Int.(dims)...)

Execution.on_backend(b::_KABackend, x) = Adapt.adapt(b.backend, x)

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
    Execution.exclusive_scan!(out, counts, backend::GPUBackend{<:KernelAbstractions.Backend}; init = 1)

The CSR offset array on the device, in the three phases the threaded scan uses: span sums on the device,
those scanned into one base per span, then each span written from its base.

`out[k]` depends on every earlier count, so no single launch computes it. Splitting at the spans leaves
each work-item a serial walk of its own range with nothing shared, and the span count is bounded, so the
middle phase is `O(1)` in `n`.
"""
function Execution.exclusive_scan!(
    out::AbstractVector, counts::AbstractVector, b::_KABackend; init::Integer = 1,
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
        sums = KernelAbstractions.allocate(b.backend, T, ngroups)
        _span_sum_kernel(b.backend)(sums, counts, n, span; ndrange = ngroups)
        KernelAbstractions.synchronize(b.backend)

        # One number per span, so this stage is bounded however long `counts` is.
        host = Array(sums)
        @inbounds for g in 1:ngroups
            s = host[g]
            host[g] = acc
            acc += s
        end
        bases = KernelAbstractions.allocate(b.backend, T, ngroups)
        copyto!(bases, host)

        _span_scan_kernel(b.backend)(out, counts, bases, n, span; ndrange = ngroups)
        KernelAbstractions.synchronize(b.backend)
    end
    # The total sits past the last count, where a CSR row bound reads it.
    total = acc
    Execution.run_indices(1, b) do _
        @inbounds out[n + 1] = total
    end
    return out
end

end # module FlowGeometriesKernelAbstractionsExt
