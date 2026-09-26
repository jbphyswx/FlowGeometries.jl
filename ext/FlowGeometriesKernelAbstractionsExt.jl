module FlowGeometriesKernelAbstractionsExt

using KernelAbstractions: KernelAbstractions
# `@kernel` rewrites `@index`, `@localmem` and `@synchronize` inside its body by name, so these arrive
# unqualified.
using KernelAbstractions: @kernel, @index, @localmem, @synchronize
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

# Work-items per group. Each group's local buffers are `_W` long, and KernelAbstractions sizes local
# memory at compile time, so this is a constant; a power of two, for the pairwise tree.
const _W = 256
# Levels of the pairwise tree that combines a group's `_W` values.
const _LOG2W = trailing_zeros(_W)
# The most groups a reduction's first stage launches. Its second stage is one group folding their
# `_MAX_GROUPS` partials, `cld(_MAX_GROUPS, _W)` rounds at most.
const _MAX_GROUPS = 65_536
# The most groups a scan launches. Each group sums the totals of the groups before it to find where its
# block starts, so that work grows with the square of this.
const _SCAN_GROUPS = 1024

# Element `i` of an array, as a callable: the fold kernel takes it in place of `f` to fold an array's
# elements (a reduction's partials, a scan's counts).
struct _Read{A}
    a::A
end
@inline (r::_Read)(i::Integer) = @inbounds r.a[i]
Adapt.adapt_structure(to, r::_Read) = _Read(Adapt.adapt(to, r.a))

# Group `g` folds indices `((g-1)·rounds + r - 1)·W + t`, `r ∈ 1:rounds`, `t ∈ 1:W`, into `partials[g]`.
# Each round is `W` consecutive indices, combined by an adjacent-pair tree, and the rounds are combined
# in order, so the fold is `op` over the group's block in index order. An index past `n` contributes
# `init`, `op`'s identity.
@kernel function _fold_kernel(partials, f, op, init, n, rounds)
    g = @index(Group, Linear)
    t = @index(Local, Linear)
    vals = @localmem eltype(partials) (_W,)
    acc = @localmem eltype(partials) (1,)
    if t == 1
        @inbounds acc[1] = init
    end
    for r in 1:rounds
        i = ((g - 1) * rounds + r - 1) * _W + t
        @inbounds vals[t] = i <= n ? f(i) : init
        @synchronize
        for k in 1:_LOG2W
            s = 1 << (k - 1)
            if (t - 1) & (2s - 1) == 0
                @inbounds vals[t] = op(vals[t], vals[t + s])
            end
            @synchronize
        end
        if t == 1
            @inbounds acc[1] = op(acc[1], vals[1])
        end
        @synchronize
    end
    if t == 1
        @inbounds partials[g] = acc[1]
    end
end

"""
    Execution.reduce_indices(f, op, init, n, backend::GPUBackend{<:KernelAbstractions.Backend})

Reduce `f(i)` over `1:n` on the device, in two launches: each group folds a contiguous block in index
order, then one group folds the group results in order. Only the result crosses to the host.

The grouping is fixed by `n` alone, so `op` need only be associative, and the result does not depend on
scheduling. `op`'s results are stored as `typeof(init)`.
"""
function Execution.reduce_indices(f::F, op::O, init, n::Integer, b::_KABackend) where {F,O}
    n = Int(n)
    n > 0 || return init
    T = typeof(init)
    ngroups = min(cld(n, _W), _MAX_GROUPS)
    kernel = _fold_kernel(b.backend, _W)
    partials = KernelAbstractions.allocate(b.backend, T, ngroups)
    kernel(partials, f, op, init, n, cld(n, ngroups * _W); ndrange = ngroups * _W)
    KernelAbstractions.synchronize(b.backend)
    total = KernelAbstractions.allocate(b.backend, T, 1)
    kernel(total, _Read(partials), op, init, ngroups, cld(ngroups, _W); ndrange = _W)
    KernelAbstractions.synchronize(b.backend)
    return Array(total)[1]
end

# A buffer the launched passes can write, which is device memory.
Execution.allocate(b::_KABackend, ::Type{T}, dims::Integer...) where {T} =
    KernelAbstractions.allocate(b.backend, T, Int.(dims)...)

Execution.on_backend(b::_KABackend, x) = Adapt.adapt(b.backend, x)

# Group `g` scans the block `_fold_kernel` summed into `sums[g]`. Its base is `init` plus the sums of the
# groups before it; each round of `W` counts is scanned in local memory (Hillis–Steele) and written from
# the running offset. The last group writes the total past the last count.
@kernel function _scan_kernel(out, counts, sums, init, n, rounds, ngroups, brounds)
    g = @index(Group, Linear)
    t = @index(Local, Linear)
    vals = @localmem eltype(out) (_W,)
    tmp = @localmem eltype(out) (_W,)
    run = @localmem eltype(out) (1,)
    if t == 1
        @inbounds run[1] = init
    end
    for r in 1:brounds
        j = (r - 1) * _W + t
        @inbounds vals[t] = j < g ? sums[j] : zero(eltype(out))
        @synchronize
        for k in 1:_LOG2W
            s = 1 << (k - 1)
            if (t - 1) & (2s - 1) == 0
                @inbounds vals[t] += vals[t + s]
            end
            @synchronize
        end
        if t == 1
            @inbounds run[1] += vals[1]
        end
        @synchronize
    end
    for r in 1:rounds
        i = ((g - 1) * rounds + r - 1) * _W + t
        @inbounds vals[t] = i <= n ? counts[i] : zero(eltype(out))
        @synchronize
        for k in 1:_LOG2W
            s = 1 << (k - 1)
            @inbounds tmp[t] = t > s ? vals[t - s] : zero(eltype(out))
            @synchronize
            @inbounds vals[t] += tmp[t]
            @synchronize
        end
        # A section between synchronizations is its own work-item loop on a CPU backend, and a local
        # assigned in an earlier one is not visible here.
        o = ((g - 1) * rounds + r - 1) * _W + t
        if o <= n
            @inbounds out[o] = run[1] + vals[t] - counts[o]
        end
        @synchronize
        if t == 1
            @inbounds run[1] += vals[_W]
        end
        @synchronize
    end
    if g == ngroups && t == 1
        @inbounds out[n + 1] = run[1]
    end
end

"""
    Execution.exclusive_scan!(out, counts, backend::GPUBackend{<:KernelAbstractions.Backend}; init = 1)

The CSR offset array on the device, in two launches: each group's sum of its block of `counts`, then
each group scanning its block from a base it forms from the sums before it. Nothing passes through the
host.
"""
function Execution.exclusive_scan!(
    out::AbstractVector, counts::AbstractVector, b::_KABackend; init::Integer = 1,
)
    n = length(counts)
    length(out) == n + 1 || throw(DimensionMismatch(
        "out must be one longer than counts: got $(length(out)) and $n",
    ))
    T = eltype(out)
    base = convert(T, init)
    if n == 0
        Execution.run_indices(1, b) do _
            @inbounds out[1] = base
        end
        return out
    end
    ngroups = min(cld(n, _W), _SCAN_GROUPS)
    rounds = cld(n, ngroups * _W)
    sums = KernelAbstractions.allocate(b.backend, T, ngroups)
    _fold_kernel(b.backend, _W)(sums, _Read(counts), +, zero(T), n, rounds; ndrange = ngroups * _W)
    KernelAbstractions.synchronize(b.backend)
    _scan_kernel(b.backend, _W)(out, counts, sums, base, n, rounds, ngroups, cld(ngroups, _W);
                                ndrange = ngroups * _W)
    KernelAbstractions.synchronize(b.backend)
    return out
end

end # module FlowGeometriesKernelAbstractionsExt
