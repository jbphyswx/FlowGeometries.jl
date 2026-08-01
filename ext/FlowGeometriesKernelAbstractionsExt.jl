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

end # module FlowGeometriesKernelAbstractionsExt
