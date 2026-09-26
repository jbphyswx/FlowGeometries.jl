module FlowGeometriesDistributedExt

# A loop under `DistributedBackend(inner)` runs one contiguous share of `1:n` per worker, each under
# `inner`, against the worker's copy of what the loop captures. A worker returns the spans its share
# wrote of every declared `Written` output, and the caller copies them into place; a reduction returns
# its partial, and the caller combines the partials in share order. Every worker needs FlowGeometries
# loaded, `Distributed.@everywhere using FlowGeometries`.

using Distributed: Distributed
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries.Execution: Execution

# `f` over a share starting past `off`, as a callable struct so the offset is a field and the body
# stays device-compatible under a GPU `inner`.
struct _Shift{F}
    f::F
    off::Int
end
@inline (s::_Shift)(i::Integer) = s.f(i + s.off)

struct _ShiftRange{F}
    f::F
    off::Int
end
@inline (s::_ShiftRange)(r::UnitRange{Int}) = s.f(r .+ s.off)

_shares(n::Int) = Execution.chunk_ranges(n, Distributed.nworkers())

# What a share wrote of one output, as a host array to send back.
_written_part(o::Execution.Written, rng::UnitRange{Int}) =
    Array(vec(o.array)[Execution._host_span(o.span)(rng)])

function _place!(outs::Tuple, shares, parts)
    for (k, o) in enumerate(outs)
        span = Execution._host_span(o.span)
        for (rng, written) in zip(shares, parts)
            s = span(rng)
            isempty(s) || copyto!(o.array, first(s), written[k], 1, length(s))
        end
    end
    return nothing
end

@noinline _no_outputs(b) = throw(ArgumentError(
    "a loop under $(nameof(typeof(b))) returns its results through the arrays it declares as " *
    "`Execution.Written` outputs; this one declares none, so its writes would stay on the workers."))

function Execution.run_indices(
    f::F, n::Integer, b::CB.AbstractDistributedBackend, outs::Execution.Written...,
) where {F}
    isempty(outs) && _no_outputs(b)
    inner = CB.local_backend(b)
    shares = _shares(Int(n))
    parts = Distributed.pmap(shares) do rng
        Execution.run_indices(_Shift(f, first(rng) - 1), length(rng), inner)
        return map(o -> _written_part(o, rng), outs)
    end
    _place!(outs, shares, parts)
    return nothing
end

function Execution.run_chunks(
    f::F, n::Integer, b::CB.AbstractDistributedBackend, outs::Execution.Written...,
) where {F}
    isempty(outs) && _no_outputs(b)
    inner = CB.local_backend(b)
    shares = _shares(Int(n))
    parts = Distributed.pmap(shares) do rng
        Execution.run_chunks(_ShiftRange(f, first(rng) - 1), length(rng), inner)
        return map(o -> _written_part(o, rng), outs)
    end
    _place!(outs, shares, parts)
    return nothing
end

# Each share's chunk results, concatenated in share order: the chunk order the caller combines in.
function Execution.map_chunks(f::F, n::Integer, b::CB.AbstractDistributedBackend) where {F}
    inner = CB.local_backend(b)
    parts = Distributed.pmap(_shares(Int(n))) do rng
        return Execution.map_chunks(_ShiftRange(f, first(rng) - 1), length(rng), inner)
    end
    return reduce(vcat, parts)
end

function Execution.reduce_indices(
    f::F, op::O, init, n::Integer, b::CB.AbstractDistributedBackend,
) where {F,O}
    inner = CB.local_backend(b)
    parts = Distributed.pmap(_shares(Int(n))) do rng
        return Execution.reduce_indices(_Shift(f, first(rng) - 1), op, init, length(rng), inner)
    end
    acc = init
    for p in parts
        acc = op(acc, p)
    end
    return acc
end

end # module FlowGeometriesDistributedExt
