module FlowGeometriesMPIExt

# A loop under `MPIBackend(inner)` runs one contiguous share of `1:n` per rank, each under `inner`. The
# shares of every declared `Written` output are then exchanged in place, so each rank ends with the whole
# array; a reduction gathers every rank's partial and combines them in rank order, so every rank returns
# the same value. Every rank calls the loop, as it calls every collective.

using MPI: MPI
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries.Execution: Execution

_comm(b::CB.MPIBackend) = b.comm === nothing ? MPI.COMM_WORLD : b.comm
_comm(::CB.AbstractMPIBackend) = MPI.COMM_WORLD

# Rank `r`'s share is entry `r + 1`, one per rank, empty where `1:n` has fewer indices than ranks.
function _shares(n::Int, nranks::Int)
    ranges = Execution.chunk_ranges(n, nranks)
    return [k ≤ length(ranges) ? ranges[k] : ((n + 1):n) for k in 1:nranks]
end

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

# Every rank's share of `o`, exchanged in place. The spans of consecutive shares are consecutive, so each
# rank's contribution already sits where the exchange reads it. Anything but a host `Array` is staged
# through one, so a device array or a strided view is exchanged by its values.
function _allgather!(o::Execution.Written, shares, comm)
    span = Execution._host_span(o.span)
    spans = map(span, shares)
    counts = Cint[length(s) for s in spans]
    displs = Cint[first(s) - 1 for s in spans]
    A = o.array
    buf = A isa Array ? vec(A) : Array(vec(A))
    MPI.Allgatherv!(MPI.VBuffer(buf, counts, displs), comm)
    A isa Array || copyto!(A, buf)
    return nothing
end

# Every rank's object, in rank order, on every rank.
function _allgather_objects(x, comm)
    isbitstype(typeof(x)) && return MPI.Allgather(x, comm)
    return MPI.bcast(MPI.gather(x, comm; root = 0), 0, comm)
end

function Execution.run_indices(
    f::F, n::Integer, b::CB.AbstractMPIBackend, outs::Execution.Written...,
) where {F}
    comm = _comm(b)
    shares = _shares(Int(n), MPI.Comm_size(comm))
    rng = shares[MPI.Comm_rank(comm) + 1]
    Execution.run_indices(_Shift(f, first(rng) - 1), length(rng), CB.local_backend(b))
    foreach(o -> _allgather!(o, shares, comm), outs)
    return nothing
end

function Execution.run_chunks(
    f::F, n::Integer, b::CB.AbstractMPIBackend, outs::Execution.Written...,
) where {F}
    comm = _comm(b)
    shares = _shares(Int(n), MPI.Comm_size(comm))
    rng = shares[MPI.Comm_rank(comm) + 1]
    Execution.run_chunks(_ShiftRange(f, first(rng) - 1), length(rng), CB.local_backend(b))
    foreach(o -> _allgather!(o, shares, comm), outs)
    return nothing
end

# This rank's chunk results, and every other rank's, concatenated in rank order.
function Execution.map_chunks(f::F, n::Integer, b::CB.AbstractMPIBackend) where {F}
    comm = _comm(b)
    rng = _shares(Int(n), MPI.Comm_size(comm))[MPI.Comm_rank(comm) + 1]
    mine = Execution.map_chunks(_ShiftRange(f, first(rng) - 1), length(rng), CB.local_backend(b))
    return reduce(vcat, MPI.bcast(MPI.gather(mine, comm; root = 0), 0, comm))
end

function Execution.reduce_indices(
    f::F, op::O, init, n::Integer, b::CB.AbstractMPIBackend,
) where {F,O}
    comm = _comm(b)
    rng = _shares(Int(n), MPI.Comm_size(comm))[MPI.Comm_rank(comm) + 1]
    mine = Execution.reduce_indices(_Shift(f, first(rng) - 1), op, init, length(rng), CB.local_backend(b))
    acc = init
    for p in _allgather_objects(mine, comm)
        acc = op(acc, p)
    end
    return acc
end

end # module FlowGeometriesMPIExt
