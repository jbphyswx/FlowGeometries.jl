module FlowGeometriesPencilArraysExt

# PencilArrays over a grid split across MPI ranks: the `Pencil` holding the split's blocks, and the
# transpose between two splits of one grid. A split's Cartesian communicator is the one PencilArrays'
# `MPITopology` reads, and its blocks are the ranges PencilArrays places on it, so the two agree.

using MPI: MPI
using PencilArrays: PencilArrays
using FlowGeometries.Decomposition: Decomposition as D

D.pencil(s::D.AbstractRankSplit) = _pencil(_topology(s, D.split_dims(s)), s)

# The topology of a split's communicator. Building one is collective, and a transpose requires both of
# its pencils to hold the same one.
_topology(s::D.AbstractRankSplit, ::NTuple{M,Int}) where {M} = PencilArrays.MPITopology{M}(D.cartesian_comm(s))

# The `Pencil` of `s`'s blocks on `topo`, a transpose onto it staging through `s`'s buffers.
function _pencil(topo::PencilArrays.MPITopology, s::D.AbstractRankSplit)
    send_buf, recv_buf = D._buffers(s)
    p = PencilArrays.Pencil(topo, size(D.global_grid(s)), D.split_dims(s); send_buf, recv_buf)
    PencilArrays.range_local(p) == D.block(s).ranges || throw(ArgumentError(
        "the split's blocks are not the ranges PencilArrays places on its communicator"))
    return p
end

D.pencil_array(f::D.AbstractRankField) = PencilArrays.PencilArray(D.pencil(D.split_of(f)), Array(_own_cells(f)))

function D._rank_redistribute!(::D.PencilTranspose, dst::D.AbstractRankField{T,N},
                               src::D.AbstractRankField{T,N}) where {T,N}
    sd, ss = D.split_of(dst), D.split_of(src)
    D.global_grid(sd) === D.global_grid(ss) ||
        throw(ArgumentError("redistribute! moves a field between splits of one grid"))
    topo = _topology(ss, D.split_dims(ss))
    o = _target(_pencil(topo, sd), D.local_array(dst), _own_cells(dst))
    PencilArrays.transpose!(o, _source(_pencil(topo, ss), D.local_array(src), _own_cells(src)))
    _settle!(D.local_array(dst), _own_cells(dst), o)
    return dst
end

# This rank's own cells of `f`, halos left out.
_own_cells(f::D.AbstractRankField) = (A = D.local_array(f); view(A, D._own(D.block(D.split_of(f)), A)...))

# A field's own cells as a `PencilArray`: the field's memory where it is a host `Array`; otherwise a host
# copy of a source, and a host array a target is copied back from.
_source(p::PencilArrays.Pencil, ::Array, v::AbstractArray) = PencilArrays.PencilArray(p, v)
_source(p::PencilArrays.Pencil, ::AbstractArray, v::AbstractArray) = PencilArrays.PencilArray(p, Array(v))
_target(p::PencilArrays.Pencil, ::Array, v::AbstractArray) = PencilArrays.PencilArray(p, v)
_target(p::PencilArrays.Pencil, ::AbstractArray, v::AbstractArray{T}) where {T} =
    PencilArrays.PencilArray(p, Array{T}(undef, size(v)))
_settle!(::Array, ::AbstractArray, ::PencilArrays.PencilArray) = nothing
_settle!(::AbstractArray, v::AbstractArray, o::PencilArrays.PencilArray) = (copyto!(v, parent(o)); nothing)

end # module FlowGeometriesPencilArraysExt
