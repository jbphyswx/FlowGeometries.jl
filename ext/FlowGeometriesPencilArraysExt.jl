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

function _pencil(topo::PencilArrays.MPITopology, s::D.AbstractRankSplit)
    p = PencilArrays.Pencil(topo, size(D.global_grid(s)), D.split_dims(s))
    PencilArrays.range_local(p) == D.block(s).ranges || throw(ArgumentError(
        "the split's blocks are not the ranges PencilArrays places on its communicator"))
    return p
end

D.pencil_array(f::D.AbstractRankField) = _pencil_array(D.pencil(D.split_of(f)), f)

function _pencil_array(p::PencilArrays.Pencil, f::D.AbstractRankField)
    A = D.local_array(f)
    return PencilArrays.PencilArray(p, Array(view(A, D._own(D.block(D.split_of(f)), A)...)))
end

function D._rank_redistribute!(::D.PencilTranspose, dst::D.AbstractRankField{T,N},
                               src::D.AbstractRankField{T,N}) where {T,N}
    sd, ss = D.split_of(dst), D.split_of(src)
    D.global_grid(sd) === D.global_grid(ss) ||
        throw(ArgumentError("redistribute! moves a field between splits of one grid"))
    topo = _topology(ss, D.split_dims(ss))
    A = D.local_array(src)
    o = PencilArrays.PencilArray{T}(undef, _pencil(topo, sd), D._batch(A, Val(N))...)
    PencilArrays.transpose!(o, _pencil_array(_pencil(topo, ss), src))
    B = D.local_array(dst)
    copyto!(view(B, D._own(D.block(sd), B)...), parent(o))
    return dst
end

end # module FlowGeometriesPencilArraysExt
