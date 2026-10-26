module FlowGeometriesMPIExt

# A grid split over the ranks of an `MPIBackend`. The blocks lie on a Cartesian communicator, one per
# rank, and halos pass between neighbouring ranks by `MPI.Sendrecv!`. Every rank holds its block of every
# field, and every rank calls every function here, as it calls every collective.

using MPI: MPI
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries.Execution: Execution
using FlowGeometries.Grids: Grids
using FlowGeometries.Decomposition: Decomposition as D

"""
    MPISplitGrid

A grid split over the ranks of a Cartesian communicator: the whole grid's description, this rank's
[`Decomposition.Block`](@ref), the split directions in the communicator's axis order, and this rank's
neighbour along each direction below and above (`MPI.PROC_NULL` where there is none).
"""
struct MPISplitGrid{N,M,G,B<:D.Block{N},X} <: D.AbstractRankSplit{N}
    grid::G
    block::B
    comm::MPI.Comm
    dims::NTuple{M,Int}
    halo::NTuple{N,Int}
    neighbours::NTuple{N,Tuple{Int,Int}}
    execution::X
end

"""
    MPISplitField

A field on an [`MPISplitGrid`](@ref): this rank's block, halos and batch axes included.
"""
struct MPISplitField{T,N,A<:AbstractArray{T},S<:MPISplitGrid{N}} <: D.AbstractRankField{T,N}
    data::A
    split::S
end

D.split_of(f::MPISplitField) = f.split
D.global_grid(s::MPISplitGrid) = s.grid
D.split_dims(s::MPISplitGrid) = s.dims
D.block(s::MPISplitGrid) = s.block
D.cartesian_comm(s::MPISplitGrid) = s.comm
D.local_array(f::MPISplitField) = f.data
D._halo(s::MPISplitGrid) = s.halo

_comm(b::CB.MPIBackend) = b.comm === nothing ? MPI.COMM_WORLD : b.comm
_comm(::CB.AbstractMPIBackend) = MPI.COMM_WORLD

function D.split_grid(
    grid::Grids.StructuredGrid{T,G,N}, backend::CB.AbstractMPIBackend;
    dims::NTuple{M,Int} = (N,), blocks::Union{Nothing,NTuple{M,Int}} = nothing, halo::Integer = 2,
) where {T,G,N,M}
    comm = _comm(backend)
    nr = MPI.Comm_size(comm)
    P = blocks === nothing ? ntuple(k -> Int(MPI.Dims_create(nr, zeros(Int, M))[k]), Val(M)) : blocks
    prod(P) == nr || throw(ArgumentError("$(prod(P)) blocks for $nr ranks"))
    cart = MPI.Cart_create(comm, collect(P); periodic = fill(false, M), reorder = false)
    return _split(grid, cart, dims, P, Int(halo), backend)
end

function D.split_grid(s::MPISplitGrid{N,M}; dims::NTuple{M,Int}) where {N,M}
    P = ntuple(k -> Int(MPI.Cart_get(s.comm)[1][k]), Val(M))
    return _split(s.grid, s.comm, dims, P, maximum(s.halo), s.execution)
end

function _split(grid::Grids.StructuredGrid{T,G,N}, cart::MPI.Comm, dims::NTuple{M,Int}, P::NTuple{M,Int},
                halo::Int, backend) where {T,G,N,M}
    count, h = D._plan(grid, dims, P, halo)
    me = ntuple(k -> MPI.Cart_coords(cart)[k] + 1, Val(M))
    sz = size(grid)
    # A direction `dims` does not name is one block, whose neighbour across a periodic seam is itself.
    at(d) = (k = findfirst(==(d), dims); k === nothing ? 1 : me[k])
    ranges = ntuple(d -> D._block_range(sz[d], count[d], at(d)), Val(N))
    block = D._block(grid, ranges, h, backend)
    self = MPI.Comm_rank(cart)
    nb = ntuple(Val(N)) do d
        k = findfirst(==(d), dims)
        wraps = Grids.isperiodic(grid, d)
        rank(c) = k === nothing ? self :
            MPI.Cart_rank(cart, [j == k ? mod1(c, P[k]) - 1 : me[j] - 1 for j in 1:M])
        side(c) = (1 ≤ c ≤ count[d] || wraps) ? rank(c) : Int(MPI.PROC_NULL)
        (side(at(d) - 1), side(at(d) + 1))
    end
    return MPISplitGrid(grid, block, cart, dims, h, nb, backend)
end

function D.split_field(s::MPISplitGrid, ::Type{T}, batch::Integer...) where {T}
    return MPISplitField(Execution.allocate(s.block.execution, T, size(s.block.grid)..., batch...), s)
end

function D.distribute(s::MPISplitGrid{N}, A::AbstractArray{T}) where {N,T}
    b = s.block
    sz = size(s.grid)
    ntuple(d -> size(A, d), Val(N)) == sz || throw(DimensionMismatch(
        "the array's leading axes $(ntuple(d -> size(A, d), Val(N))) are not the grid's $sz"))
    idx = ntuple(d -> mod1.(_padded_range(b, d), sz[d]), Val(N))
    part = A[idx..., ntuple(_ -> Colon(), Val(ndims(A) - N))...]
    return MPISplitField(copyto!(Execution.allocate(b.execution, T, size(part)...), part), s)
end

# The whole grid's indices of a block's local positions along `d`, before wrapping.
@inline _padded_range(b::D.Block, d::Int) =
    (first(b.ranges[d]) - (first(b.interior[d]) - 1)):(last(b.ranges[d]) + size(b.grid, d) - last(b.interior[d]))

function D.gather(f::MPISplitField{T,N}) where {T,N}
    b = f.split.block
    mine = (b.ranges, Array(view(f.data, D._own(b, f.data)...)))
    parts = MPI.gather(mine, f.split.comm; root = 0)
    parts === nothing && return nothing
    out = Array{T}(undef, size(f.split.grid)..., D._batch(f.data, Val(N))...)
    for (r, a) in parts
        out[r..., ntuple(_ -> Colon(), Val(ndims(a) - N))...] = a
    end
    return out
end

D.exchange_halo!(f::MPISplitField{T,N}) where {T,N} = (foreach(d -> _exchange!(f, d), 1:N); f)
D.exchange_halo!(f::MPISplitField, dim::Integer) = (_exchange!(f, Int(dim)); f)

# Along `d`: the first `h` own cells go down and the upper halo is filled from above, then the last `h`
# go up and the lower halo is filled from below. A slab spans every other direction whole, halos
# included, and passes through a host buffer, so a device-resident block exchanges by its values.
function _exchange!(f::MPISplitField{T,N}, d::Int) where {T,N}
    s = f.split
    h = s.halo[d]
    h == 0 && return nothing
    lo, hi = s.neighbours[d]
    A = f.data
    r = s.block.interior[d]
    pad = (first(r) - 1, size(A, d) - last(r))
    slab(rng) = ntuple(k -> k == d ? rng : (1:size(A, k)), Val(ndims(A)))
    _swap!(A, slab(first(r):(first(r) + h - 1)), slab((last(r) + 1):(last(r) + pad[2])), lo, hi, s.comm)
    _swap!(A, slab((last(r) - h + 1):last(r)), slab((first(r) - pad[1]):(first(r) - 1)), hi, lo, s.comm)
    return nothing
end

# `A[send...]` to `dest`, and from `source` into `A[recv...]`. A missing neighbour is `MPI.PROC_NULL`, to
# which nothing is sent and from which nothing arrives; a bounded end has an empty `recv`.
function _swap!(A, send, recv, dest::Int, source::Int, comm)
    out = Array(view(A, send...))
    into = similar(out, map(length, recv))
    MPI.Sendrecv!(out, into, comm; dest = dest, source = source)
    source == MPI.PROC_NULL || copyto!(view(A, recv...), into)
    return nothing
end

function D.foreach_block(f::F, s::MPISplitGrid, fields::MPISplitField...) where {F}
    all(x -> x.split === s, fields) || throw(ArgumentError("a field lives on another split"))
    f(s.block, map(x -> x.data, fields)...)
    return nothing
end

function D.map_blocks(f::F, s::MPISplitGrid, fields::MPISplitField...) where {F}
    all(x -> x.split === s, fields) || throw(ArgumentError("a field lives on another split"))
    return MPI.bcast(MPI.gather(f(s.block, map(x -> x.data, fields)...), s.comm; root = 0), 0, s.comm)
end

end # module FlowGeometriesMPIExt
