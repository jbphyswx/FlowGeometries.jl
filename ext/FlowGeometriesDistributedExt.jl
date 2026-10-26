module FlowGeometriesDistributedExt

# A loop under `DistributedBackend(inner)` runs one contiguous share of `1:n` per worker, each under
# `inner`, against the worker's copy of what the loop captures. A worker returns the spans its share
# wrote of every declared `Written` output, and the caller copies them into place; a reduction returns
# its partial, and the caller combines the partials in share order. Every worker needs FlowGeometries
# loaded, `Distributed.@everywhere using FlowGeometries`.
#
# A grid split over the workers keeps each worker's block and its part of every split field as the
# value of a `Future` that worker owns; the caller holds the futures. Work runs where the data is, and
# halos and transposes pass worker to worker, a slab at a time.

using Distributed: Distributed
using ComputationalBackends: ComputationalBackends as CB
using FlowGeometries.Execution: Execution
using FlowGeometries.Grids: Grids
using FlowGeometries.Decomposition: Decomposition as D

# ---------------------------------------------------------------------------
# A share of the indices per worker
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# A grid split over the workers
# ---------------------------------------------------------------------------

"""
    DistributedSplitGrid

A grid split over workers: the whole grid's description, the worker holding each block, each block's
cells in the whole grid, its own cells' local positions and its array size, halos included, and the
futures whose values are the blocks, one per worker.
"""
struct DistributedSplitGrid{
    N,M,G,
    P<:AbstractArray{Int,N},
    R<:AbstractArray{NTuple{N,UnitRange{Int}},N},
    Z<:AbstractArray{NTuple{N,Int},N},
    F<:AbstractArray{Distributed.Future,N},
    X,
} <: D.AbstractWorkerSplit{N}
    grid::G
    dims::NTuple{M,Int}
    procs::P
    ranges::R
    interior::R
    sizes::Z
    halo::NTuple{N,Int}
    blocks::F
    execution::X
end

"""
    DistributedSplitField

A field on a [`DistributedSplitGrid`](@ref): one future per block, owned by the block's worker, whose
value is the block's array, halos and batch axes included.
"""
struct DistributedSplitField{T,N,K,F<:AbstractArray{Distributed.Future,N},S<:DistributedSplitGrid{N}} <:
       D.AbstractWorkerField{T,N}
    parts::F
    batch::NTuple{K,Int}
    split::S
end

_field(::Type{T}, parts::AbstractArray{Distributed.Future,N}, batch::NTuple{K,Int},
       s::DistributedSplitGrid{N}) where {T,N,K} =
    DistributedSplitField{T,N,K,typeof(parts),typeof(s)}(parts, batch, s)

D.split_of(f::DistributedSplitField) = f.split
D.global_grid(s::DistributedSplitGrid) = s.grid
D.split_dims(s::DistributedSplitGrid) = s.dims
D._halo(s::DistributedSplitGrid) = s.halo
Base.eltype(::DistributedSplitField{T}) where {T} = T

function D.split_grid(
    grid::Grids.StructuredGrid{T,G,N}, backend::CB.AbstractDistributedBackend;
    dims::NTuple{M,Int} = (N,), blocks::Union{Nothing,NTuple{M,Int}} = nothing, halo::Integer = 2,
) where {T,G,N,M}
    P = blocks === nothing ? _balanced(Distributed.nworkers(), Val(M)) : blocks
    prod(P) ≤ Distributed.nworkers() ||
        throw(ArgumentError("$(prod(P)) blocks for $(Distributed.nworkers()) workers"))
    return _split(grid, dims, P, Int(halo), backend, Distributed.workers()[1:prod(P)])
end

D.split_grid(s::DistributedSplitGrid{N,M}; dims::NTuple{M,Int},
             blocks::NTuple{M,Int} = _balanced(length(s.procs), Val(M))) where {N,M} =
    _split(s.grid, dims, blocks, maximum(s.halo), s.execution, vec(s.procs)[1:prod(blocks)])

# `P` workers over `M` directions, each prime factor going to the direction holding fewest blocks so far.
function _balanced(P::Int, ::Val{M}) where {M}
    counts = ones(Int, M)
    for f in _factors(P)
        counts[argmin(counts)] *= f
    end
    return ntuple(k -> counts[k], Val(M))
end

function _factors(n::Int)
    out = Int[]
    p = 2
    while n > 1
        while n % p == 0
            push!(out, p)
            n ÷= p
        end
        p += 1
    end
    return sort!(out; rev = true)
end

function _split(grid::Grids.StructuredGrid{T,G,N}, dims::NTuple{M,Int}, P::NTuple{M,Int}, halo::Int,
                backend, workers::Vector{Int}) where {T,G,N,M}
    count, h = D._plan(grid, dims, P, halo)
    sz = size(grid)
    cells = CartesianIndices(count)
    ranges = [ntuple(d -> D._block_range(sz[d], count[d], I[d]), Val(N)) for I in cells]
    pads = map(r -> D._pads(grid, r, h), ranges)
    interior = map((r, p) -> map((rd, pd) -> (pd[1] + 1):(pd[1] + length(rd)), r, p), ranges, pads)
    sizes = map((r, p) -> map((rd, pd) -> length(rd) + pd[1] + pd[2], r, p), ranges, pads)
    procs = reshape(copy(workers), count)
    # Each block is cut where the grid is and sent to its worker, which keeps it.
    blocks = [Distributed.remotecall(identity, procs[I], D._block(grid, ranges[I], h, backend)) for I in cells]
    return DistributedSplitGrid(grid, dims, procs, ranges, interior, sizes, h, blocks, backend)
end

function D.split_field(s::DistributedSplitGrid, ::Type{T}, batch::Integer...) where {T}
    nb = map(Int, batch)
    parts = [Distributed.remotecall(_allocate, s.procs[I], s.blocks[I], T, (s.sizes[I]..., nb...))
             for I in CartesianIndices(s.procs)]
    return _field(T, parts, nb, s)
end

_allocate(block::Distributed.Future, ::Type{T}, dims) where {T} =
    Execution.allocate(fetch(block).execution, T, dims...)

function D.distribute(s::DistributedSplitGrid{N}, A::AbstractArray{T}) where {N,T}
    sz = size(s.grid)
    ntuple(d -> size(A, d), Val(N)) == sz || throw(DimensionMismatch(
        "the array's leading axes $(ntuple(d -> size(A, d), Val(N))) are not the grid's $sz"))
    nb = ntuple(k -> size(A, N + k), Val(ndims(A) - N))
    parts = map(CartesianIndices(s.procs)) do I
        lo = ntuple(d -> first(s.ranges[I][d]) - first(s.interior[I][d]) + 1, Val(N))
        idx = ntuple(d -> mod1.(lo[d]:(lo[d] + s.sizes[I][d] - 1), sz[d]), Val(N))
        Distributed.remotecall(_placed, s.procs[I], s.blocks[I], A[idx..., ntuple(_ -> Colon(), Val(length(nb)))...])
    end
    return _field(T, parts, nb, s)
end

_placed(block::Distributed.Future, part::AbstractArray{T}) where {T} =
    copyto!(Execution.allocate(fetch(block).execution, T, size(part)...), part)

# A block's own cells of one part, halos left out, as a host array; run where the part is.
_own_copy(part::Distributed.Future, interior::Tuple) = _own_copy(fetch(part), interior)
_own_copy(A::AbstractArray, interior::Tuple) =
    Array(view(A, interior..., ntuple(_ -> Colon(), Val(ndims(A) - length(interior)))...))

function D.gather(f::DistributedSplitField{T,N}) where {T,N}
    s = f.split
    out = Array{T}(undef, size(s.grid)..., f.batch...)
    colons = ntuple(_ -> Colon(), Val(length(f.batch)))
    @sync for I in CartesianIndices(s.procs)
        @async out[s.ranges[I]..., colons...] =
            Distributed.remotecall_fetch(_own_copy, s.procs[I], f.parts[I], s.interior[I])
    end
    return out
end

D.exchange_halo!(f::DistributedSplitField{T,N}) where {T,N} = (foreach(d -> _exchange!(f, d), 1:N); f)
D.exchange_halo!(f::DistributedSplitField, dim::Integer) = (_exchange!(f, Int(dim)); f)

# Along `d`, every worker at once: each fills its two halos from its neighbours' edge cells, fetching a
# slab from each. A slab spans every other direction whole, halos included; a neighbour along `d` has
# the same extent there. The pass ends before any worker reads a halo it fills.
function _exchange!(f::DistributedSplitField{T,N}, d::Int) where {T,N}
    s = f.split
    s.halo[d] == 0 && return nothing
    P = size(s.procs, d)
    wraps = Grids.isperiodic(s.grid, d)
    @sync for I in CartesianIndices(s.procs)
        below, above = _neighbour(I, d, -1, P, wraps), _neighbour(I, d, 1, P, wraps)
        # The neighbour's own cells nearest this block: the last below it, the first above.
        lo = below === nothing ? nothing :
            (s.procs[below], f.parts[below], (last(s.interior[below][d]) - _pad(s, I, d, 1) + 1):last(s.interior[below][d]))
        hi = above === nothing ? nothing :
            (s.procs[above], f.parts[above], first(s.interior[above][d]):(first(s.interior[above][d]) + _pad(s, I, d, 2) - 1))
        @async Distributed.remotecall_wait(_fill_halos!, s.procs[I], f.parts[I], d, s.interior[I][d], lo, hi)
    end
    return nothing
end

@inline _pad(s::DistributedSplitGrid, I, d::Int, side::Int) =
    side == 1 ? first(s.interior[I][d]) - 1 : s.sizes[I][d] - last(s.interior[I][d])

function _neighbour(I::CartesianIndex{N}, d::Int, step::Int, P::Int, wraps::Bool) where {N}
    c = I[d] + step
    (1 ≤ c ≤ P || wraps) || return nothing
    return CartesianIndex(ntuple(k -> k == d ? mod1(c, P) : I[k], Val(N)))
end

# On the worker owning `part`: its halos along `d` from the neighbours' cells `rows` along `d`.
function _fill_halos!(part::Distributed.Future, d::Int, own::UnitRange{Int}, lo, hi)
    A = fetch(part)
    slab(rng) = ntuple(k -> k == d ? rng : (1:size(A, k)), Val(ndims(A)))
    lo === nothing || copyto!(view(A, slab((first(own) - length(lo[3])):(first(own) - 1))...),
                              Distributed.remotecall_fetch(_slab, lo[1], lo[2], d, lo[3]))
    hi === nothing || copyto!(view(A, slab((last(own) + 1):(last(own) + length(hi[3])))...),
                              Distributed.remotecall_fetch(_slab, hi[1], hi[2], d, hi[3]))
    return nothing
end

function _slab(part::Distributed.Future, d::Int, rows)
    A = fetch(part)
    return Array(view(A, ntuple(k -> k == d ? rows : (1:size(A, k)), Val(ndims(A)))...))
end

function D.redistribute!(dst::DistributedSplitField{T,N}, src::DistributedSplitField{T,N}) where {T,N}
    sd, ss = dst.split, src.split
    sd.grid === ss.grid || throw(ArgumentError("redistribute! moves a field between splits of one grid"))
    dst.batch == src.batch || throw(DimensionMismatch("the fields carry batch axes $(dst.batch) and $(src.batch)"))
    @sync for I in CartesianIndices(sd.procs)
        # The source blocks overlapping this destination block, each with the overlap in its own local
        # positions and in the destination's.
        pieces = Tuple{Int,Distributed.Future,NTuple{N,UnitRange{Int}},NTuple{N,UnitRange{Int}}}[]
        for J in CartesianIndices(ss.procs)
            o = map(intersect, sd.ranges[I], ss.ranges[J])
            any(isempty, o) && continue
            at(r, own, glob) = (first(own) + first(r) - first(glob)):(first(own) + last(r) - first(glob))
            push!(pieces, (ss.procs[J], src.parts[J], map(at, o, ss.interior[J], ss.ranges[J]),
                           map(at, o, sd.interior[I], sd.ranges[I])))
        end
        @async Distributed.remotecall_wait(_gather_pieces!, sd.procs[I], dst.parts[I], pieces)
    end
    return dst
end

function _gather_pieces!(part::Distributed.Future, pieces)
    A = fetch(part)
    colons = ntuple(_ -> Colon(), Val(ndims(A) - length(first(pieces)[3])))
    for (p, src, from, into) in pieces
        copyto!(view(A, into..., colons...), Distributed.remotecall_fetch(_own_copy, p, src, from))
    end
    return nothing
end

function D.foreach_block(f::F, s::DistributedSplitGrid, fields::DistributedSplitField...) where {F}
    all(x -> x.split === s, fields) || throw(ArgumentError("a field lives on another split"))
    @sync for I in CartesianIndices(s.procs)
        @async Distributed.remotecall_wait(_run_block, s.procs[I], f, s.blocks[I], map(x -> x.parts[I], fields))
    end
    return nothing
end

function D.map_blocks(f::F, s::DistributedSplitGrid, fields::DistributedSplitField...) where {F}
    all(x -> x.split === s, fields) || throw(ArgumentError("a field lives on another split"))
    return vec(asyncmap(CartesianIndices(s.procs)) do I
        Distributed.remotecall_fetch(_run_block, s.procs[I], f, s.blocks[I], map(x -> x.parts[I], fields))
    end)
end

# On the worker: `f` on its block and parts, behind a barrier on their types.
_run_block(f::F, block::Distributed.Future, parts::Tuple) where {F} = _call(f, fetch(block), map(fetch, parts))
_call(f::F, block::D.Block, arrays::Tuple) where {F} = f(block, arrays...)

end # module FlowGeometriesDistributedExt
