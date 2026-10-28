module Decomposition

# A structured grid split into blocks, one per MPI rank or Distributed worker, and fields held block by
# block. Each block holds its own cells and a halo of its neighbours' cells along every split direction,
# so a stencil operator runs on each block as it runs on a whole grid, after one halo exchange. The
# block slicing and the operators are here. The MPI extension splits over ranks and the Distributed
# extension over workers, each with its own library alone; the PencilArrays extension adds transposes
# between rank splits.

using ComputationalBackends: ComputationalBackends as CB
using ..Execution: Execution
using ..Axes: Axes
using ..Grids: Grids
using ..Operators: Operators

public AbstractSplitGrid, AbstractSplitField, AbstractRankSplit, AbstractRankField, AbstractWorkerSplit,
       AbstractWorkerField, Block, split_grid, split_field, distribute, gather, exchange_halo!,
       redistribute!, foreach_block, map_blocks, split_of, global_grid, split_dims, block, local_array,
       cartesian_comm, pencil, pencil_array

"""
    AbstractSplitGrid{N}

A structured grid of `N` directions split into blocks over processes: [`split_grid`](@ref) builds one.
"""
abstract type AbstractSplitGrid{N} end

"""
    AbstractSplitField{T,N}

A field on an [`AbstractSplitGrid`](@ref), each process holding its block's cells, a halo along every
split direction and any trailing batch axes. [`split_field`](@ref) and [`distribute`](@ref) build one.
"""
abstract type AbstractSplitField{T,N} end

"""
    AbstractRankSplit{N}

A grid split over the ranks of an MPI communicator, one block per rank, every rank holding its own
[`block`](@ref) on the Cartesian communicator [`cartesian_comm`](@ref).
"""
abstract type AbstractRankSplit{N} <: AbstractSplitGrid{N} end

"""
    AbstractRankField{T,N}

A field on an [`AbstractRankSplit`](@ref): each rank holds its block's [`local_array`](@ref).
"""
abstract type AbstractRankField{T,N} <: AbstractSplitField{T,N} end

"""
    AbstractWorkerSplit{N}

A grid split over Distributed workers, one block per worker, held on the worker; the caller holds a
handle.
"""
abstract type AbstractWorkerSplit{N} <: AbstractSplitGrid{N} end

"""
    AbstractWorkerField{T,N}

A field on an [`AbstractWorkerSplit`](@ref), each worker holding its block's array.
"""
abstract type AbstractWorkerField{T,N} <: AbstractSplitField{T,N} end

Base.eltype(::Type{<:AbstractSplitField{T}}) where {T} = T

"""
    split_dims(split) -> NTuple{M,Int}

The directions `split` divides, in the order of its block layout's axes.
"""
function split_dims end

"""
    block(split::AbstractRankSplit) -> Block

This rank's [`Block`](@ref).
"""
function block end

"""
    local_array(field::AbstractRankField) -> AbstractArray

This rank's array of `field`: its block's cells, halos and batch axes.
"""
function local_array end

"""
    cartesian_comm(split::AbstractRankSplit) -> MPI.Comm

The Cartesian communicator the blocks lie on: block coordinates are rank coordinates, no rank
reordered.
"""
function cartesian_comm end

"""
    pencil(split::AbstractRankSplit) -> PencilArrays.Pencil

With PencilArrays loaded: the `Pencil` holding `split`'s blocks, on the same communicator.
"""
function pencil end

"""
    pencil_array(field::AbstractRankField) -> PencilArrays.PencilArray

With PencilArrays loaded: this rank's own cells of `field`, halos left out, as a `PencilArray` on
[`pencil`](@ref)`(split_of(field))`, a copy.
"""
function pencil_array end

"""
    split_grid(grid, backend; dims = (ndims(grid),), blocks = nothing, halo = 2) -> AbstractSplitGrid
    split_grid(split; dims) -> AbstractSplitGrid

`grid` split into blocks along the directions `dims`, `blocks[k]` of them along `dims[k]`, one block per
rank of an `MPIBackend` (`using MPI`) or per worker of a `DistributedBackend` (`using Distributed`,
FlowGeometries loaded on every worker). `blocks = nothing` divides the ranks over `dims` as
`MPI.Dims_create` does, or the workers likewise. Block `p` of `P` along a direction of `n` cells holds
cells `(n(p-1))÷P + 1 : (np)÷P`, the split PencilArrays makes.

Each block carries `halo` cells of its neighbours' each side of a split direction, wrapping across a
periodic seam and stopping at a bounded end, so a stencil of up to `halo` nodes along that direction sees
every sample it sees on the whole grid. A block holds at least `halo` cells.

The second form splits the same grid along other directions over the same processes, the pair
[`redistribute!`](@ref) moves fields between.
"""
function split_grid end

"""
    split_field(split, T, batch...) -> AbstractSplitField

An uninitialised field of element type `T` on `split`, with trailing batch axes of lengths `batch`.
"""
function split_field end

"""
    distribute(split, A) -> AbstractSplitField

The field on `split` holding the whole array `A`, of `split`'s global size and any trailing batch axes,
halos included. Under MPI every rank passes the same `A`; under Distributed the caller does, and each
worker is sent its own block.
"""
function distribute end

"""
    gather(field) -> Array or nothing

The field's cells, halos left out, as one array of the global size: on rank 0 under MPI (`nothing`
elsewhere), on the caller under Distributed.
"""
function gather end

"""
    exchange_halo!(field[, dim]) -> field

Fill every halo cell of `field` from the block that owns it: along every split direction, or along
`dim` alone. Directions are exchanged one after another, each slab spanning the halos of the
directions before it, so the corner cells arrive too.
"""
function exchange_halo! end

"""
    redistribute!(dst, src) -> dst

Copy `src` into `dst`, a field on another split of the same grid over the same processes: the
distributed transpose. Between rank splits this needs PencilArrays loaded. `dst`'s halos hold no data
until exchanged.
"""
function redistribute! end

# Between rank splits the transpose is PencilArrays', which its extension supplies for this tag.
struct PencilTranspose end
redistribute!(dst::AbstractRankField, src::AbstractRankField) = _rank_redistribute!(PencilTranspose(), dst, src)
_rank_redistribute!(::Any, _, _) = throw(ArgumentError(
    "a transpose between MPI splits is PencilArrays': `using PencilArrays`"))

"""
    foreach_block(f, split, fields...) -> nothing

Call `f(block, arrays...)` on every process with its [`Block`](@ref) and each field's local array, halos
and batch axes included.
"""
function foreach_block end

"""
    map_blocks(f, split, fields...) -> Vector

`f(block, arrays...)` on every process, one value per block, collected in block order: on every rank
under MPI, on the caller under Distributed.
"""
function map_blocks end

"""
    split_of(field) -> AbstractSplitGrid

The split grid `field` lives on.
"""
function split_of end

"""
    global_grid(split) -> StructuredGrid

The grid `split` divides.
"""
function global_grid end

"""
    Block

One process's share of a split grid: `grid`, a `StructuredGrid` of its cells and halos; `core`, the same
cells without halos; `interior`, the local indices of its own cells in `grid`; `ranges`, their indices
in the whole grid; and `execution`, the backend its loops run under.
"""
struct Block{N,G<:Grids.StructuredGrid,C<:Grids.StructuredGrid,B<:CB.AbstractExecutionBackend}
    grid::G
    core::C
    interior::NTuple{N,UnitRange{Int}}
    ranges::NTuple{N,UnitRange{Int}}
    execution::B
end

# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

# Block `p` of `P` along a direction of `n` cells: PencilArrays' `local_data_range`.
@inline _block_range(n::Int, P::Int, p::Int) = ((n * (p - 1)) ÷ P + 1):((n * p) ÷ P)

# The checked split: the block count along each direction, one where `dims` does not name it, and the
# halo width along each, `halo` wherever a block has a neighbour (another block, or itself across a
# periodic seam) and none along a bounded direction held whole.
function _plan(grid::Grids.StructuredGrid{T,G,N}, dims::NTuple{M,Int}, blocks::NTuple{M,Int},
               halo::Int) where {T,G,N,M}
    allunique(dims) && all(d -> 1 ≤ d ≤ N, dims) ||
        throw(ArgumentError("split directions $dims are not distinct directions of a $N-direction grid"))
    halo ≥ 0 || throw(ArgumentError("a halo is a cell count, got $halo"))
    sz = size(grid)
    count = ntuple(d -> (k = findfirst(==(d), dims); k === nothing ? 1 : blocks[k]), Val(N))
    h = ntuple(d -> (count[d] > 1 || Grids.isperiodic(grid, d)) ? halo : 0, Val(N))
    for d in 1:N
        P = count[d]
        P ≥ 1 || throw(ArgumentError("direction $d is split into $P blocks"))
        w = sz[d] ÷ P
        w ≥ max(h[d], 1) || throw(ArgumentError(
            "direction $d of $(sz[d]) cells in $P blocks leaves a block of $w cells, fewer than the halo of " *
            "$(h[d]) or none; use fewer blocks or a narrower halo"))
    end
    return count, h
end

# The halo widths below and above a block: none at a bounded end of the grid.
function _pads(grid::Grids.StructuredGrid{T,G,N}, ranges::NTuple{N,UnitRange{Int}},
               halo::NTuple{N,Int}) where {T,G,N}
    sz = size(grid)
    return ntuple(Val(N)) do d
        h = halo[d]
        wraps = Grids.isperiodic(grid, d)
        ((wraps || first(ranges[d]) > 1) ? h : 0, (wraps || last(ranges[d]) < sz[d]) ? h : 0)
    end
end

# The grid's indices a block's local positions stand for, before wrapping: its cells and halos.
@inline _padded(r::UnitRange{Int}, pad::Tuple{Int,Int}) = (first(r) - pad[1]):(last(r) + pad[2])

"""
    _block(grid, ranges, halo, execution) -> Block

The block of `grid` holding cells `ranges` with `halo` cells each side of every direction that has one.
Coordinates, measure and mask are the grid's own values at those cells. Every direction of a block is
bounded: across a periodic seam its halo continues the coordinate one period on. A block's type follows
the grid's type alone, whichever directions are split.
"""
function _block(grid::Grids.StructuredGrid{T,G,N}, ranges::NTuple{N,UnitRange{Int}}, halo::NTuple{N,Int},
                execution::CB.AbstractExecutionBackend) where {T,G,N}
    pads = _pads(grid, ranges, halo)
    padded = map(_padded, ranges, pads)
    interior = map((r, p) -> (p[1] + 1):(p[1] + length(r)), ranges, pads)
    b = Execution.local_backend(execution)
    g = Execution.on_backend(b, _sliced(grid, padded))
    core = Execution.on_backend(b, _sliced(grid, ranges))
    return Block(g, core, interior, ranges, b)
end

# `grid` on the index ranges `idx`, every direction bounded.
function _sliced(grid::Grids.StructuredGrid{T,G,N}, idx::NTuple{N,UnitRange{Int}}) where {T,G,N}
    sz = size(grid)
    wrapped = ntuple(d -> mod1.(idx[d], sz[d]), Val(N))
    ax = map((x, j, P, t) -> _slice_axis(x, j, P, t), Grids.coordinates(grid), idx, getfield(grid, :period),
             Grids.topology(grid))
    return Grids.rebuild(grid, (; coordinates = ax, measure = _slice_measure(Grids.measure(grid), wrapped),
                                mask = _slice_mask(Grids.mask(grid), wrapped),
                                topology = ntuple(_ -> Grids.Bounded(), Val(N)), period = ntuple(_ -> zero(T), Val(N)),
                                sampling = nothing, stats = map(Grids._axis_summary, ax)))
end

# An axis on indices `j`. Along a periodic direction `j` may run past either end, an index `n` past the
# end standing one period further on in the axis's direction: a periodic axis continues uniformly, its
# seam being one step, and any other becomes a vector. A bounded direction's block lies within its ends.
_slice_axis(x::Axes.PeriodicUniformAxis{T}, j::UnitRange{Int}, ::Real, ::Grids.Periodic) where {T} =
    Axes.UniformAxis{T}(first(x) + T(first(j) - 1) * step(x), step(x), length(j))

function _slice_axis(x::AbstractVector{T}, j::UnitRange{Int}, P::Real, ::Grids.Periodic) where {T}
    n = length(x)
    s = Axes.wrap_sign(x) * T(P)
    return [x[mod1(i, n)] + T(fld(i - 1, n)) * s for i in j]
end

_slice_axis(x::AbstractVector, j::UnitRange{Int}, ::Real, ::Grids.Bounded) = x[j]

_slice_measure(m::Grids.SeparableMeasure, idx::Tuple) = Grids.SeparableMeasure(map(_take, m.factors, idx))
_slice_measure(m::Grids.SlabMeasure, idx::Tuple) =
    Grids.SlabMeasure(_take(m.lead, idx[1]), m.slab[idx[2], idx[3]], map(_take, m.rest, Base.tail(Base.tail(Base.tail(idx)))))
_slice_measure(m::AbstractArray, idx::Tuple) = m[idx...]

@inline _take(v::AbstractVector, i) = v[i]

_slice_mask(::Grids.AllActive, idx::Tuple) = Grids.AllActive(map(length, idx))
_slice_mask(m::AbstractArray{Bool}, idx::Tuple) = m[idx...]

# Each field's local array, halos included, and the field's batch shape past the grid's directions.
@inline _batch(A::AbstractArray, ::Val{N}) where {N} = ntuple(k -> size(A, N + k), Val(ndims(A) - N))

# The local positions of a block's own cells in an array carrying batch axes.
@inline _own(b::Block{N}, A::AbstractArray) where {N} =
    (b.interior..., ntuple(_ -> Colon(), Val(ndims(A) - N))...)

# ---------------------------------------------------------------------------
# Operators on split fields
# ---------------------------------------------------------------------------

function _same_split(fields::AbstractSplitField...)
    s = split_of(first(fields))
    all(f -> split_of(f) === s, fields) ||
        throw(ArgumentError("the fields live on different splits; redistribute! one onto the other's"))
    return s
end

# A stencil of `nodes` along a direction with halos reads that many samples beyond a block's edge cell
# at most; along a bounded direction held whole there is nothing beyond the block.
function _check_reach(split::AbstractSplitGrid, dim::Int, nodes::Int)
    h = _halo(split)[dim]
    (h == 0 || h ≥ nodes) || throw(ArgumentError(
        "a stencil of $nodes nodes along direction $dim needs a halo of $nodes; this split has $h"))
    return nothing
end

"""
    Operators.derivative!(out::AbstractSplitField, field::AbstractSplitField, dim; order, nodes, masked,
                          policy) -> out

[`Operators.derivative!`](@ref) on each block of a split field, after exchanging `field`'s halo along
`dim`: every block's own cells take the values the whole grid gives them, to rounding where a window is
rebuilt from coordinates, which along a periodic uniform axis a block holds as a slice of its own.
`out`'s halo holds no data.
"""
function Operators.derivative!(
    out::AbstractSplitField, field::AbstractSplitField, dim::Integer; order::Integer = 1,
    nodes::Integer = Int(order) + 1, masked = zero(eltype(out)),
    policy::Operators.AbstractMaskPolicy = Operators.BlankMasked(),
)
    s = _same_split(out, field)
    d, ord, k = Int(dim), Int(order), Int(nodes)
    _check_reach(s, d, k)
    exchange_halo!(field, d)
    foreach_block(s, out, field) do b, o, x
        Operators.derivative!(o, x, b.grid, d; order = ord, nodes = k, masked = masked, policy = policy,
                              execution = b.execution)
        return nothing
    end
    return out
end

"""
    Operators.apply_stencil!(out::AbstractSplitField, field::AbstractSplitField, dim; order, nodes,
                             masked, policy) -> out

[`Operators.apply_stencil!`](@ref) on each block of a split field, after exchanging `field`'s halo along
`dim`. `out`'s halo holds no data.
"""
function Operators.apply_stencil!(
    out::AbstractSplitField, field::AbstractSplitField, dim::Integer; order::Integer = 1,
    nodes::Integer = Int(order) + 1, masked = zero(eltype(out)),
    policy::Operators.AbstractMaskPolicy = Operators.BlankMasked(),
)
    s = _same_split(out, field)
    d, ord, k = Int(dim), Int(order), Int(nodes)
    _check_reach(s, d, k)
    exchange_halo!(field, d)
    foreach_block(s, out, field) do b, o, x
        Operators.apply_stencil!(o, x, b.grid, d; order = ord, nodes = k, masked = masked, policy = policy,
                                 execution = b.execution)
        return nothing
    end
    return out
end

"""
    Grids.integrate(field::AbstractSplitField; active_only = true)

[`Grids.integrate`](@ref) over every block's own cells, the blocks' integrals summed in block order: a
number, or one per batch element.
"""
function Grids.integrate(field::AbstractSplitField; active_only::Bool = true)
    parts = map_blocks(split_of(field), field) do b, x
        return Grids.integrate(view(x, _own(b, x)...), b.core; active_only = active_only,
                               execution = b.execution)
    end
    return reduce(+, parts)
end

# The halo width along each direction; each extension's split grid answers it.
function _halo end

# A rank split's two byte buffers, which a transpose onto it stages through.
function _buffers end

end # module Decomposition
