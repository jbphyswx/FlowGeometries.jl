# ---------------------------------------------------------------------------
# Stencil resolution
# ---------------------------------------------------------------------------

# Offsets come from `Stencils`, which generates them from the stencil's type, so any shape, radius and
# dimension is available and the loop over them still unrolls.
@inline _stencil_offsets(::Val{N}, s::Stencils.AbstractStencil) where {N} =
    Stencils.offsets(s, Val(N))

# A stencil is named by its type, so the neighbour iterator built from it is concretely typed and a
# traversal allocates nothing per cell. This signature is where that is enforced. The `stencil` keyword
# stays unannotated: `::AbstractStencil` on it declares an abstract type and loses the inference.
@inline _stencil_val(s::Stencils.AbstractStencil) = s

@inline function _wrap_or_clip(i::Int, di::Int, n::Int, periodic::Bool)
    j = i + di
    if periodic
        return mod1(j, n)
    elseif 1 ≤ j ≤ n
        return j
    else
        return 0
    end
end

# The grid carries its topology in its type, so this const-folds from singleton types.
@inline _periodic_flags(grid::Grids.AbstractGrid) = Grids.periodic_flags(grid)

# ---------------------------------------------------------------------------
# neighbors! / nneighbors / Grids.neighbors — dispatch on grid
# ---------------------------------------------------------------------------

"""
    neighbors!(out, grid, I...; stencil=Axial(1), active_only=true) -> n_written

Cell `I`'s neighbors, as linear indices into the grid's mask, written into `out`. This is the form to
use on a hot path: [`nneighbors`](@ref) counts them without writing, and
[`Grids.neighbors`](@ref) returns a fresh vector. `stencil` is any
[`Stencils.AbstractStencil`](@ref).
"""
function neighbors! end

"""
    nneighbors(grid, I...; stencil=Axial(1), active_only=true) -> Int

How many neighbors [`neighbors!`](@ref) writes for cell `I`, counted without writing them.
"""
function nneighbors end

# One entry point each, resolved by `Grids.adjacency_source`. The arity stays a type parameter because a
# `Vararg{Integer}` without one is not specialized on arity and allocates per call.

neighbors!(
    out::AbstractVector{<:Integer}, grid::Grids.AbstractGrid, I::Vararg{Integer,NI};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {NI} = _neighbors!(out, grid, Grids._cell_named_by(grid, I), _stencil_val(stencil),
                           active_only, Grids.adjacency_source(grid))

nneighbors(
    grid::Grids.AbstractGrid, I::Vararg{Integer,NI};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {NI} = _nneighbors(grid, Grids._cell_named_by(grid, I), _stencil_val(stencil),
                           active_only, Grids.adjacency_source(grid))

function Grids.neighbors(
    grid::Grids.AbstractGrid, I::Vararg{Integer,NI};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {NI}
    return _neighbors(grid, Grids._cell_named_by(grid, I), _stencil_val(stencil), active_only,
                      Grids.adjacency_source(grid))
end

# ---- Adjacency from index-space offsets --------------------------------------

@inline _neighbors!(out, grid, I, sten::Stencils.AbstractStencil, active_only::Bool,
                    ::Grids.IndexStencilNeighbors) =
    neighbors!(out, IndexTopology(grid), I, sten, active_only)

@inline _nneighbors(grid, I, sten::Stencils.AbstractStencil, active_only::Bool,
                    ::Grids.IndexStencilNeighbors) =
    _nneighbors(IndexTopology(grid), I, sten, active_only)

@inline function _neighbors(grid, I, sten::Stencils.AbstractStencil, active_only::Bool,
                            ::Grids.IndexStencilNeighbors)
    Grids._cell_checkbounds(grid, I)
    return _stencil_neighbors(grid, I, sten, active_only)
end

# ---- Adjacency read from stored incidence ------------------------------------
#
# A stored graph is the mesh's own neighbour relation, so a stencil does not enter it. `active_only`
# does, on the same rule the offset walk keeps: a masked cell has no neighbours, and a masked neighbour
# is not one. `Grids.incident_nodes` is the unfiltered storage underneath.

"""
    MeshNeighbors{GR}

Lazy neighbour sequence of one node of a stored-incidence layout: iterating it walks the node's CSR
block and yields each active neighbour.

The counterpart of [`StencilNeighbors`](@ref): a traversal over every node allocates nothing.
"""
struct MeshNeighbors{GR}
    grid::GR
    node::Int
    active_only::Bool
end

Base.IteratorSize(::Type{<:MeshNeighbors}) = Base.HasLength()
Base.IteratorEltype(::Type{<:MeshNeighbors}) = Base.HasEltype()
Base.eltype(::Type{<:MeshNeighbors}) = Int
Base.length(m::MeshNeighbors) =
    _nneighbors(m.grid, m.node, nothing, m.active_only, Grids.StoredMeshNeighbors())

@inline function Base.iterate(m::MeshNeighbors, k::Int = 0)
    # A masked node has no neighbours at all, matching the offset walk.
    (k == 0 && m.active_only && !Grids.isactive(m.grid, m.node)) && return nothing
    nbr = Grids.incident_nodes(m.grid, m.node)
    @inbounds while k < length(nbr)
        k += 1
        j = Int(nbr[k])
        m.active_only && !Grids.isactive(m.grid, j) && continue
        return j, k
    end
    return nothing
end

@inline function _nneighbors(grid, i::Int, _sten, active_only::Bool, ::Grids.StoredMeshNeighbors)
    nbr = Grids.incident_nodes(grid, i)
    active_only || return length(nbr)
    Grids.isactive(grid, i) || return 0
    n = 0
    @inbounds for k in eachindex(nbr)
        Grids.isactive(grid, Int(nbr[k])) && (n += 1)
    end
    return n
end

@inline _neighbors(grid, i::Int, _sten, active_only::Bool, ::Grids.StoredMeshNeighbors) =
    MeshNeighbors(grid, i, active_only)

function _neighbors!(out::AbstractVector{<:Integer}, grid, i::Int, _sten, active_only::Bool,
                     ::Grids.StoredMeshNeighbors)
    nbr = Grids.incident_nodes(grid, i)
    (active_only && !Grids.isactive(grid, i)) && return 0
    n = 0
    @inbounds for k in eachindex(nbr)
        j = Int(nbr[k])
        active_only && !Grids.isactive(grid, j) && continue
        n += 1
        n ≤ length(out) || throw(ArgumentError("out too short (need ≥ $n)"))
        out[n] = j
    end
    return n
end

# ---- Index topology ----------------------------------------------------------

"""
    IndexTopology(size, periodic, mask)
    IndexTopology(grid)

Extent, wrapping and activity per dimension — the whole of what a neighbor computation reads.
Coordinates, cell measure and geometry never enter one, so a sampling hands this over directly, with no
grid (axes, dense measure, full mask) built to be read once and dropped. A curvilinear grid is the
`N = 2` case of the same algorithm.

`mask === nothing` means every cell is active, and costs no storage and no load.
"""
struct IndexTopology{N,M}
    size::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    mask::M
end

# `Grids.mask` is used here; dot access on a grid resolves coordinate names first. Defined where a
# cell's index space and the coordinate directions coincide, so one `periodic` flag per axis of `size`
# is meaningful.
@inline IndexTopology(grid::Grids.AbstractGrid) = IndexTopology(grid, Grids.cell_address(grid))
@inline IndexTopology(grid, ::Grids.CartesianCells) =
    IndexTopology(Grids.size_tuple(grid), _periodic_flags(grid), Grids.mask(grid))

IndexTopology(grid, ::Grids.FlatCells) = throw(ArgumentError(
    "$(nameof(typeof(grid))) enumerates its cells flatly, so it has no index space for a stencil to " *
    "range over; its adjacency is $(Grids.adjacency_source(grid))",
))

@inline _active(::IndexTopology{N,Nothing}, ::Vararg{Int,N}) where {N} = true
@inline _active(t::IndexTopology{N}, I::Vararg{Int,N}) where {N} = @inbounds t.mask[I...]

# Both queries below read only `(size, periodic, mask)`, so structured and curvilinear grids share one
# implementation. Constructing the topology copies two tuples and a mask reference, allocating nothing,
# and `N` stays a type parameter so the offset loop still unrolls.

@inline function _check_index(t::IndexTopology{N}, Ii::NTuple{N,Int}) where {N}
    @inbounds for d in 1:N
        (1 ≤ Ii[d] ≤ t.size[d]) || throw(BoundsError(t.mask, Ii))
    end
    return nothing
end

function _nneighbors(
    t::IndexTopology{N}, Ii::NTuple{N,Int}, sten::Stencils.AbstractStencil, active_only::Bool,
) where {N}
    _check_index(t, Ii)
    active_only && !_active(t, Ii...) && return 0
    sz = t.size
    per = t.periodic
    # A fold: the offsets are unrolled into the body, so a wide stencil materializes no tuple, and the
    # count is threaded through as a value with nothing captured.
    return Stencils.fold_offsets(0, sten, Val(N)) do k, δ
        J = ntuple(d -> _wrap_or_clip(Ii[d], δ[d], sz[d], per[d]), Val(N))
        any(==(0), J) && return k
        active_only && !_active(t, J...) && return k
        return k + 1
    end
end

function neighbors!(
    out::AbstractVector{<:Integer}, t::IndexTopology{N}, Ii::NTuple{N,Int},
    sten::Stencils.AbstractStencil, active_only::Bool,
) where {N}
    _check_index(t, Ii)
    active_only && !_active(t, Ii...) && return 0
    sz = t.size
    per = t.periodic
    return Stencils.fold_offsets(0, sten, Val(N)) do k, δ
        J = ntuple(d -> _wrap_or_clip(Ii[d], δ[d], sz[d], per[d]), Val(N))
        any(==(0), J) && return k
        active_only && !_active(t, J...) && return k
        k += 1
        k ≤ length(out) || throw(ArgumentError("out too short for stencil (need ≥ $k)"))
        @inbounds out[k] = _linidx(sz, J...)
        return k
    end
end


"""
    StencilNeighbors{G,N,S}

Lazy neighbor sequence of one cell of an index-topology grid: iterating it walks the stencil offsets
and yields the linear index of each in-range, active neighbor.

Nothing is stored, so a traversal that visits every cell allocates nothing at all. Use
[`neighbors!`](@ref) to write into a caller-supplied buffer, or `collect` this to materialize it.
"""
struct StencilNeighbors{GR,N,S<:Stencils.AbstractStencil}
    grid::GR
    I::NTuple{N,Int}
    stencil::S
    active_only::Bool
end

Base.IteratorSize(::Type{<:StencilNeighbors}) = Base.HasLength()
Base.IteratorEltype(::Type{<:StencilNeighbors}) = Base.HasEltype()
Base.eltype(::Type{<:StencilNeighbors}) = Int

@inline function Base.iterate(s::StencilNeighbors{GR,N}, k::Int = 0) where {GR,N}
    grid = s.grid
    # A masked-out cell has no neighbors at all, matching `nneighbors`.
    (k == 0 && s.active_only && !Grids.isactive(grid, s.I...)) && return nothing
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    offs = _stencil_offsets(Val{N}(), s.stencil)
    @inbounds while k < length(offs)
        k += 1
        δ = offs[k]
        J = ntuple(d -> _wrap_or_clip(s.I[d], δ[d], sz[d], per[d]), Val(N))
        any(==(0), J) && continue
        s.active_only && !Grids.isactive(grid, J...) && continue
        return _linidx(sz, J...), k
    end
    return nothing
end

# The iterator counts through the same kernel its own walk uses.
Base.length(s::StencilNeighbors) =
    _nneighbors(s.grid, s.I, s.stencil, s.active_only, Grids.adjacency_source(s.grid))


@inline _stencil_neighbors(grid::GR, I::NTuple{N,Int}, sten::S, active_only::Bool) where {GR,N,S} =
    StencilNeighbors{GR,N,S}(grid, I, sten, active_only)
