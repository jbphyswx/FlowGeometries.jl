# ---------------------------------------------------------------------------
# Adjacency that is arithmetic
# ---------------------------------------------------------------------------
#
# A layout with `Grids.FormulaNeighbors` supplies `Grids.formula_neighbors(grid, cell)`, returning a
# fixed-width tuple and a count. Everything below is written against that, so a layout adds its
# arithmetic and gets the whole neighbour API — the buffer form, the count, and the lazy sequence.

"""
    FormulaNeighborSeq{GR,K}

Lazy neighbour sequence of one cell of a layout whose adjacency is arithmetic.

Holds the tuple `Grids.formula_neighbors` returned, so iterating it touches no heap: the counterpart of
[`StencilNeighbors`](@ref) and [`MeshNeighbors`](@ref) for the third kind of adjacency.
"""
struct FormulaNeighborSeq{GR,K}
    grid::GR
    ids::NTuple{K,Int}
    n::Int
    active_only::Bool
end

Base.IteratorSize(::Type{<:FormulaNeighborSeq}) = Base.HasLength()
Base.IteratorEltype(::Type{<:FormulaNeighborSeq}) = Base.HasEltype()
Base.eltype(::Type{<:FormulaNeighborSeq}) = Int

function Base.length(s::FormulaNeighborSeq)
    s.active_only || return s.n
    m = 0
    @inbounds for t in 1:(s.n)
        Grids.isactive(s.grid, s.ids[t]) && (m += 1)
    end
    return m
end

@inline function Base.iterate(s::FormulaNeighborSeq, t::Int = 0)
    @inbounds while t < s.n
        t += 1
        j = s.ids[t]
        s.active_only && !Grids.isactive(s.grid, j) && continue
        return j, t
    end
    return nothing
end

# A masked cell has no neighbours at all, which is the rule the offset walk and the stored graph keep.
@inline function _formula_ids(grid, i::Int, active_only::Bool)
    (active_only && !Grids.isactive(grid, i)) &&
        return (ntuple(_ -> 0, Val(Grids.max_neighbors(grid))), 0)
    return Grids.formula_neighbors(grid, i)
end

@inline function _nneighbors(grid, i::Int, _sten, active_only::Bool, ::Grids.FormulaNeighbors)
    ids, n = _formula_ids(grid, i, active_only)
    active_only || return n
    m = 0
    @inbounds for t in 1:n
        Grids.isactive(grid, ids[t]) && (m += 1)
    end
    return m
end

@inline function _neighbors(grid, i::Int, _sten, active_only::Bool, ::Grids.FormulaNeighbors)
    ids, n = _formula_ids(grid, i, active_only)
    return FormulaNeighborSeq(grid, ids, n, active_only)
end

function _neighbors!(
    out::AbstractVector{<:Integer}, grid, i::Int, _sten, active_only::Bool,
    ::Grids.FormulaNeighbors,
)
    ids, n = _formula_ids(grid, i, active_only)
    m = 0
    @inbounds for t in 1:n
        j = ids[t]
        active_only && !Grids.isactive(grid, j) && continue
        m += 1
        m ≤ length(out) || throw(ArgumentError("out too short (need ≥ $m)"))
        out[m] = j
    end
    return m
end

# ---- HEALPix ----------------------------------------------------------------

# Eight compass directions, of which the eight pixels at a face corner have only seven.
@inline Grids.max_neighbors(::Grids.HEALPixGrid) = 8

@inline function Grids.formula_neighbors(grid::Grids.HEALPixGrid, i::Integer)
    ids, n = healpix_neighbor_ids(Grids.nside(grid), Int(i) - 1)
    # The pixel walk speaks 0-based ids; a cell here is 1-based.
    return (ntuple(t -> @inbounds(t ≤ n ? ids[t] + 1 : 0), Val(8)), n)
end
