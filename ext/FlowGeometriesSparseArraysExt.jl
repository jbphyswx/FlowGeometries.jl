module FlowGeometriesSparseArraysExt

using SparseArrays: SparseArrays
using FlowGeometries.Connectivity: Connectivity
using FlowGeometries.Grids: Grids
using FlowGeometries.Stencils: Stencils

# A `CSRConnectivity` and a `SparseMatrixCSC` are the same layout transposed, so the matrix is built
# straight from the neighbor list by `sparse_adjacency_csc!` — no coordinate triples, no sort, no
# permutation vector. The only allocations are the three arrays the matrix takes ownership of, and
# `sparse_adjacency_matrix!` lets the caller supply even those.

# `Ti` defaults to the width the connectivity already carries, so a narrowed CSR gives a narrowed
# matrix and the two hold the same number of bytes.
function Connectivity.sparse_adjacency_matrix(
    conn::Connectivity.CSRConnectivity; Ti::Union{Nothing,Type{<:Integer}} = nothing, Tv::Type = Bool,
)
    Ti === nothing && (Ti = promote_type(eltype(conn.ptr), eltype(conn.nbrs)))
    n = Connectivity.nnodes(conn)
    ne = Connectivity.nedges(conn)
    return Connectivity.sparse_adjacency_matrix!(
        Vector{Ti}(undef, n + 1), Vector{Ti}(undef, ne), Vector{Tv}(undef, ne), conn,
    )
end

# For a symmetric graph with sorted indices the CSR and CSC arrays are the same arrays, so the matrix
# wraps the neighbour list and holds no second copy. The connectivity is built here and discarded, so
# its buffers are this call's to hand over.
#
# Whether the graph is symmetric is read from the layout, before it is built. Under
# `Grids.IndexStencilNeighbors` the answer is the stencil's own symmetry, which its type carries: a
# stencil need not be symmetric — a caller's forward-only one is the package's documented
# extension-point example — and wrapping such a neighbour list gives the transpose of the adjacency, a
# different matrix. A formula layout answers with `Grids.has_symmetric_adjacency`. A node set's
# adjacency is caller-supplied and a truncated k-nearest one is asymmetric whatever stencil was passed,
# so both stay at that trait's `false` and take the transpose.
#
# `Connectivity.is_symmetric_adjacency` answers the same question about a built graph by constructing
# its transpose, which costs more than transposing outright.
@inline function _wraps_as_csc(grid::Grids.AbstractGrid, st, ::Grids.IndexStencilNeighbors)
    return Stencils.is_symmetric(st, Val(Grids.ncoordinates(grid)))
end

@inline _wraps_as_csc(grid::Grids.AbstractGrid, _st, ::Grids.AbstractAdjacency) =
    Grids.has_symmetric_adjacency(grid)

# `Ti` defaults to the width the builder chose, so the wrap below is reachable: a builder narrows its
# ids to `Int32` wherever the node and edge counts allow
function Connectivity.sparse_adjacency_matrix(
    grid::Grids.AbstractGrid; Ti::Union{Nothing,Type{<:Integer}} = nothing, Tv::Type = Bool,
    kwargs...,
)
    conn = Connectivity.build_connectivity(grid; kwargs...)
    Ti === nothing && (Ti = promote_type(eltype(conn.ptr), eltype(conn.nbrs)))
    # The stencil is read for the guard, never forwarded
    st = get(values(kwargs), :stencil, Stencils.Axial(1))
    if _wraps_as_csc(grid, st, Grids.adjacency_source(grid)) &&
       conn.ptr isa Vector{Ti} && conn.nbrs isa Vector{Ti}
        Connectivity.sort_neighbors!(conn)
        n, ne = Connectivity.nnodes(conn), Connectivity.nedges(conn)
        return SparseArrays.SparseMatrixCSC{Tv,Ti}(n, n, conn.ptr, conn.nbrs, fill(one(Tv), ne))
    end
    return Connectivity.sparse_adjacency_matrix(conn; Ti = Ti, Tv = Tv)
end

# The concrete `Vector` is forced by `SparseMatrixCSC`, which declares its own fields as `Vector`; those
# are the only buffers it wraps without a copy, and `resize!` below needs them too.
function Connectivity.sparse_adjacency_matrix!(
    colptr::Vector{Ti}, rowval::Vector{Ti}, nzval::Vector{Tv}, conn::Connectivity.CSRConnectivity,
) where {Ti<:Integer, Tv}
    n = Connectivity.nnodes(conn)
    ne = Connectivity.sparse_adjacency_csc!(colptr, rowval, conn)
    length(nzval) ≥ ne || throw(DimensionMismatch("nzval needs length ≥ nedges = $ne"))
    fill!(view(nzval, 1:ne), one(Tv))
    # A matrix must own arrays of exactly the right length; an oversized reusable buffer is trimmed.
    length(colptr) == n + 1 || resize!(colptr, n + 1)
    length(rowval) == ne || resize!(rowval, ne)
    length(nzval) == ne || resize!(nzval, ne)
    return SparseArrays.SparseMatrixCSC{Tv,Ti}(n, n, colptr, rowval, nzval)
end

end # module
