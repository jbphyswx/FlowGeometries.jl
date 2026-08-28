module FlowGeometriesSparseArraysExt

using SparseArrays: SparseArrays
using FlowGeometries.Connectivity: Connectivity
using FlowGeometries.Grids: Grids
using FlowGeometries.Stencils: Stencils

# A `CSRConnectivity` and a `SparseMatrixCSC` are the same layout transposed, so the matrix is built
# straight from the neighbor list by `sparse_adjacency_csc!` — no coordinate triples, no sort, no
# permutation vector. The only allocations are the three arrays the matrix takes ownership of, and
# `sparse_adjacency_matrix!` lets the caller supply even those.

function Connectivity.sparse_adjacency_matrix(
    conn::Connectivity.CSRConnectivity; Ti::Type{<:Integer} = Int, Tv::Type = Bool,
)
    n = Connectivity.nnodes(conn)
    ne = Connectivity.nedges(conn)
    return Connectivity.sparse_adjacency_matrix!(
        Vector{Ti}(undef, n + 1), Vector{Ti}(undef, ne), Vector{Tv}(undef, ne), conn,
    )
end

# For a SYMMETRIC graph with sorted indices the CSR and CSC arrays are the same arrays, so the matrix
# can wrap the neighbour list instead of transposing it into a second permanent copy. The connectivity
# is built here and discarded, so its buffers are this call's to hand over.
#
# The guard is the STENCIL's symmetry, read from its type, not the built graph's. A stencil need not be
# symmetric — a caller's own forward-only one is the package's documented extension-point example — and
# wrapping ITS neighbour list returns the transpose of the adjacency, which is a different matrix.
# Asking the graph instead would answer the same question by building the very transpose the wrap
# exists to avoid: measured, that costs more in total than transposing outright.
function Connectivity.sparse_adjacency_matrix(
    grid::Grids.AbstractGrid; Ti::Type{<:Integer} = Int, Tv::Type = Bool, kwargs...,
)
    conn = Connectivity.build_connectivity(grid; kwargs...)
    # The stencil is read for the guard, never forwarded
    st = get(values(kwargs), :stencil, Stencils.Axial(1))
    # …and the guard holds only where the graph COMES from a stencil. A node set's adjacency is stored,
    # and a truncated k-nearest one is not symmetric however symmetric the stencil it ignored; a
    # formula layout's is its own. Both take the transpose.
    if Grids.adjacency_source(grid) isa Grids.IndexStencilNeighbors &&
       Stencils.is_symmetric(st, Val(Grids.ncoordinates(grid))) &&
       conn.ptr isa Vector{Ti} && conn.nbrs isa Vector{Ti}
        Connectivity.sort_neighbors!(conn)
        n, ne = Connectivity.nnodes(conn), Connectivity.nedges(conn)
        return SparseArrays.SparseMatrixCSC{Tv,Ti}(n, n, conn.ptr, conn.nbrs, fill(one(Tv), ne))
    end
    return Connectivity.sparse_adjacency_matrix(conn; Ti = Ti, Tv = Tv)
end

# Concrete `Vector` here is forced, not sloppiness: `SparseMatrixCSC` declares its own fields as
# `Vector`, so those are the only buffers it can wrap without a copy — and `resize!` needs them too.
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
