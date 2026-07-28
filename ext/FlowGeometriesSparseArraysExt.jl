module FlowGeometriesSparseArraysExt

using SparseArrays: SparseArrays
using FlowGeometries.Connectivity: Connectivity
using FlowGeometries.Grids: Grids

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

Connectivity.sparse_adjacency_matrix(grid::Grids.AbstractGrid; kwargs...) =
    Connectivity.sparse_adjacency_matrix(Connectivity.build_connectivity(grid; kwargs...))

# A stencil adjacency is symmetric — `J` is reachable from `I` under offset `δ` exactly when `I` is
# reachable from `J` under `−δ`, and masking or clipping removes both directions together. For a
# symmetric matrix with sorted indices the CSR and CSC arrays are THE SAME arrays, so transposing
# here would allocate a second copy of the structure just built. The intermediate connectivity is
# owned by this call and discarded, so its buffers can be handed to the matrix directly.
function Connectivity.sparse_adjacency_matrix(
    grid::Union{Grids.AbstractStructuredGrid,Grids.AbstractCurvilinearGrid};
    Ti::Type{<:Integer} = Int, Tv::Type = Bool, kwargs...,
)
    conn = Connectivity.build_connectivity(grid; kwargs...)
    conn.ptr isa Vector{Ti} && conn.nbrs isa Vector{Ti} || return Connectivity.sparse_adjacency_matrix(
        conn; Ti = Ti, Tv = Tv,
    )
    Connectivity.sort_neighbors!(conn)
    n, ne = Connectivity.nnodes(conn), Connectivity.nedges(conn)
    return SparseArrays.SparseMatrixCSC{Tv,Ti}(n, n, conn.ptr, conn.nbrs, fill(one(Tv), ne))
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
