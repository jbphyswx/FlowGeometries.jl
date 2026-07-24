module FlowGeometriesSparseArraysExt

using SparseArrays: SparseArrays
using FlowGeometries.Connectivity: Connectivity

function Connectivity.sparse_adjacency_matrix(conn::Connectivity.CSRConnectivity)
    ne = Connectivity.nedges(conn)
    I = Vector{Int}(undef, ne)
    J = Vector{Int}(undef, ne)
    Connectivity.sparse_adjacency_coo!(I, J, conn)
    # values length == nedges (not N²); CSC assemble allocates once
    return SparseArrays.sparse(I, J, trues(ne), Connectivity.nnodes(conn), Connectivity.nnodes(conn))
end

end # module
