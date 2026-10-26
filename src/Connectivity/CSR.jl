
# ---------------------------------------------------------------------------
# Sparse CSR storage
# ---------------------------------------------------------------------------

"""
    CSRConnectivity{VN,VP}

Sparse neighbor list: node `i` owns `nbrs[ptr[i]:ptr[i+1]-1]`. The two buffers are typed
independently. The builders take the ids' integer type as their first argument, `Int32` by default
(half the memory and bandwidth of `Int64`, and the width GPU kernels want), and store the offsets as
`Int`, since they run to the edge count.
"""
struct CSRConnectivity{VN<:AbstractVector{<:Integer}, VP<:AbstractVector{<:Integer}}
    nbrs::VN
    ptr::VP
end

"""
    _csr_total(deg, n, backend) -> Int

The CSR's edge count, `sum(deg)`, reduced where `deg` is held: under `backend`'s
[`Execution.local_backend`](@ref), since the counting pass leaves `deg` whole on every process that
reads it.
"""
@inline function _csr_total(deg::AbstractVector, n::Int, backend::CB.AbstractExecutionBackend)
    return Execution.reduce_indices(+, 0, n, Execution.local_backend(backend)) do k
        return @inbounds Int(deg[k])
    end
end

"""
    csr_connectivity(nbrs, ptr; validate=true) -> CSRConnectivity

Wrap CSR buffers. `validate=false` skips O(nnz) checks (internal / trusted data).
"""
function csr_connectivity(
    nbrs::AbstractVector{<:Integer}, ptr::AbstractVector{<:Integer}; validate::Bool = true,
)
    if validate
        length(ptr) ≥ 1 || throw(ArgumentError("ptr must be non-empty (length nnodes+1)"))
        ptr[1] == 1 || throw(ArgumentError("ptr[1] must be 1 (got $(ptr[1]))"))
        ptr[end] - 1 == length(nbrs) || throw(ArgumentError(
            "CSR length mismatch: length(nbrs)=$(length(nbrs)), ptr[end]-1=$(ptr[end] - 1)",
        ))
        n = length(ptr) - 1
        @inbounds for i in 1:n
            ptr[i] ≤ ptr[i + 1] || throw(ArgumentError("ptr must be nondecreasing at i=$i"))
            for k in ptr[i]:(ptr[i + 1] - 1)
                j = nbrs[k]
                (1 ≤ j ≤ n) || throw(ArgumentError("neighbor index $j out of range 1:$n"))
            end
        end
    end
    return CSRConnectivity(nbrs, ptr)
end

"""
    empty_csr(nnodes, [I=Int32]) -> CSRConnectivity

Adjacency in which every node has no neighbors, with `I`-typed ids and `Int` offsets.
"""
empty_csr(nnodes::Integer, ::Type{I} = Int32) where {I<:Integer} =
    csr_connectivity(I[], ones(Int, Int(nnodes) + 1); validate = false)

@inline nnodes(conn::CSRConnectivity) = length(conn.ptr) - 1
@inline nedges(conn::CSRConnectivity) = length(conn.nbrs)

@inline function nneighbors(conn::CSRConnectivity, i::Integer)
    @boundscheck checkbounds(conn.ptr, Int(i) + 1)
    return @inbounds conn.ptr[i + 1] - conn.ptr[i]
end

@inline function Grids.neighbors(conn::CSRConnectivity, i::Integer)
    @boundscheck checkbounds(conn.ptr, Int(i) + 1)
    lo = @inbounds conn.ptr[i]
    hi = @inbounds conn.ptr[i + 1] - 1
    return view(conn.nbrs, lo:hi)
end

# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

# Column-major linear index, for any number of dimensions. Written as a fold over the tuple so it
# unrolls: `i₁ + (i₂-1)·n₁ + (i₃-1)·n₁n₂ + …`.
@inline _linidx(sz::NTuple{1,Int}, i::Int) = i
@inline _linidx(sz::NTuple{2,Int}, i::Int, j::Int) = i + (j - 1) * sz[1]
@inline _linidx(sz::NTuple{3,Int}, i::Int, j::Int, k::Int) =
    i + (j - 1) * sz[1] + (k - 1) * sz[1] * sz[2]

@inline function _linidx(sz::NTuple{N,Int}, I::Vararg{Int,N}) where {N}
    lin = 1
    stride = 1
    @inbounds for d in 1:N
        lin += (I[d] - 1) * stride
        stride *= sz[d]
    end
    return lin
end

@inline linear_index(grid::Grids.AbstractStructuredGrid, I::Vararg{Integer,K}) where {K} =
    _linidx(Grids.size_tuple(grid), map(Int, I)...)

@inline linear_index(grid::Grids.AbstractCurvilinearGrid, I::Vararg{Integer,K}) where {K} =
    _linidx(Grids.size_tuple(grid), map(Int, I)...)

@inline cartesian_index(grid::Union{Grids.AbstractStructuredGrid,Grids.AbstractCurvilinearGrid}, lin::Integer) =
    CartesianIndices(Grids.size_tuple(grid))[lin]
