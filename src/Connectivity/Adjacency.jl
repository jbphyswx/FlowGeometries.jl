# ---------------------------------------------------------------------------
# Dense adjacency — bang first; grid path fills from stencil (no CSR)
# ---------------------------------------------------------------------------

"""
    adjacency_matrix!(A, conn) -> A
    adjacency_matrix!(A, grid; stencil=Axial(1), active_only=true) -> A

Fill preallocated `N×N` `A`. Grid overload uses the stencil directly (no CSR alloc).
"""
function adjacency_matrix!(A::AbstractMatrix{Bool}, conn::CSRConnectivity)
    n = nnodes(conn)
    size(A) == (n, n) || throw(DimensionMismatch(
        "adjacency buffer size $(size(A)) != ($n, $n) = nnodes×nnodes",
    ))
    fill!(A, false)
    @inbounds for i in 1:n
        for j in Grids.neighbors(conn, i)
            A[i, j] = true
        end
    end
    return A
end

# Structured and curvilinear read only `(size, periodic, mask)` here, so they share one N-generic
# implementation through the topology, exactly as the neighbour queries above do.
adjacency_matrix!(
    A::AbstractMatrix{Bool}, grid::Union{Grids.StructuredGrid,Grids.CurvilinearGrid};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) = adjacency_matrix!(A, IndexTopology(grid); stencil = stencil, active_only = active_only)

function adjacency_matrix!(
    A::AbstractMatrix{Bool}, t::IndexTopology{N};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {N}
    n = prod(t.size)
    size(A) == (n, n) || throw(DimensionMismatch(
        "adjacency buffer size $(size(A)) != ($n, $n)",
    ))
    fill!(A, false)
    sz = t.size
    per = t.periodic
    offs = _stencil_offsets(Val{N}(), _stencil_val(stencil))
    @inbounds for ci in CartesianIndices(sz)
        I = Tuple(ci)
        active_only && !_active(t, I...) && continue
        row = _linidx(sz, I...)
        for δ in offs
            J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
            any(==(0), J) && continue
            active_only && !_active(t, J...) && continue
            A[row, _linidx(sz, J...)] = true
        end
    end
    return A
end

function adjacency_matrix!(A::AbstractMatrix{Bool}, grid::Grids.UnstructuredGrid; _...)
    return adjacency_matrix!(A, build_connectivity(grid))
end

"""
    adjacency_matrix(grid_or_conn; kwargs...) -> Matrix{Bool}

Dense `n × n` adjacency over the `n` nodes.

**This allocates n² bytes**, which is quadratic in the node count and therefore quartic in the side
of a 2-D grid: a 1000×1000 grid has 10⁶ nodes and so needs ~10¹² bytes. Dense adjacency is for small
node counts and for testing. For anything else use [`sparse_adjacency_matrix`](@ref), which stores
`nedges` entries instead of `n²`, or [`neighbors!`](@ref), which answers neighbour queries from the
index stencil with no graph storage at all.
"""
adjacency_matrix(conn::CSRConnectivity) =
    adjacency_matrix!(Matrix{Bool}(undef, nnodes(conn), nnodes(conn)), conn)

adjacency_matrix(grid::Grids.AbstractGrid; kwargs...) =
    adjacency_matrix!(Matrix{Bool}(undef, length(Grids.mask(grid)), length(Grids.mask(grid))),
                      grid; kwargs...)

# ---------------------------------------------------------------------------
# SparseMatrixCSC — extension; COO bang in core for reuse
# ---------------------------------------------------------------------------

"""
    sparse_adjacency_coo!(I, J, conn) -> ne
    sparse_adjacency_coo!(I, J, V, conn) -> ne

Fill preallocated COO buffers (`length ≥ nedges(conn)`). `V`, if given, is set to `true`.
"""
function sparse_adjacency_coo!(I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}, conn::CSRConnectivity)
    ne = nedges(conn)
    length(I) ≥ ne && length(J) ≥ ne || throw(DimensionMismatch(
        "COO buffers need length ≥ nedges=$ne (got $(length(I)), $(length(J)))",
    ))
    k = 0
    n = nnodes(conn)
    @inbounds for i in 1:n
        for j in Grids.neighbors(conn, i)
            k += 1
            I[k] = i
            J[k] = j
        end
    end
    return k
end

function sparse_adjacency_coo!(
    I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}, V::AbstractVector{Bool}, conn::CSRConnectivity,
)
    ne = sparse_adjacency_coo!(I, J, conn)
    length(V) ≥ ne || throw(DimensionMismatch("V length must be ≥ nedges=$ne"))
    @inbounds for k in 1:ne
        V[k] = true
    end
    return ne
end

"""
    sort_neighbors!(conn) -> conn

Sort each node's neighbor block ascending, in place. Rows are stencil-short, so an insertion sort
per row is `O(nedges)` overall and needs no scratch.
"""
function sort_neighbors!(conn::CSRConnectivity)
    ptr, nbrs = conn.ptr, conn.nbrs
    @inbounds for i in 1:nnodes(conn)
        lo, hi = ptr[i], ptr[i + 1] - 1
        for a in (lo + 1):hi
            v = nbrs[a]
            b = a - 1
            while b ≥ lo && nbrs[b] > v
                nbrs[b + 1] = nbrs[b]
                b -= 1
            end
            nbrs[b + 1] = v
        end
    end
    return conn
end

"""
    is_symmetric_adjacency(conn) -> Bool

Whether `j ∈ N(i)` implies `i ∈ N(j)` throughout. `O(nedges + nnodes)`, by comparing the graph with its
transpose rather than searching a row per edge; guards the shortcut that reads a CSR as a CSC.
"""
function is_symmetric_adjacency(conn::CSRConnectivity)
    ptr, nbrs = conn.ptr, conn.nbrs
    n = nnodes(conn)
    n == 0 && return true
    # Transpose by counting sort, then compare each row against it. Searching row `j` for `i` per edge
    # instead is `O(nedges·degree)`, which a stencil graph hides (degree 6) and a ball graph does not
    # (degree in the hundreds).
    deg = zeros(Int, n)
    @inbounds for k in eachindex(nbrs)
        j = Int(nbrs[k])
        (1 ≤ j ≤ n) || return false
        deg[j] += 1
    end
    tptr = Vector{Int}(undef, n + 1)
    tptr[1] = 1
    @inbounds for i in 1:n
        tptr[i + 1] = tptr[i] + deg[i]
    end
    cursor = copy(tptr)
    tnbrs = Vector{Int}(undef, length(nbrs))
    @inbounds for i in 1:n
        for k in ptr[i]:(ptr[i + 1] - 1)
            j = Int(nbrs[k])
            tnbrs[cursor[j]] = i
            cursor[j] += 1
        end
    end
    # Row `i` of the transpose is exactly the set of nodes naming `i`. Equal degrees plus every
    # transpose entry present in row `i` means the two rows agree, which is the symmetry claim.
    stamp = zeros(Int, n)
    @inbounds for i in 1:n
        (ptr[i + 1] - ptr[i]) == (tptr[i + 1] - tptr[i]) || return false
        for k in ptr[i]:(ptr[i + 1] - 1)
            stamp[Int(nbrs[k])] = i
        end
        for k in tptr[i]:(tptr[i + 1] - 1)
            stamp[tnbrs[k]] == i || return false
        end
    end
    return true
end

"""
    sparse_adjacency_csc!(colptr, rowval, conn) -> nedges

Fill caller-owned CSC structure arrays (`length(colptr) ≥ nnodes+1`, `length(rowval) ≥ nedges`) for
the adjacency of `conn`, so that entry `(i, j)` is set iff `j` is a neighbor of `i`.

This is the direct route to a sparse matrix: CSR and CSC are the same layout transposed, so the
structure is obtained by one counting pass and one placement pass over the existing neighbor list —
no coordinate triples are materialized, nothing is sorted, and no permutation vector is built. Row
indices come out ascending within each column for free, because the placement pass walks nodes in
order. The running cursors live in `colptr` itself and are shifted back at the end, so this needs no
scratch beyond the two output arrays.
"""
function sparse_adjacency_csc!(
    colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer}, conn::CSRConnectivity,
)
    n = nnodes(conn)
    ne = nedges(conn)
    length(colptr) ≥ n + 1 || throw(DimensionMismatch("colptr needs length ≥ nnodes+1 = $(n + 1)"))
    length(rowval) ≥ ne || throw(DimensionMismatch("rowval needs length ≥ nedges = $ne"))
    ptr, nbrs = conn.ptr, conn.nbrs

    @inbounds for j in 1:(n + 1)
        colptr[j] = 0
    end
    @inbounds for k in 1:ne
        colptr[nbrs[k] + 1] += 1
    end
    @inbounds colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] += colptr[j]
    end
    # Place each (i → j) as row i of column j, advancing that column's cursor in place.
    @inbounds for i in 1:n
        for k in ptr[i]:(ptr[i + 1] - 1)
            j = nbrs[k]
            rowval[colptr[j]] = i
            colptr[j] += 1
        end
    end
    # Cursors now sit one past each column; shift them back into proper offsets.
    @inbounds for j in n:-1:1
        colptr[j + 1] = colptr[j]
    end
    @inbounds colptr[1] = 1
    return ne
end

"""
    sparse_adjacency_matrix(grid_or_conn; kwargs...) -> SparseMatrixCSC

Requires `using SparseArrays` (extension). Prefer [`sparse_adjacency_coo!`](@ref)
when reusing COO buffers.
"""
function sparse_adjacency_matrix end

"""
    sparse_adjacency_matrix!(colptr, rowval, nzval, conn) -> SparseMatrixCSC

Assemble the adjacency matrix into caller-owned buffers and wrap them without copying, so a repeated
build reuses storage instead of allocating a new matrix each time. Requires `using SparseArrays`.

The returned matrix ALIASES the three buffers, so reusing them for a later call invalidates any
matrix built from them earlier. Buffers longer than needed are trimmed to fit (`nnodes+1` and
`nedges`), since a matrix must own arrays of exactly the right length.
"""
function sparse_adjacency_matrix! end
