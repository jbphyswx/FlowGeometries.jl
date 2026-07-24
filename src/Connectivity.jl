module Connectivity

using ..Grids: Grids

# Connectivity is a property of the *grid architecture* or of a *spherical sampling*
# that defines its own mesh topology:
#   Structured / Curvilinear → index-topology stencil (periodicity + mask)
#   Unstructured            → stored CSR on the grid
#   Cubed-sphere / Yin–Yang / HEALPix / icosahedral / tensor-product samplings
#                           → `build_connectivity(sampling, …)` (see ConnectivitySpherical.jl)
#
# Primary API dispatches on `grid` or `sampling`. `CSRConnectivity` is the sparse storage
# format from `build_connectivity` when you need a flat graph.

# ---------------------------------------------------------------------------
# Sparse CSR storage
# ---------------------------------------------------------------------------

"""
    CSRConnectivity{VI}

Sparse neighbor list: node `i` owns `nbrs[ptr[i]:ptr[i+1]-1]`.
"""
struct CSRConnectivity{VI<:AbstractVector{Int}}
    nbrs::VI
    ptr::VI
end

"""
    csr_connectivity(nbrs, ptr; validate=true) -> CSRConnectivity

Wrap CSR buffers. `validate=false` skips O(nnz) checks (internal / trusted data).
"""
function csr_connectivity(nbrs::AbstractVector{Int}, ptr::AbstractVector{Int}; validate::Bool = true)
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

empty_csr(nnodes::Integer) = csr_connectivity(Int[], ones(Int, Int(nnodes) + 1); validate = false)

@inline nnodes(conn::CSRConnectivity) = length(conn.ptr) - 1
@inline nedges(conn::CSRConnectivity) = length(conn.nbrs)

@inline function nneighbors(conn::CSRConnectivity, i::Integer)
    @boundscheck checkbounds(conn.ptr, Int(i) + 1)
    return @inbounds conn.ptr[i + 1] - conn.ptr[i]
end

@inline function neighbors(conn::CSRConnectivity, i::Integer)
    @boundscheck checkbounds(conn.ptr, Int(i) + 1)
    lo = @inbounds conn.ptr[i]
    hi = @inbounds conn.ptr[i + 1] - 1
    return view(conn.nbrs, lo:hi)
end

# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

@inline function _linidx(sz::NTuple{1,Int}, i::Int)
    return i
end
@inline function _linidx(sz::NTuple{2,Int}, i::Int, j::Int)
    return i + (j - 1) * sz[1]
end
@inline function _linidx(sz::NTuple{3,Int}, i::Int, j::Int, k::Int)
    return i + (j - 1) * sz[1] + (k - 1) * sz[1] * sz[2]
end

@inline linear_index(grid::Grids.AbstractStructuredGrid, I::Vararg{Integer}) =
    _linidx(Grids.size_tuple(grid), map(Int, I)...)

@inline linear_index(grid::Grids.AbstractCurvilinearGrid, i::Integer, j::Integer) =
    _linidx(Grids.size_tuple(grid), Int(i), Int(j))

@inline cartesian_index(grid::Union{Grids.AbstractStructuredGrid,Grids.AbstractCurvilinearGrid}, lin::Integer) =
    CartesianIndices(Grids.size_tuple(grid))[lin]

# ---------------------------------------------------------------------------
# Const stencils (Val — no Symbol / no heap in hot path)
# ---------------------------------------------------------------------------

const FACE_1D = ((-1,), (1,))
const FACE_2D = ((-1, 0), (1, 0), (0, -1), (0, 1))
const FACE_3D = (
    (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
)
const VERTEX_1D = FACE_1D
const VERTEX_2D = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)
const VERTEX_3D = let
    offs = NTuple{3,Int}[]
    for dz in -1:1, dy in -1:1, dx in -1:1
        (dx == 0 && dy == 0 && dz == 0) && continue
        push!(offs, (dx, dy, dz))
    end
    Tuple(offs)
end

@inline _stencil_offsets(::Val{1}, ::Val{:face}) = FACE_1D
@inline _stencil_offsets(::Val{2}, ::Val{:face}) = FACE_2D
@inline _stencil_offsets(::Val{3}, ::Val{:face}) = FACE_3D
@inline _stencil_offsets(::Val{1}, ::Val{:vertex}) = VERTEX_1D
@inline _stencil_offsets(::Val{2}, ::Val{:vertex}) = VERTEX_2D
@inline _stencil_offsets(::Val{3}, ::Val{:vertex}) = VERTEX_3D

function _stencil_val(stencil::Symbol)
    stencil === :face && return Val{:face}()
    stencil === :vertex && return Val{:vertex}()
    throw(ArgumentError("stencil must be :face or :vertex, got $stencil"))
end

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

@inline function _periodic_flags(grid::Grids.StructuredGrid{G,T,N}) where {G,T,N}
    return ntuple(d -> Grids.isperiodic(grid, d), N)
end

# ---------------------------------------------------------------------------
# neighbors! / nneighbors / Grids.neighbors — dispatch on grid
# ---------------------------------------------------------------------------

"""
    neighbors!(out, grid, I...; stencil=:face, active_only=true) -> n_written
    nneighbors(grid, I...; …) -> Int
    Grids.neighbors(grid, I...; …) -> Vector{Int}

Neighbors as linear indices into `grid.mask`. Prefer `neighbors!` on hot paths.
`stencil` is `:face` or `:vertex` (converted to `Val` once per call).
"""
function neighbors! end
function nneighbors end

# ---- Unstructured ------------------------------------------------------------

@inline nneighbors(grid::Grids.UnstructuredGrid, idx::Integer; _...) =
    length(Grids.neighbors(grid, idx))

function neighbors!(out::AbstractVector{Int}, grid::Grids.UnstructuredGrid, idx::Integer; _...)
    nbr = Grids.neighbors(grid, idx)
    n = length(nbr)
    n ≤ length(out) || throw(ArgumentError("out too short (need ≥ $n)"))
    @inbounds for k in 1:n
        out[k] = nbr[k]
    end
    return n
end

# ---- Structured --------------------------------------------------------------

function neighbors!(
    out::AbstractVector{Int}, grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil::Symbol = :face, active_only::Bool = true,
) where {G,T,N}
    return neighbors!(out, grid, I, _stencil_val(stencil), active_only)
end

function neighbors!(
    out::AbstractVector{Int},
    grid::Grids.StructuredGrid{G,T,N},
    I::NTuple{N,Integer},
    ::Val{S},
    active_only::Bool,
) where {G,T,N,S}
    sz = Grids.size_tuple(grid)
    Ii = map(Int, I)
    @inbounds for d in 1:N
        (1 ≤ Ii[d] ≤ sz[d]) || throw(BoundsError(grid.mask, I))
    end
    if active_only && !(@inbounds grid.mask[Ii...])
        return 0
    end
    per = _periodic_flags(grid)
    offs = _stencil_offsets(Val{N}(), Val{S}())
    k = 0
    @inbounds for δ in offs
        J = ntuple(d -> _wrap_or_clip(Ii[d], δ[d], sz[d], per[d]), N)
        any(==(0), J) && continue
        active_only && !grid.mask[J...] && continue
        k += 1
        k ≤ length(out) || throw(ArgumentError("out too short for stencil (need ≥ $k)"))
        out[k] = _linidx(sz, J...)
    end
    return k
end

function nneighbors(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil::Symbol = :face, active_only::Bool = true,
) where {G,T,N}
    return _nneighbors(grid, map(Int, I), _stencil_val(stencil), active_only)
end

function _nneighbors(
    grid::Grids.StructuredGrid{G,T,N},
    Ii::NTuple{N,Int},
    ::Val{S},
    active_only::Bool,
) where {G,T,N,S}
    sz = Grids.size_tuple(grid)
    @inbounds for d in 1:N
        (1 ≤ Ii[d] ≤ sz[d]) || throw(BoundsError(grid.mask, Ii))
    end
    if active_only && !(@inbounds grid.mask[Ii...])
        return 0
    end
    per = _periodic_flags(grid)
    offs = _stencil_offsets(Val{N}(), Val{S}())
    k = 0
    @inbounds for δ in offs
        J = ntuple(d -> _wrap_or_clip(Ii[d], δ[d], sz[d], per[d]), N)
        any(==(0), J) && continue
        active_only && !grid.mask[J...] && continue
        k += 1
    end
    return k
end

function Grids.neighbors(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil::Symbol = :face, active_only::Bool = true,
) where {G,T,N}
    sv = _stencil_val(stencil)
    buf = Vector{Int}(undef, length(_stencil_offsets(Val{N}(), sv)))
    n = neighbors!(buf, grid, map(Int, I), sv, active_only)
    return resize!(buf, n)
end

# ---- Curvilinear -------------------------------------------------------------

function neighbors!(
    out::AbstractVector{Int}, grid::Grids.CurvilinearGrid, i::Integer, j::Integer;
    stencil::Symbol = :face, active_only::Bool = true,
)
    return neighbors!(out, grid, Int(i), Int(j), _stencil_val(stencil), active_only)
end

@inline function _periodic_flags(grid::Grids.CurvilinearGrid)
    return grid.periodic
end

function neighbors!(
    out::AbstractVector{Int},
    grid::Grids.CurvilinearGrid,
    i::Int, j::Int,
    ::Val{S},
    active_only::Bool,
) where {S}
    sz = Grids.size_tuple(grid)
    (1 ≤ i ≤ sz[1] && 1 ≤ j ≤ sz[2]) || throw(BoundsError(grid.mask, (i, j)))
    if active_only && !(@inbounds grid.mask[i, j])
        return 0
    end
    per = _periodic_flags(grid)
    offs = _stencil_offsets(Val{2}(), Val{S}())
    k = 0
    @inbounds for δ in offs
        ii = _wrap_or_clip(i, δ[1], sz[1], per[1])
        jj = _wrap_or_clip(j, δ[2], sz[2], per[2])
        (ii == 0 || jj == 0) && continue
        active_only && !grid.mask[ii, jj] && continue
        k += 1
        k ≤ length(out) || throw(ArgumentError("out too short for stencil (need ≥ $k)"))
        out[k] = _linidx(sz, ii, jj)
    end
    return k
end

function nneighbors(
    grid::Grids.CurvilinearGrid, i::Integer, j::Integer;
    stencil::Symbol = :face, active_only::Bool = true,
)
    return _nneighbors(grid, Int(i), Int(j), _stencil_val(stencil), active_only)
end

function _nneighbors(
    grid::Grids.CurvilinearGrid, i::Int, j::Int, ::Val{S}, active_only::Bool,
) where {S}
    sz = Grids.size_tuple(grid)
    (1 ≤ i ≤ sz[1] && 1 ≤ j ≤ sz[2]) || throw(BoundsError(grid.mask, (i, j)))
    if active_only && !(@inbounds grid.mask[i, j])
        return 0
    end
    per = _periodic_flags(grid)
    offs = _stencil_offsets(Val{2}(), Val{S}())
    k = 0
    @inbounds for δ in offs
        ii = _wrap_or_clip(i, δ[1], sz[1], per[1])
        jj = _wrap_or_clip(j, δ[2], sz[2], per[2])
        (ii == 0 || jj == 0) && continue
        active_only && !grid.mask[ii, jj] && continue
        k += 1
    end
    return k
end

function Grids.neighbors(
    grid::Grids.CurvilinearGrid, i::Integer, j::Integer;
    stencil::Symbol = :face, active_only::Bool = true,
)
    sv = _stencil_val(stencil)
    buf = Vector{Int}(undef, length(_stencil_offsets(Val{2}(), sv)))
    n = neighbors!(buf, grid, Int(i), Int(j), sv, active_only)
    return resize!(buf, n)
end

# ---------------------------------------------------------------------------
# build_connectivity
# ---------------------------------------------------------------------------

"""
    build_connectivity(grid; stencil=:face, active_only=true) -> CSRConnectivity

Materialize CSR adjacency. Unstructured wraps existing buffers without re-validation.
"""
function build_connectivity end

build_connectivity(grid::Grids.UnstructuredGrid; _...) =
    csr_connectivity(grid.neighbor_nbrs, grid.neighbor_ptr; validate = false)

function build_connectivity(
    grid::Grids.StructuredGrid{G,T,N};
    stencil::Symbol = :face, active_only::Bool = true,
) where {G,T,N}
    return _build_connectivity_structured(grid, _stencil_val(stencil), active_only)
end

function _build_connectivity_structured(
    grid::Grids.StructuredGrid{G,T,N}, ::Val{S}, active_only::Bool,
) where {G,T,N,S}
    sz = Grids.size_tuple(grid)
    n = length(grid.mask)
    offs = _stencil_offsets(Val{N}(), Val{S}())
    per = _periodic_flags(grid)
    # deg[i] = degree; then exclusive scan into ptr; reuse deg as write heads
    deg = zeros(Int, n)
    @inbounds for ci in CartesianIndices(sz)
        I = Tuple(ci)
        active_only && !grid.mask[I...] && continue
        lin = _linidx(sz, I...)
        for δ in offs
            J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), N)
            any(==(0), J) && continue
            active_only && !grid.mask[J...] && continue
            deg[lin] += 1
        end
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    fill!(deg, 0)  # reuse as per-row insertion counters
    @inbounds for ci in CartesianIndices(sz)
        I = Tuple(ci)
        active_only && !grid.mask[I...] && continue
        lin = _linidx(sz, I...)
        for δ in offs
            J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), N)
            any(==(0), J) && continue
            active_only && !grid.mask[J...] && continue
            slot = ptr[lin] + deg[lin]
            nbrs[slot] = _linidx(sz, J...)
            deg[lin] += 1
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

function build_connectivity(
    grid::Grids.CurvilinearGrid;
    stencil::Symbol = :face, active_only::Bool = true,
)
    return _build_connectivity_curvilinear(grid, _stencil_val(stencil), active_only)
end

function _build_connectivity_curvilinear(
    grid::Grids.CurvilinearGrid, ::Val{S}, active_only::Bool,
) where {S}
    sz = Grids.size_tuple(grid)
    n = length(grid.mask)
    offs = _stencil_offsets(Val{2}(), Val{S}())
    per = _periodic_flags(grid)
    deg = zeros(Int, n)
    @inbounds for j in 1:sz[2], i in 1:sz[1]
        active_only && !grid.mask[i, j] && continue
        lin = _linidx(sz, i, j)
        for δ in offs
            ii = _wrap_or_clip(i, δ[1], sz[1], per[1])
            jj = _wrap_or_clip(j, δ[2], sz[2], per[2])
            (ii == 0 || jj == 0) && continue
            active_only && !grid.mask[ii, jj] && continue
            deg[lin] += 1
        end
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    fill!(deg, 0)
    @inbounds for j in 1:sz[2], i in 1:sz[1]
        active_only && !grid.mask[i, j] && continue
        lin = _linidx(sz, i, j)
        for δ in offs
            ii = _wrap_or_clip(i, δ[1], sz[1], per[1])
            jj = _wrap_or_clip(j, δ[2], sz[2], per[2])
            (ii == 0 || jj == 0) && continue
            active_only && !grid.mask[ii, jj] && continue
            slot = ptr[lin] + deg[lin]
            nbrs[slot] = _linidx(sz, ii, jj)
            deg[lin] += 1
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

# ---------------------------------------------------------------------------
# Dense adjacency — bang first; grid path fills from stencil (no CSR)
# ---------------------------------------------------------------------------

"""
    adjacency_matrix!(A, conn) -> A
    adjacency_matrix!(A, grid; stencil=:face, active_only=true) -> A

Fill preallocated `N×N` `A`. Grid overload uses the stencil directly (no CSR alloc).
"""
function adjacency_matrix!(A::AbstractMatrix{Bool}, conn::CSRConnectivity)
    n = nnodes(conn)
    size(A) == (n, n) || throw(DimensionMismatch(
        "adjacency buffer size $(size(A)) != ($n, $n) = nnodes×nnodes",
    ))
    fill!(A, false)
    @inbounds for i in 1:n
        for j in neighbors(conn, i)
            A[i, j] = true
        end
    end
    return A
end

function adjacency_matrix!(
    A::AbstractMatrix{Bool}, grid::Grids.StructuredGrid{G,T,N};
    stencil::Symbol = :face, active_only::Bool = true,
) where {G,T,N}
    n = length(grid.mask)
    size(A) == (n, n) || throw(DimensionMismatch(
        "adjacency buffer size $(size(A)) != ($n, $n)",
    ))
    fill!(A, false)
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    offs = _stencil_offsets(Val{N}(), _stencil_val(stencil))
    @inbounds for ci in CartesianIndices(sz)
        I = Tuple(ci)
        active_only && !grid.mask[I...] && continue
        row = _linidx(sz, I...)
        for δ in offs
            J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), N)
            any(==(0), J) && continue
            active_only && !grid.mask[J...] && continue
            A[row, _linidx(sz, J...)] = true
        end
    end
    return A
end

function adjacency_matrix!(
    A::AbstractMatrix{Bool}, grid::Grids.CurvilinearGrid;
    stencil::Symbol = :face, active_only::Bool = true,
)
    n = length(grid.mask)
    size(A) == (n, n) || throw(DimensionMismatch(
        "adjacency buffer size $(size(A)) != ($n, $n)",
    ))
    fill!(A, false)
    sz = Grids.size_tuple(grid)
    offs = _stencil_offsets(Val{2}(), _stencil_val(stencil))
    per = _periodic_flags(grid)
    @inbounds for j in 1:sz[2], i in 1:sz[1]
        active_only && !grid.mask[i, j] && continue
        row = _linidx(sz, i, j)
        for δ in offs
            ii = _wrap_or_clip(i, δ[1], sz[1], per[1])
            jj = _wrap_or_clip(j, δ[2], sz[2], per[2])
            (ii == 0 || jj == 0) && continue
            active_only && !grid.mask[ii, jj] && continue
            A[row, _linidx(sz, ii, jj)] = true
        end
    end
    return A
end

function adjacency_matrix!(A::AbstractMatrix{Bool}, grid::Grids.UnstructuredGrid; _...)
    return adjacency_matrix!(A, build_connectivity(grid))
end

adjacency_matrix(conn::CSRConnectivity) =
    adjacency_matrix!(Matrix{Bool}(undef, nnodes(conn), nnodes(conn)), conn)

adjacency_matrix(grid::Grids.AbstractGrid; kwargs...) =
    adjacency_matrix!(Matrix{Bool}(undef, length(grid.mask), length(grid.mask)), grid; kwargs...)

# ---------------------------------------------------------------------------
# SparseMatrixCSC — extension; COO bang in core for reuse
# ---------------------------------------------------------------------------

"""
    sparse_adjacency_coo!(I, J, conn) -> ne
    sparse_adjacency_coo!(I, J, V, conn) -> ne

Fill preallocated COO buffers (`length ≥ nedges(conn)`). `V`, if given, is set to `true`.
"""
function sparse_adjacency_coo!(I::AbstractVector{Int}, J::AbstractVector{Int}, conn::CSRConnectivity)
    ne = nedges(conn)
    length(I) ≥ ne && length(J) ≥ ne || throw(DimensionMismatch(
        "COO buffers need length ≥ nedges=$ne (got $(length(I)), $(length(J)))",
    ))
    k = 0
    n = nnodes(conn)
    @inbounds for i in 1:n
        for j in neighbors(conn, i)
            k += 1
            I[k] = i
            J[k] = j
        end
    end
    return k
end

function sparse_adjacency_coo!(
    I::AbstractVector{Int}, J::AbstractVector{Int}, V::AbstractVector{Bool}, conn::CSRConnectivity,
)
    ne = sparse_adjacency_coo!(I, J, conn)
    length(V) ≥ ne || throw(DimensionMismatch("V length must be ≥ nedges=$ne"))
    @inbounds for k in 1:ne
        V[k] = true
    end
    return ne
end

"""
    sparse_adjacency_matrix(grid_or_conn; kwargs...) -> SparseMatrixCSC

Requires `using SparseArrays` (extension). Prefer [`sparse_adjacency_coo!`](@ref)
when reusing COO buffers.
"""
function sparse_adjacency_matrix end

sparse_adjacency_matrix(grid::Grids.AbstractGrid; kwargs...) =
    sparse_adjacency_matrix(build_connectivity(grid; kwargs...))

include("ConnectivitySpherical.jl")

end # module
