module Connectivity

using ..Execution: Execution
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
    CSRConnectivity{VN,VP}

Sparse neighbor list: node `i` owns `nbrs[ptr[i]:ptr[i+1]-1]`. The two buffers are typed
independently, and their element type is any `Integer`, so a large mesh can hold `Int32` indices
(half the memory and bandwidth of `Int64`, and the width GPU kernels want).
"""
struct CSRConnectivity{VN<:AbstractVector{<:Integer}, VP<:AbstractVector{<:Integer}}
    nbrs::VN
    ptr::VP
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
    empty_csr(nnodes, [I=Int]) -> CSRConnectivity

Adjacency in which every node has no neighbors, with `I`-typed indices.
"""
empty_csr(nnodes::Integer, ::Type{I} = Int) where {I<:Integer} =
    csr_connectivity(I[], ones(I, Int(nnodes) + 1); validate = false)

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

function neighbors!(out::AbstractVector{<:Integer}, grid::Grids.UnstructuredGrid, idx::Integer; _...)
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
    out::AbstractVector{<:Integer}, grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil::Symbol = :face, active_only::Bool = true,
) where {G,T,N}
    return neighbors!(out, grid, I, _stencil_val(stencil), active_only)
end

# ---- Index topology ----------------------------------------------------------

"""
    IndexTopology(size, periodic, mask)
    IndexTopology(grid)

Extent, wrapping and activity per dimension — the whole of what a neighbor computation reads.
Coordinates, cell measure and geometry never enter one, so a sampling can hand this over directly
rather than materializing a grid (axes, dense measure, full mask) to be read once and dropped. A
curvilinear grid is the `N = 2` case of the same algorithm, not a separate one.

`mask === nothing` means every cell is active, and costs no storage and no load.
"""
struct IndexTopology{N,M}
    size::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    mask::M
end

@inline IndexTopology(grid::Grids.StructuredGrid{G,T,N}) where {G,T,N} =
    IndexTopology(Grids.size_tuple(grid), _periodic_flags(grid), grid.mask)
@inline IndexTopology(grid::Grids.CurvilinearGrid) =
    IndexTopology(Grids.size_tuple(grid), _periodic_flags(grid), grid.mask)

@inline _active(::IndexTopology{N,Nothing}, ::Vararg{Int,N}) where {N} = true
@inline _active(t::IndexTopology{N}, I::Vararg{Int,N}) where {N} = @inbounds t.mask[I...]

# Both queries below read only `(size, periodic, mask)`, so structured and curvilinear grids share
# one implementation. Constructing the topology copies two tuples and a mask REFERENCE — nothing is
# allocated, and `N` stays a type parameter so the offset loop still unrolls.

@inline function _check_index(t::IndexTopology{N}, Ii::NTuple{N,Int}) where {N}
    @inbounds for d in 1:N
        (1 ≤ Ii[d] ≤ t.size[d]) || throw(BoundsError(t.mask, Ii))
    end
    return nothing
end

function _nneighbors(t::IndexTopology{N}, Ii::NTuple{N,Int}, ::Val{S}, active_only::Bool) where {N,S}
    _check_index(t, Ii)
    active_only && !_active(t, Ii...) && return 0
    sz = t.size
    per = t.periodic
    k = 0
    @inbounds for δ in _stencil_offsets(Val{N}(), Val{S}())
        J = ntuple(d -> _wrap_or_clip(Ii[d], δ[d], sz[d], per[d]), N)
        any(==(0), J) && continue
        active_only && !_active(t, J...) && continue
        k += 1
    end
    return k
end

function neighbors!(
    out::AbstractVector{<:Integer}, t::IndexTopology{N}, Ii::NTuple{N,Int},
    ::Val{S}, active_only::Bool,
) where {N,S}
    _check_index(t, Ii)
    active_only && !_active(t, Ii...) && return 0
    sz = t.size
    per = t.periodic
    k = 0
    @inbounds for δ in _stencil_offsets(Val{N}(), Val{S}())
        J = ntuple(d -> _wrap_or_clip(Ii[d], δ[d], sz[d], per[d]), N)
        any(==(0), J) && continue
        active_only && !_active(t, J...) && continue
        k += 1
        k ≤ length(out) || throw(ArgumentError("out too short for stencil (need ≥ $k)"))
        out[k] = _linidx(sz, J...)
    end
    return k
end

@inline neighbors!(
    out::AbstractVector{<:Integer}, grid::Grids.StructuredGrid{G,T,N},
    I::NTuple{N,Integer}, v::Val, active_only::Bool,
) where {G,T,N} = neighbors!(out, IndexTopology(grid), map(Int, I), v, active_only)

function nneighbors(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil::Symbol = :face, active_only::Bool = true,
) where {G,T,N}
    return _nneighbors(IndexTopology(grid), map(Int, I), _stencil_val(stencil), active_only)
end

"""
    StencilNeighbors{G,N,S}

Lazy neighbor sequence of one cell of an index-topology grid: iterating it walks the stencil offsets
and yields the linear index of each in-range, active neighbor.

Nothing is stored, so a traversal that visits every cell allocates nothing at all — where returning a
freshly built `Vector` per cell would cost two heap allocations per cell. Use [`neighbors!`](@ref) to
write into a caller-supplied buffer, or `collect` this to materialize it.
"""
struct StencilNeighbors{GR,N,S}
    grid::GR
    I::NTuple{N,Int}
    active_only::Bool
end

Base.IteratorSize(::Type{<:StencilNeighbors}) = Base.HasLength()
Base.IteratorEltype(::Type{<:StencilNeighbors}) = Base.HasEltype()
Base.eltype(::Type{<:StencilNeighbors}) = Int
Base.length(s::StencilNeighbors{GR,N,S}) where {GR,N,S} =
    _nneighbors(s.grid, s.I, Val{S}(), s.active_only)

@inline function Base.iterate(s::StencilNeighbors{GR,N,S}, k::Int = 0) where {GR,N,S}
    grid = s.grid
    # A masked-out cell has no neighbors at all, matching `nneighbors`.
    (k == 0 && s.active_only && !Grids.isactive(grid, s.I...)) && return nothing
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    offs = _stencil_offsets(Val{N}(), Val{S}())
    @inbounds while k < length(offs)
        k += 1
        δ = offs[k]
        J = ntuple(d -> _wrap_or_clip(s.I[d], δ[d], sz[d], per[d]), N)
        any(==(0), J) && continue
        s.active_only && !Grids.isactive(grid, J...) && continue
        return _linidx(sz, J...), k
    end
    return nothing
end

# The iterator counts through the same topology kernel, for either grid type.
@inline _nneighbors(
    grid::Union{Grids.StructuredGrid,Grids.CurvilinearGrid}, I::NTuple{N,Int}, sv::Val, active_only::Bool,
) where {N} = _nneighbors(IndexTopology(grid), I, sv, active_only)

function Grids.neighbors(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil::Symbol = :face, active_only::Bool = true,
) where {G,T,N}
    Ii = map(Int, I)
    sz = Grids.size_tuple(grid)
    @inbounds for d in 1:N
        (1 ≤ Ii[d] ≤ sz[d]) || throw(BoundsError(Grids.mask(grid), I))
    end
    return _stencil_neighbors(grid, Ii, _stencil_val(stencil), active_only)
end

@inline _stencil_neighbors(grid::GR, I::NTuple{N,Int}, ::Val{S}, active_only::Bool) where {GR,N,S} =
    StencilNeighbors{GR,N,S}(grid, I, active_only)

# ---- Curvilinear -------------------------------------------------------------

function neighbors!(
    out::AbstractVector{<:Integer}, grid::Grids.CurvilinearGrid, i::Integer, j::Integer;
    stencil::Symbol = :face, active_only::Bool = true,
)
    return neighbors!(out, grid, Int(i), Int(j), _stencil_val(stencil), active_only)
end

@inline function _periodic_flags(grid::Grids.CurvilinearGrid)
    return grid.periodic
end

@inline neighbors!(
    out::AbstractVector{<:Integer}, grid::Grids.CurvilinearGrid,
    i::Int, j::Int, v::Val, active_only::Bool,
) = neighbors!(out, IndexTopology(grid), (i, j), v, active_only)

function nneighbors(
    grid::Grids.CurvilinearGrid, i::Integer, j::Integer;
    stencil::Symbol = :face, active_only::Bool = true,
)
    return _nneighbors(IndexTopology(grid), (Int(i), Int(j)), _stencil_val(stencil), active_only)
end

function Grids.neighbors(
    grid::Grids.CurvilinearGrid, i::Integer, j::Integer;
    stencil::Symbol = :face, active_only::Bool = true,
)
    sz = Grids.size_tuple(grid)
    (1 ≤ i ≤ sz[1] && 1 ≤ j ≤ sz[2]) || throw(BoundsError(Grids.mask(grid), (i, j)))
    return _stencil_neighbors(grid, (Int(i), Int(j)), _stencil_val(stencil), active_only)
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
    grid::Grids.StructuredGrid;
    stencil::Symbol = :face, active_only::Bool = true, backend = nothing,
)
    return _build_connectivity_topology(
        IndexTopology(grid), _stencil_val(stencil), active_only; backend = backend,
    )
end

function build_connectivity(
    t::IndexTopology;
    stencil::Symbol = :face, active_only::Bool = true, backend = nothing,
)
    return _build_connectivity_topology(t, _stencil_val(stencil), active_only; backend = backend)
end

function _build_connectivity_topology(
    t::IndexTopology{N,M}, ::Val{S}, active_only::Bool; backend = nothing,
) where {N,M,S}
    sz = t.size
    per = t.periodic
    n = prod(sz)
    ci = CartesianIndices(sz)
    offs = _stencil_offsets(Val{N}(), Val{S}())
    # Column-major linear order, so the k-th Cartesian index IS linear index k — no `_linidx` needed
    # for the owning cell, and chunking over `k` chunks over contiguous slots of every output.
    #
    # Both cell passes write only to slots that cell owns (`deg[k]`, then `nbrs` inside that cell's
    # own `ptr` range), so they parallelize without coordination. The prefix scan between them is
    # inherently sequential, and O(n) against the O(n·stencil) passes it separates.
    deg = zeros(Int, n)
    Execution.run_chunks(n, backend) do rng
        @inbounds for k in rng
            I = Tuple(ci[k])
            active_only && !_active(t, I...) && continue
            c = 0
            for δ in offs
                J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), N)
                any(==(0), J) && continue
                active_only && !_active(t, J...) && continue
                c += 1
            end
            deg[k] = c
        end
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    Execution.run_chunks(n, backend) do rng
        @inbounds for k in rng
            I = Tuple(ci[k])
            active_only && !_active(t, I...) && continue
            slot = ptr[k]
            for δ in offs
                J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), N)
                any(==(0), J) && continue
                active_only && !_active(t, J...) && continue
                nbrs[slot] = _linidx(sz, J...)
                slot += 1
            end
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

# A curvilinear grid's connectivity is its index topology's, exactly as a structured grid's is —
# neighbors never consult coordinates, so the N = 2 case needs no separate implementation.
function build_connectivity(
    grid::Grids.CurvilinearGrid;
    stencil::Symbol = :face, active_only::Bool = true, backend = nothing,
)
    return _build_connectivity_topology(
        IndexTopology(grid), _stencil_val(stencil), active_only; backend = backend,
    )
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
        for j in Grids.neighbors(conn, i)
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

Whether `j ∈ N(i)` implies `i ∈ N(j)` throughout. `O(nedges·degree)`; used to guard the shortcut
that reads a CSR as a CSC.
"""
function is_symmetric_adjacency(conn::CSRConnectivity)
    ptr, nbrs = conn.ptr, conn.nbrs
    @inbounds for i in 1:nnodes(conn)
        for k in ptr[i]:(ptr[i + 1] - 1)
            j = nbrs[k]
            found = false
            for q in ptr[j]:(ptr[j + 1] - 1)
                if nbrs[q] == i
                    found = true
                    break
                end
            end
            found || return false
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

include("ConnectivitySpherical.jl")

end # module
