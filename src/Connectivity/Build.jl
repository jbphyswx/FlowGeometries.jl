# ---------------------------------------------------------------------------
# build_connectivity
# ---------------------------------------------------------------------------

"""
    build_connectivity([I], grid; stencil=Axial(1), active_only=true) -> CSRConnectivity

Materialize CSR adjacency, its node ids of integer type `I` and its offsets `Int`. `I` defaults to the
type a stored mesh holds, and to `Int32` elsewhere.

Which builder runs is [`Grids.adjacency_source`](@ref), the trait the per-cell neighbour queries resolve
through, so a layout gets this by declaring how its adjacency is defined and nothing else:
[`Grids.IndexStencilNeighbors`](@ref) ranges the stencil over an index space,
[`Grids.FormulaNeighbors`](@ref) evaluates the arithmetic per cell, and
[`Grids.StoredMeshNeighbors`](@ref) wraps the mesh's own buffers. `stencil` selects among the offsets of
the first, which is the one whose adjacency is a stencil shape.

For adjacency by physical distance, see [`build_connectivity_within`](@ref).
"""
function build_connectivity end

# Which builder runs is `Grids.adjacency_source`, the same trait the per-cell queries resolve through.
build_connectivity(grid::Grids.AbstractGrid; kwargs...) =
    build_connectivity(_default_index_type(grid), grid; kwargs...)

build_connectivity(::Type{I}, grid::Grids.AbstractGrid; kwargs...) where {I<:Integer} =
    build_connectivity(I, grid, Grids.adjacency_source(grid); kwargs...)

# A stored mesh's own id type, so its buffers are wrapped; every graph built here takes `Int32`.
@inline _default_index_type(grid::Grids.AbstractGrid) =
    _default_index_type(grid, Grids.adjacency_source(grid))
@inline _default_index_type(grid, ::Grids.StoredMeshNeighbors) = eltype(Grids.neighbor_nbrs(grid))
@inline _default_index_type(_grid, ::Grids.AbstractAdjacency) = Int32

@inline _ids_as(::Type{I}, v::AbstractVector{I}) where {I<:Integer} = v
@inline _ids_as(::Type{I}, v::AbstractVector{<:Integer}) where {I<:Integer} = I.(v)

# A fully active mesh's stored buffers already are the answer, and are wrapped in place. A mask makes
# them a superset, and `active_only` means what it means everywhere else here.
function build_connectivity(
    ::Type{I}, grid::Grids.AbstractGrid, ::Grids.StoredMeshNeighbors; active_only::Bool = true, _...,
) where {I<:Integer}
    msk = Grids.mask(grid)
    n = length(msk)
    Grids._check_ids(I, n)
    (!active_only || msk isa Grids.AllActive) && return csr_connectivity(
        _ids_as(I, Grids.neighbor_nbrs(grid)), Grids.neighbor_ptr(grid); validate = false,
    )
    ptr = Vector{Int}(undef, n + 1)
    nbrs = I[]
    sizehint!(nbrs, length(Grids.neighbor_nbrs(grid)))
    @inbounds ptr[1] = 1
    @inbounds for i in 1:n
        for j in Grids.neighbors(grid, i; active_only = true)
            push!(nbrs, j)
        end
        ptr[i + 1] = length(nbrs) + 1
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

function build_connectivity(
    ::Type{I}, grid::Grids.AbstractGrid, ::Grids.IndexStencilNeighbors;
    stencil = Stencils.Axial(1), active_only::Bool = true,
    execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {I<:Integer}
    return _build_connectivity_topology(I, IndexTopology(grid), _stencil_val(stencil), active_only,
                                        execution)
end

"""
    build_connectivity_within([I], grid; ball, active_only=true) -> CSRConnectivity

Materialize the CSR adjacency of every pair of cells within `ball` of each other — the bulk form of
[`neighbors_within`](@ref), row `k` holding exactly what the per-cell query returns for cell `k`. The
cell ids are of integer type `I`, `Int32` by default, and the offsets `Int`.

Symmetric by construction, since the metric is.

Where a separable window can bound the candidates the default `topology` needs no index; elsewhere it
carries a cell list, built once and amortized over the `n` rows, so a row costs `O(log n + m)`. Pass
`topology = MetricTopology(grid)` for the scanning build, or an [`indexed`](@ref) topology for the k-d
tree.

Rows are balls, i.e. [`Unrestricted`](@ref), and there is no [`Connected`](@ref) form: reachability
within one cell's ball is asymmetric — a bridge cell can lie in one cell's ball while that cell lies
outside its own — so such a graph is not an adjacency.

# For a mesh-degree radius

The result is `∑ᵢ degree(i)` integers, and a ball's degree grows with the cube (or square) of its
radius. At a radius of a few cell widths that is mesh-degree adjacency, built once and read row by row
many times. At a radius of tens of widths the degree reaches `10³`–`10⁴`, and on a million cells the
graph is `10⁹`–`10¹⁰` integers: tens of gigabytes.

Ask [`neighbors_within!`](@ref), [`fold_within`](@ref) or [`mapreduce_within`](@ref) instead. Those go
through the same index and visit the same cells, one row at a time, and never hold more than one row.
"""
function build_connectivity_within end

build_connectivity(t::IndexTopology; kwargs...) = build_connectivity(Int32, t; kwargs...)

function build_connectivity(
    ::Type{I}, t::IndexTopology;
    stencil = Stencils.Axial(1), active_only::Bool = true,
    execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {I<:Integer}
    return _build_connectivity_topology(I, t, _stencil_val(stencil), active_only, execution)
end

function _build_connectivity_topology(
    ::Type{I}, t::IndexTopology{N,M}, sten::Stencils.AbstractStencil, active_only::Bool,
    execution::CB.AbstractExecutionBackend,
) where {I<:Integer,N,M}
    sz = t.size
    per = t.periodic
    n = prod(sz)
    Grids._check_ids(I, n)
    ci = CartesianIndices(sz)
    offs = _stencil_offsets(Val{N}(), sten)
    # Column-major linear order, so the k-th Cartesian index IS linear index k — no `_linidx` needed
    # for the owning cell, and chunking over `k` chunks over contiguous slots of every output.
    #
    # Both cell passes write only to slots that cell owns (`deg[k]`, then `nbrs` inside that cell's
    # own `ptr` range), so they parallelize without coordination, and the scan between them splits at
    # its spans. Every buffer comes from `Execution.allocate`, so all three land wherever the passes
    # filling them run.
    deg = Execution.allocate(execution, Int, n)
    # Per index: nothing here carries across cells, so the same body runs as a device kernel under a
    # device backend. Every cell writes its own degree, so the buffer needs no zeroing pass.
    Execution.run_indices(n, execution) do k
        @inbounds begin
            Ik = Tuple(ci[k])
            c = 0
            if !(active_only && !_active(t, Ik...))
                for δ in offs
                    J = ntuple(d -> _wrap_or_clip(Ik[d], δ[d], sz[d], per[d]), Val(N))
                    any(==(0), J) && continue
                    active_only && !_active(t, J...) && continue
                    c += 1
                end
            end
            deg[k] = c
        end
    end
    total = _csr_total(deg, n, execution)
    ptr = Execution.exclusive_scan!(Execution.allocate(execution, Int, n + 1), deg, execution)
    nbrs = Execution.allocate(execution, I, total)
    Execution.run_indices(n, execution) do k
        @inbounds begin
            Ik = Tuple(ci[k])
            if !(active_only && !_active(t, Ik...))
                slot = ptr[k]
                for δ in offs
                    J = ntuple(d -> _wrap_or_clip(Ik[d], δ[d], sz[d], per[d]), Val(N))
                    any(==(0), J) && continue
                    active_only && !_active(t, J...) && continue
                    nbrs[slot] = _linidx(sz, J...)
                    slot += 1
                end
            end
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

# A sweep amortizes an index over `n` rows, so this builds one by default where one can be built, and
# the whole build is `O(n log n)` against the `O(n²)` of a scan per row. A single query costs
# `O(n log n)` to build and saves one `O(n)` scan. `topology` overrides the default either way.
#
# Which default applies is `Grids.candidate_source`: a separable window needs no index to be `O(1)` per
# row, and everything else amortizes one.
default_sweep_topology(grid::Grids.AbstractGrid, ball, active_only::Bool = true) =
    default_sweep_topology(grid, ball, active_only, Grids.candidate_source(grid))

default_sweep_topology(grid, _ball, _active_only::Bool, ::Grids.SeparableWindow) =
    MetricTopology(grid)

# A cell list: it needs no package, it builds and queries faster than the tree here, and it enumerates
# through a fold, so the sweep holds no candidate buffer. `indexed(grid)` selects the tree.
#
# The sweep's own `active_only` is passed on, so a sweep over the active region indexes that region and
# no more.
default_sweep_topology(grid, ball, active_only::Bool, ::Grids.IndexedCandidates) =
    MetricTopology(grid; index = Grids.cell_list(grid; ball = _ball_radius(ball),
                                                 active_only = active_only))

# One cell pass. Which form it takes is [`_buffered_candidates`](@ref) — per index where the candidates
# need no storage, per chunk where each task needs its own buffer — and the choice folds away, being a
# property of the topology's type. `body(k, scratch)` handles cell `k`.
@inline function _within_pass(body::F, n::Int, topology::MetricTopology, backend) where {F}
    if _buffered_candidates(topology.index)
        Execution.run_chunks(n, backend) do rng
            s = ball_scratch()
            @inbounds for k in rng
                body(k, s)
            end
        end
    else
        Execution.run_indices(n, backend) do k
            body(k, nothing)
        end
    end
    return nothing
end

build_connectivity_within(grid::Grids.AbstractGrid; kwargs...) =
    build_connectivity_within(Int32, grid; kwargs...)

# One body for every layout. How a cell is named is `Grids.cell_address`, what bounds the candidates is
# the topology, and `_within_scan` takes the same arguments on every layout — `scratch` reaches a
# separable window too, which enumerates without a buffer and leaves it unused.
function build_connectivity_within(
    ::Type{I}, grid::Grids.AbstractGrid; ball, active_only::Bool = true,
    execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
    topology = default_sweep_topology(grid, ball, active_only),
) where {I<:Integer}
    cs = Grids.cells(grid)
    n = length(cs)
    Grids._check_ids(I, n)
    return _within_build(I, grid, cs, n, ball, active_only, topology, Execution.resolve(execution))
end

# On the host each row's ball is queried once: every chunk appends its rows to a buffer of its own, and
# the chunks, being contiguous and in order, concatenate into the CSR. A serial build is one chunk, whose
# buffer is the CSR's.
function _within_build(
    ::Type{I}, grid, cs, n::Int, ball, active_only::Bool, topology,
    backend::Union{CB.AbstractSerialBackend,CB.AbstractThreadedBackend},
) where {I<:Integer}
    deg = Vector{Int}(undef, n)
    bufs = Execution.map_chunks(n, backend) do rng
        buf = I[]
        s = ball_scratch(topology)
        @inbounds for k in rng
            m = length(buf)
            _within_append!(buf, grid, Grids.cell_at(grid, cs[k]), ball, active_only, topology, s)
            deg[k] = length(buf) - m
        end
        return buf
    end
    ptr = Execution.exclusive_scan!(Vector{Int}(undef, n + 1), deg)
    length(bufs) == 1 && return csr_connectivity(only(bufs), ptr; validate = false)
    nbrs = Vector{I}(undef, sum(length, bufs; init = 0))
    pos = 1
    @inbounds for c in eachindex(bufs)
        b = bufs[c]
        copyto!(nbrs, pos, b, 1, length(b))
        pos += length(b)
        bufs[c] = similar(b, 0)                        # the chunk's rows are copied; let them go
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

@inline _within_append!(buf, grid, I, ball, active_only::Bool, mt, scratch) =
    _route_fold(buf, Unrestricted(), grid, I, ball, NearestImage(), active_only, false, mt,
                scratch) do v, J, _
        push!(v, _sweep_linear(grid, J))
        return v
    end

# A device, or a set of processes, counts every row and then fills it: the count → prefix-scan → fill
# shape of the stencil builder, both passes writing only slots their own cell owns.
function _within_build(
    ::Type{I}, grid, cs, n::Int, ball, active_only::Bool, topology, backend,
) where {I<:Integer}
    # Every cell writes its own degree, so the buffer needs no zeroing pass before the count.
    deg = Execution.allocate(backend, Int, n)
    _within_pass(n, topology, backend) do k, s
        @inbounds deg[k] = _within_scan(nothing, grid, Grids.cell_at(grid, cs[k]), ball,
                                        active_only, topology, s)
        return nothing
    end
    ptr = Execution.exclusive_scan!(Execution.allocate(backend, Int, n + 1), deg, backend)
    nbrs = Execution.allocate(backend, I, _csr_total(deg, n, backend))
    _within_pass(n, topology, backend) do k, s
        @inbounds if deg[k] != 0
            _within_scan(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid,
                         Grids.cell_at(grid, cs[k]), ball, active_only, topology, s)
        end
        return nothing
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end
