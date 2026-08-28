# ---------------------------------------------------------------------------
# build_connectivity
# ---------------------------------------------------------------------------

"""
    build_connectivity(grid; stencil=Axial(1), active_only=true) -> CSRConnectivity

Materialize CSR adjacency.

Which builder runs is [`Grids.adjacency_source`](@ref), the trait the per-cell neighbour queries resolve
through, so a layout gets this by declaring how its adjacency is defined and nothing else:
[`Grids.IndexStencilNeighbors`](@ref) ranges the stencil over an index space,
[`Grids.FormulaNeighbors`](@ref) evaluates the arithmetic per cell, and
[`Grids.StoredMeshNeighbors`](@ref) wraps the mesh's own buffers. `stencil` selects among the offsets of
the first, which is the one whose adjacency is a stencil shape.

For adjacency by physical distance rather than by stencil, see [`build_connectivity_within`](@ref).
"""
function build_connectivity end

# A fully active mesh's stored buffers already are the answer, so they are wrapped rather than copied.
# A mask makes them a superset, and `active_only` means what it means everywhere else here.
function build_connectivity(
    grid::Grids.AbstractGrid, ::Grids.StoredMeshNeighbors; active_only::Bool = true, _...,
)
    msk = Grids.mask(grid)
    (!active_only || msk isa Grids.AllActive) &&
        return csr_connectivity(Grids.neighbor_nbrs(grid), Grids.neighbor_ptr(grid); validate = false)
    n = length(msk)
    ptr = Vector{Int}(undef, n + 1)
    nbrs = Int[]
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

# Which builder runs is `Grids.adjacency_source`, the same trait the per-cell queries resolve through.
function build_connectivity(grid::Grids.AbstractGrid; kwargs...)
    return build_connectivity(grid, Grids.adjacency_source(grid); kwargs...)
end

function build_connectivity(
    grid::Grids.AbstractGrid, ::Grids.IndexStencilNeighbors;
    stencil = Stencils.Axial(1), active_only::Bool = true, backend = nothing,
)
    return _build_connectivity_topology(
        IndexTopology(grid), _stencil_val(stencil), active_only; backend = backend,
    )
end

"""
    build_connectivity_within(grid; ball, active_only=true) -> CSRConnectivity

Materialize the CSR adjacency of every pair of cells within `ball` of each other — the bulk form of
[`neighbors_within`](@ref), row `k` holding exactly what the per-cell query returns for cell `k`.

Symmetric by construction, since the metric is.

Where a separable window can bound the candidates the default `topology` needs no index; elsewhere it
carries a cell list, built once and amortized over the `n` rows, so a row costs `O(log n + m)`. Pass
`topology = MetricTopology(grid)` for the scanning build, or an [`indexed`](@ref) topology for the k-d
tree.

Rows are balls, i.e. [`Unrestricted`](@ref), and there is no [`Connected`](@ref) form: reachability
within one cell's ball is not a symmetric relation — a bridge cell can lie in one ball and not the
other — so such a graph would not be an adjacency.
"""
function build_connectivity_within end

function build_connectivity(
    t::IndexTopology;
    stencil = Stencils.Axial(1), active_only::Bool = true, backend = nothing,
)
    return _build_connectivity_topology(t, _stencil_val(stencil), active_only; backend = backend)
end

function _build_connectivity_topology(
    t::IndexTopology{N,M}, sten::Stencils.AbstractStencil, active_only::Bool; backend = nothing,
) where {N,M}
    sz = t.size
    per = t.periodic
    n = prod(sz)
    ci = CartesianIndices(sz)
    offs = _stencil_offsets(Val{N}(), sten)
    # Column-major linear order, so the k-th Cartesian index IS linear index k — no `_linidx` needed
    # for the owning cell, and chunking over `k` chunks over contiguous slots of every output.
    #
    # Both cell passes write only to slots that cell owns (`deg[k]`, then `nbrs` inside that cell's
    # own `ptr` range), so they parallelize without coordination. The prefix scan between them is
    # inherently sequential, and O(n) against the O(n·stencil) passes it separates.
    deg = zeros(Int, n)
    # Per index rather than per chunk: nothing here carries across cells, so the same body runs as a
    # device kernel when the backend is one.
    Execution.run_indices(n, backend) do k
        @inbounds begin
            I = Tuple(ci[k])
            if !(active_only && !_active(t, I...))
                c = 0
                for δ in offs
                    J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
                    any(==(0), J) && continue
                    active_only && !_active(t, J...) && continue
                    c += 1
                end
                deg[k] = c
            end
        end
    end
    ptr = Execution.exclusive_scan!(Vector{Int}(undef, n + 1), deg, backend)
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    Execution.run_indices(n, backend) do k
        @inbounds begin
            I = Tuple(ci[k])
            if !(active_only && !_active(t, I...))
                slot = ptr[k]
                for δ in offs
                    J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
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

# A sweep is where an index pays for itself — it amortizes over `n` rows, where a single query would
# pay `O(n log n)` to save one `O(n)` scan. So this builds one by default when one can be built, which
# turns the whole build from `O(n²)` into `O(n log n)`. `topology` overrides that either way.
#
# Which default applies is `Grids.candidate_source`: a separable window needs no index to be `O(1)` per
# row, and everything else amortizes one.
default_sweep_topology(grid::Grids.AbstractGrid, ball, active_only::Bool = true) =
    default_sweep_topology(grid, ball, active_only, Grids.candidate_source(grid))

default_sweep_topology(grid, _ball, _active_only::Bool, ::Grids.SeparableWindow) =
    MetricTopology(grid)

# A cell list, not a tree: it needs no package, it builds and queries faster here (2.75 ms and 1.08 µs
# at 65k cells, against 8.95 ms and 1.25 µs), and it enumerates through a fold, so the sweep holds no
# candidate buffer. `indexed(grid)` is still there for the tree.
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

# The same count → prefix-scan → fill shape as the stencil builder: both cell passes write only slots
# their own cell owns, so they parallelize without coordination.
#
# One body for every layout. How a cell is named is `Grids.cell_address`, what bounds the candidates is
# the topology, and `_within_scan` takes the same arguments on every layout — `scratch` reaches a
# separable window too, which enumerates without a buffer and leaves it unused.
function build_connectivity_within(
    grid::Grids.AbstractGrid; ball, active_only::Bool = true, backend = nothing,
    topology = default_sweep_topology(grid, ball, active_only),
)
    cs = Grids.cells(grid)
    n = length(cs)
    deg = zeros(Int, n)
    _within_pass(n, topology, backend) do k, s
        @inbounds deg[k] = _within_scan(nothing, grid, Grids.cell_at(grid, cs[k]), ball,
                                        active_only, topology, s)
        return nothing
    end
    ptr = Execution.exclusive_scan!(Vector{Int}(undef, n + 1), deg, backend)
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    _within_pass(n, topology, backend) do k, s
        @inbounds if deg[k] != 0
            _within_scan(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid, Grids.cell_at(grid, cs[k]),
                         ball, active_only, topology, s)
        end
        return nothing
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end
