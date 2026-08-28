# ---------------------------------------------------------------------------
# build_connectivity
# ---------------------------------------------------------------------------

"""
    build_connectivity(grid; stencil=Axial(1), active_only=true) -> CSRConnectivity

Materialize CSR adjacency. Unstructured wraps existing buffers without re-validation.

For adjacency by physical distance rather than by stencil, see [`build_connectivity_within`](@ref).
"""
function build_connectivity end

# A fully active mesh's stored buffers already are the answer, so they are wrapped rather than copied.
# A mask makes them a superset, and `active_only` means what it means everywhere else here.
function build_connectivity(
    grid::Grids.AbstractUnstructuredGrid; active_only::Bool = true, _...,
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

function build_connectivity(
    grid::Grids.StructuredGrid;
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

On the architectures with no separable axes to bound a window with — curvilinear and node grids — the
default `topology` is [`indexed`](@ref) when `NearestNeighbors` is loaded, making the build
`O(n log n)` rather than `O(n²)`; pass `topology = MetricTopology(grid)` for the scanning build.

Rows are balls, i.e. [`Unrestricted`](@ref), and there is no [`Connected`](@ref) form: reachability
within one cell's ball is not a symmetric relation — a bridge cell can lie in one ball and not the
other — so such a graph would not be an adjacency.
"""
function build_connectivity_within end

# The same count → prefix-scan → fill shape as the stencil builder: both passes write only slots the
# cell owns, so they chunk without coordination.
function build_connectivity_within(
    grid::Grids.StructuredGrid{T, G,N}; ball, active_only::Bool = true, backend = nothing,
) where {G,T,N}
    sz = Grids.size_tuple(grid)
    n = prod(sz)
    ci = CartesianIndices(sz)
    deg = zeros(Int, n)
    mt = MetricTopology(grid)     # a grid invariant: built once, not once per row
    Execution.run_chunks(n, backend) do rng
        @inbounds for k in rng
            deg[k] = _within_scan(nothing, grid, Tuple(ci[k]), ball, active_only, mt)
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
            deg[k] == 0 && continue
            _within_scan(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid, Tuple(ci[k]), ball, active_only, mt)
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

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
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
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

# A curvilinear grid's connectivity is its index topology's, exactly as a structured grid's is —
# neighbors never consult coordinates, so the N = 2 case needs no separate implementation.
function build_connectivity(
    grid::Grids.CurvilinearGrid;
    stencil = Stencils.Axial(1), active_only::Bool = true, backend = nothing,
)
    return _build_connectivity_topology(
        IndexTopology(grid), _stencil_val(stencil), active_only; backend = backend,
    )
end

# A sweep is where an index pays for itself — it amortizes over `n` rows, where a single query would
# pay `O(n log n)` to save one `O(n)` scan. So this builds one by default when one can be built, which
# turns the whole build from `O(n²)` into `O(n log n)`. `topology` overrides that either way.
#
# `default_sweep_topology` is not a silent fallback: with no extension loaded there is no index to
# have, and the unindexed topology computes the same rows.
default_sweep_topology(grid::Grids.StructuredGrid, _ball) = MetricTopology(grid)

# A cell list, not a tree: it needs no package, it builds and queries faster here (2.75 ms and 1.08 µs
# at 65k cells, against 8.95 ms and 1.25 µs), and it enumerates through a fold, so the sweep holds no
# candidate buffer. `indexed(grid)` is still there for the tree.
default_sweep_topology(grid::Grids.AbstractGrid, ball) =
    MetricTopology(grid; index = Grids.cell_list(grid; ball = _ball_radius(ball)))

function build_connectivity_within(
    grid::Grids.CurvilinearGrid; ball, active_only::Bool = true, backend = nothing,
    topology = default_sweep_topology(grid, ball),
)
    sz = Grids.size_tuple(grid)
    n = prod(sz)
    ci = CartesianIndices(sz)
    deg = zeros(Int, n)
    # One candidate buffer per chunk, since the topology is shared read-only across them.
    Execution.run_chunks(n, backend) do rng
        s = ball_scratch()
        @inbounds for k in rng
            deg[k] = _within_scan(nothing, grid, Tuple(ci[k]), ball, active_only, topology, s)
        end
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    Execution.run_chunks(n, backend) do rng
        s = ball_scratch()
        @inbounds for k in rng
            deg[k] == 0 && continue
            _within_scan(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid, Tuple(ci[k]), ball,
                         active_only, topology, s)
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

function build_connectivity_within(
    grid::Grids.UnstructuredGrid; ball, active_only::Bool = true, backend = nothing,
    topology = default_sweep_topology(grid, ball),
)
    n = length(Grids.mask(grid))
    deg = zeros(Int, n)
    Execution.run_chunks(n, backend) do rng
        s = ball_scratch()
        @inbounds for k in rng
            deg[k] = _within_scan(nothing, grid, k, ball, active_only, topology, s)
        end
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    Execution.run_chunks(n, backend) do rng
        s = ball_scratch()
        @inbounds for k in rng
            deg[k] == 0 && continue
            _within_scan(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid, k, ball,
                         active_only, topology, s)
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end
