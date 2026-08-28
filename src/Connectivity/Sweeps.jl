# ---------------------------------------------------------------------------
# Sweeps — every cell's ball, with the per-grid work done once
# ---------------------------------------------------------------------------

# What a sweep iterates, and how one of its elements becomes the cell the per-cell kernels take.
@inline _sweep_cells(grid::Grids.AbstractGrid) = _sweep_cells(grid, Grids.cell_address(grid))
@inline _sweep_cells(grid, ::Grids.CartesianCells) = CartesianIndices(Grids.size_tuple(grid))
@inline _sweep_cells(grid, ::Grids.FlatCells) = Base.OneTo(length(Grids.mask(grid)))

@inline _sweep_index(grid::Grids.AbstractGrid, c) = _sweep_index(grid, c, Grids.cell_address(grid))
@inline _sweep_index(_grid, ci::CartesianIndex, ::Grids.CartesianCells) = Tuple(ci)
@inline _sweep_index(_grid, k::Integer, ::Grids.FlatCells) = Int(k)


# Only the separable architectures carry an image convention; refused rather than ignored elsewhere.
@inline _check_sweep_images(::Grids.StructuredGrid, ::AbstractImageConvention) = nothing
@inline _check_sweep_images(grid, images::AbstractImageConvention) =
    images isa NearestImage || throw(ArgumentError(
        "`images = $(images)` is only meaningful on a `StructuredGrid`; $(typeof(grid)) visits each cell once",
    ))

"""
    mapreduce_within(f, op, init, grid; ball, …) -> value

Reduce `f(I, J, d)` with `op` over every cell `I` of `grid` and every cell `J` within `ball` of it, `d`
being the distance. The bulk counterpart of [`fold_within`](@ref).

Everything that depends on the grid rather than the query is built **once** and reused across all `n`
cells — above all the spatial index, which is what makes the sweep `O(n log n)` instead of `O(n²)` on a
curvilinear or node grid. Writing the loop by hand gets the topology for free, since that is `O(1)`, but
not the index; measured at 9× on a 9 216-cell curvilinear grid.

`op` must be associative; chunks are reduced in index order, so a threaded `backend` gives the same
answer as the serial default rather than one that depends on scheduling.
"""
function mapreduce_within(
    f::F, op::O, init, grid::Grids.AbstractGrid;
    ball, images::AbstractImageConvention = NearestImage(), active_only::Bool = true,
    self::Bool = false, topology = default_sweep_topology(grid, ball, active_only),
    reach::AbstractReach = Unrestricted(), backend = nothing,
) where {F,O}
    _check_sweep_images(grid, images)
    cells = _sweep_cells(grid)
    n = length(cells)
    return Execution._reduce_chunks(op, n, backend) do rng
        acc = init
        s = ball_scratch()
        @inbounds for t in rng
            I = _sweep_index(grid, cells[t])
            acc = _route_fold(acc, reach, grid, I, ball, images, active_only, self, topology, s) do a, J, d
                return op(a, f(I, J, d))
            end
        end
        return acc
    end
end

"""
    foreach_within(f, grid; ball, …) -> nothing

Call `f(I, J, d)` for every cell `I` of `grid` and every cell `J` within `ball` of it. The same hoisting
as [`mapreduce_within`](@ref); use this one when `f` writes rather than reduces.

Under a threaded `backend`, `f` runs on disjoint spans of cells concurrently, so what it writes has to be
determined by `I` — the same contract the connectivity builders keep.
"""
function foreach_within(
    f::F, grid::Grids.AbstractGrid;
    ball, images::AbstractImageConvention = NearestImage(), active_only::Bool = true,
    self::Bool = false, topology = default_sweep_topology(grid, ball, active_only),
    reach::AbstractReach = Unrestricted(), backend = nothing,
) where {F}
    _check_sweep_images(grid, images)
    cells = _sweep_cells(grid)
    _sweep_cells_with(f, grid, cells, ball, images, active_only, self, topology, reach, backend)
    return nothing
end

# Without an index there is no candidate buffer to own, so the sweep is one body per cell and runs
# wherever `run_indices` runs — a device included. With one, each task needs its own buffer, which only
# the chunked form can give it.
function _sweep_cells_with(
    f::F, grid, cells, ball, images, active_only, self,
    topology::MetricTopology{N,T,Nothing}, reach, backend,
) where {F,N,T}
    Execution.run_indices(length(cells), backend) do t
        I = _sweep_index(grid, @inbounds cells[t])
        _route_fold(nothing, reach, grid, I, ball, images, active_only, self, topology, nothing) do _, J, d
            f(I, J, d)
            return nothing
        end
    end
    return nothing
end

function _sweep_cells_with(
    f::F, grid, cells, ball, images, active_only, self, topology::MetricTopology, reach, backend,
) where {F}
    Execution.run_chunks(length(cells), backend) do rng
        s = ball_scratch()
        @inbounds for t in rng
            I = _sweep_index(grid, cells[t])
            _route_fold(nothing, reach, grid, I, ball, images, active_only, self, topology, s) do _, J, d
                f(I, J, d)
                return nothing
            end
        end
    end
    return nothing
end
