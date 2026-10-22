# ---------------------------------------------------------------------------
# k nearest
# ---------------------------------------------------------------------------

# A bounded max-heap over (distance, index), held in the caller's two buffers. Keeping the k smallest
# needs no sort of the candidate set and no allocation, and it is the same code for every architecture
# because it consumes `fold_within`.
# Ordered by (distance, index), so an equal-distance tie resolves the same way whatever order the
# traversal visited in, and an indexed query and a scan return the same cells.
@inline _knn_after(d1, i1, d2, i2) = d1 > d2 || (d1 == d2 && i1 > i2)

@inline function _heap_sift_down!(ds, is, n::Int, root::Int)
    @inbounds while true
        c = 2root
        c > n && break
        (c < n && _knn_after(ds[c + 1], is[c + 1], ds[c], is[c])) && (c += 1)
        _knn_after(ds[c], is[c], ds[root], is[root]) || break
        ds[root], ds[c] = ds[c], ds[root]
        is[root], is[c] = is[c], is[root]
        root = c
    end
    return nothing
end

@inline function _heap_offer!(ds, is, n::Int, k::Int, d, i)
    @inbounds if n < k
        n += 1
        ds[n] = d
        is[n] = i
        c = n
        while c > 1                       # sift up
            p = c >> 1
            _knn_after(ds[c], is[c], ds[p], is[p]) || break
            ds[p], ds[c] = ds[c], ds[p]
            is[p], is[c] = is[c], is[p]
            c = p
        end
    elseif k > 0 && _knn_after(ds[1], is[1], d, i)
        ds[1] = d
        is[1] = i
        _heap_sift_down!(ds, is, n, 1)
    end
    return n
end

# Nearest first, which a heap does not give on its own.
@inline function _heap_sort!(ds, is, n::Int)
    @inbounds for last in n:-1:2
        ds[1], ds[last] = ds[last], ds[1]
        is[1], is[last] = is[last], is[1]
        _heap_sift_down!(ds, is, last - 1, 1)
    end
    return nothing
end

# A radius to start the search from, holding about `cells` cells the size of cell `I`.
@inline _knn_seed_radius(grid::Grids.AbstractGrid{G,T}, I, cells::Int) where {G,T} =
    _knn_ball_radius(T(Grids.measure(grid, I...)), cells, Grids.ncoordinates(grid))

# The radius of a ball holding about `2k` cells of measure `m` in `D` coordinate directions:
# `V_D ρᴰ = 2k m`, with `V_D` the measure of the unit ball. A search opening there finishes in one round
# where the cells are even, and doubles where they are sparse; the doubling makes the result
# independent of the guess. The `D`-th root turns a measure into a length: a spherical cell's measure is
# an area and its square root a distance, whatever the rank of the index space.
@inline function _knn_ball_radius(m::T, k::Int, D::Int) where {T}
    return (2 * max(k, 1) * (m > 0 ? m : one(T)) / _unit_ball(T, D))^(one(T) / D)
end

# `V_D = (2π/D) V_{D-2}`, from `V_0 = 1` and `V_1 = 2`.
@inline function _unit_ball(::Type{T}, D::Int) where {T}
    v = isodd(D) ? T(2) : one(T)
    for d in (isodd(D) ? 3 : 2):2:D
        v *= 2 * T(π) / d
    end
    return v
end

"""
    k_nearest!(idx, dist, grid, I...; k, active_only=true, …) -> n

Write the `k` cells nearest to cell `I` into `idx`, and their distances into `dist`, nearest first.
Returns how many were written, which is fewer than `k` only when the grid holds fewer candidates. The
cell itself is excluded, as in [`neighbors_within!`](@ref).

Exact under the geometry's own metric, on every architecture. It searches a ball, widens it until `k`
cells have been seen, and keeps the `k` smallest in a bounded heap — so the answer never depends on the
starting radius, and no candidate list is materialized. `topology`, `scratch` and `reach` behave as they
do for [`neighbors_within!`](@ref); an [`indexed`](@ref) topology makes each round a range query.

Ties at equal distance are broken by linear index, so the result is reproducible.
"""
function k_nearest! end

"""
    k_nearest(grid, I...; k, …) -> (idx, dist)

Allocating form of [`k_nearest!`](@ref): returns the indices and their distances, nearest first.
"""
function k_nearest end

function _k_nearest!(
    idx::AbstractVector{<:Integer}, dist::AbstractVector, grid::Grids.AbstractGrid, Ii, Iraw,
    kk::Int, active_only::Bool, topology, scratch, reach::AbstractReach,
)
    kk ≥ 0 || throw(ArgumentError("k must be non-negative, got $kk"))
    (kk == 0 || isempty(idx)) && return 0
    (length(idx) ≥ kk && length(dist) ≥ kk) || throw(ArgumentError(
        "idx and dist must hold at least k = $kk entries; got $(length(idx)) and $(length(dist))",
    ))
    r = _knn_seed_radius(grid, Iraw, kk + 1)      # the ball holds the cell itself
    rmax = _knn_radius_ceiling(grid, topology)
    n = 0
    while true
        n = _route_fold(0, reach, grid, Ii, r, NearestImage(), active_only, false, topology, scratch) do m, J, d
            return _heap_offer!(dist, idx, m, kk, d, _sweep_linear(grid, J))
        end
        (n ≥ kk || r ≥ rmax) && break
        r = min(r * 2, rmax)
    end
    _heap_sort!(dist, idx, n)
    return n
end

# One pair of methods for every layout: how a cell is named is `Grids.cell_address`, which
# `_cell_named_by` resolves, and the search itself reads only coordinates and the mask. The arity stays a
# type parameter because a `Vararg{Integer}` with no length is not specialized on arity and allocates on
# every call.
function k_nearest!(
    idx::AbstractVector{<:Integer}, dist::AbstractVector, grid::Grids.AbstractGrid,
    I::Vararg{Integer,NI};
    k::Integer, active_only::Bool = true, topology = MetricTopology(grid),
    scratch = nothing, reach::AbstractReach = Unrestricted(),
) where {NI}
    Iraw = map(Int, I)
    return _k_nearest!(idx, dist, grid, Grids._cell_named_by(grid, Iraw), Iraw,
                       Int(k), active_only, topology, scratch, reach)
end

function k_nearest(
    grid::Grids.AbstractGrid{G,T}, I::Vararg{Integer,NI}; k::Integer, kwargs...,
) where {G,T,NI}
    kk = Int(k)
    idx = Vector{Int}(undef, kk)
    dist = Vector{T}(undef, kk)
    n = k_nearest!(idx, dist, grid, I...; k = kk, kwargs...)
    return resize!(idx, n), resize!(dist, n)
end

# Every cell is within this of every other, so the widening always terminates.
@inline function _knn_radius_ceiling(grid::Grids.AbstractGrid{G,T}, mt) where {G<:Geometry.AbstractSphericalGeometry,T}
    return T(π) * T(Geometry.radius(Grids.grid_geometry(grid)))
end
@inline function _knn_radius_ceiling(grid::Grids.AbstractGrid{G,T}, mt) where {G<:Geometry.AbstractEllipsoidalGeometry,T}
    return T(π) * T(Geometry.semimajor_axis(Grids.grid_geometry(grid)))
end
# The diagonal of the coordinate extents, so every coordinate direction contributes: on a node set
# `size_tuple` counts nodes, and the coordinate count is what spans the domain.
@inline function _knn_radius_ceiling(grid::Grids.AbstractGrid{G,T}, mt) where {G,T}
    D = Grids.ncoordinates(grid)
    s = zero(T)
    for d in 1:D
        e = @inbounds T(mt.span[d])
        s += e * e
    end
    return sqrt(s) + one(T)
end

# How a cell is named — an index tuple or a single integer — is `Grids.cell_address`, so each of these
# three is two methods that serve every layout.

# The cell as a linear index, which is how every traversal reports one.
@inline _sweep_linear(grid::Grids.AbstractGrid, J) = _sweep_linear(grid, J, Grids.cell_address(grid))
@inline _sweep_linear(grid, J, ::Grids.CartesianCells) = _linidx(Grids.size_tuple(grid), J...)
@inline _sweep_linear(_grid, k::Integer, ::Grids.FlatCells) = Int(k)
