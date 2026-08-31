# ---------------------------------------------------------------------------
# Queries seeded by a point
# ---------------------------------------------------------------------------
#
# Observation data does not arrive on the grid: a float, a station, a ship track is a coordinate. The
# queries above are seeded by a cell index, and these take a coordinate, on every layout — a curvilinear
# or node grid included, where there are no axes to compose `locate` along.
#
# The gate is the same `distance ≤ r` under the geometry's own metric; only the enumeration differs,
# a point having no cell whose window to take.

"""
    fold_at(f, init, grid, p; ball, active_only=true, topology, scratch) -> acc

Fold `acc = f(acc, J, d)` over every cell `J` within `ball` of the **point** `p`, `d` being its distance.
`p` is a coordinate in the grid's own coordinates, written any way a point is accepted elsewhere;
[`fold_within`](@ref) is this at a cell centre.

There is no cell to exclude, so unlike the cell-seeded form every cell within `ball` is visited.
"""
function fold_at end

# A window centred on the point's own cell, widened by how far the point sits from that cell's centre —
# so the window still covers every cell within `ball` of the point, and stays `O(1)` per direction.
function fold_at(
    f::F, init, grid::Grids.StructuredGrid{T, G,N}, p::NTuple{N,Real};
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
) where {F,G,T,N}
    geo = Grids.grid_geometry(grid)
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    p0 = ntuple(d -> T(p[d]), Val(N))
    # A point outside a bounded direction has no cell; the nearest one still centres a window that
    # covers the ball, since the widening below accounts for how far the point sits from its centre.
    raw = Grids.locate(grid, p0)
    I0 = ntuple(d -> clamp(raw[d], 1, sz[d]), Val(N))
    c0 = Grids._raw_coords(grid, I0...)
    prd = map(T, _wrap_lengths(grid, Val(N)))
    d0 = Geometry.distance(geo, p0, _min_image(p0, c0, prd))
    w = _ball_window(grid, I0, r + d0, topology, NearestImage())
    msk = Grids.mask(grid)
    acc = init
    rng = ntuple(d -> _delta_range(w[d], I0[d], sz[d], per[d], NearestImage()), Val(N))
    @inbounds for ci in CartesianIndices(rng)
        δ = Tuple(ci)
        J = ntuple(d -> per[d] ? mod1(I0[d] + δ[d], sz[d]) : I0[d] + δ[d], Val(N))
        any(d -> J[d] < 1 || J[d] > sz[d], 1:N) && continue
        active_only && !msk[J...] && continue
        q = _min_image(p0, Grids._raw_coords(grid, J...), prd)
        dist = Geometry.distance(geo, p0, q)
        dist ≤ r || continue
        acc = f(acc, J, dist)
    end
    return acc
end

# Every layout whose candidates come from an index answers a point query the same way: the index
# reports a superset and the exact distance gate below decides. What differs is how a cell is named,
# which is [`Grids.cell_address`](@ref) — a curvilinear mesh and a panel layout by index tuple, a node
# set and a pixelization by one integer.
function fold_at(
    f::F, init, grid::Grids.AbstractGrid{G,T}, p::NTuple{D,Real};
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
) where {F,G,T,D}
    return _fold_at_scattered(f, init, grid, ntuple(d -> T(p[d]), Val(D)), ball, active_only,
                              topology, scratch, _linear_shape(grid))
end

# The shape a linear candidate index is resolved against, or `nothing` where a cell already is one.
@inline _linear_shape(grid::Grids.AbstractGrid) = _linear_shape(grid, Grids.cell_address(grid))
@inline _linear_shape(grid, ::Grids.CartesianCells) = Grids.size_tuple(grid)
@inline _linear_shape(_grid, ::Grids.FlatCells) = nothing

function _fold_at_scattered(
    f::F, init, grid::Grids.AbstractGrid{G,T}, p0::NTuple{D,T}, ball, active_only::Bool,
    mt::MetricTopology, scratch, sz,
) where {F,G,T,D}
    _check_index_covers(mt.index, active_only)
    geo = Grids.grid_geometry(grid)
    r = T(_ball_radius(ball))
    prd = map(T, _wrap_lengths(grid, Val(D)))
    msk = Grids.mask(grid)
    return _fold_candidates_at(init, mt, grid, p0, r, scratch) do acc, lin
        J = sz === nothing ? lin : Tuple(@inbounds CartesianIndices(sz)[lin])
        @inbounds active_only && !(sz === nothing ? msk[lin] : msk[J...]) && return acc
        q = _min_image(p0, sz === nothing ? Grids._raw_coords(grid, lin) :
                           Grids._raw_coords(grid, J...), prd)
        dist = Geometry.distance(geo, p0, q)
        dist ≤ r || return acc
        return f(acc, J, dist)
    end
end

# Candidates around a point. Only a cell list can answer this without a cell to look up, so anything
# else falls back to every cell — correct, and the reason `cell_list` is worth passing here.
@inline _fold_candidates_at(f::F, acc, mt::MetricTopology, grid, p0, r, scratch) where {F} =
    _fold_at_index(f, acc, mt.index, grid, p0, r, scratch)

@inline function _fold_at_index(f::F, acc, ::Nothing, grid, _p0, _r, _scratch) where {F}
    @inbounds for k in Base.OneTo(length(Grids.mask(grid)))
        acc = f(acc, k)
    end
    return acc
end

# Anything else goes through the index's own point query, which raises where it has none. A caller who
# passed an index gets a range query or an error, never a silent scan of every cell.
@inline _fold_at_index(f::F, acc, index, grid, p0, r, scratch) where {F} =
    Grids.fold_candidates_at(f, acc, index, Grids.embed_point(grid, p0), r, scratch)

"""
    neighbors_within(grid, p; ball, …) -> Vector{Int}
    nneighbors_within(grid, p; ball, …) -> Int

Cells within `ball` of the **point** `p`, and how many. `p` is a coordinate, where the cell-seeded
methods take index arguments; it may be a `Tuple`, `NamedTuple`, `AbstractVector` or `SVector`, as
everywhere else a point is accepted.
"""
function neighbors_within(
    grid::Grids.AbstractGrid, p::NTuple{D,Real}; ball, active_only::Bool = true,
    topology = MetricTopology(grid), scratch = nothing,
) where {D}
    sz = grid isa Grids.UnstructuredGrid ? nothing : Grids.size_tuple(grid)
    return fold_at(Int[], grid, p; ball = ball, active_only = active_only,
                   topology = topology, scratch = scratch) do v, J, _
        push!(v, sz === nothing ? J : _linidx(sz, J...))
        return v
    end
end

function nneighbors_within(
    grid::Grids.AbstractGrid, p::NTuple{D,Real}; ball, active_only::Bool = true,
    topology = MetricTopology(grid), scratch = nothing,
) where {D}
    return fold_at(0, grid, p; ball = ball, active_only = active_only,
                   topology = topology, scratch = scratch) do n, _J, _d
        return n + 1
    end
end

"""
    k_nearest(grid, p; k, …) -> (idx, dist)

The `k` cells nearest the **point** `p`, nearest first, exact under the geometry's own metric. The
cell-seeded [`k_nearest`](@ref) is this at a cell centre, minus the cell itself.
"""
function k_nearest(
    grid::Grids.AbstractGrid{G,T}, p::NTuple{D,Real}; k::Integer, active_only::Bool = true,
    topology = MetricTopology(grid), scratch = nothing,
) where {G,T,D}
    kk = Int(k)
    idx = Vector{Int}(undef, kk)
    dist = Vector{T}(undef, kk)
    n = k_nearest!(idx, dist, grid, p; k = kk, active_only = active_only,
                   topology = topology, scratch = scratch)
    return resize!(idx, n), resize!(dist, n)
end

function k_nearest!(
    idx::AbstractVector{<:Integer}, dist::AbstractVector, grid::Grids.AbstractGrid{G,T},
    p::NTuple{D,Real}; k::Integer, active_only::Bool = true,
    topology = MetricTopology(grid), scratch = nothing,
) where {G,T,D}
    kk = Int(k)
    kk ≥ 0 || throw(ArgumentError("k must be non-negative, got $k"))
    (kk == 0 || isempty(idx)) && return 0
    sz = grid isa Grids.UnstructuredGrid ? nothing : Grids.size_tuple(grid)
    r = _knn_seed_radius_at(grid, p, kk, topology)
    rmax = _knn_radius_ceiling(grid, topology)
    n = 0
    while true
        n = fold_at(0, grid, p; ball = r, active_only = active_only,
                    topology = topology, scratch = scratch) do m, J, d
            return _heap_offer!(dist, idx, m, kk, d, sz === nothing ? J : _linidx(sz, J...))
        end
        (n ≥ kk || r ≥ rmax) && break
        r = min(r * 2, rmax)
    end
    _heap_sort!(dist, idx, n)
    return n
end

# Off the rectilinear grids there are no axes to bracket the point along, so the containing cell is the
# nearest centre — exactly right for a node set, whose cells are its nodes' Voronoi regions. The radius
# widens as `k_nearest` does, and tracking one best needs no heap and no buffer.
function Grids.locate(
    grid::Grids.AbstractGrid{G,T}, p::NTuple{D,Real};
    active_only::Bool = false, topology = MetricTopology(grid), scratch = nothing,
) where {G,T,D}
    sz = _linear_shape(grid)
    r = _knn_seed_radius_at(grid, p, 1, topology)
    rmax = _knn_radius_ceiling(grid, topology)
    while true
        best = fold_at((0, T(Inf)), grid, p; ball = r, active_only = active_only,
                       topology = topology, scratch = scratch) do acc, J, d
            lin = sz === nothing ? J : _linidx(sz, J...)
            # Ties go to the lower index, so an equidistant pair resolves the same way every call.
            return d < acc[2] || (d == acc[2] && lin < acc[1]) ? (lin, d) : acc
        end
        best[1] != 0 && return sz === nothing ? best[1] : Tuple(CartesianIndices(sz)[best[1]])
        r ≥ rmax && return sz === nothing ? 0 : ntuple(_ -> 0, Val(D))
        r = min(r * 2, rmax)
    end
end

# A point may be written any of the ways the rest of the package accepts one. Normalizing at these entry
# points keeps one tuple method per kernel, and the cell-seeded forms take integers, so nothing accepted
# here can be mistaken for one. `Geometry.PointLike` excludes `Tuple`, which the tuple methods take.

for fn in (:neighbors_within, :nneighbors_within, :k_nearest)
    @eval @inline $fn(grid::Grids.AbstractGrid, p::Geometry.PointLike; kwargs...) =
        $fn(grid, Geometry.as_ntuple(p); kwargs...)
end

@inline k_nearest!(idx::AbstractVector{<:Integer}, dist::AbstractVector,
                   grid::Grids.AbstractGrid, p::Geometry.PointLike; kwargs...) =
    k_nearest!(idx, dist, grid, Geometry.as_ntuple(p); kwargs...)

@inline fold_at(f::F, init, grid::Grids.AbstractGrid, p::Geometry.PointLike; kwargs...) where {F} =
    fold_at(f, init, grid, Geometry.as_ntuple(p); kwargs...)

@inline Grids.locate(grid::Grids.AbstractGrid, p::Geometry.PointLike; kwargs...) =
    Grids.locate(grid, Geometry.as_ntuple(p); kwargs...)

# The cell the point falls in gives the local cell size to start from, exactly as the cell-seeded form
# uses its own cell's measure.
@inline function _knn_seed_radius_at(grid::Grids.StructuredGrid{T, G,N}, p, k::Int, mt) where {G,T,N}
    sz = Grids.size_tuple(grid)
    raw = Grids.locate(grid, ntuple(d -> T(p[d]), Val(N)))
    return _knn_seed_radius(grid, ntuple(d -> clamp(raw[d], 1, sz[d]), Val(N)), k)
end

# Elsewhere there are no axes to locate along, so the scale comes from the mean cell. Only the starting
# radius depends on it; the doubling reaches the rest.
@inline function _knn_seed_radius_at(grid::Grids.AbstractGrid{G,T}, p, k::Int, mt) where {G,T}
    D = Grids.ncoordinates(grid)
    ncell = length(Grids.mask(grid))
    vol = one(T)
    # The span comes from the topology, which reduced it once. Off a rectilinear grid the coordinates
    # are fields with no order, so asking the grid would scan them on every query.
    for d in 1:D
        e = @inbounds T(mt.span[d])
        e > 0 && (vol *= e)
    end
    cell = ncell > 0 ? (vol / ncell)^(one(T) / D) : one(T)
    return T(1.5) * cell * T(max(k, 1))^(one(T) / D)
end
