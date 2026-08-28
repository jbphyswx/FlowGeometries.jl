# ---------------------------------------------------------------------------
# Unstructured Grid
# ---------------------------------------------------------------------------

"""
    UnstructuredGrid{T, G, V, VA, B, VI}

Unstructured mesh (e.g. radial data, finite volume, or triangular mesh) where coords are 1D vectors.

# Type parameters
- `T`: coordinate float type. `G<:AbstractGeometry{T}` is tied to it (a mismatched-eltype geometry is
  a type error, not a silent promotion) — hence `T` precedes `G` (Julia forbids the forward reference
  `G<:AbstractGeometry{T}, T` needed to keep the `{G,T}` order), matching the same convention
  [`CurvilinearGrid`](@ref) uses.
- `C`: tuple type of the per-direction node-coordinate vectors (a node set's own coordinate vectors
  are legitimately almost always the same concrete type).
- `VA`: vector type of the derived `measure` field — independent of `C`, since it is frequently a
  computed field (Voronoi tessellation) with no reason to match the coordinate vectors' storage type.
- `B`: mask storage type.
- `VN`/`VP`: CSR neighbor-list and offset storage types, independent of each other. Their element
  type is a free `Integer`, so a large mesh can carry `Int32` indices (half the memory and bandwidth
  of `Int64`, and the width GPU kernels want) without needing a separate grid type.

Neighbor adjacency is stored CSR-style (flat `neighbor_nbrs` + `neighbor_ptr` offsets, node `t` owns
`neighbor_ptr[t]:neighbor_ptr[t+1]-1`) rather than as a vector of per-node vectors — the data is
immutable after construction, so there's no reason to pay for `Nnodes` separately-heap-allocated
`Vector`s (cache-unfriendly pointer-chasing, one allocation per node) when one contiguous block (two
allocations total) holds the same information.
"""
struct UnstructuredGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T},
    N,
    C<:NTuple{N,AbstractVector{T}},
    VA<:AbstractVector{T},
    B<:AbstractVector{Bool},
    TP<:NTuple{N,AbstractTopology},
    VN<:AbstractVector{<:Integer},
    VP<:AbstractVector{<:Integer},
} <: AbstractUnstructuredGrid{G, T}
    geometry::G
    coordinates::C     # node coordinate vector per direction (Nnodes)
    measure::VA        # control-volume size of each node (Nnodes)
    mask::B            # active mask (true = active/included) (Nnodes)
    neighbor_nbrs::VN  # flat neighbor-index array (CSR)
    neighbor_ptr::VP   # CSR offsets, length Nnodes+1
    topology::TP       # per-direction closure of the enclosing domain (singletons: no storage)
    period::NTuple{N,T}       # wrap length per direction; meaningless where Bounded
    stats::NTuple{N,AxisStats{T}}   # span per direction; gaps are undefined for scattered nodes
end

@inline _from_fields(
    geometry::G, coordinates::C, measure::VA, mask::B, neighbor_nbrs::VN, neighbor_ptr::VP,
    topology::TP, period::NTuple{N,T}, stats::NTuple{N,AxisStats{T}},
) where {T,G<:Geometry.AbstractGeometry{T},N,C,VA,B,TP,VN,VP} =
    UnstructuredGrid{T,G,N,C,VA,B,TP,VN,VP}(geometry, coordinates, measure, mask, neighbor_nbrs,
                                            neighbor_ptr, topology, period, stats)

@inline topology(grid::UnstructuredGrid) = getfield(grid, :topology)

"""
    UnstructuredGrid(geometry, coords::Tuple, measure, mask[, neighbor_nbrs, neighbor_ptr]; periodic, period)
    UnstructuredGrid(geometry, x, y, measure, mask[, neighbor_nbrs, neighbor_ptr]; periodic, period)

Build a node grid in **any** number of directions from one coordinate vector per direction and CSR
adjacency. Coordinates come as a tuple; the two-direction case may pass `x, y` positionally.

Omitting the CSR pair gives a grid with no adjacency — every node reports zero neighbours, which is
enough for scattered-point spectral methods that never query it. Real-space neighbourhood operations
need adjacency: build it (e.g. through the k-d-tree constructor below) and pass it in, or query by
distance with `Connectivity.neighbors_within`, which reads coordinates rather than edges.

`periodic` declares that the enclosing domain wraps in a direction, and `period` gives the wrap
length there. A scattered point set carries no axis to infer this from, so both are explicit —
except on a sphere, where longitude wraps at 2π by construction and is the default. See
[`isperiodic`](@ref) and [`period`](@ref).
"""
function UnstructuredGrid(
    geometry::G, coords::NTuple{N,AbstractVector}, measure::AbstractVector,
    mask::AbstractVector{Bool} = AllActive((length(first(coords)),)), nbrs = nothing, ptr = nothing;
    periodic = nothing, period = nothing,
) where {N, T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    c = ntuple(d -> _to_axis(T, coords[d]), N)
    n = length(c[1])
    all(v -> length(v) == n, c) || throw(ArgumentError(
        "coordinate vectors must have the same length; got $(map(length, c))",
    ))
    length(mask) == n || throw(ArgumentError("coordinates and mask must have the same length"))
    length(measure) == n || throw(ArgumentError("coordinates and measure must have the same length"))
    # No CSR pair given: every node reports zero neighbours, which `ptr` all-ones expresses.
    p = ptr === nothing ? ones(Int, n + 1) : ptr
    nb = ptr === nothing ? Int[] : nbrs
    length(p) == n + 1 || throw(ArgumentError(
        "neighbor_ptr must have length Nnodes+1 = $(n + 1); got $(length(p))",
    ))
    per, prd = _node_periodicity(geometry, Val(N), periodic, period)
    m = _to_axis(T, measure)
    return UnstructuredGrid{
        T, G, N, typeof(c), typeof(m), typeof(mask), typeof(per), typeof(nb), typeof(p),
    }(geometry, c, m, mask, nb, p, per, prd, ntuple(d -> _axis_stats(c[d]), Val(N)))
end

# Two-direction convenience forms. Node coordinates are all `AbstractVector`s, so — unlike a
# `CurvilinearGrid`, where `ndims(mask)` counts them — nothing in the types says how many of a run of
# vectors are coordinates: `(x, y, mask)` and `(x, measure, mask)` are the same call. Hence the tuple
# above for the general case, and explicit methods for the two-direction one.
UnstructuredGrid(
    geometry::Geometry.AbstractGeometry, x::AbstractVector, y::AbstractVector,
    measure::AbstractVector, mask::AbstractVector{Bool},
    nbrs::AbstractVector{<:Integer}, ptr::AbstractVector{<:Integer}; kwargs...,
) = UnstructuredGrid(geometry, (x, y), measure, mask, nbrs, ptr; kwargs...)

UnstructuredGrid(
    geometry::Geometry.AbstractGeometry, x::AbstractVector, y::AbstractVector,
    measure::AbstractVector, mask::AbstractVector{Bool}; kwargs...,
) = UnstructuredGrid(geometry, (x, y), measure, mask; kwargs...)

# Longitude on a sphere wraps at 2π whatever the point set looks like; a Cartesian box has no
# intrinsic period, so wrapping there is opt-in and the length must be given.
function _node_periodicity(
    ::Union{Geometry.AbstractSphericalGeometry{T},Geometry.AbstractEllipsoidalGeometry{T}},
    ::Val{N}, periodic, period,
) where {N, T<:AbstractFloat}
    per = _as_topology(Val(N), periodic === nothing ?
        ntuple(d -> d == 1, Val(N)) : periodic)
    prd = period === nothing ? ntuple(d -> d == 1 ? T(2π) : zero(T), Val(N)) :
        _node_period_tuple(Val(N), T, period, per)
    return per, prd
end

function _node_periodicity(
    ::Geometry.AbstractCartesianGeometry{T}, ::Val{N}, periodic, period,
) where {N, T<:AbstractFloat}
    per = _as_topology(Val(N), periodic === nothing ? ntuple(_ -> false, Val(N)) : periodic)
    any(_is_periodic, per) || return per, ntuple(_ -> zero(T), Val(N))
    period === nothing && throw(ArgumentError(
        "a periodic Cartesian node grid needs an explicit `period` (the wrap length per direction): " *
        "scattered points carry no axis to infer it from",
    ))
    return per, _node_period_tuple(Val(N), T, period, per)
end

function _node_period_tuple(::Val{N}, ::Type{T}, period, per) where {N,T}
    prd = period isa Real ? ntuple(d -> d == 1 ? T(period) : zero(T), Val(N)) : NTuple{N,T}(period)
    all(d -> !_is_periodic(per[d]) || prd[d] > 0, 1:N) || throw(ArgumentError(
        "period must be positive in every periodic direction; got $prd for topology=$per",
    ))
    return prd
end

"""
    period(grid, d) -> T

Wrap length of coordinate direction `d`, meaningful only where [`isperiodic`](@ref) holds.
"""
@inline period(grid::UnstructuredGrid, d::Integer) =
    @inbounds getfield(grid, :period)[_checked_direction(getfield(grid, :period), d)]


# ---------------------------------------------------------------------------
# Unstructured grid construction: k-d-tree adjacency + (optional) Voronoi areas
# ---------------------------------------------------------------------------
#
# Two extension hook points (fallbacks that throw until a consumer package loads the relevant
# weakdep and overrides these methods): a k-d-tree neighbor query (`NearestNeighbors.jl`) and a
# per-node Voronoi-cell area (`DelaunayTriangulation.jl` for Cartesian, `Quickhull.jl` for spherical,
# dispatched on the geometry type since each needs a different tessellation library).

"""
    _build_kdtree_neighbors(geometry, coords::Tuple; k=6, radius=nothing) -> (nbrs, ptr)

Extension hook: build CSR neighbor adjacency via a k-d tree. Overridden by
a consumer NearestNeighbors extension (load `using NearestNeighbors`). `radius`, if given,
switches to an all-neighbors-within-`radius` query (mutually exclusive with `k`); `radius` is in the
grid's physical distance units (`Geometry.distance` — meters for `SphericalGeometry`, geometry's own
units for `CartesianGeometry`), NOT a raw chord/angle.
"""
function _build_kdtree_neighbors(
    geometry::Geometry.AbstractGeometry, coords::NTuple{D,AbstractVector}; _...,
) where {D}
    throw(ArgumentError(
        "k-d-tree neighbor construction requires NearestNeighbors.jl — run `using NearestNeighbors` " *
        "(or build adjacency explicitly and pass it to `UnstructuredGrid` alongside the coordinates).",
    ))
end

# Every architecture caches its per-direction reductions, so the span accessors are reads rather than
# `extrema` over the coordinates. They sit under the search-radius bound a query evaluates, which is why
# it matters that they are `O(1)` and not merely correct.
const _StatGrid = Union{StructuredGrid,CurvilinearGrid,UnstructuredGrid}

@inline axis_stats(grid::Union{CurvilinearGrid,UnstructuredGrid}) = getfield(grid, :stats)
@inline axis_stats(grid::Union{CurvilinearGrid,UnstructuredGrid}, d::Integer) =
    @inbounds axis_stats(grid)[d]

@inline bounds(grid::_StatGrid, d::Integer) =
    (st = axis_stats(grid, d); (st.min_value, st.max_value))
@inline function extent(grid::_StatGrid, d::Integer)
    st = axis_stats(grid, d)
    return st.max_value - st.min_value
end
@inline origin(grid::_StatGrid, d::Integer) = axis_stats(grid, d).first_value
