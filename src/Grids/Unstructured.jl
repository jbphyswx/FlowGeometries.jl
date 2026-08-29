# ---------------------------------------------------------------------------
# Cell incidence
# ---------------------------------------------------------------------------

"""
    CellMesh(cell_ptr, cell_nodes, node_ptr, node_cells)
    CellMesh(cell_ptr, cell_nodes, nnodes)

A node set's CELLS, and the nodes each one joins — both directions, each as CSR.

Node→node adjacency says which nodes are linked; it does not say which nodes bound a face. Only the
cells do, and they are what an area, a flux through an edge, or an interpolation inside a triangle is
defined on. A triangulation is what produces them, and the tessellation that computes a node set's
Voronoi areas has already built one: this is that mesh, kept rather than discarded.

- `cell_nodes[cell_ptr[c] : cell_ptr[c+1]-1]` are the nodes of cell `c`, in order around it
- `node_cells[node_ptr[i] : node_ptr[i+1]-1]` are the cells incident on node `i`

Both are `O(n)`: a triangulation of `n` nodes has about `2n` triangles and `6n` entries either way.
The second is the transpose of the first and the two-argument form builds it.

The cells are a triangulation or the caller's own. They are NOT a `k`-nearest graph: that is not
planar, not symmetric once truncated, and its "cells" do not tile anything, so every quantity defined
on a cell would be defined on a fiction.
"""
struct CellMesh{VI<:AbstractVector{<:Integer},VP<:AbstractVector{<:Integer}}
    cell_ptr::VP
    cell_nodes::VI
    node_ptr::VP
    node_cells::VI
end

function CellMesh(cell_ptr::AbstractVector{<:Integer}, cell_nodes::AbstractVector{<:Integer},
                  nnodes::Integer)
    nc = length(cell_ptr) - 1
    nc ≥ 0 || throw(ArgumentError("cell_ptr must have length ncells+1; got $(length(cell_ptr))"))
    n = Int(nnodes)
    I = eltype(cell_nodes)
    # Count, scan, place: the transpose of a CSR without materializing a pair per entry.
    node_ptr = similar(cell_ptr, n + 1)
    fill!(node_ptr, 0)
    @inbounds for k in eachindex(cell_nodes)
        v = Int(cell_nodes[k])
        1 ≤ v ≤ n || throw(ArgumentError("cell_nodes holds node $v, outside 1:$n"))
        node_ptr[v + 1] += 1
    end
    @inbounds node_ptr[1] = 1
    @inbounds for i in 1:n
        node_ptr[i + 1] += node_ptr[i]
    end
    node_cells = similar(cell_nodes, Int(node_ptr[end]) - 1)
    cursor = copy(node_ptr)
    @inbounds for c in 1:nc
        for k in cell_ptr[c]:(cell_ptr[c + 1] - 1)
            v = Int(cell_nodes[k])
            node_cells[cursor[v]] = I(c)
            cursor[v] += 1
        end
    end
    return CellMesh(cell_ptr, cell_nodes, node_ptr, node_cells)
end

"""
    ncells(mesh) -> Int

How many cells the mesh has.
"""
@inline ncells(m::CellMesh) = length(m.cell_ptr) - 1

"""
    cell_nodes(mesh, c) -> AbstractVector{<:Integer}

The nodes of cell `c`, in order around it, as a view into the mesh's own storage.
"""
@inline function cell_nodes(m::CellMesh, c::Integer)
    @boundscheck (1 ≤ c ≤ ncells(m)) || throw(BoundsError(m, c))
    @inbounds return view(m.cell_nodes, Int(m.cell_ptr[c]):(Int(m.cell_ptr[c + 1]) - 1))
end

"""
    node_cells(mesh, i) -> AbstractVector{<:Integer}

The cells incident on node `i`, as a view into the mesh's own storage.
"""
@inline function node_cells(m::CellMesh, i::Integer)
    @boundscheck (1 ≤ i ≤ length(m.node_ptr) - 1) || throw(BoundsError(m, i))
    @inbounds return view(m.node_cells, Int(m.node_ptr[i]):(Int(m.node_ptr[i + 1]) - 1))
end

Base.show(io::IO, m::CellMesh) =
    print(io, "CellMesh(", ncells(m), " cells, ", length(m.node_ptr) - 1, " nodes, ",
          length(m.cell_nodes), " incidences)")

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
    MH<:Union{Nothing,CellMesh},
} <: AbstractUnstructuredGrid{G, T}
    geometry::G
    coordinates::C     # node coordinate vector per direction (Nnodes)
    measure::VA        # control-volume size of each node (Nnodes)
    mask::B            # active mask (true = active/included) (Nnodes)
    neighbor_nbrs::VN  # flat neighbor-index array (CSR)
    neighbor_ptr::VP   # CSR offsets, length Nnodes+1
    topology::TP       # per-direction closure of the enclosing domain (singletons: no storage)
    period::NTuple{N,T}       # wrap length per direction; meaningless where Bounded
    mesh::MH           # the cells, where a triangulation produced them; see `CellMesh`
end

@inline _from_fields(
    ::Type{<:UnstructuredGrid},
    geometry::G, coordinates::C, measure::VA, mask::B, neighbor_nbrs::VN, neighbor_ptr::VP,
    topology::TP, period::NTuple{N,T}, mesh::MH,
) where {T,G<:Geometry.AbstractGeometry{T},N,C,VA,B,TP,VN,VP,MH} =
    UnstructuredGrid{T,G,N,C,VA,B,TP,VN,VP,MH}(geometry, coordinates, measure, mask, neighbor_nbrs,
                                               neighbor_ptr, topology, period, mesh)

"""
    cell_mesh(grid) -> CellMesh | Nothing

The grid's [`CellMesh`](@ref), or `nothing` where it has none.

A node set built by tessellation keeps the cells that tessellation produced; one given its adjacency
directly has only that adjacency, and no cells to speak of unless it is handed some.
"""
@inline cell_mesh(grid::UnstructuredGrid) = getfield(grid, :mesh)
@inline cell_mesh(::AbstractGrid) = nothing

"""
    has_cell_mesh(grid) -> Bool

Whether [`cell_mesh`](@ref) has cells to give — what the operators that need a cell dispatch on.
"""
@inline has_cell_mesh(grid) = cell_mesh(grid) !== nothing

@inline topology(grid::UnstructuredGrid) = getfield(grid, :topology)

# ---------------------------------------------------------------------------
# Spatial ordering
# ---------------------------------------------------------------------------

# Interleave the bits of `N` quantized coordinates, low bit first: the Morton (Z-curve) key. Points
# close in space share a long prefix, so sorting by it puts them close in memory.
@inline function _morton_key(u::NTuple{N,UInt64}, bits::Int) where {N}
    key = zero(UInt64)
    @inbounds for b in 0:(bits - 1), d in 1:N
        key |= ((u[d] >> b) & one(UInt64)) << (b * N + (d - 1))
    end
    return key
end

"""
    spatial_order(grid) -> Vector{Int}

A permutation of the grid's nodes along a Morton (Z-order) curve: `perm[k]` is the node that belongs
in position `k`.

Scattered nodes usually arrive in whatever order they were generated, and a neighbour traversal then
jumps across the whole array per edge. Along a space-filling curve, neighbours in space are neighbours
in memory, so `neighbor_nbrs[ptr[i]:ptr[i+1]-1]` reads mostly-adjacent addresses.

The order is a property of the POINTS, not of the order they came in: the same set, differently
shuffled, sorts to the same sequence. So this repairs an incoherent input order and does not improve
on one that is already coherent.

This RETURNS the permutation rather than applying it, and construction does not apply it either. The
node index is the caller's handle on their own data: a grid that quietly renumbered would leave every
field they hold pointing at the wrong node. Apply it with [`reorder`](@ref) and permute those fields
the same way.
"""
function spatial_order(grid::UnstructuredGrid{T,G,N}) where {T,G,N}
    n = length(mask(grid))
    n < 2 && return collect(1:n)
    # Keyed on the EMBEDDED position, not the raw coordinates: on a sphere longitude wraps, and two
    # nodes either side of the seam are neighbours in space whose `λ` differ by a whole turn. The
    # ambient position has no seam, so the curve does not cut there.
    pts = [embed_point(grid, _raw_coords(grid, i)) for i in 1:n]
    D = length(first(pts))
    bits = min(64 ÷ D, 21)
    span = UInt64(1) << bits - one(UInt64)
    lo = ntuple(d -> minimum(p[d] for p in pts), D)
    hi = ntuple(d -> maximum(p[d] for p in pts), D)
    # A direction of zero extent quantizes to one bucket rather than dividing by nothing.
    scale = ntuple(d -> hi[d] > lo[d] ? T(span) / T(hi[d] - lo[d]) : zero(T), D)
    keys = Vector{UInt64}(undef, n)
    @inbounds for i in 1:n
        p = pts[i]
        u = ntuple(d -> UInt64(round(clamp((T(p[d]) - T(lo[d])) * scale[d], zero(T), T(span)))), D)
        keys[i] = _morton_key(u, bits)
    end
    return sortperm(keys)
end

"""
    reorder(grid, perm) -> UnstructuredGrid

`grid` with its nodes in the order `perm` gives — coordinates, measure and mask permuted, and the
adjacency and any [`CellMesh`](@ref) renumbered to match, so every index the grid reports is an index
into the new order.

Pair it with [`spatial_order`](@ref), and permute your own fields by the same `perm`: `f[perm]` is
that field on the reordered grid.
"""
function reorder(grid::UnstructuredGrid{T,G,N}, perm::AbstractVector{<:Integer}) where {T,G,N}
    n = length(mask(grid))
    length(perm) == n || throw(DimensionMismatch(
        "the permutation has $(length(perm)) entries for a grid of $n nodes",
    ))
    inv = zeros(Int, n)
    @inbounds for k in 1:n
        p = Int(perm[k])
        1 ≤ p ≤ n || throw(ArgumentError("the permutation names node $p, outside 1:$n"))
        inv[p] = k
    end
    all(>(0), inv) || throw(ArgumentError("the permutation names a node twice and another not at all"))

    c = coordinates(grid)
    newc = ntuple(d -> c[d][perm], Val(N))
    newm = measure(grid)[perm]
    msk = mask(grid)
    newmask = msk isa AllActive ? msk : msk[perm]

    optr, onbrs = neighbor_ptr(grid), neighbor_nbrs(grid)
    I = eltype(onbrs)
    newptr = similar(optr, n + 1)
    newnbrs = similar(onbrs, length(onbrs))
    @inbounds newptr[1] = 1
    @inbounds for k in 1:n
        o = Int(perm[k])
        lo, hi = Int(optr[o]), Int(optr[o + 1]) - 1
        newptr[k + 1] = newptr[k] + (hi - lo + 1)
        w = Int(newptr[k]) - 1
        for t in lo:hi
            newnbrs[w + t - lo + 1] = I(inv[Int(onbrs[t])])
        end
    end

    # The cells keep their own numbering; only the nodes they name change.
    mh = cell_mesh(grid)
    newmesh = mh === nothing ? nothing :
        CellMesh(mh.cell_ptr, map(v -> eltype(mh.cell_nodes)(inv[Int(v)]), mh.cell_nodes), n)

    return UnstructuredGrid(grid_geometry(grid), newc, newm, newmask, newnbrs, newptr;
                            periodic = topology(grid),
                            period = ntuple(d -> period(grid, d), Val(N)), mesh = newmesh)
end

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
    periodic = nothing, period = nothing, mesh::Union{Nothing,CellMesh} = nothing,
) where {N, T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    c = ntuple(d -> _to_axis(T, coords[d]), N)
    n = length(c[1])
    all(v -> length(v) == n, c) || throw(ArgumentError(
        "coordinate vectors must have the same length; got $(map(length, c))",
    ))
    length(mask) == n || throw(ArgumentError("coordinates and mask must have the same length"))
    length(measure) == n || throw(ArgumentError("coordinates and measure must have the same length"))
    # No CSR pair given: every node reports zero neighbours, which is `ptr` all-ones — and a constant
    # vector says that in one number rather than `n + 1` copies of it. `neighbors` reads it through the
    # same `ptr[i]:ptr[i+1]-1` slice and gets an empty range.
    p = ptr === nothing ? Axes.ConstantVector(1, n + 1) : ptr
    nb = ptr === nothing ? Int[] : nbrs
    length(p) == n + 1 || throw(ArgumentError(
        "neighbor_ptr must have length Nnodes+1 = $(n + 1); got $(length(p))",
    ))
    per, prd = _node_periodicity(geometry, Val(N), periodic, period)
    m = _to_axis(T, measure)
    mesh === nothing || length(mesh.node_ptr) == n + 1 || throw(ArgumentError(
        "the mesh is over $(length(mesh.node_ptr) - 1) nodes and the grid has $n",
    ))
    return UnstructuredGrid{
        T, G, N, typeof(c), typeof(m), typeof(mask), typeof(per), typeof(nb), typeof(p), typeof(mesh),
    }(geometry, c, m, mask, nb, p, per, prd, mesh)
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
