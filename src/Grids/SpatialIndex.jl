# ---------------------------------------------------------------------------
# Cell-list index
# ---------------------------------------------------------------------------

"""
    CellListIndex

A uniform-bin spatial index over the embedded cell centres: points are bucketed by which bin of side `h`
they fall in, and a query visits the bins its ball can reach.

Three properties distinguish it from a tree, and all three are why it is the one that runs on a device:

  * it is **arrays only** — bin offsets, point ids and the bin each sits in — so `Adapt` moves it like
    any other field, and it holds no copy of the coordinates, which stay on the grid;
  * every point lands in **exactly one** bin, because periodicity wraps the bin coordinate rather than
    replicating the point, so a query emits each cell once and needs no candidate buffer to deduplicate
    into. That is what lets [`fold_candidates`](@ref) be a fold rather than a list;
  * bins are hashed into `O(n)` buckets, so the memory does not depend on `h`. A sphere binned at
    100 km would otherwise need `(2R/h)³ ≈ 2×10⁶` mostly empty cells, and far more as `h` shrinks.

Build it for the radius you intend to query at: `h` is that radius, so a query touches `3ᴰ` bins. A much
larger radius still works and costs `(2⌈r/h⌉+1)ᴰ` bins.
"""
struct CellListIndex{D,T<:AbstractFloat,E<:AbstractEmbedding,VI<:AbstractVector{Int},
                     VB<:AbstractVector{NTuple{D,Int}}}
    n::Int                  # cells indexed, one entry each, never replicated
    lo::NTuple{D,T}         # bin-lattice origin
    h::NTuple{D,T}          # bin width per direction
    nbins::NTuple{D,Int}    # per-direction bin count where the direction wraps; 0 otherwise
    wrap::NTuple{D,Bool}
    starts::VI              # nbucket + 1, CSR over buckets
    items::VI               # point ids, grouped by bucket
    bins::VB                # each item's bin, in the same order — see `fold_candidates_at`
    embedding::E
    active_only::Bool       # masked cells were left out; see `cell_list`
end

@inline _nbuckets(ix::CellListIndex) = length(ix.starts) - 1

"""
    embedded_at(grid, cell) -> NTuple{D,T}

A cell's centre in the space an index searches, from the grid rather than from a stored copy: `O(1)`
arithmetic where the coordinates are a formula, and a read where they are data.
"""
@inline embedded_at(grid::AbstractGrid, cell) = embed_point(grid, _cell_coords(grid, cell))

"""
    cells(grid)

What a traversal of every cell iterates, and `cell_at` turns one of its elements into a cell. Together
they are how a layout is walked without knowing how it names a cell — see [`cell_address`](@ref).
"""
@inline cells(grid::AbstractGrid) = cells(grid, cell_address(grid))
@inline cells(grid, ::CartesianCells) = CartesianIndices(size_tuple(grid))
@inline cells(grid, ::FlatCells) = Base.OneTo(length(mask(grid)))

@inline cell_at(grid::AbstractGrid, c) = cell_at(grid, c, cell_address(grid))
@inline cell_at(_grid, ci::CartesianIndex, ::CartesianCells) = Tuple(ci)
@inline cell_at(_grid, k::Integer, ::FlatCells) = Int(k)

@inline first_cell(grid::AbstractGrid) = cell_at(grid, first(cells(grid)))

"""
    _linear_of(grid, cell) -> Int

The linear index a cell reports as, and the inverse of [`_cell_from_linear`](@ref).
"""
@inline _linear_of(grid::AbstractGrid, cell) = _linear_of(grid, cell, cell_address(grid))
@inline _linear_of(grid, I::Tuple, ::CartesianCells) =
    @inbounds LinearIndices(size_tuple(grid))[I...]
@inline _linear_of(_grid, k::Integer, ::FlatCells) = Int(k)

# Bins in a query window, saturating: a radius far wider than the bin side gives a per-direction reach
# whose plain product overflows `Int` and would wrap to a small — or negative — number.
@inline function _window_bins(reach::NTuple{D,Int}) where {D}
    m = 1
    @inbounds for d in 1:D
        v = reach[d]
        v ≤ 0 && return 0
        v > typemax(Int) ÷ m && return typemax(Int)
        m *= v
    end
    return m
end

# Bin coordinate of a point, wrapped where the direction does. A wrapping direction's width divides the
# period exactly, so `mod` lands on a real lattice rather than aliasing a partial bin onto bin zero.
@inline function _bin_of(ix::CellListIndex{D,T}, x::NTuple{D,T}) where {D,T}
    return ntuple(Val(D)) do d
        @inbounds b = Base.unsafe_trunc(Int, floor((x[d] - ix.lo[d]) / ix.h[d]))
        @inbounds ix.wrap[d] ? mod(b, ix.nbins[d]) : b
    end
end

# A cheap integer mix. Only the bucket assignment depends on it: a collision costs a few extra items to
# skip, never a wrong answer, because the bin coordinate is compared before a point is emitted.
@inline function _bin_hash(b::NTuple{D,Int}, nbucket::Int) where {D}
    h = UInt(0x9e3779b97f4a7c15)
    @inbounds for d in 1:D
        h = (h ⊻ (reinterpret(UInt, b[d] * 0x27220a95) + 0x165667b19e3779f9 + (h << 6) + (h >> 2)))
    end
    return Int(h % UInt(nbucket)) + 1
end

"""
    cell_list(grid; ball, active_only = false) -> CellListIndex

Build a [`CellListIndex`](@ref) over `grid`'s cell centres, binned at side `ball` — the radius you mean
to query at. Needs no external package.

`active_only = true` leaves masked cells out, so a mostly-masked grid — a basin, a catchment — is
indexed at the size of its active region rather than of its bounding box. It is not the default because
such an index answers only queries at that same policy: it holds no masked cell, so a query asking to
see one raises rather than returning a short answer. The default covers every cell and therefore serves
both policies, an `active_only = true` query filtering what it is handed.

Where the policy is known — a sweep, which passes its own — narrowing is worth asking for.
"""
function cell_list(grid::AbstractGrid{G,T}; ball::Real, active_only::Bool = false) where {G,T}
    h = T(ball)
    h > 0 || throw(ArgumentError("the bin side must be positive, got $ball"))
    embedding = embedding_of(grid)
    hemb = T(embedded_radius(embedding, h))
    hemb > 0 || throw(ArgumentError("radius $ball is degenerate in this embedding"))
    # A function barrier on the dimension: it is a runtime value here, so building the index type from
    # it inline leaves the whole construction loop dynamically dispatched — measured at 34 MiB and
    # 196 ms for 65k points, against 3 ms once the dimension is a type.
    D = length(embedded_at(grid, first_cell(grid)))
    return _build_cell_list(grid, embedding, hemb, active_only, Val(D))
end

# Streamed: a cell's position is read from the grid three times rather than materialized once into a
# `D × n` matrix. That is what keeps the index `O(n)` integers on a layout whose coordinates are a
# formula, where holding them would reintroduce exactly the array the layout exists to avoid — and on a
# node set it is a read of vectors that already exist.
function _build_cell_list(
    grid, embedding::E, hemb::T, active_only::Bool, ::Val{D},
) where {T,E<:AbstractEmbedding,D}
    wrap, nbins, lo, hd = _cell_lattice(grid, embedding, hemb, Val(D))
    everything = !active_only || mask(grid) isa AllActive
    n = everything ? length(cells(grid)) : count(mask(grid))

    @inline function binof(c)
        x = map(T, embedded_at(grid, cell_at(grid, c)))
        return ntuple(Val(D)) do d
            @inbounds b = Base.unsafe_trunc(Int, floor((x[d] - lo[d]) / hd[d]))
            @inbounds wrap[d] ? mod(b, nbins[d]) : b
        end
    end

    nbucket = max(1, nextpow(2, max(n, 1)))
    counts = zeros(Int, nbucket + 1)
    cellbin = Vector{NTuple{D,Int}}(undef, n)
    ids = Vector{Int}(undef, n)
    k = 0
    @inbounds for c in cells(grid)
        cell = cell_at(grid, c)
        (everything || _cell_active(grid, cell)) || continue
        k += 1
        ids[k] = _linear_of(grid, cell)
        cellbin[k] = binof(c)
        counts[_bin_hash(cellbin[k], nbucket) + 1] += 1
    end
    starts = Vector{Int}(undef, nbucket + 1)
    starts[1] = 1
    @inbounds for b in 1:nbucket
        starts[b + 1] = starts[b] + counts[b + 1]
    end
    cursor = copy(starts)
    items = Vector{Int}(undef, n)
    bins = Vector{NTuple{D,Int}}(undef, n)
    @inbounds for k in 1:n
        b = _bin_hash(cellbin[k], nbucket)
        items[cursor[b]] = ids[k]
        bins[cursor[b]] = cellbin[k]      # parallel to `items`, so a walk reads both contiguously
        cursor[b] += 1
    end
    return CellListIndex{D,T,E,Vector{Int},Vector{NTuple{D,Int}}}(
        n, lo, hd, nbins, wrap, starts, items, bins, embedding, active_only,
    )
end

"""
    _embedded_floor(grid, embedding, h, ::Val{D}) -> NTuple{D,T}

A lower bound on the embedded coordinate in each direction, less one bin, which is where the lattice
starts. `O(1)`: no cell is read.

It only has to be a bound, not the tightest one — the lattice is unbounded along a direction that does
not wrap, and bins are hashed, so a loose origin shifts every bin index equally and changes nothing. The
bound comes from the geometry, which is what lets a layout be indexed without a pass over its cells.
"""
function _embedded_floor end

# The embedding is the coordinates themselves, and their span is an axis reduction the grid already has.
@inline _embedded_floor(grid, ::CartesianEmbedding, h::T, ::Val{D}) where {T,D} =
    ntuple(d -> T(bounds(grid, d)[1]) - h, Val(D))

# A point on a sphere of radius `ρ` has every Cartesian component in `[-ρ, ρ]`.
@inline _embedded_floor(grid, e::ArcEmbedding, h::T, ::Val{D}) where {T,D} =
    ntuple(_ -> -T(e.radius) - h, Val(D))

@inline _embedded_floor(grid, ::ChordEmbedding, h::T, ::Val{D}) where {T,D} =
    ntuple(_ -> -_ambient_bound(grid) - h, Val(D))

"""
    _ambient_bound(grid) -> T

The largest `|x|`, `|y|` or `|z|` any of this grid's cells can embed to.
"""
@inline function _ambient_bound(grid::AbstractGrid{G,T}) where {G<:Geometry.AbstractSphericalGeometry,T}
    # Beyond the surface the third direction is the absolute radius, so the outermost one bounds it.
    ncoordinates(grid) ≥ 3 || return T(Geometry.radius(grid_geometry(grid)))
    lo, hi = bounds(grid, 3)
    return max(abs(T(lo)), abs(T(hi)))
end

@inline function _ambient_bound(grid::AbstractGrid{G,T}) where {G<:Geometry.AbstractEllipsoidalGeometry,T}
    geo = grid_geometry(grid)
    a = T(Geometry.semimajor_axis(geo))
    ncoordinates(grid) ≥ 3 || return a
    lo, hi = bounds(grid, 3)          # geodetic height, which offsets the surface
    return a + max(abs(T(lo)), abs(T(hi)))
end

# Cartesian directions wrap with the grid; every other embedding is closed by its own transform.
#
# A wrapping direction's width is the period divided by a whole number of bins, not `h` itself: binning
# by `h` and then reducing mod `nbins` would fold a partial bin onto bin zero, and the span a query walks
# would no longer be a lattice neighbourhood.
function _cell_lattice(grid, ::CartesianEmbedding, h::T, ::Val{D}) where {T,D}
    wrap = ntuple(d -> isperiodic(grid, d), Val(D))
    nbins = ntuple(Val(D)) do d
        wrap[d] ? max(1, Base.unsafe_trunc(Int, floor(T(period(grid, d)) / h))) : 0
    end
    hd = ntuple(d -> wrap[d] ? T(period(grid, d)) / nbins[d] : h, Val(D))
    floor_ = _embedded_floor(grid, CartesianEmbedding(), h, Val(D))
    lo = ntuple(d -> wrap[d] ? T(origin(grid, d)) : floor_[d], Val(D))
    return wrap, nbins, lo, hd
end

function _cell_lattice(grid, embedding::AbstractEmbedding, h::T, ::Val{D}) where {T,D}
    lo = _embedded_floor(grid, embedding, h, Val(D))
    return ntuple(_ -> false, Val(D)), ntuple(_ -> 0, Val(D)), lo, ntuple(_ -> h, Val(D))
end

"""
    fold_candidates(f, acc, index, grid, I, r) -> acc

Thread `acc = f(acc, k)` over every cell `k` the index reports near cell `I`, without building a list. A
**superset** of the ball, each cell exactly once; the caller's exact distance gate decides membership.

A fold rather than a returned list is the whole point: it allocates nothing and needs no per-query
buffer, which is what a kernel requires and what a tree cannot offer, since a tree walk has to
deduplicate the periodic images it searches over.
"""
function fold_candidates end

function fold_candidates(f::F, acc, ix::CellListIndex{D,T}, grid, I, r) where {F,D,T}
    return fold_candidates_at(f, acc, ix, map(T, embedded_at(grid, I)), r, nothing)
end

"""
    fold_candidates_at(f, acc, index, q, r, scratch) -> acc

[`fold_candidates`](@ref) around an arbitrary point `q`, already in the index's embedding, rather than
around a cell. A cell query is this one at the cell's own centre, so there is one traversal.

`scratch` is a candidate buffer for an index that has to materialize one — a tree does, since it must
deduplicate the periodic images it searches over. A cell list folds directly and ignores it.
"""
function fold_candidates_at(f, acc, index, q, r, scratch)
    throw(ArgumentError(
        "$(typeof(index)) cannot be queried at a point; build a `cell_list` for point-seeded queries",
    ))
end

function fold_candidates_at(f::F, acc, ix::CellListIndex{D,T}, q::NTuple{D,T}, r, _scratch) where {F,D,T}
    remb = T(embedded_radius(ix.embedding, r))
    # Per direction, since a wrapping direction's bins are the period divided evenly and so are not
    # exactly `h` wide.
    span = ntuple(d -> max(0, Base.unsafe_trunc(Int, ceil(remb / @inbounds(ix.h[d])))), Val(D))
    b0 = _bin_of(ix, q)
    nbucket = _nbuckets(ix)
    # A wrapping direction is capped at the lattice width, since beyond that the offsets revisit bins
    # already covered.
    reach = ntuple(d -> @inbounds(ix.wrap[d]) ? min(2 * span[d] + 1, ix.nbins[d]) : 2 * span[d] + 1, Val(D))
    # A ball much wider than the bin side walks more bins than the lattice holds cells, and most of them
    # are empty — at which point every cell is a candidate anyway and enumerating them directly is both
    # cheaper and bounded. Without this a query at 10× the bin side costs `10^D` times its own answer,
    # and an index built for one radius and queried at a far larger one degenerates without limit.
    if _window_bins(reach) > nbucket
        # Every indexed cell is a candidate. That is `items`, not `1:n`: a masked grid indexes a subset,
        # and `items` holds the ids it kept.
        @inbounds for t in eachindex(ix.items)
            acc = f(acc, ix.items[t])
        end
        return acc
    end
    @inbounds for off in CartesianIndices(map(m -> 0:(m - 1), reach))
        b = ntuple(Val(D)) do d
            j = b0[d] - span[d] + off[d]
            ix.wrap[d] ? mod(j, ix.nbins[d]) : j
        end
        bucket = _bin_hash(b, nbucket)
        for t in ix.starts[bucket]:(ix.starts[bucket + 1] - 1)
            # A bucket is a hash of a bin and two bins can share one, so the bin decides membership.
            # It is stored alongside the item, which is what keeps this an integer compare rather than
            # a coordinate lookup and `D` divisions.
            ix.bins[t] == b || continue
            acc = f(acc, ix.items[t])
        end
    end
    return acc
end

# The buffered hook, for callers that want a list. No sort and no dedup: every cell sits in exactly one
# bin, so the walk already emits each at most once.
function index_within!(buf::AbstractVector{<:Integer}, ix::CellListIndex, grid, I, r)
    empty!(buf)
    fold_candidates(nothing, ix, grid, I, r) do _, k
        push!(buf, k)
        return nothing
    end
    return buf
end

"""
    locate(grid, p) -> cell index
    locate(grid, p; active_only=false, topology, scratch) -> cell index

The cell of `grid` that `p` belongs to, as an `NTuple` of indices on a rectilinear or curvilinear grid
and a node number on an `UnstructuredGrid`. `p` is a coordinate tuple in the grid's own coordinates.

On a `StructuredGrid` this is [`Discretization.locate`](@ref) per direction, so the answer is the cell
whose faces bracket `p` — an `O(1)` lookup on a uniform axis and a bisection otherwise, with a periodic
direction wrapped first. A direction `p` lies outside reports `0` for that direction.

Elsewhere there are no axes to bracket along and it is the **nearest cell centre**, which is exactly the
containing cell for a node set, whose cells are the Voronoi regions of its nodes. On a curvilinear grid
the two can differ where cells are strongly sheared, so read it as nearest-centre rather than
point-in-quadrilateral. That form takes the keywords: `topology` carrying an index — [`cell_list`](@ref)
— makes it a bin lookup rather than a scan, `scratch` is a `Connectivity.ball_scratch` buffer,
and `active_only` restricts the answer to unmasked cells (`false` here, unlike the ball queries, since
the cell a point falls in is a question about the grid rather than about the active region).
"""
function locate end

function locate(grid::StructuredGrid{T, G,N}, p::NTuple{N,Real}) where {G,T,N}
    return ntuple(Val(N)) do d
        x = coordinates(grid, d)
        v = T(p[d])
        per = isperiodic(grid, d)
        if per
            L = T(period(grid, d))
            lo = bounds(grid, d)[1]
            L > 0 && (v = lo + mod(v - lo, L))
        end
        i = Discretization.locate(x, v)
        # The cell straddling the seam has half its extent on each side, so after wrapping into one
        # period part of it lies beyond the outermost face and `locate` reports "outside". It is
        # whichever end cell is nearer across the seam — the comparison, not `1`, so a descending axis
        # is right too.
        if i == 0 && per
            n = length(x)
            L = T(period(grid, d))
            df = abs(rem(v - T(@inbounds x[1]), L, RoundNearest))
            dl = abs(rem(v - T(@inbounds x[n]), L, RoundNearest))
            i = df ≤ dl ? 1 : n
        end
        i
    end
end

"""
    has_spatial_index(grid) -> Bool

Whether the k-d tree behind [`spatial_index`](@ref) can be built for this grid — `false` until the
NearestNeighbors extension is loaded. Answers "is the tree available?" without calling `spatial_index`
speculatively and catching its error.

This is not a test for whether a grid can be indexed at all: [`cell_list`](@ref) needs no extension and
is what the sweeps build.
"""
has_spatial_index(::AbstractGrid) = false

"""
    spatial_index(grid) -> opaque index

Extension hook: a range-queryable spatial index over the grid's cell centres, overridden by the
NearestNeighbors extension. Paired with [`index_within!`](@ref).
"""
function spatial_index(grid::AbstractGrid)
    throw(ArgumentError(
        "a spatial index requires NearestNeighbors.jl — run `using NearestNeighbors`. Without it a ball " *
        "query on this grid scans every cell, which is correct but linear per query.",
    ))
end

"""
    index_within!(buffer, index, grid, I, r) -> candidate cell indices
    index_within(index, grid, I, r) -> candidate cell indices

Extension hook: the cells an index reports near `I`, as linear indices. It must return a **superset** of
the cells within `r`; the caller applies the exact distance gate, so over-returning is safe and
under-returning is not.

`index_within!` overwrites and returns `buffer`, which is how a sweep over many cells avoids one heap
allocation per query — nothing at all, against 480 bytes on a small ball and 6.1 KB on a 310-candidate
one. `index_within` is the same query into a fresh vector.
"""
function index_within!(buffer::AbstractVector{<:Integer}, index, grid, I, r)
    throw(ArgumentError("no `index_within!` method for $(typeof(index)); build one with `spatial_index`"))
end

index_within(index, grid, I, r) = index_within!(Int[], index, grid, I, r)

"""
    _voronoi_areas(geometry, x, y) -> Vector{T}

Extension hook: exact per-node Voronoi-cell area from a Delaunay/convex-hull tessellation of the node
coordinates. Dispatched on the geometry type (each needs a different tessellation library): overridden
for `CartesianGeometry` by a consumer DelaunayTriangulation extension (load
`using DelaunayTriangulation`, planar Voronoi clipped to the point set's convex hull) and for
`SphericalGeometry` by a consumer Quickhull extension (load `using Quickhull`, spherical
Voronoi from the dual of the 3D convex hull of the unit-sphere embedding).
"""
function _voronoi_areas(geometry::Geometry.AbstractCartesianGeometry, x::AbstractVector, y::AbstractVector)
    throw(ArgumentError(
        "Cartesian Voronoi-cell areas require DelaunayTriangulation.jl — run `using DelaunayTriangulation` " *
        "(or supply `areas` explicitly to the `UnstructuredGrid` constructor).",
    ))
end
function _voronoi_areas(geometry::Geometry.AbstractSphericalGeometry, x::AbstractVector, y::AbstractVector)
    throw(ArgumentError(
        "Spherical Voronoi-cell areas for an arbitrary point set require Quickhull.jl — run " *
        "`using Quickhull` (or supply `areas` explicitly to the `UnstructuredGrid` constructor).",
    ))
end

"""
    UnstructuredGrid(geometry, x, y, mask; k=6, radius=nothing, areas=nothing)

Build an `UnstructuredGrid` with REAL neighbor adjacency, via a k-d-tree nearest-neighbor query
(`NearestNeighbors.jl`; brute-force O(N²) doesn't scale) — either the `k` nearest neighbors per node
(default `k=6`), or every neighbor within a physical `radius` (pass `radius` to switch; mutually
exclusive with `k`). For `SphericalGeometry` the tree is built on the 3D Cartesian embedding of the
nodes (nearest-by-chord-distance is exactly nearest-by-great-circle-distance — exact, not an
approximation).

`areas`: supply per-node cell areas explicitly (common for a real dataset that ships its own), or
leave `nothing` to auto-compute exact Voronoi-cell areas from a Delaunay/convex-hull tessellation
(`DelaunayTriangulation.jl` for Cartesian, `Quickhull.jl` for spherical — see [`_voronoi_areas`](@ref)).

`periodic`/`period` declare a wrapping domain, and the neighbor search honors it: a node near one
face finds the nodes across the opposite face as genuine neighbors. Spherical longitude wraps by
default; a Cartesian box is opt-in and needs its `period`.
"""
UnstructuredGrid(
    geometry::Geometry.AbstractGeometry, x::AbstractVector, y::AbstractVector,
    mask::AbstractVector{Bool} = AllActive((length(x),)); kwargs...,
) = UnstructuredGrid(geometry, (x, y), mask; kwargs...)

function UnstructuredGrid(
    geometry::Geometry.AbstractGeometry{T}, coords::NTuple{N,AbstractVector},
    mask::AbstractVector{Bool} = AllActive((length(first(coords)),));
    k::Integer = 6, radius::Union{Nothing,Real} = nothing, areas::Union{Nothing,AbstractVector} = nothing,
    periodic = nothing, period = nothing,
) where {N, T<:AbstractFloat}
    c = ntuple(d -> _to_axis(T, coords[d]), Val(N))
    per, prd = _node_periodicity(geometry, Val(N), periodic, period)
    nbrs, ptr = _build_kdtree_neighbors(
        geometry, c; k = k, radius = radius,
        periodic = map(_is_periodic, per), period = prd,
    )
    # Voronoi tessellation is a 2-D algorithm (planar Delaunay / spherical convex hull), so past two
    # directions the control volumes are the caller's to supply — the neighbour search is not, and is
    # built for any `N` above.
    areas_T = if areas !== nothing
        _to_axis(T, areas)
    elseif N == 2
        _voronoi_areas(geometry, c[1], c[2])
    else
        throw(ArgumentError(
            "a $N-direction node grid has no control volumes to derive: the Voronoi tessellation " *
            "behind them is a 2-D algorithm (planar Delaunay / spherical convex hull). Pass `areas`.",
        ))
    end
    return UnstructuredGrid(geometry, c, areas_T, mask, nbrs, ptr; periodic = per, period = prd)
end

"""
    neighbor_nbrs(grid::UnstructuredGrid) -> AbstractVector{<:Integer}
    neighbor_ptr(grid::UnstructuredGrid) -> AbstractVector{<:Integer}

The CSR adjacency arrays: the flat neighbour indices, and the per-node offsets into them.
"""
@inline neighbor_nbrs(grid::UnstructuredGrid) = getfield(grid, :neighbor_nbrs)
@inline neighbor_ptr(grid::UnstructuredGrid) = getfield(grid, :neighbor_ptr)

"""
    neighbors(grid, I...; stencil = Stencils.Axial(1), active_only = true)

The neighbours of cell `I`, as a lazy sequence of linear indices that allocates nothing.

Where they come from is the layout's [`adjacency_source`](@ref): index-space offsets, which is what
`stencil` selects among, or the mesh's own stored incidence, which no stencil ranges over.

`active_only` holds either way — a masked cell has no neighbours, and a masked cell is not one. See
[`incident_nodes`](@ref) for the unfiltered storage behind a stored graph.
"""
function neighbors end

"""
    incident_nodes(grid, idx::Integer) -> AbstractVector{<:Integer}

The nodes stored as incident to node `idx`, as a zero-copy view into the CSR adjacency.

Storage, not a query: it reports what the mesh holds, with no regard for the mask. `neighbors` is the
query, and it honours `active_only` on every layout.
"""
@inline function incident_nodes(grid::AbstractUnstructuredGrid, idx::Integer)
    ptr = neighbor_ptr(grid)
    @inbounds lo = ptr[idx]
    @inbounds hi = ptr[idx+1] - 1
    return view(neighbor_nbrs(grid), lo:hi)
end

@inline function _raw_coords(grid::UnstructuredGrid{T,G,N}, idx::Integer) where {T,G,N}
    c = coordinates(grid)
    @boundscheck checkbounds(c[1], idx)
    return ntuple(d -> @inbounds(c[d][idx]), Val(N))
end
