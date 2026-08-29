
"""
    AbstractGrid{G<:AbstractGeometry, T<:AbstractFloat}

Supertype for all grid architectures.
"""
abstract type AbstractGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} end

"""
    AbstractStructuredGrid{G, T} <: AbstractGrid{G,T}

Rectilinear grids.  Default: [`StructuredGrid`](@ref).
"""
abstract type AbstractStructuredGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} <: AbstractGrid{G,T} end

"""
    AbstractCurvilinearGrid{G,T} <: AbstractGrid{G,T}

Curvilinear grids.  Default: [`CurvilinearGrid`](@ref).
"""
abstract type AbstractCurvilinearGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} <: AbstractGrid{G,T} end

"""
    AbstractUnstructuredGrid{G,T} <: AbstractGrid{G,T}

Unstructured / node grids.  Default: [`UnstructuredGrid`](@ref).
"""
abstract type AbstractUnstructuredGrid{G<:Geometry.AbstractGeometry, T<:AbstractFloat} <: AbstractGrid{G,T} end

# ---------------------------------------------------------------------------
# What a traversal asks of a layout
# ---------------------------------------------------------------------------
#
# Three questions, and every neighbourhood algorithm here is a function of the answers rather than of
# which grid type it was handed: how a cell is named, where its adjacency comes from, and how a
# distance query enumerates candidates. They are independent — a layout picks one answer to each — so
# they are three traits and not one, and a new layout supplies three methods instead of joining every
# traversal's dispatch table.

"""
    AbstractCellAddress

How a cell of a grid is named: [`CartesianCells`](@ref) or [`FlatCells`](@ref). Read it with
[`cell_address`](@ref).
"""
abstract type AbstractCellAddress end

"""
    CartesianCells()

A cell is an `NTuple{N,Int}` into an `N`-dimensional array of cells, so a traversal walks
`CartesianIndices` and converts to a linear index to report. Rectilinear, curvilinear and every
panel layout.
"""
struct CartesianCells <: AbstractCellAddress end

"""
    FlatCells()

A cell is one integer into a flat list, so the index a traversal walks and the index it reports are
the same number. Node sets, and the pixelizations whose cells are enumerated by a single id.
"""
struct FlatCells <: AbstractCellAddress end

"""
    cell_address(grid) -> AbstractCellAddress

Whether `grid`'s cells are named by an index tuple or by a single integer.
"""
function cell_address end

@inline cell_address(::AbstractStructuredGrid) = CartesianCells()
@inline cell_address(::AbstractCurvilinearGrid) = CartesianCells()
@inline cell_address(::AbstractUnstructuredGrid) = FlatCells()

"""
    AbstractAdjacency

Where a cell's neighbours come from: [`IndexStencilNeighbors`](@ref), [`FormulaNeighbors`](@ref) or
[`StoredMeshNeighbors`](@ref). Read it with [`adjacency_source`](@ref).

Distinct from [`AbstractCandidateSource`](@ref): adjacency is the mesh's own neighbour relation, and a
candidate source answers a question about *distance*, which no adjacency determines.
"""
abstract type AbstractAdjacency end

"""
    IndexStencilNeighbors()

Neighbours are offsets in the cell index space, wrapped or clipped per direction — the whole of what
`Connectivity.IndexTopology` carries. Coordinates never enter.
"""
struct IndexStencilNeighbors <: AbstractAdjacency end

"""
    FormulaNeighbors()

Neighbours are closed-form arithmetic on the cell id, with no graph and no coordinates stored: a
pixelization's face tables, a ring grid's in-ring and adjacent-ring maps, a panel seam.

A layout with this trait supplies [`formula_neighbors`](@ref).
"""
struct FormulaNeighbors <: AbstractAdjacency end

"""
    formula_neighbors(grid, cell) -> (ids::NTuple{K,Int}, n)

A cell's neighbours as linear indices, in the first `n` entries of a fixed-width tuple, where `K` is
[`max_neighbors`](@ref)`(grid)`.

A tuple rather than a buffer because it is a stack value: a traversal over every cell of a layout whose
adjacency is arithmetic then allocates nothing at all, and needs no scratch to be threaded through. `n`
varies where a layout has singular cells — a HEALPix pixel at a face corner has seven neighbours, not
eight.

The one method a [`FormulaNeighbors`](@ref) layout supplies.
"""
function formula_neighbors end

"""
    max_neighbors(grid) -> Int

The widest neighbour count any cell of `grid` can have, as a compile-time constant. It fixes the tuple
width [`formula_neighbors`](@ref) returns.
"""
function max_neighbors end

"""
    StoredMeshNeighbors()

Neighbours are read from stored incidence, because the mesh is genuinely arbitrary and no formula
describes it.
"""
struct StoredMeshNeighbors <: AbstractAdjacency end

"""
    adjacency_source(grid) -> AbstractAdjacency

Where `grid`'s neighbour relation comes from.
"""
function adjacency_source end

@inline adjacency_source(::AbstractStructuredGrid) = IndexStencilNeighbors()
@inline adjacency_source(::AbstractCurvilinearGrid) = IndexStencilNeighbors()
@inline adjacency_source(::AbstractUnstructuredGrid) = StoredMeshNeighbors()

"""
    AbstractCandidateSource

How a distance query enumerates the cells it must test: [`SeparableWindow`](@ref) or
[`IndexedCandidates`](@ref). Read it with [`candidate_source`](@ref).

Either way the caller's exact `distance ≤ r` gate decides membership, so both must return a SUPERSET
of the ball — which is what makes the two enumerations agree by construction rather than by two
implementations happening to match.
"""
abstract type AbstractCandidateSource end

"""
    SeparableWindow()

The grid has separable axes, so a bounding index window per direction contains every cell within a
given distance and is `O(1)` to compute — see `Connectivity.metric_window`. No index is needed and
nothing is buffered.
"""
struct SeparableWindow <: AbstractCandidateSource end

"""
    IndexedCandidates()

The grid has no separable axes to bound with, so candidates come from a spatial index over its cell
centres.
"""
struct IndexedCandidates <: AbstractCandidateSource end

"""
    candidate_source(grid) -> AbstractCandidateSource

How a distance query on `grid` enumerates candidates.
"""
function candidate_source end

@inline candidate_source(::AbstractStructuredGrid) = SeparableWindow()
@inline candidate_source(::AbstractCurvilinearGrid) = IndexedCandidates()
@inline candidate_source(::AbstractUnstructuredGrid) = IndexedCandidates()

# ---------------------------------------------------------------------------
# Common interface
# ---------------------------------------------------------------------------
# Coordinate STORAGE is positional: every grid holds an `NTuple{N,<:AbstractArray{T}}` of coordinate
# arrays, one per coordinate direction, reached by index through `coordinates(grid, d)`. Coordinate
# NAMES come from the geometry (`Geometry.point_names`), so `grid.x` exists on a Cartesian grid and
# `grid.λ` on a spherical one — never both, and never one standing in for the other.

# These reach fields with `getfield`, never `grid.field`: `getproperty` below is overloaded to
# resolve coordinate names, so property access from inside the interface would recurse.
@inline grid_geometry(grid::AbstractGrid) = getfield(grid, :geometry)

"""
    coordinates(grid) -> NTuple{N,AbstractArray}
    coordinates(grid, d::Integer) -> AbstractArray

The grid's coordinate arrays, or just direction `d`'s. Shape depends on the grid architecture: a
`StructuredGrid` stores one 1-D axis vector per direction, a `CurvilinearGrid` one `N`-D array
per direction, an `UnstructuredGrid` one value per node. Direction order matches
[`Geometry.point_names`](@ref) — `(x, y, z)` for Cartesian, `(λ, φ, r)` for spherical.

See also [`axis`](@ref) (the rectilinear spelling), [`coords`](@ref) (a single point).
"""
@inline coordinates(grid::AbstractGrid) = getfield(grid, :coordinates)
@inline coordinates(grid::AbstractGrid, d::Integer) = @inbounds coordinates(grid)[d]

"""
    materialize(grid) -> NTuple{D,Vector}

Every cell's coordinates, as one dense vector per direction.

How to ask a layout whose coordinates are a formula for them as data — for writing a file, or handing
them to something that takes point clouds. It allocates `D·n` numbers, so it is an explicit call.
"""
function materialize(grid::AbstractGrid{G,T}) where {G,T}
    D = ncoordinates(grid)
    n = length(mask(grid))
    out = ntuple(_ -> Vector{T}(undef, n), D)
    @inbounds for c in cells(grid)
        cell = cell_at(grid, c)
        k = _linear_of(grid, cell)
        p = _cell_coords(grid, cell)
        for d in 1:D
            out[d][k] = p[d]
        end
    end
    return out
end

"""
    ncoordinates(grid) -> Int

How many coordinate directions a cell of `grid` has — `2` for `(λ, φ)`, `3` for `(x, y, z)`.

Separate from counting `coordinates(grid)` because a layout whose coordinates are a formula stores no
arrays to count, and answers this from its parameters instead.
"""
@inline ncoordinates(grid::AbstractGrid) = length(coordinates(grid))

"""
    axis(grid::AbstractStructuredGrid, d::Integer) -> AbstractVector

Direction `d`'s coordinate axis. Only rectilinear grids have axes; this is
[`coordinates`](@ref) under the name that is exact for them.
"""
@inline axis(grid::AbstractStructuredGrid, d::Integer) = coordinates(grid, d)

# ---------------------------------------------------------------------------
# Spacing and span
# ---------------------------------------------------------------------------

"""
    spacing_trait(grid, ::Val{d}) -> UniformSpacing() | NonuniformSpacing()

Whether direction `d`'s spacing is known from its type, as a value a method can **dispatch** on.

[`isuniform`](@ref) answers the same question, but as a `Bool`, and a `Bool` can only be branched on.
A kernel that wants the uniform form of an expression — a stencil row that is the same at every
interior sample, an `O(1)` locate — has to select it by dispatch, which needs the direction in the
type. Hence the `Val`: it is what makes the answer available where the branch would be too late.

`isuniform(grid, d::Integer)` remains for a direction chosen at run time, where no such selection is
possible and a value is the only available form.
"""
@inline spacing_trait(grid::AbstractGrid, ::Val{d}) where {d} =
    Axes.spacing_trait(@inbounds coordinates(grid)[d])

"""
    isuniform(grid, d) -> Bool
    isuniform(grid) -> Bool

Whether coordinate direction `d` has constant spacing known from its TYPE (all directions, for the
no-`d` form). No code path inspects coordinate VALUES to decide this; the answer comes from the type
alone. See [`spacing_trait`](@ref) for the form a method can dispatch on.

A curvilinear or unstructured grid is never uniform: its coordinates are per-cell fields, not axes.
"""
@inline isuniform(grid::AbstractGrid, d::Integer) = _at_axis(Axes.isuniform, coordinates(grid), d)
@inline isuniform(grid::AbstractGrid) = all(Axes.isuniform, coordinates(grid))

"""
    spacing(grid, d) -> T

The constant spacing of coordinate direction `d`, available without reading any coordinate. Signed,
so a descending axis reports a negative spacing. Raises for a direction that is not
[`isuniform`](@ref) — for a nonuniform one use [`minimum_spacing`](@ref) / [`maximum_spacing`](@ref)
for its range of gaps, [`local_spacing`](@ref) for the gaps at one index, or [`cell_width`](@ref) /
[`cell_widths`](@ref) for the width of a cell.
"""
@inline spacing(grid::AbstractGrid, d::Integer) = _at_axis(Axes.spacing, coordinates(grid), d)

"""
    origin(grid, d) -> T

The first coordinate along direction `d`.
"""
@inline origin(grid::AbstractGrid, d::Integer) = _at_axis(first, coordinates(grid), d)

"""
    bounds(grid, d) -> (lo, hi)

Smallest and largest coordinate along direction `d`, ordered `lo ≤ hi` regardless of whether the
direction is stored ascending or descending. These are the extreme SAMPLE positions (cell centres),
not the outer cell boundaries.

`O(1)` on a rectilinear grid, whose directions are AXES: an axis is monotone — every search here
bisects it, which requires that — so its extremes are its endpoints and there is nothing to scan.
`O(N)` where the coordinates are per-cell fields instead, which have no such order; a query that wants
that repeatedly reads it from a [`MetricTopology`](@ref FlowGeometries.Connectivity.MetricTopology), built once.
"""
@inline bounds(grid::AbstractGrid, d::Integer) = extrema(coordinates(grid, d))

@inline bounds(grid::AbstractStructuredGrid, d::Integer) = _at_axis(_axis_bounds, coordinates(grid), d)

@inline function _axis_bounds(x::AbstractVector{T}) where {T}
    isempty(x) && return (T(Inf), T(-Inf))
    @inbounds a, b = first(x), last(x)
    return a ≤ b ? (a, b) : (b, a)
end

"""
    extent(grid, d) -> T

`hi - lo` from [`bounds`](@ref): the span covered by direction `d`'s samples. Zero for a singleton
direction.
"""
@inline function extent(grid::AbstractGrid, d::Integer)
    lo, hi = bounds(grid, d)
    return hi - lo
end

"""
    minimum_spacing(grid, d) -> T

Smallest gap between consecutive samples along direction `d`, as a non-negative magnitude. `O(1)` when
the direction is [`isuniform`](@ref) and `O(N_d)` otherwise. With [`maximum_spacing`](@ref) it bounds
how far an index window must reach to cover a given physical distance, which is what a
neighbourhood-by-distance query needs on a stretched axis. They are also the exact test of whether a
stretched axis happens to be equally spaced: its gaps are identical when the two are equal.

A direction of fewer than two samples has no gap, and reports `Inf` — the identity for `min`.
"""
@inline minimum_spacing(grid::AbstractStructuredGrid, d::Integer) =
    _at_axis(_min_gap, coordinates(grid), d)

"""
    maximum_spacing(grid, d) -> T

Largest gap between consecutive samples along direction `d`, the counterpart of
[`minimum_spacing`](@ref). A direction of fewer than two samples reports `0`, the identity for `max`.
"""
@inline maximum_spacing(grid::AbstractStructuredGrid, d::Integer) =
    _at_axis(_max_gap, coordinates(grid), d)

# Selecting ONE axis by a runtime direction out of the coordinate tuple, whose entries may be different
# types. `ntuple(…, Val(N))` unrolls where every direction is wanted; only one is here, so the tuple is
# walked by a recursive tail-split instead — the same shape as `_checkaxes`. Each branch sees a
# concretely typed axis and every branch returns the same concrete type, so the result infers and
# nothing allocates, where `coordinates(grid, d)` alone would be a dynamic lookup.
@inline _at_axis(::F, ::Tuple{}, d::Integer) where {F} =
    throw(ArgumentError("direction $d is outside this grid's directions"))
@inline _at_axis(f::F, c::Tuple, d::Integer) where {F} =
    d == 1 ? f(first(c)) : _at_axis(f, Base.tail(c), d - 1)

"""
    local_spacing(grid, d, i) -> (h_m, h_p)

The one-sided coordinate gaps around index `i` along direction `d`:
[`Discretization.local_spacing`](@ref) on that direction's axis, with the wrap period taken from the
grid, so a periodic seam is right without the caller supplying it.

Signed, so a descending axis reports negative gaps — see the axis-level form for why, and
[`cell_width`](@ref) for the non-negative width built from them. Allocation-free, so this is the
per-point form to call inside a loop assembling a finite-difference operator.
"""
@inline function local_spacing(grid::AbstractStructuredGrid, d::Integer, i::Integer)
    # Branching here rather than passing a `Union{Nothing,T}` period: both arms return the same
    # concrete tuple type, where the union would have to be split inside the callee on every call.
    return isperiodic(grid, d) ?
        _at_axis(x -> Discretization.local_spacing(x, i, period(grid, d)), coordinates(grid), d) :
        _at_axis(x -> Discretization.local_spacing(x, i, nothing), coordinates(grid), d)
end

"""
    cell_width(grid, d, i) -> T

The coordinate width of cell `i` along direction `d`: [`Discretization.cell_width`](@ref) on that
direction's axis, with the grid's own wrap period. Non-negative whichever way the axis is stored.

This is the coordinate width, not the cell measure — on a sphere [`measure`](@ref) is `R²cosφ·Δλ·Δφ`
and this is the `Δλ` or `Δφ` in it, which [`measure_factors`](@ref) does not expose separately
because it folds the metric into the factor it multiplies.
"""
@inline function cell_width(grid::AbstractStructuredGrid, d::Integer, i::Integer)
    return isperiodic(grid, d) ?
        _at_axis(x -> Discretization.cell_width(x, i, period(grid, d)), coordinates(grid), d) :
        _at_axis(x -> Discretization.cell_width(x, i, nothing), coordinates(grid), d)
end

"""
    cell_widths(grid, d) -> AbstractVector

[`cell_width`](@ref) along the whole of direction `d`, as
[`Discretization.cell_widths`](@ref) on its axis with the grid's own wrap period. A uniform direction
gets an [`Axes.ConstantVector`](@ref), so nothing is materialized.
"""
@inline function cell_widths(grid::AbstractStructuredGrid, d::Integer)
    return isperiodic(grid, d) ?
        Discretization.cell_widths(coordinates(grid, d), period(grid, d)) :
        Discretization.cell_widths(coordinates(grid, d), nothing)
end

"""
    coordinate_names(grid) -> NTuple{N,Symbol}

The grid's coordinate names, from its geometry: `(:x, :y[, :z])` or `(:λ, :φ[, :r])`.
"""
@inline coordinate_names(grid::AbstractGrid) =
    Geometry.point_names(grid_geometry(grid), Val(ncoordinates(grid)))

# Geometry-correct named access (`grid.x` / `grid.λ`), resolved from `coordinate_names`. A name that
# does not belong to this geometry is a `FieldError`, not a silent alias for the wrong quantity.
@inline function Base.getproperty(grid::AbstractGrid, name::Symbol)
    d = findfirst(==(name), coordinate_names(grid))
    d === nothing && return getfield(grid, name)
    return @inbounds coordinates(grid)[d]
end

@inline Base.propertynames(grid::AbstractGrid) = (coordinate_names(grid)..., fieldnames(typeof(grid))...)

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

"""
    AbstractTopology

How a coordinate direction closes: [`Periodic`](@ref) or [`Bounded`](@ref).

Carried in the grid's type, so [`isperiodic`](@ref) is a compile-time answer and callers can dispatch
on it. The cell measure depends on it — a wrapped boundary cell has a width a bounded one does not —
so it is part of what the grid is, not a flag attached to it.
"""
abstract type AbstractTopology end

"""
    Periodic()

A direction that wraps: the last cell's neighbour is the first, one [`period`](@ref) away.
"""
struct Periodic <: AbstractTopology end

"""
    Bounded()

A direction that ends: its first and last cells each have one neighbour rather than two.
"""
struct Bounded <: AbstractTopology end

"""
    topology(grid) -> NTuple{N,AbstractTopology}
    topology(grid, d) -> AbstractTopology

The grid's per-direction topology. Singletons, so this occupies no storage.
"""
topology(grid::AbstractGrid) = map(_ -> Bounded(), coordinates(grid))
@inline topology(grid::AbstractGrid, d::Integer) = topology(grid)[d]

"""
    periodic_flags(grid) -> NTuple{N,Bool}

[`topology`](@ref) as one `Bool` per direction. Const-folds, and unlike indexing the heterogeneous
topology tuple it stays type-stable under a runtime direction.
"""
@inline periodic_flags(grid::AbstractGrid) = map(_is_periodic, topology(grid))

@inline _is_periodic(::Periodic) = true
@inline _is_periodic(::Bounded) = false

"""
    isperiodic(grid, d) -> Bool

Whether coordinate direction `d` wraps. See [`topology`](@ref) for the type-level form and
[`period`](@ref) for the wrap length.
"""
@inline isperiodic(grid::AbstractGrid, d::Integer) =
    @inbounds periodic_flags(grid)[_checked_direction(periodic_flags(grid), d)]

# A direction is a dimension selector, not an array index, so an out-of-range one is an
# `ArgumentError` — the same one `_at_axis` raises — rather than whatever the underlying tuple read
# happens to do. Without this the read is `@inbounds` on a runtime index: under `--check-bounds=yes`
# it raises `BoundsError` from an internal, and under the default it is simply out of bounds.
@inline function _checked_direction(t::Tuple, d::Integer)
    1 ≤ d ≤ length(t) ||
        throw(ArgumentError("direction $d is outside this grid's directions"))
    return d
end

# Normalize a caller's `topology` to `NTuple{N,AbstractTopology}`. `Bool`s are accepted, and a scalar
# or short tuple applies to the leading directions with the rest `Bounded`.
_as_topology(::Val{N}, t::AbstractTopology) where {N} = ntuple(d -> d == 1 ? t : Bounded(), Val(N))
_as_topology(::Val{N}, b::Bool) where {N} = _as_topology(Val(N), b ? Periodic() : Bounded())
_as_topology(::Val{N}, t::NTuple{N,AbstractTopology}) where {N} = t
_as_topology(::Val{N}, t::Union{Tuple,AbstractVector}) where {N} =
    ntuple(d -> d ≤ length(t) ? _topology_of(t[d]) : Bounded(), Val(N))

@inline _topology_of(t::AbstractTopology) = t
@inline _topology_of(b::Bool) = b ? Periodic() : Bounded()

"""
    SeparableMeasure(factors)

The cell measure of a rectilinear grid, stored as its per-axis factors rather than materialized.

Every measure this package supports on such a grid is a product of one factor per axis (see
[`_measure_factors`](@ref)), so the `∏ Nᵈ` entries carry only `∑ Nᵈ` numbers. It is a genuine
`AbstractArray`: indexing, broadcasting and `collect` behave as for the dense equivalent.

`sum` is specialized to `∏ᵈ ∑ᵢ wᵈᵢ`, which is `O(∑ Nᵈ)` rather than `O(∏ Nᵈ)`.
"""
struct SeparableMeasure{T,N,F<:NTuple{N,AbstractVector{T}}} <: AbstractArray{T,N}
    factors::F
end

@inline Base.size(m::SeparableMeasure) = map(length, m.factors)
Base.IndexStyle(::Type{<:SeparableMeasure}) = IndexCartesian()

@inline function Base.getindex(m::SeparableMeasure{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(m, I...)
    return prod(ntuple(d -> @inbounds(m.factors[d][I[d]]), Val(N)))
end

"""
    SlabMeasure(lead, slab, rest)

The cell measure of a rectilinear grid in which ONE pair of directions is coupled and the rest are
not: `measure[i, j, k, l…] == lead[i] · slab[j, k] · rest[1][l] · …`.

A geodetic `(λ, φ, h)` volume element is `(N(φ)+h)·cosφ·(M(φ)+h)·Δλ·Δφ·Δh`. The height offsets BOTH
curvature radii, so no product of per-axis factors reproduces it and a
[`SeparableMeasure`](@ref) cannot hold it — but longitude enters none of it, so only `(φ, h)` need be
stored together. That is `Nφ·Nh + Nλ + …` numbers where the dense array is `∏ Nᵈ`: at a degree of
longitude and a hundred levels, a hundred and forty kilobytes rather than fifty megabytes.

`sum` and `extrema` are specialized, on the same argument as the separable case: the whole is a product
of independent groups, so its total is the product of their totals and its extremes are attained with
every group at one of its own.
"""
struct SlabMeasure{T,N,V<:AbstractVector{T},S<:AbstractMatrix{T},R<:Tuple} <: AbstractArray{T,N}
    lead::V
    slab::S
    rest::R
end

function SlabMeasure(lead::AbstractVector{T}, slab::AbstractMatrix{T}, rest::Tuple) where {T}
    N = 3 + length(rest)
    return SlabMeasure{T,N,typeof(lead),typeof(slab),typeof(rest)}(lead, slab, rest)
end

@inline Base.size(m::SlabMeasure) = (length(m.lead), size(m.slab)..., map(length, m.rest)...)
Base.IndexStyle(::Type{<:SlabMeasure}) = IndexCartesian()

@inline function Base.getindex(m::SlabMeasure{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(m, I...)
    v = @inbounds m.lead[I[1]] * m.slab[I[2], I[3]]
    @inbounds for d in 1:(N - 3)
        v *= m.rest[d][I[d + 3]]
    end
    return v
end

# ∑_{i,j,k} lead_i·slab_jk·… = (∑ lead)·(∑ slab)·∏(∑ rest), which is `O(Nλ + Nφ·Nh + ∑)` rather than
# `O(∏ Nᵈ)` — the same factorization `SeparableMeasure` uses, with one group of rank two.
Base.sum(m::SlabMeasure) = sum(m.lead) * sum(m.slab) * prod(sum, m.rest; init = one(eltype(m)))

function Base.extrema(m::SlabMeasure{T}) where {T}
    isempty(m) && throw(ArgumentError("extrema of an empty SlabMeasure is undefined"))
    ends = (extrema(m.lead), extrema(m.slab), map(extrema, m.rest)...)
    K = length(ends)
    lo = hi = prod(e -> e[1], ends)
    for corner in 0:(2^K - 1)
        v = one(T)
        for g in 1:K
            v *= ends[g][1 + ((corner >> (g - 1)) & 1)]
        end
        v < lo && (lo = v)
        v > hi && (hi = v)
    end
    return (lo, hi)
end
