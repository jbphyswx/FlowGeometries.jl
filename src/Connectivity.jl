module Connectivity

using ..Execution: Execution
using ..Geometry: Geometry
using ..Stencils: Stencils
using ..Discretization: Discretization
using ..Grids: Grids

# Connectivity is a property of the *grid architecture* or of a *spherical sampling*
# that defines its own mesh topology:
#   Structured / Curvilinear → index-topology stencil (periodicity + mask)
#   Unstructured            → stored CSR on the grid
#   Cubed-sphere / Yin–Yang / HEALPix / icosahedral / tensor-product samplings
#                           → `build_connectivity(sampling, …)` (see ConnectivitySpherical.jl)
#
# Primary API dispatches on `grid` or `sampling`. `CSRConnectivity` is the sparse storage
# format from `build_connectivity` when you need a flat graph.

# ---------------------------------------------------------------------------
# Sparse CSR storage
# ---------------------------------------------------------------------------

"""
    CSRConnectivity{VN,VP}

Sparse neighbor list: node `i` owns `nbrs[ptr[i]:ptr[i+1]-1]`. The two buffers are typed
independently, and their element type is any `Integer`, so a large mesh can hold `Int32` indices
(half the memory and bandwidth of `Int64`, and the width GPU kernels want).
"""
struct CSRConnectivity{VN<:AbstractVector{<:Integer}, VP<:AbstractVector{<:Integer}}
    nbrs::VN
    ptr::VP
end

"""
    csr_connectivity(nbrs, ptr; validate=true) -> CSRConnectivity

Wrap CSR buffers. `validate=false` skips O(nnz) checks (internal / trusted data).
"""
function csr_connectivity(
    nbrs::AbstractVector{<:Integer}, ptr::AbstractVector{<:Integer}; validate::Bool = true,
)
    if validate
        length(ptr) ≥ 1 || throw(ArgumentError("ptr must be non-empty (length nnodes+1)"))
        ptr[1] == 1 || throw(ArgumentError("ptr[1] must be 1 (got $(ptr[1]))"))
        ptr[end] - 1 == length(nbrs) || throw(ArgumentError(
            "CSR length mismatch: length(nbrs)=$(length(nbrs)), ptr[end]-1=$(ptr[end] - 1)",
        ))
        n = length(ptr) - 1
        @inbounds for i in 1:n
            ptr[i] ≤ ptr[i + 1] || throw(ArgumentError("ptr must be nondecreasing at i=$i"))
            for k in ptr[i]:(ptr[i + 1] - 1)
                j = nbrs[k]
                (1 ≤ j ≤ n) || throw(ArgumentError("neighbor index $j out of range 1:$n"))
            end
        end
    end
    return CSRConnectivity(nbrs, ptr)
end

"""
    empty_csr(nnodes, [I=Int]) -> CSRConnectivity

Adjacency in which every node has no neighbors, with `I`-typed indices.
"""
empty_csr(nnodes::Integer, ::Type{I} = Int) where {I<:Integer} =
    csr_connectivity(I[], ones(I, Int(nnodes) + 1); validate = false)

@inline nnodes(conn::CSRConnectivity) = length(conn.ptr) - 1
@inline nedges(conn::CSRConnectivity) = length(conn.nbrs)

@inline function nneighbors(conn::CSRConnectivity, i::Integer)
    @boundscheck checkbounds(conn.ptr, Int(i) + 1)
    return @inbounds conn.ptr[i + 1] - conn.ptr[i]
end

@inline function Grids.neighbors(conn::CSRConnectivity, i::Integer)
    @boundscheck checkbounds(conn.ptr, Int(i) + 1)
    lo = @inbounds conn.ptr[i]
    hi = @inbounds conn.ptr[i + 1] - 1
    return view(conn.nbrs, lo:hi)
end

# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

# Column-major linear index, for any number of dimensions. Written as a fold over the tuple so it
# unrolls: `i₁ + (i₂-1)·n₁ + (i₃-1)·n₁n₂ + …`.
@inline _linidx(sz::NTuple{1,Int}, i::Int) = i
@inline _linidx(sz::NTuple{2,Int}, i::Int, j::Int) = i + (j - 1) * sz[1]
@inline _linidx(sz::NTuple{3,Int}, i::Int, j::Int, k::Int) =
    i + (j - 1) * sz[1] + (k - 1) * sz[1] * sz[2]

@inline function _linidx(sz::NTuple{N,Int}, I::Vararg{Int,N}) where {N}
    lin = 1
    stride = 1
    @inbounds for d in 1:N
        lin += (I[d] - 1) * stride
        stride *= sz[d]
    end
    return lin
end

@inline linear_index(grid::Grids.AbstractStructuredGrid, I::Vararg{Integer}) =
    _linidx(Grids.size_tuple(grid), map(Int, I)...)

@inline linear_index(grid::Grids.AbstractCurvilinearGrid, I::Vararg{Integer}) =
    _linidx(Grids.size_tuple(grid), map(Int, I)...)

@inline cartesian_index(grid::Union{Grids.AbstractStructuredGrid,Grids.AbstractCurvilinearGrid}, lin::Integer) =
    CartesianIndices(Grids.size_tuple(grid))[lin]

# ---------------------------------------------------------------------------
# Stencil resolution
# ---------------------------------------------------------------------------

# Offsets come from `Stencils`, which generates them from the stencil's type, so any shape, radius and
# dimension is available and the loop over them still unrolls.
@inline _stencil_offsets(::Val{N}, s::Stencils.AbstractStencil) where {N} =
    Stencils.offsets(s, Val(N))

# A stencil is named by its TYPE, never by a symbol: a symbol cannot be resolved until run time, so the
# neighbour iterator built from it would not be concretely typed and every cell of a traversal would
# allocate. This signature is where that is enforced — the `stencil` keyword is deliberately left
# UNANNOTATED, because annotating it `::AbstractStencil` would declare it abstract and destroy the very
# inference the type-level design exists to get.
@inline _stencil_val(s::Stencils.AbstractStencil) = s

@inline function _wrap_or_clip(i::Int, di::Int, n::Int, periodic::Bool)
    j = i + di
    if periodic
        return mod1(j, n)
    elseif 1 ≤ j ≤ n
        return j
    else
        return 0
    end
end

# The grid carries its topology in its type, so this is a const-fold of singleton types rather than a
# tuple rebuilt per call.
@inline _periodic_flags(grid::Grids.AbstractGrid) = Grids.periodic_flags(grid)

# ---------------------------------------------------------------------------
# neighbors! / nneighbors / Grids.neighbors — dispatch on grid
# ---------------------------------------------------------------------------

"""
    neighbors!(out, grid, I...; stencil=Axial(1), active_only=true) -> n_written
    nneighbors(grid, I...; …) -> Int
    Grids.neighbors(grid, I...; …) -> Vector{Int}

Neighbors as linear indices into `grid.mask`. Prefer `neighbors!` on hot paths.
`stencil` is any [`Stencils.AbstractStencil`](@ref).
"""
function neighbors! end
function nneighbors end

# ---- Unstructured ------------------------------------------------------------

@inline nneighbors(grid::Grids.UnstructuredGrid, idx::Integer; _...) =
    length(Grids.neighbors(grid, idx))

function neighbors!(out::AbstractVector{<:Integer}, grid::Grids.UnstructuredGrid, idx::Integer; _...)
    nbr = Grids.neighbors(grid, idx)
    n = length(nbr)
    n ≤ length(out) || throw(ArgumentError("out too short (need ≥ $n)"))
    @inbounds for k in 1:n
        out[k] = nbr[k]
    end
    return n
end

# ---- Structured --------------------------------------------------------------

function neighbors!(
    out::AbstractVector{<:Integer}, grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {G,T,N}
    return neighbors!(out, grid, I, _stencil_val(stencil), active_only)
end

# ---- Index topology ----------------------------------------------------------

"""
    IndexTopology(size, periodic, mask)
    IndexTopology(grid)

Extent, wrapping and activity per dimension — the whole of what a neighbor computation reads.
Coordinates, cell measure and geometry never enter one, so a sampling can hand this over directly
rather than materializing a grid (axes, dense measure, full mask) to be read once and dropped. A
curvilinear grid is the `N = 2` case of the same algorithm, not a separate one.

`mask === nothing` means every cell is active, and costs no storage and no load.
"""
struct IndexTopology{N,M}
    size::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    mask::M
end

# `Grids.mask`, not `grid.mask`: dot access on a grid resolves coordinate names first, which is not free.
@inline IndexTopology(grid::Grids.StructuredGrid{G,T,N}) where {G,T,N} =
    IndexTopology(Grids.size_tuple(grid), _periodic_flags(grid), Grids.mask(grid))
@inline IndexTopology(grid::Grids.CurvilinearGrid) =
    IndexTopology(Grids.size_tuple(grid), _periodic_flags(grid), Grids.mask(grid))

@inline _active(::IndexTopology{N,Nothing}, ::Vararg{Int,N}) where {N} = true
@inline _active(t::IndexTopology{N}, I::Vararg{Int,N}) where {N} = @inbounds t.mask[I...]

# Both queries below read only `(size, periodic, mask)`, so structured and curvilinear grids share
# one implementation. Constructing the topology copies two tuples and a mask REFERENCE — nothing is
# allocated, and `N` stays a type parameter so the offset loop still unrolls.

@inline function _check_index(t::IndexTopology{N}, Ii::NTuple{N,Int}) where {N}
    @inbounds for d in 1:N
        (1 ≤ Ii[d] ≤ t.size[d]) || throw(BoundsError(t.mask, Ii))
    end
    return nothing
end

function _nneighbors(
    t::IndexTopology{N}, Ii::NTuple{N,Int}, sten::Stencils.AbstractStencil, active_only::Bool,
) where {N}
    _check_index(t, Ii)
    active_only && !_active(t, Ii...) && return 0
    sz = t.size
    per = t.periodic
    # Folded rather than looped over a returned offset tuple: the offsets are unrolled into the body, so
    # a wide stencil materializes nothing, and the count is threaded through as a value rather than a
    # captured local.
    return Stencils.fold_offsets(0, sten, Val(N)) do k, δ
        J = ntuple(d -> _wrap_or_clip(Ii[d], δ[d], sz[d], per[d]), Val(N))
        any(==(0), J) && return k
        active_only && !_active(t, J...) && return k
        return k + 1
    end
end

function neighbors!(
    out::AbstractVector{<:Integer}, t::IndexTopology{N}, Ii::NTuple{N,Int},
    sten::Stencils.AbstractStencil, active_only::Bool,
) where {N}
    _check_index(t, Ii)
    active_only && !_active(t, Ii...) && return 0
    sz = t.size
    per = t.periodic
    return Stencils.fold_offsets(0, sten, Val(N)) do k, δ
        J = ntuple(d -> _wrap_or_clip(Ii[d], δ[d], sz[d], per[d]), Val(N))
        any(==(0), J) && return k
        active_only && !_active(t, J...) && return k
        k += 1
        k ≤ length(out) || throw(ArgumentError("out too short for stencil (need ≥ $k)"))
        @inbounds out[k] = _linidx(sz, J...)
        return k
    end
end

@inline neighbors!(
    out::AbstractVector{<:Integer}, grid::Grids.StructuredGrid{G,T,N},
    I::NTuple{N,Integer}, sten::Stencils.AbstractStencil, active_only::Bool,
) where {G,T,N} = neighbors!(out, IndexTopology(grid), map(Int, I), sten, active_only)

function nneighbors(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {G,T,N}
    return _nneighbors(IndexTopology(grid), map(Int, I), _stencil_val(stencil), active_only)
end

"""
    StencilNeighbors{G,N,S}

Lazy neighbor sequence of one cell of an index-topology grid: iterating it walks the stencil offsets
and yields the linear index of each in-range, active neighbor.

Nothing is stored, so a traversal that visits every cell allocates nothing at all — where returning a
freshly built `Vector` per cell would cost two heap allocations per cell. Use [`neighbors!`](@ref) to
write into a caller-supplied buffer, or `collect` this to materialize it.
"""
struct StencilNeighbors{GR,N,S<:Stencils.AbstractStencil}
    grid::GR
    I::NTuple{N,Int}
    stencil::S
    active_only::Bool
end

Base.IteratorSize(::Type{<:StencilNeighbors}) = Base.HasLength()
Base.IteratorEltype(::Type{<:StencilNeighbors}) = Base.HasEltype()
Base.eltype(::Type{<:StencilNeighbors}) = Int
Base.length(s::StencilNeighbors) = _nneighbors(s.grid, s.I, s.stencil, s.active_only)

@inline function Base.iterate(s::StencilNeighbors{GR,N}, k::Int = 0) where {GR,N}
    grid = s.grid
    # A masked-out cell has no neighbors at all, matching `nneighbors`.
    (k == 0 && s.active_only && !Grids.isactive(grid, s.I...)) && return nothing
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    offs = _stencil_offsets(Val{N}(), s.stencil)
    @inbounds while k < length(offs)
        k += 1
        δ = offs[k]
        J = ntuple(d -> _wrap_or_clip(s.I[d], δ[d], sz[d], per[d]), Val(N))
        any(==(0), J) && continue
        s.active_only && !Grids.isactive(grid, J...) && continue
        return _linidx(sz, J...), k
    end
    return nothing
end

# The iterator counts through the same topology kernel, for either grid type.
@inline _nneighbors(
    grid::Union{Grids.StructuredGrid,Grids.CurvilinearGrid}, I::NTuple{N,Int},
    sten::Stencils.AbstractStencil, active_only::Bool,
) where {N} = _nneighbors(IndexTopology(grid), I, sten, active_only)

function Grids.neighbors(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {G,T,N}
    Ii = map(Int, I)
    sz = Grids.size_tuple(grid)
    @inbounds for d in 1:N
        (1 ≤ Ii[d] ≤ sz[d]) || throw(BoundsError(Grids.mask(grid), I))
    end
    return _stencil_neighbors(grid, Ii, _stencil_val(stencil), active_only)
end

@inline _stencil_neighbors(grid::GR, I::NTuple{N,Int}, sten::S, active_only::Bool) where {GR,N,S} =
    StencilNeighbors{GR,N,S}(grid, I, sten, active_only)

# ---- Curvilinear -------------------------------------------------------------

function neighbors!(
    out::AbstractVector{<:Integer}, grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {T,G,N}
    return neighbors!(out, grid, map(Int, I), _stencil_val(stencil), active_only)
end

@inline neighbors!(
    out::AbstractVector{<:Integer}, grid::Grids.CurvilinearGrid{T,G,N},
    I::NTuple{N,Int}, sten::Stencils.AbstractStencil, active_only::Bool,
) where {T,G,N} = neighbors!(out, IndexTopology(grid), I, sten, active_only)

function nneighbors(
    grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {T,G,N}
    return _nneighbors(IndexTopology(grid), map(Int, I), _stencil_val(stencil), active_only)
end

function Grids.neighbors(
    grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {T,G,N}
    Ii = map(Int, I)
    sz = Grids.size_tuple(grid)
    @inbounds for d in 1:N
        (1 ≤ Ii[d] ≤ sz[d]) || throw(BoundsError(Grids.mask(grid), Ii))
    end
    return _stencil_neighbors(grid, Ii, _stencil_val(stencil), active_only)
end

# ---------------------------------------------------------------------------
# Metric neighbourhoods — queries by physical distance
# ---------------------------------------------------------------------------
#
# A stencil query reads only `(size, periodic, mask)`. A distance query cannot: it needs coordinates and
# the geometry's own `distance`, so these take the grid rather than an `IndexTopology`. That is the whole
# reason they are a separate entry point instead of another `stencil` keyword — a
# [`Stencils.MetricBall`](@ref) has no fixed offset set, because how many cells lie within a given
# distance varies from cell to cell.

@inline _ball_radius(b::Stencils.MetricBall) = Stencils.radius(b)
@inline _ball_radius(r::Real) = r ≥ 0 ? r : throw(ArgumentError("a search radius must be ≥ 0, got $r"))

# Index half-width covering physical distance `r` when one index step spans AT LEAST `s`: a cell more
# than `w` steps out is then at least `w·s ≥ r` away. Clamped to the axis, and a degenerate direction
# (single sample, or no positive spacing) opens to the whole axis rather than silently excluding cells.
@inline function _steps(r::T, s::T, n::Int) where {T<:AbstractFloat}
    (s > 0 && isfinite(s)) || return n
    w = ceil(r / s)
    return w ≥ n ? n : Int(w)
end

# The chord between points at radii ≥ ρ separated by central angle σ is ≥ 2ρ·c·sin(σ/2), with `c` the
# `cosφ` attenuation (1 where none applies). Inverting it: the angle beyond which every cell is farther
# than `r` in the chord metric — and in the arc metric too, since arc ≥ chord. `n` when no angle is far
# enough, e.g. `r` reaching the antipode.
@inline function _angle_steps(r::T, scale::T, Δ::T, n::Int) where {T<:AbstractFloat}
    (scale > 0 && isfinite(scale)) || return n
    x = r / scale
    x < 1 || return n
    return _steps(one(T), Δ / (2 * asin(x)), n)   # w = ceil(σ_cut / Δ), through the same clamp
end

# The smallest interior gap of direction `d`, including the seam gap where it wraps — a periodic axis's
# seam can be its narrowest gap, and a window bound built without it could under-cover across the seam.
#
# `Grids.minimum_spacing` is `O(1)` on a uniform axis and an `O(N)` scan on a stretched one, so this is
# computed ONCE into a `MetricTopology` rather than per query. Called per query it made a stretched-axis
# ball 53.6× slower than a uniform one at N = 16384, growing without bound, for a window holding the same
# number of cells either way.
@inline function _min_step_scan(grid::Grids.StructuredGrid{G,T}, d::Int) where {G,T}
    st = Grids.axis_stats(grid, d)
    s = st.min_gap
    if Grids.isperiodic(grid, d)
        p = T(Grids.period(grid, d))
        if @inbounds(Grids.size_tuple(grid)[d]) ≥ 2 && p > 0
            seam = p - (st.max_value - st.min_value)
            seam > 0 && (s = min(s, seam))
        end
    end
    return s
end

"""
    MetricTopology(grid; index = nothing)

Everything a distance query reads that depends on the grid alone — the counterpart of
[`IndexTopology`](@ref) for the metric path.

A stencil query reads `(size, periodic, mask)` and `IndexTopology` carries it. A ball query additionally
needs the tightest per-direction step bound, which sizes the search window, and — where there are no
separable axes to bound with — a spatial index.

Constructing one is `O(1)` and allocates nothing: the per-axis reductions live on the grid already, as
[`Grids.AxisStats`](@ref), computed once when it was built. So the default `topology` on every query
costs nothing and there is no hoisting to remember. What is still worth hoisting is the **index**, which
is not built by default because a k-d tree inside a single query would cost more than the scan it
replaces — [`foreach_within`](@ref) and [`mapreduce_within`](@ref) do that hoisting for a sweep, and
[`indexed`](@ref) does it explicitly.

It is not cached on the grid: grid types are immutable and the `Adapt` extension reconstructs them
field-by-field for a device, so a mutable cache field would break both that and thread safety.

`index` holds a spatial index when one is available, for the architectures with no separable axes to bound.
"""
struct MetricTopology{N,T,S}
    steps::NTuple{N,T}   # smallest index step per direction, seam included
    min3::T              # global minimum of direction 3 (geodetic height); zero below 3 directions
    index::S
end

function MetricTopology(grid::Grids.StructuredGrid{G,T,N}; index = nothing) where {G,T,N}
    steps = ntuple(d -> _min_step_scan(grid, d), Val(N))
    m3 = N ≥ 3 ? Grids.axis_stats(grid, 3).min_value : zero(T)
    return MetricTopology{N,T,typeof(index)}(steps, m3, index)
end

@inline _min_step(mt::MetricTopology, d::Int) = @inbounds mt.steps[d]

# Curvilinear and node grids have no separable axes, so there is no step bound to carry — the topology
# exists for them to hold the spatial index.
function MetricTopology(
    grid::Union{Grids.CurvilinearGrid{T,G,N},Grids.UnstructuredGrid{T,G,N}};
    index = nothing,
) where {T,G,N}
    return MetricTopology{N,T,typeof(index)}(ntuple(_ -> zero(T), Val(N)), zero(T), index)
end

"""
    indexed(grid) -> MetricTopology

A [`MetricTopology`](@ref) carrying a spatial index, so ball queries on it cost `O(log n + m)` instead of
scanning every cell. Requires `NearestNeighbors`; without it [`Grids.spatial_index`](@ref) raises and the
unindexed topology still works, just linearly.
"""
indexed(grid::Grids.AbstractGrid) = MetricTopology(grid; index = Grids.spatial_index(grid))

# Candidate enumeration. Without an index every cell is a candidate; with one, the index returns a
# superset of the ball and the caller's exact gate does the rest.
#
# `scratch` is the index's candidate buffer, owned by the CALLER rather than by the topology: one
# `MetricTopology` is shared by every task in a threaded sweep, so a buffer inside it would be a data
# race. Read-only sharing is the property that makes that safe, and it is worth one argument to keep.
@inline _candidates(mt::MetricTopology, grid, I, r, scratch) =
    _index_candidates(mt.index, grid, I, r, scratch)

@inline _index_candidates(::Nothing, grid, _I, _r, _scratch) = Base.OneTo(length(Grids.mask(grid)))

# `scratch === nothing` is a type test, not a runtime one, so the branch is resolved at compile time.
@inline _index_candidates(index, grid, I, r, scratch) =
    scratch === nothing ? Grids.index_within(index, grid, I, r) :
                          Grids.index_within!(scratch, index, grid, I, r)

# Threading the accumulator over the candidates rather than over a list of them. An index that can
# enumerate without materializing says so by defining `Grids.fold_candidates`, and then the whole
# traversal holds no buffer — which is what makes it runnable inside a kernel.
@inline _fold_candidates(f::F, acc, mt::MetricTopology, grid, I, r, scratch) where {F} =
    _fold_over_index(f, acc, mt.index, grid, I, r, scratch)

@inline function _fold_over_index(f::F, acc, ::Nothing, grid, _I, _r, _scratch) where {F}
    @inbounds for k in Base.OneTo(length(Grids.mask(grid)))
        acc = f(acc, k)
    end
    return acc
end

@inline _fold_over_index(f::F, acc, ix::Grids.CellListIndex, grid, I, r, _scratch) where {F} =
    Grids.fold_candidates(f, acc, ix, grid, I, r)

# Anything else hands back a list, which then has to exist somewhere.
@inline function _fold_over_index(f::F, acc, index, grid, I, r, scratch) where {F}
    @inbounds for k in _index_candidates(index, grid, I, r, scratch)
        acc = f(acc, k)
    end
    return acc
end

"""
    ball_scratch() -> Vector{Int}

A candidate buffer to hand to repeated ball queries through their `scratch` argument, so an indexed
query reuses one allocation instead of making one per call. One buffer per task.
"""
ball_scratch() = Int[]

# The smallest `f(x[j])` over the index window, which is what bounds a scale factor that varies across
# it. Walks the clamped window; a periodic direction can reach every sample, so it walks all of them.
@inline function _window_min(f::F, x::AbstractVector{T}, i::Int, w::Int, periodic::Bool) where {F,T}
    n = length(x)
    lo, hi = periodic || w ≥ n ? (1, n) : (max(1, i - w), min(n, i + w))
    m = T(Inf)
    @inbounds for j in lo:hi
        v = f(x[j])
        v < m && (m = v)
    end
    return m
end

"""
    metric_window(grid, I, ball) -> NTuple{N,Int}

Per-direction index half-width guaranteed to contain every cell within `ball` of cell `I`.

Each direction is bounded through its smallest gap — [`Grids.minimum_spacing`](@ref), and the seam gap
where the direction wraps — which is one number on a uniform axis and an `O(N)` scan on a stretched one.
[`MetricTopology`](@ref) holds those gaps, so the four-argument form does no scanning at all; the
three-argument form builds a topology per call. On a spherical or ellipsoidal grid the longitude cut
additionally walks the latitude window for its smallest `cosφ`, so it costs the window it returns.

This is a *bound*, not the answer: it is the window [`neighbors_within!`](@ref) scans before filtering on
the geometry's own `distance`. It never under-covers, which is why it is geometry-specific. On a
spherical grid one longitude step spans `R·cosφ·Δλ`, so the λ half-width is taken at the latitude in the
window nearest a pole rather than at the cell's own latitude — at a polar cell every longitude is in
range, and the window says so.
"""
function metric_window end

metric_window(grid::Grids.StructuredGrid, I::NTuple, ball) =
    metric_window(grid, I, ball, MetricTopology(grid))

# A radius, however it is spelled. The grid-level forms below take this rather than an untyped `ball`
# so they cannot be confused with the per-cell forms, whose second argument is the cell index.
const _BallLike = Union{Real,Stencils.MetricBall}

"""
    metric_window(grid, ball) -> NTuple{N,Int}
    metric_window(grid, ball, topology) -> NTuple{N,Int}

The window valid for **every** cell of `grid`, rather than for one of them: the per-cell form
maximised over the grid, which is what sizing a cache or a footprint table needs.

`O(1)`. Taking `maximum` of the per-cell form would be `O(N)` for something the cached
[`Grids.AxisStats`](@ref) already determines: the smallest gap per axis is stored, and the extreme
`|cos φ|` over a latitude axis is at one of its two ends, since `|cos|` on `[-π/2, π/2]` is largest in
the middle. No `cos` per row, and no scan.

Conservative by construction — it is the per-cell window at the worst cell — so it never under-covers.
"""
metric_window(grid::Grids.StructuredGrid, ball::_BallLike) =
    metric_window(grid, ball, MetricTopology(grid))

function metric_window(
    grid::Grids.StructuredGrid{G,T,N}, ball::_BallLike, mt::MetricTopology,
) where {G<:Geometry.AbstractCartesianGeometry,T,N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    return ntuple(d -> _steps(r, _min_step(mt, d), sz[d]), Val(N))   # already point-independent
end

function metric_window(
    grid::Grids.StructuredGrid{G,T,N}, ball::_BallLike, mt::MetricTopology,
) where {G<:Geometry.AbstractSphericalGeometry,T,N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    wrest = ntuple(d -> _steps(r, _min_step(mt, d + 2), sz[d + 2]), Val(max(N - 2, 0)))
    # The smallest radius anywhere, which is where a given arc spans the least distance.
    ρ = if N ≥ 3
        st = Grids.axis_stats(grid, 3)
        min(abs(st.min_value), abs(st.max_value))
    else
        T(Geometry.radius(Grids.grid_geometry(grid)))
    end
    wφ = N ≥ 2 ? _angle_steps(r, 2ρ, _min_step(mt, 2), sz[2]) : 0
    wλ = _angle_steps(r, 2ρ * _cos_extreme(grid, Val(N)), _min_step(mt, 1), sz[1])
    return ntuple(d -> d == 1 ? wλ : d == 2 ? wφ : wrest[d - 2], Val(N))
end

function metric_window(
    grid::Grids.StructuredGrid{G,T,N}, ball::_BallLike, mt::MetricTopology,
) where {G<:Geometry.AbstractEllipsoidalGeometry,T,N}
    geo = Grids.grid_geometry(grid)
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    a = T(Geometry.semimajor_axis(geo))
    M0 = a * (one(T) - T(Geometry.eccentricity²(geo)))
    wrest = ntuple(d -> _steps(r, _min_step(mt, d + 2), sz[d + 2]), Val(max(N - 2, 0)))
    cosmin = _cos_extreme(grid, Val(N))
    if N ≥ 3
        hmin = mt.min3
        wφ = _steps(r, (M0 + hmin - r) * _min_step(mt, 2), sz[2])
        wλ = _angle_steps(r, 2 * (M0 + min(hmin, zero(T))) * cosmin, _min_step(mt, 1), sz[1])
        return ntuple(d -> d == 1 ? wλ : d == 2 ? wφ : wrest[d - 2], Val(N))
    end
    wφ = N ≥ 2 ? _steps(r, M0 * _min_step(mt, 2), sz[2]) : 0
    wλ = _steps(r, a * cosmin * _min_step(mt, 1), sz[1])
    return ntuple(d -> d == 1 ? wλ : d == 2 ? wφ : wrest[d - 2], Val(N))
end

# The smallest `|cos φ|` anywhere on the latitude axis, in `O(1)`. `|cos|` on `[-π/2, π/2]` falls away
# from the middle in both directions, so over an interval its minimum is at whichever end is farther
# from the equator — the two stored extremes are enough, and no row is visited.
@inline function _cos_extreme(grid::Grids.StructuredGrid{G,T,N}, ::Val{N}) where {G,T,N}
    N ≥ 2 || return one(T)
    st = Grids.axis_stats(grid, 2)
    return min(abs(cos(T(st.min_value))), abs(cos(T(st.max_value))))
end

"""
    gradient_plan(grid; stencil=Stencils.Axial(1), active_only=true, conn=nothing) -> GradientPlan

Build the least-squares gradient of `grid` — the geometry of it, with no field involved. See
[`Discretization.GradientPlan`](@ref) for what it is and why it is that; apply it with
[`Discretization.gradient!`](@ref).

This is the counterpart of `apply_stencil!` for the two architectures that have no separable axis to
difference along: a `CurvilinearGrid`, whose neighbours come from its index topology, and an
`UnstructuredGrid`, whose come from its stored adjacency. `conn` overrides the neighbour set; otherwise
one is built, from `stencil` where the architecture takes one.

Surface fields only — the tangent plane is two-dimensional, so the grid's coordinates must be a
`(λ, φ)` or `(x, y)` pair.

A masked cell gets no coefficients at all and reads zero gradient, and an inactive neighbour is not
offered to the fit, on the same rule as everywhere else: not determined by the active data, so not
invented.
"""
function gradient_plan end

function gradient_plan(
    grid::Grids.AbstractGrid{G,T}; stencil = Stencils.Axial(1), active_only::Bool = true,
    conn = nothing,
) where {G,T}
    length(Grids.coordinates(grid)) == 2 || throw(ArgumentError(
        "a least-squares gradient is built in the tangent plane, so it needs a 2-coordinate grid; " *
        "got $(length(Grids.coordinates(grid)))",
    ))
    # `build_connectivity` already resolves this per architecture: an index-space stencil on a
    # curvilinear grid, the stored adjacency on a node set, which ignores the stencil.
    c = conn === nothing ?
        build_connectivity(grid; stencil = stencil, active_only = active_only) : conn
    geo = Grids.grid_geometry(grid)
    msk = Grids.mask(grid)
    sz = grid isa Grids.UnstructuredGrid ? nothing : Grids.size_tuple(grid)
    n = length(msk)
    ptr = Vector{Int}(undef, n + 1)
    nbr = Int[]
    c1 = T[]
    c2 = T[]
    sizehint!(nbr, length(c.nbrs)); sizehint!(c1, length(c.nbrs)); sizehint!(c2, length(c.nbrs))
    # Scratch for one cell's neighbours: the displacements are needed twice, once to accumulate `A`
    # and once to weight it by `A⁺`, and re-projecting them would double the trigonometry.
    d1 = T[]; d2 = T[]; wk = T[]
    @inbounds ptr[1] = 1
    @inbounds for i in 1:n
        empty!(d1); empty!(d2); empty!(wk)
        if !(active_only && !msk[i])
            p0 = _grad_coords(grid, sz, i)
            for t in c.ptr[i]:(c.ptr[i + 1] - 1)
                j = Int(c.nbrs[t])
                active_only && !msk[j] && continue
                Δ = Geometry.project_to_tangent_plane(geo, p0, _grad_coords(grid, sz, j))
                δ1, δ2 = T(Δ[1]), T(Δ[2])
                q = δ1 * δ1 + δ2 * δ2
                q > 0 || continue                     # a coincident neighbour carries no direction
                push!(d1, δ1); push!(d2, δ2); push!(wk, inv(q)); push!(nbr, j)
            end
            a = zero(T); b = zero(T); cc = zero(T)
            for t in eachindex(wk)
                a += wk[t] * d1[t] * d1[t]
                b += wk[t] * d1[t] * d2[t]
                cc += wk[t] * d2[t] * d2[t]
            end
            # Relative tolerance: `A` scales with the weights, and `wₖ = 1/|Δrₖ|²` makes it O(number
            # of neighbours), so the cut has to be against its own size rather than an absolute number.
            tol = max(a + cc, one(T)) * sqrt(eps(T))
            p11, p12, p22 = Discretization._sympinv2(a, b, cc, tol)
            for t in eachindex(wk)
                push!(c1, wk[t] * (p11 * d1[t] + p12 * d2[t]))
                push!(c2, wk[t] * (p12 * d1[t] + p22 * d2[t]))
            end
        end
        ptr[i + 1] = ptr[i] + length(wk)
    end
    names = Geometry.point_names(geo, Val(2))
    return Discretization.GradientPlan(ptr, nbr, c1, c2, names)
end

# The scalar fit: one value per cell in, one value out. Reached through the rank-matched methods below,
# which is what keeps `interpolate` type-stable — a scalar for an unbatched field, a vector for a
# batched one, decided by the rank rather than by a length at run time.
function _interp_scattered(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}},
    p::NTuple{D,Real}; k::Integer = 8, active_only::Bool = true, masked = T(NaN),
    topology = MetricTopology(grid), scratch = nothing,
    policy::Discretization.AbstractMaskPolicy = Discretization.BlankMasked(),
) where {T,D}
    policy isa Discretization.ShiftWithinRun && Discretization._interp_mask_error(policy)
    length(Grids.coordinates(grid)) == 2 || throw(ArgumentError(
        "interpolation off a rectilinear grid is fitted in the tangent plane, so it needs a " *
        "2-coordinate grid; got $(length(Grids.coordinates(grid)))",
    ))
    n = length(Grids.mask(grid))
    length(field) == n || throw(DimensionMismatch(
        "field has $(length(field)) values for a grid of $n cells",
    ))
    geo = Grids.grid_geometry(grid)
    p0 = ntuple(d -> T(p[d]), Val(D))
    idx, dist = k_nearest(grid, p0; k = k, active_only = active_only, topology = topology,
                          scratch = scratch)
    isempty(idx) && return masked
    # `BlankMasked` refuses where the neighbourhood is not wholly active, on the same rule a stencil
    # uses; `k_nearest` with `active_only` has already dropped those, so the test is whether doing so
    # left a hole — a cell nearer than the farthest one kept, that was skipped.
    if policy isa Discretization.BlankMasked && active_only
        msk = Grids.mask(grid)
        rmax = dist[end]
        n_in = nneighbors_within(grid, p0; ball = rmax, active_only = false, topology = topology,
                                 scratch = scratch)
        n_in > length(idx) && return masked
    end
    return _scattered_fit(field, grid, geo, p0, idx, 0, masked)
end

# The fit at one batch element, `off` into the field. The neighbour set and the mask verdict are
# properties of the POINT and the geometry, so a batched call solves those once — they are the k-d tree
# query, the expensive part — and calls this per element. The per-neighbour projections are eight
# arithmetic ops and are recomputed rather than buffered, which keeps this allocation-free.
function _scattered_fit(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, geo,
    p0::NTuple{D,T}, idx, off::Int, masked,
) where {T,D}
    # Weighted least squares for `f ≈ a + g·Δr` in the tangent plane at `p`. `a` is the value there,
    # and including `g` is what makes it exact for a linear field rather than a smoothed average.
    m11 = zero(T); m12 = zero(T); m13 = zero(T)
    m22 = zero(T); m23 = zero(T); m33 = zero(T)
    b1 = zero(T); b2 = zero(T); b3 = zero(T)
    sz = grid isa Grids.UnstructuredGrid ? nothing : Grids.size_tuple(grid)
    fsum = zero(T); wsum = zero(T)
    @inbounds for t in eachindex(idx)
        j = Int(idx[t])
        Δ = Geometry.project_to_tangent_plane(geo, p0, _grad_coords(grid, sz, j))
        δ1, δ2 = T(Δ[1]), T(Δ[2])
        # Inverse-square distance, floored so a query exactly on a cell centre stays finite.
        q = δ1 * δ1 + δ2 * δ2
        w = inv(max(q, eps(T)))
        fv = T(field[off + j])
        m11 += w;            m12 += w * δ1;      m13 += w * δ2
        m22 += w * δ1 * δ1;  m23 += w * δ1 * δ2; m33 += w * δ2 * δ2
        b1 += w * fv;        b2 += w * δ1 * fv;  b3 += w * δ2 * fv
        fsum += w * fv;      wsum += w
    end
    # A symmetric 3×3 by its adjugate. Where it is singular — collinear neighbours, or a single one —
    # the plane is not determined and only its constant is, which is the weighted mean.
    a11 = m22 * m33 - m23 * m23
    a12 = m13 * m23 - m12 * m33
    a13 = m12 * m23 - m13 * m22
    det = m11 * a11 + m12 * a12 + m13 * a13
    scale = max(m11 * m22 * m33, one(T))
    abs(det) ≤ scale * sqrt(eps(T)) && return wsum > 0 ? fsum / wsum : masked
    return (a11 * b1 + a12 * b2 + a13 * b3) / det        # the constant term: the value at `p`
end

# One value per cell — a curvilinear grid's cells are an `N`-D array, a node grid's a vector — so the
# field's rank matching the cells' says this is a single field and the answer is a scalar.
@inline Discretization.interpolate(
    field::AbstractArray{<:Any,N}, grid::Grids.CurvilinearGrid{T,G,N}, p::NTuple{D,Real}; kwargs...,
) where {T,G,N,D} = _interp_scattered(field, grid, p; kwargs...)

@inline Discretization.interpolate(
    field::AbstractVector, grid::Grids.UnstructuredGrid{T}, p::NTuple{D,Real}; kwargs...,
) where {T,D} = _interp_scattered(field, grid, p; kwargs...)

# A higher rank than the cells means trailing batch axes, and the answer is one value per element. The
# allocating form of [`interpolate!`](@ref), as everywhere else in the package.
function Discretization.interpolate(
    field::AbstractArray{<:Any,NA}, grid::Grids.CurvilinearGrid{T,G,N}, p::NTuple{D,Real}; kwargs...,
) where {T,G,N,NA,D}
    n = length(Grids.mask(grid))
    return Discretization.interpolate!(Vector{T}(undef, length(field) ÷ n), field, grid, p; kwargs...)
end

function Discretization.interpolate(
    field::AbstractArray{<:Any,NA}, grid::Grids.UnstructuredGrid{T}, p::NTuple{D,Real}; kwargs...,
) where {T,NA,D}
    n = length(Grids.mask(grid))
    return Discretization.interpolate!(Vector{T}(undef, length(field) ÷ n), field, grid, p; kwargs...)
end

@inline Discretization.interpolate(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid,Grids.UnstructuredGrid},
    p::Geometry.PointLike; kwargs...,
) = Discretization.interpolate(field, grid, Geometry.as_ntuple(p); kwargs...)

"""
    interpolate!(out, field, grid, p; k=8, …) -> out

[`Discretization.interpolate`](@ref) off a rectilinear grid for a field carrying trailing BATCH axes,
writing one value per batch element.

The `k` nearest cells and the mask verdict are a property of the point and the geometry — and the k-d
tree query is the expensive part — so they are solved once here and the tangent-plane fit is then
applied to every element.
"""
function Discretization.interpolate!(
    out::AbstractVector, field::AbstractArray,
    grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, p::NTuple{D,Real};
    k::Integer = 8, active_only::Bool = true, masked = T(NaN),
    topology = MetricTopology(grid), scratch = nothing,
    policy::Discretization.AbstractMaskPolicy = Discretization.BlankMasked(),
) where {T,D}
    policy isa Discretization.ShiftWithinRun && Discretization._interp_mask_error(policy)
    length(Grids.coordinates(grid)) == 2 || throw(ArgumentError(
        "interpolation off a rectilinear grid is fitted in the tangent plane, so it needs a " *
        "2-coordinate grid; got $(length(Grids.coordinates(grid)))",
    ))
    n = length(Grids.mask(grid))
    (length(field) % n == 0) || throw(DimensionMismatch(
        "grid has $n cells; a field of $(length(field)) is not a whole number of them",
    ))
    nb = length(field) ÷ n
    length(out) == nb || throw(DimensionMismatch(
        "out holds $(length(out)) values but the field carries $nb batch elements",
    ))
    geo = Grids.grid_geometry(grid)
    p0 = ntuple(d -> T(p[d]), Val(D))
    idx, dist = k_nearest(grid, p0; k = k, active_only = active_only, topology = topology,
                          scratch = scratch)
    if isempty(idx)
        fill!(out, masked)
        return out
    end
    if policy isa Discretization.BlankMasked && active_only
        rmax = dist[end]
        n_in = nneighbors_within(grid, p0; ball = rmax, active_only = false, topology = topology,
                                 scratch = scratch)
        if n_in > length(idx)
            fill!(out, masked)
            return out
        end
    end
    @inbounds for b in 1:nb
        out[b] = _scattered_fit(field, grid, geo, p0, idx, (b - 1) * n, masked)
    end
    return out
end

@inline Discretization.interpolate!(
    out::AbstractVector, field::AbstractArray,
    grid::Union{Grids.CurvilinearGrid,Grids.UnstructuredGrid}, p::Geometry.PointLike; kwargs...,
) = Discretization.interpolate!(out, field, grid, Geometry.as_ntuple(p); kwargs...)

@inline _grad_coords(grid, ::Nothing, i::Int) = Grids._raw_coords(grid, i)
@inline _grad_coords(grid, sz::Tuple, i::Int) =
    Grids._raw_coords(grid, Tuple(@inbounds CartesianIndices(sz)[i])...)

"""
    metric_band(grid, dim, coord_t, coord_n, ball) -> T

The **exact** half-width along direction `dim` of the part of the row at `coord_n` that lies within
`ball` of a point at `coord_t`, in that direction's own coordinate units. `coord_t` and `coord_n` are
coordinates on the *other* direction of a two-direction grid.

[`metric_window`](@ref) returns a bounding box, which is the right answer for a query that then filters
on distance. A **separable sweep** — a prefix sum along a row, a row-by-row convolution — cannot filter,
and using the box instead of the exact extent costs it the exactness that made it worth doing. This is
the same geodesic solve, resolved per row rather than maximised into a box.

Returns a negative number where the row is out of reach entirely, so `band < 0` is the empty test. A
row the ball covers completely gives the half-width of the whole direction (`π` in longitude).

On a sphere, for `dim = 1`, this inverts the spherical law of cosines:

```math
|Δλ| ≤ \\arccos\\left(\\frac{\\cos(r/R) - \\sin φ_t \\sin φ_n}{\\cos φ_t \\cos φ_n}\\right)
```

with the empty band, the whole circle and a pole at either end all falling out of the same expression —
the pole case being where the denominator vanishes and the separation stops depending on `λ` at all,
handled here once rather than by each caller.
"""
function metric_band end

function metric_band(
    grid::Grids.StructuredGrid{G,T,2}, dim::Integer, coord_t::Real, coord_n::Real, ball,
) where {G<:Geometry.AbstractCartesianGeometry,T}
    r = T(_ball_radius(ball))
    d = abs(T(coord_n) - T(coord_t))
    d > r && return -one(T)                       # the row is farther than the ball reaches
    return sqrt(r * r - d * d)                    # a circle's half-chord at that offset, exactly
end

function metric_band(
    grid::Grids.StructuredGrid{G,T,2}, dim::Integer, coord_t::Real, coord_n::Real, ball,
) where {G<:Geometry.AbstractSphericalGeometry,T}
    dim == 1 || throw(ArgumentError(
        "a spherical band is solved along longitude; got dim = $dim. The latitude extent at fixed " *
        "longitudes is not a closed form in the same way — use `metric_window` for a bound.",
    ))
    R = T(Geometry.radius(Grids.grid_geometry(grid)))
    rad = T(_ball_radius(ball)) / R               # the ball as a central angle
    rad ≥ T(π) && return T(π)                     # reaches the antipode: the whole circle
    φt, φn = T(coord_t), T(coord_n)
    sφt, cφt = sincos(φt)
    sφn, cφn = sincos(φn)
    den = cφt * cφn
    num = cos(rad) - sφt * sφn
    # A pole at either end: the separation is `|φt - φn|` whatever the longitudes are, so the row is
    # either wholly in or wholly out. The generic expression divides by zero here.
    if abs(den) ≤ eps(T)
        return abs(φt - φn) ≤ rad ? T(π) : -one(T)
    end
    c = num / den
    c > one(T) && return -one(T)                  # even the nearest longitude is too far
    c < -one(T) && return T(π)                    # every longitude is within reach
    return acos(c)
end

function metric_window(
    grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Integer}, ball, mt::MetricTopology,
) where {G<:Geometry.AbstractCartesianGeometry, T, N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    return ntuple(d -> _steps(r, _min_step(mt, d), sz[d]), Val(N))
end

# Spherical `(λ, φ, r, …)`, distance the great-circle arc (2-D) or the chord (3-D). Both are bounded
# below by the chord, so one angular cut serves both. Solved in dependency order: the radial window
# fixes the smallest radius `ρ` the chords are taken at (chord ≥ |Δr|, so cells beyond the radial
# window are already excluded); the latitude cut follows (central angle ≥ |Δφ|); and the λ cut uses the
# smallest `cosφ` in the latitude window (`sin(σ/2) ≥ cosφ·sin(|Δλ|/2)` with both endpoint latitudes in
# that window).
function metric_window(
    grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Integer}, ball, mt::MetricTopology,
) where {G<:Geometry.AbstractSphericalGeometry, T, N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    wrest = ntuple(d -> _steps(r, _min_step(mt, d + 2), sz[d + 2]), Val(max(N - 2, 0)))
    ρ = if N ≥ 3
        _window_min(abs, Grids.coordinates(grid, 3), Int(I[3]), wrest[1], per[3])
    else
        T(Geometry.radius(Grids.grid_geometry(grid)))
    end
    wφ = N ≥ 2 ? _angle_steps(r, 2ρ, _min_step(mt, 2), sz[2]) : 0
    cosmin = N ≥ 2 ?
        _window_min(φ -> abs(cos(φ)), Grids.coordinates(grid, 2), Int(I[2]), wφ, per[2]) : one(T)
    wλ = _angle_steps(r, 2ρ * cosmin, _min_step(mt, 1), sz[1])
    return ntuple(d -> d == 1 ? wλ : d == 2 ? wφ : wrest[d - 2], Val(N))
end

# Geodetic `(λ, φ, h, …)`, distance Vincenty's geodesic (2-D) or the ECEF chord (3-D).
# The bounds, each provable without knowing the path:
#   h — geodetic height is the distance to the ellipsoid surface, and distance-to-a-set is 1-Lipschitz,
#       so chord ≥ |Δh|.
#   φ — 2-D: the surface element gives ds ≥ M(φ)|dφ| ≥ M(0)|dφ| along any path, M being smallest at the
#       equator. 3-D: |∇φ| = 1/(M+h), and everything within `r` of a grid point has h ≥ hmin - r, so
#       chord ≥ (M(0) + hmin - r)|Δφ| while that factor is positive — it reaches zero exactly where the
#       geodetic coordinate chart stops being regular, and the window opens fully.
#   λ — 2-D: ds ≥ N(φ)cosφ|dλ| ≥ a·cosφ|dλ|, with cosφ taken over the latitudes reachable within `r`
#       (|Δφ| ≤ r/M(0) along any path of length ≤ r). 3-D: the chord bound with geocentric radius
#       ≥ b²/a + hmin and cosψ ≥ cosφ (geocentric latitude is nearer the equator than geodetic).
function metric_window(
    grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Integer}, ball, mt::MetricTopology,
) where {G<:Geometry.AbstractEllipsoidalGeometry, T, N}
    geo = Grids.grid_geometry(grid)
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    a = T(Geometry.semimajor_axis(geo))
    M0 = a * (one(T) - T(Geometry.eccentricity²(geo)))   # M(0) = b²/a, the smallest curvature radius
    wrest = ntuple(d -> _steps(r, _min_step(mt, d + 2), sz[d + 2]), Val(max(N - 2, 0)))
    if N ≥ 3
        hmin = mt.min3
        wφ = _steps(r, (M0 + hmin - r) * _min_step(mt, 2), sz[2])
        cosmin = _window_min(φ -> abs(cos(φ)), Grids.coordinates(grid, 2), Int(I[2]), wφ, per[2])
        wλ = _angle_steps(r, 2 * (M0 + min(hmin, zero(T))) * cosmin, _min_step(mt, 1), sz[1])
        return ntuple(d -> d == 1 ? wλ : d == 2 ? wφ : wrest[d - 2], Val(N))
    end
    wφ = N ≥ 2 ? _steps(r, M0 * _min_step(mt, 2), sz[2]) : 0
    cosreach = if N ≥ 2
        φ0 = T(@inbounds Grids.coordinates(grid, 2)[I[2]])
        reach = r / M0
        lo, hi = max(φ0 - reach, -T(π) / 2), min(φ0 + reach, T(π) / 2)
        lo ≤ 0 ≤ hi ? min(cos(lo), cos(hi)) : min(abs(cos(lo)), abs(cos(hi)))
    else
        one(T)
    end
    wλ = _steps(r, a * cosreach * _min_step(mt, 1), sz[1])
    return ntuple(d -> d == 1 ? wλ : d == 2 ? wφ : wrest[d - 2], Val(N))
end

"""
    AbstractImageConvention

How a ball query treats a periodic direction: [`NearestImage`](@ref) or [`AllImages`](@ref).

Singleton **types** rather than a `Bool`, for the reason given above about stencils: the traversal branches
on this per candidate, so a runtime value leaves the coordinate expression unresolved and the whole walk
allocates — measured at 14 KB for one query on a 32² grid, against nothing when it is a type.
"""
abstract type AbstractImageConvention end

"""
    NearestImage()

Visit each cell at most once, at its nearest image. The neighbour-set convention, and the default.
"""
struct NearestImage <: AbstractImageConvention end

"""
    AllImages()

Visit every image of a cell that lands inside the ball, each carrying its own displacement rather than a
reduced one — what a periodic convolution sums over. See [`fold_within`](@ref).
"""
struct AllImages <: AbstractImageConvention end

"""
    AbstractReach

Which of the cells within `ball` a query returns: [`Unrestricted`](@ref), every one of them, or
[`Connected`](@ref), those reachable from the seed without leaving the ball.

A **type** rather than a flag, like [`AbstractImageConvention`](@ref): the two are different sets computed
by different algorithms, so which one you get should be visible in the call and fixed at compile time,
never inferred from a runtime value.
"""
abstract type AbstractReach end

"""
    Unrestricted()

Every cell whose centre lies within `ball`. The default, and a purely spatial query: with a spatial index
it costs `O(log n + m)` and never consults adjacency.

Note that the result is a ball, **not** a connected patch — with a mask, or a concave domain, it can
contain cells that are near the seed in space but reachable from it only by leaving the ball.
"""
struct Unrestricted <: AbstractReach end

"""
    Connected()
    Connected(stencil)

The cells within `ball` reachable from the seed through cells that are themselves within `ball` and
active: the connected component of the ball that contains the seed. A graph query — it expands along
adjacency and prunes at the ball's edge — so it is a subset of [`Unrestricted`](@ref). The two agree
whenever the ball is connected under the adjacency, which a maskless Cartesian `StructuredGrid` always is:
each index step moves monotonically in one coordinate, so a staircase path to a cell never leaves that
cell's own distance. `Connected` is strictly smaller wherever a mask or a boundary separates two parts of
the ball.

Adjacency has to be named, since only a node set carries its own: with no argument, direction-1
adjacency (`Stencils.Axial(1)`) on the index-space architectures and the stored neighbour lists on an
`UnstructuredGrid`. `Connected(stencil)` sets it explicitly for the former, and is an error for the
latter, which has no index space for a stencil to mean anything in.

**This is not a cheaper way to compute `Unrestricted`.** Take cells `P` (the seed), `Q` and `R`, adjacent
only as `P–Q–R`, with `d(P,Q) = 1.2r` and `d(P,R) = 0.8r`. A walk outward from `P` that drops any cell
farther than `r` stops at `Q` and never reaches `R`, though `R` is inside the ball. That walk is not a
broken `Unrestricted` — it is exactly `Connected`, which is why it cannot produce the other one.
"""
struct Connected{S} <: AbstractReach
    adjacency::S
end
Connected() = Connected(nothing)

@inline _reach_stencil(::Connected{Nothing}) = Stencils.Axial(1)
@inline _reach_stencil(c::Connected) = _stencil_val(c.adjacency)

# Per-direction candidate offsets. Bounded: the window, clipped at the walls. Periodic under
# `NearestImage`: the window clamped to one full turn — offsets congruent mod `n` are the same cell, and
# the coordinate each is measured at is decided by minimum image, not by which offset reached it.
# Periodic under `AllImages`: no clamp, because each offset is then a distinct POSITION of that cell.
@inline function _delta_range(w::Int, i::Int, n::Int, periodic::Bool, ::NearestImage)
    if periodic
        2w + 1 ≤ n && return (-w):w
        lo = -(n >> 1)
        return lo:(lo + n - 1)
    end
    return max(1 - i, -w):min(n - i, w)
end

@inline _delta_range(w::Int, i::Int, n::Int, periodic::Bool, ::AllImages) =
    periodic ? ((-w):w) : (max(1 - i, -w):min(n - i, w))

# Consecutive images of one cell are `period` apart, which is the relevant step for a direction holding a
# single sample — there `_min_step` has no interior gap to report.
@inline function _image_step(grid::Grids.StructuredGrid{G,T}, d::Int, mt::MetricTopology) where {G,T}
    s = _min_step(mt, d)
    if Grids.isperiodic(grid, d)
        p = T(Grids.period(grid, d))
        p > 0 && (s = min(s, p))
    end
    return s
end

# The window for an image-summing walk. `_steps` caps its half-width at the axis length, which is right
# when each cell is visited once and wrong here: a kernel wider than the period reaches images several
# turns out, and capping at `n` drops every one of them. So a periodic direction gets the UNCAPPED
# `ceil(r/s)`; a bounded one is unchanged, its offsets being clipped at the walls anyway.
function _image_window(
    grid::Grids.StructuredGrid{G,T,N}, ball, mt::MetricTopology,
) where {G<:Geometry.AbstractCartesianGeometry, T, N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    return ntuple(Val(N)) do d
        s = _image_step(grid, d, mt)
        if Grids.isperiodic(grid, d)
            (s > 0 && isfinite(s)) ? Int(ceil(r / s)) : sz[d]
        else
            _steps(r, s, sz[d])
        end
    end
end

# Summing images treats a periodic direction as a TRANSLATION of the domain, which it is on a Cartesian
# torus: `x` and `x+L` are distinct positions of the same cell, so a convolution must count each. An
# angular direction is an IDENTIFICATION instead — `λ` and `λ+2π` are the same point, and the geometry's
# distance is already `2π`-periodic in it — so the images are not distinct positions and summing them
# would count one cell repeatedly. Refused rather than silently wrong.
@inline _check_images(::Grids.AbstractGrid, ::NTuple{N,Bool}, ::NearestImage) where {N} = nothing

@inline function _check_images(
    grid::Grids.AbstractGrid, per::NTuple{N,Bool}, ::AllImages,
) where {N}
    any(per) || return nothing     # nothing wraps, so there is nothing to sum
    Grids.grid_geometry(grid) isa Geometry.AbstractCartesianGeometry || throw(ArgumentError(
        "AllImages() sums periodic images, which is only meaningful where a periodic direction is a " *
        "translation of the domain. This geometry's periodic direction is an angular identification — " *
        "λ and λ+2π are the same point, and its distance is already 2π-periodic — so summing images " *
        "would count one cell repeatedly. Use NearestImage().",
    ))
    return nothing
end

@inline _ball_window(grid, I, ball, mt, ::NearestImage) = metric_window(grid, I, ball, mt)
@inline _ball_window(grid, _I, ball, mt, ::AllImages) = _image_window(grid, ball, mt)

# The candidate's position. Under `NearestImage` the stored coordinate reduced to the image nearest the
# centre; under `AllImages` the image the offset actually names, built as
# `x[mod1(raw, n)] + (periods wrapped)·L` — the same construction `Discretization.axis_stencils` uses for
# a wrapped stencil node. Reducing that by minimum image is exactly what must not happen.
@inline _cand_coords(
    grid, p0, I::NTuple{N,Int}, δ::NTuple{N,Int}, J::NTuple{N,Int}, sz, per, prd, sgn, ::NearestImage,
) where {N} = _min_image(p0, Grids._raw_coords(grid, J...), prd)

@inline function _cand_coords(
    grid, p0, I::NTuple{N,Int}, δ::NTuple{N,Int}, J::NTuple{N,Int}, sz, per, prd, sgn, ::AllImages,
) where {N}
    return ntuple(Val(N)) do d
        x = Grids.coordinates(grid, d)
        raw = I[d] + δ[d]
        @inbounds per[d] ?
            x[mod1(raw, sz[d])] + oftype(prd[d], fld(raw - 1, sz[d])) * prd[d] * sgn[d] : x[raw]
    end
end

# Minimum image and the per-direction wrap lengths live in `Grids`, which owns coordinates and periods,
# and are what `Geometry.distance(grid, I, J)` is built on as well.
using ..Grids: _min_image, _wrap_lengths

"""
    neighbors_within!(out, grid, I...; ball, active_only=true, topology, scratch, reach) -> n_written

Write the linear indices of every cell whose centre lies within `ball` of cell `I`. `ball` is a
[`Stencils.MetricBall`](@ref) or a bare radius in the geometry's length units.

Distance is the geometry's own — great-circle on a sphere, Vincenty on a spheroid, the chord where a
third direction is present — so the neighbourhood is a genuine metric ball, not a box. The cell itself
is excluded, matching stencil semantics where the zero offset is not a neighbour. A periodic direction
wraps, each cell appears at most once, and its coordinate is taken by minimum image, so the seam
neither shortens nor lengthens a distance.

Cost is [`metric_window`](@ref) — `O(1)` per direction on **any** separable axis, uniform or stretched,
given the per-direction minimum steps that `topology` carries — times one distance evaluation per
candidate. Size the buffer with [`nneighbors_within`](@ref); there is no fixed count, since how many
cells fall within a fixed distance varies from cell to cell on any non-uniform or curved grid.

Three arguments matter for anything beyond a single query:

  * `topology` — a [`MetricTopology`](@ref), the grid invariants a ball query reads. The default is `O(1)`
    and allocation-free, so leaving it out costs nothing. On a curvilinear or node grid, pass
    [`indexed`](@ref) to make each query `O(log n + m)` rather than a scan of every cell — or use
    [`foreach_within`](@ref), which builds the index once for a whole sweep.
  * `scratch` — a candidate buffer from [`ball_scratch`](@ref), one per task. Accepted on every grid type
    and used where a query goes through a spatial index, i.e. on a curvilinear or node grid; a separable
    window has no candidate list to buffer. With one an indexed query allocates **nothing**; without one
    it allocates its candidate list per call, 480 bytes whatever the grid size. The time difference is
    within noise — the allocation is the reason to pass one.
  * `reach` — [`Unrestricted`](@ref) (the ball, and the default) or [`Connected`](@ref) (the part of it
    reachable from `I` without leaving it). Note that a ball is **not** a connected patch: with a mask or
    a concave domain it can contain cells reachable from the seed only by going outside `ball`.
"""
function neighbors_within! end

"""
    nneighbors_within(grid, I...; ball, active_only=true) -> Int

The count [`neighbors_within!`](@ref) would write — how large its buffer must be.
"""
function nneighbors_within end

"""
    neighbors_within(grid, I...; ball, active_only=true) -> Vector{Int}

The allocating form of [`neighbors_within!`](@ref): counts, sizes, and fills in one call.
"""
function neighbors_within end

"""
    fold_within(f, init, grid, I...; ball, images=NearestImage(), self=false, active_only=true)

Fold `acc = f(acc, J, d)` over every cell `J` within `ball` of cell `I`, `d` being its distance. The
traversal every distance query here is built on; the accumulator is threaded through as a value rather
than mutated, so nothing is captured and nothing is boxed.

`images` selects how a periodic direction is treated — [`NearestImage`](@ref) (the default, each cell
once, the convention [`neighbors_within!`](@ref) exposes) or [`AllImages`](@ref), which visits every
image of a cell that lands inside the ball, each carrying its own displacement rather than a reduced one.

`AllImages` is what a periodic convolution needs: on a torus of period `L`,
`f̄(x) = Σₖ ∫ K(x − y − kL) f(y) dy`, so where the kernel support exceeds `L/2` one cell contributes
through several images at different displacements, and keeping only the nearest drops the rest. Below
`L/2` the two conventions coincide exactly. It also widens the search — the window becomes the uncapped
`ceil(r/s)` per periodic direction rather than [`metric_window`](@ref)'s one-turn cap — and it is refused
where a periodic direction is angular rather than a translation.

`self = true` also folds the centre cell, at distance zero. A neighbour set excludes it, which is the
default; a convolution needs it, and it carries the kernel's largest weight, so omitting it is not a
small error.

`reach` selects the ball ([`Unrestricted`](@ref)) or the part of it reachable from the seed without
leaving it ([`Connected`](@ref)); `topology` and `scratch` are as in [`neighbors_within!`](@ref).
"""
function fold_within end

@inline function _fold_within(
    f::F, init, grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Int}, ball,
    images::AbstractImageConvention, active_only::Bool, self::Bool, mt::MetricTopology,
) where {F,G,T,N}
    sz = Grids.size_tuple(grid)
    @inbounds for d in 1:N
        (1 ≤ I[d] ≤ sz[d]) || throw(BoundsError(Grids.mask(grid), I))
    end
    per = _periodic_flags(grid)
    _check_images(grid, per, images)
    (active_only && !Grids.isactive(grid, I...)) && return init
    geo = Grids.grid_geometry(grid)
    r = T(_ball_radius(ball))
    w = _ball_window(grid, I, ball, mt, images)
    prd = map(T, _wrap_lengths(grid, Val(N)))
    sgn = ntuple(d -> T(Grids._wrap_sign(Grids.coordinates(grid, d))), Val(N))
    p0 = Grids._raw_coords(grid, I...)
    msk = Grids.mask(grid)
    acc = init
    # Getting past the activity check above means the centre is active, so no further test is needed.
    self && (acc = f(acc, I, zero(T)))
    rng = ntuple(d -> _delta_range(w[d], I[d], sz[d], per[d], images), Val(N))
    @inbounds for ci in CartesianIndices(rng)
        δ = Tuple(ci)
        all(iszero, δ) && continue
        J = ntuple(d -> per[d] ? mod1(I[d] + δ[d], sz[d]) : I[d] + δ[d], Val(N))
        active_only && !msk[J...] && continue
        q = _cand_coords(grid, p0, I, δ, J, sz, per, prd, sgn, images)
        dist = Geometry.distance(geo, p0, q)
        dist ≤ r || continue
        acc = f(acc, J, dist)
    end
    return acc
end

# `f::F` rather than a bare `f`: Julia does not specialize on a function-typed argument that is only
# passed through, so the inner call would be dynamic and box the grid, the index tuple and the radius —
# 224 bytes per query, against nothing once `F` forces a specialization.
# `scratch` is accepted and unused here, so the four entry points take the same arguments on every
# architecture: a separable window enumerates candidates without a buffer to reuse.
fold_within(
    f::F, init, grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    ball, images::AbstractImageConvention = NearestImage(),
    active_only::Bool = true, self::Bool = false, topology = MetricTopology(grid),
    reach::AbstractReach = Unrestricted(), scratch = nothing,
) where {F,G,T,N} =
    _route_fold(f, init, reach, grid, map(Int, I), ball, images, active_only, self, topology)

# The counting and writing forms are the one-image fold with a counting/appending step: one traversal, so
# the window bound and the distance gate cannot drift between conventions.
@inline function _within_scan(
    out, grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Int}, ball, active_only::Bool,
    mt::MetricTopology = MetricTopology(grid), reach::AbstractReach = Unrestricted(),
) where {G,T,N}
    sz = Grids.size_tuple(grid)
    return _route_fold(0, reach, grid, I, ball, NearestImage(), active_only, false, mt) do n, J, _
        m = n + 1
        out === nothing || _keep!(out, m, _linidx(sz, J...))
        return m
    end
end

neighbors_within!(
    out::AbstractVector{<:Integer}, grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true, topology = MetricTopology(grid),
    reach::AbstractReach = Unrestricted(), scratch = nothing,
) where {G,T,N} = _within_scan(out, grid, map(Int, I), ball, active_only, topology, reach)

nneighbors_within(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true, topology = MetricTopology(grid),
    reach::AbstractReach = Unrestricted(), scratch = nothing,
) where {G,T,N} = _within_scan(nothing, grid, map(Int, I), ball, active_only, topology, reach)

function neighbors_within(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true, topology = MetricTopology(grid),
    reach::AbstractReach = Unrestricted(), scratch = nothing,
) where {G,T,N}
    Ii = map(Int, I)
    # `Connected` materializes its component to walk it, so a counting pass would repeat the whole walk;
    # append in one pass instead. `Unrestricted` keeps the two-pass form, where counting is a cheap
    # window walk and the exact size beats growing the output.
    if reach isa Connected
        sz = Grids.size_tuple(grid)
        return _route_fold(Int[], reach, grid, Ii, ball, NearestImage(), active_only, false, topology) do v, J, _
            push!(v, _linidx(sz, J...))
            return v
        end
    end
    out = Vector{Int}(undef, _within_scan(nothing, grid, Ii, ball, active_only, topology, reach))
    _within_scan(out, grid, Ii, ball, active_only, topology, reach)
    return out
end

# ---- Curvilinear and unstructured -------------------------------------------
# Neither has separable axes, so `metric_window` has nothing to bound with. Two ways to enumerate
# candidates, and the fold below is written so they cannot disagree:
#
#   * no index — every cell, `O(n)` per query.
#   * a spatial index — a range query, `O(log n + m)`.
#
# The index only has to return a SUPERSET of the ball: the exact `distance ≤ r` gate below is unchanged
# and decides membership either way. That is what makes the indexed and unindexed results identical by
# construction rather than by agreement of two implementations.

@inline function _keep!(out, n::Int, lin::Int)
    n ≤ length(out) ||
        throw(ArgumentError("out too short for this ball (need ≥ $n; size it with nneighbors_within)"))
    @inbounds out[n] = lin
    return nothing
end

# A neighbour list is a SET of cells. Which order they come out in is whatever enumerated them — a window
# walks index order, a tree walks tree order, a cell list walks bin order — and no entry point sorts,
# because that would put an `O(m log m)` pass on top of an `O(m)` query for a property nothing needs.
# `sort_neighbors!` is there for callers who do want it.

@inline function _fold_within(
    f::F, init, grid::Grids.CurvilinearGrid{T,G,N}, I::NTuple{N,Int}, ball,
    active_only::Bool, self::Bool, mt::MetricTopology, scratch = nothing,
) where {F,T,G,N}
    sz = Grids.size_tuple(grid)
    @inbounds for d in 1:N
        (1 ≤ I[d] ≤ sz[d]) || throw(BoundsError(Grids.mask(grid), I))
    end
    (active_only && !Grids.isactive(grid, I...)) && return init
    geo = Grids.grid_geometry(grid)
    r = _ball_radius(ball)
    p0 = Grids._raw_coords(grid, I...)
    prd = map(x -> oftype(first(p0), x), _wrap_lengths(grid, Val(N)))
    msk = Grids.mask(grid)
    acc = init
    self && (acc = f(acc, I, zero(eltype(p0))))
    ci = CartesianIndices(sz)
    return _fold_candidates(acc, mt, grid, I, r, scratch) do a, lin
        J = Tuple(@inbounds ci[lin])
        J == I && return a
        @inbounds active_only && !msk[J...] && return a
        q = _min_image(p0, Grids._raw_coords(grid, J...), prd)
        dist = Geometry.distance(geo, p0, q)
        dist ≤ r || return a
        return f(a, J, dist)
    end
end

fold_within(
    f::F, init, grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true, self::Bool = false, topology = MetricTopology(grid),
    scratch = nothing, reach::AbstractReach = Unrestricted(),
) where {F,T,G,N} =
    _route_fold(f, init, reach, grid, map(Int, I), ball, active_only, self, topology, scratch)

function _within_scan_curvilinear(
    out, grid::Grids.CurvilinearGrid{T,G,N}, I::NTuple{N,Int}, ball, active_only::Bool,
    mt::MetricTopology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
) where {T,G,N}
    sz = Grids.size_tuple(grid)
    return _route_fold(0, reach, grid, I, ball, active_only, false, mt, scratch) do n, J, _
        m = n + 1
        out === nothing || _keep!(out, m, _linidx(sz, J...))
        return m
    end
end

# Appending fold, for the allocating `neighbors_within`. Unlike the structured window — where the
# counting pass is a cheap walk and an exact size beats growth — here it is a second tree query or a
# second full scan, so growing the output in one pass is the cheaper of the two.
function _within_push_curvilinear(
    grid::Grids.CurvilinearGrid{T,G,N}, I::NTuple{N,Int}, ball, active_only::Bool,
    mt::MetricTopology, scratch, reach::AbstractReach,
) where {T,G,N}
    sz = Grids.size_tuple(grid)
    return _route_fold(Int[], reach, grid, I, ball, active_only, false, mt, scratch) do v, J, _
        push!(v, _linidx(sz, J...))
        return v
    end
end

@inline function _fold_within(
    f::F, init, grid::Grids.UnstructuredGrid{T,G,N}, idx::Int, ball,
    active_only::Bool, self::Bool, mt::MetricTopology, scratch = nothing,
) where {F,T,G,N}
    msk = Grids.mask(grid)
    1 ≤ idx ≤ length(msk) || throw(BoundsError(msk, idx))
    (active_only && !Grids.isactive(grid, idx)) && return init
    geo = Grids.grid_geometry(grid)
    r = _ball_radius(ball)
    p0 = Grids._raw_coords(grid, idx)
    prd = map(x -> oftype(first(p0), x), _wrap_lengths(grid, Val(N)))
    acc = init
    self && (acc = f(acc, idx, zero(eltype(p0))))
    return _fold_candidates(acc, mt, grid, idx, r, scratch) do a, k
        k == idx && return a
        @inbounds active_only && !msk[k] && return a
        q = _min_image(p0, Grids._raw_coords(grid, k), prd)
        dist = Geometry.distance(geo, p0, q)
        dist ≤ r || return a
        return f(a, k, dist)
    end
end

fold_within(
    f::F, init, grid::Grids.UnstructuredGrid, idx::Integer;
    ball, active_only::Bool = true, self::Bool = false, topology = MetricTopology(grid),
    scratch = nothing, reach::AbstractReach = Unrestricted(),
) where {F} = _route_fold(f, init, reach, grid, Int(idx), ball, active_only, self, topology, scratch)

function _within_scan_unstructured(
    out, grid::Grids.UnstructuredGrid, idx::Int, ball, active_only::Bool,
    mt::MetricTopology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
)
    return _route_fold(0, reach, grid, idx, ball, active_only, false, mt, scratch) do n, k, _
        m = n + 1
        out === nothing || _keep!(out, m, k)
        return m
    end
end

function _within_push_unstructured(
    grid::Grids.UnstructuredGrid, idx::Int, ball, active_only::Bool, mt::MetricTopology, scratch,
    reach::AbstractReach,
)
    return _route_fold(Int[], reach, grid, idx, ball, active_only, false, mt, scratch) do v, k, _
        push!(v, k)
        return v
    end
end

nneighbors_within(
    grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
) where {T,G,N} =
    _within_scan_curvilinear(nothing, grid, map(Int, I), ball, active_only, topology, scratch, reach)
neighbors_within!(
    out::AbstractVector{<:Integer}, grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
) where {T,G,N} =
    _within_scan_curvilinear(out, grid, map(Int, I), ball, active_only, topology, scratch, reach)
neighbors_within(
    grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
) where {T,G,N} =
    _within_push_curvilinear(grid, map(Int, I), ball, active_only, topology, scratch, reach)

nneighbors_within(
    grid::Grids.UnstructuredGrid, idx::Integer;
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
) = _within_scan_unstructured(nothing, grid, Int(idx), ball, active_only, topology, scratch, reach)
neighbors_within!(
    out::AbstractVector{<:Integer}, grid::Grids.UnstructuredGrid, idx::Integer;
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
) = _within_scan_unstructured(out, grid, Int(idx), ball, active_only, topology, scratch, reach)
neighbors_within(
    grid::Grids.UnstructuredGrid, idx::Integer;
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
) = _within_push_unstructured(grid, Int(idx), ball, active_only, topology, scratch, reach)

# ---------------------------------------------------------------------------
# Reach: the ball, or the part of it reachable from the seed
# ---------------------------------------------------------------------------
#
# `Unrestricted` and `Connected` are declared with the image conventions above, since the entry points'
# signatures name them; what follows is how `Connected` is computed.

# Adjacency as a callable, so the walk below is written once: `adj(lin) do nb … end` visits the
# neighbours of `lin` without materializing a list per cell.
struct _IndexSpaceAdjacency{N,O}
    size::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    offsets::O
end

@inline function (a::_IndexSpaceAdjacency{N})(cb::F, lin::Int) where {N,F}
    I = Tuple(@inbounds CartesianIndices(a.size)[lin])
    for δ in a.offsets
        J = ntuple(d -> _wrap_or_clip(I[d], δ[d], a.size[d], a.periodic[d]), Val(N))
        any(==(0), J) && continue
        cb(_linidx(a.size, J...))
    end
    return nothing
end

struct _StoredAdjacency{VN,VP}
    nbrs::VN
    ptr::VP
end

@inline function (a::_StoredAdjacency)(cb::F, k::Int) where {F}
    @inbounds for t in a.ptr[k]:(a.ptr[k + 1] - 1)
        cb(Int(a.nbrs[t]))
    end
    return nothing
end

# Sort the ball by linear index so membership is a binary search rather than a `Set`: the walk tests
# membership once per (cell, neighbour) pair, and `searchsortedfirst` over a sorted `Vector{Int}` beats
# hashing at these sizes without allocating a dictionary per query.
function _sort_ball!(idxs::Vector{Int}, ds::Vector)
    p = sortperm(idxs)
    permute!(idxs, p)
    permute!(ds, p)
    return nothing
end

# The seed's component, as positions into `idxs` in breadth-first order.
function _component(idxs::Vector{Int}, seed::Int, adj::A) where {A}
    m = length(idxs)
    pos = searchsortedfirst(idxs, seed)
    (pos ≤ m && @inbounds(idxs[pos]) == seed) || return Int[]   # inactive seed: an empty ball
    visited = falses(m)
    order = Vector{Int}()
    sizehint!(order, m)
    queue = Int[pos]
    @inbounds visited[pos] = true
    while !isempty(queue)
        p = popfirst!(queue)
        push!(order, p)
        adj(@inbounds idxs[p]) do nb
            q = searchsortedfirst(idxs, nb)
            (q ≤ m && @inbounds(idxs[q]) == nb && !@inbounds(visited[q])) || return nothing
            @inbounds visited[q] = true
            push!(queue, q)
            return nothing
        end
    end
    return order
end

# The ball as (linear indices, distances), seed included — the walk needs random access to it, which a
# fold cannot give. `self = true` here regardless of the caller's `self`, since the seed is the walk's
# root; it is dropped afterwards if the caller did not ask for it.
function _ball_lists(
    grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Int}, ball,
    images::AbstractImageConvention, active_only::Bool, mt::MetricTopology,
) where {G,T,N}
    sz = Grids.size_tuple(grid)
    idxs, ds = Int[], T[]
    _fold_within(nothing, grid, I, ball, images, active_only, true, mt) do _, J, d
        push!(idxs, _linidx(sz, J...))
        push!(ds, d)
        return nothing
    end
    _sort_ball!(idxs, ds)
    return idxs, ds
end

function _ball_lists(
    grid::Grids.CurvilinearGrid{T,G,N}, I::NTuple{N,Int}, ball, active_only::Bool,
    mt::MetricTopology, scratch,
) where {T,G,N}
    sz = Grids.size_tuple(grid)
    idxs, ds = Int[], T[]
    _fold_within(nothing, grid, I, ball, active_only, true, mt, scratch) do _, J, d
        push!(idxs, _linidx(sz, J...))
        push!(ds, d)
        return nothing
    end
    _sort_ball!(idxs, ds)
    return idxs, ds
end

function _ball_lists(
    grid::Grids.UnstructuredGrid{T,G,N}, idx::Int, ball, active_only::Bool,
    mt::MetricTopology, scratch,
) where {T,G,N}
    idxs, ds = Int[], T[]
    _fold_within(nothing, grid, idx, ball, active_only, true, mt, scratch) do _, k, d
        push!(idxs, k)
        push!(ds, d)
        return nothing
    end
    _sort_ball!(idxs, ds)
    return idxs, ds
end

# Fold over the component in breadth-first order. Distances come from the ball pass, so a `Connected`
# fold sees exactly the distances an `Unrestricted` one does.
@inline function _emit_component(
    f::F, init, idxs::Vector{Int}, ds::Vector, order::Vector{Int}, seed::Int, self::Bool, to_index::C,
) where {F,C}
    acc = init
    @inbounds for p in order
        lin = idxs[p]
        (lin == seed && !self) && continue
        acc = f(acc, to_index(lin), ds[p])
    end
    return acc
end

function _connected_fold(
    f::F, init, grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Int}, ball,
    images::AbstractImageConvention, active_only::Bool, self::Bool, mt::MetricTopology,
    reach::Connected,
) where {F,G,T,N}
    images isa NearestImage || throw(ArgumentError(
        "`Connected` is a graph query, so a cell must be one node: use `NearestImage`, not `$(images)`",
    ))
    sz = Grids.size_tuple(grid)
    idxs, ds = _ball_lists(grid, I, ball, images, active_only, mt)
    adj = _IndexSpaceAdjacency(sz, _periodic_flags(grid), _stencil_offsets(Val{N}(), _reach_stencil(reach)))
    seed = _linidx(sz, I...)
    order = _component(idxs, seed, adj)
    ci = CartesianIndices(sz)
    return _emit_component(f, init, idxs, ds, order, seed, self, lin -> Tuple(@inbounds ci[lin]))
end

function _connected_fold(
    f::F, init, grid::Grids.CurvilinearGrid{T,G,N}, I::NTuple{N,Int}, ball,
    active_only::Bool, self::Bool, mt::MetricTopology, scratch, reach::Connected,
) where {F,T,G,N}
    sz = Grids.size_tuple(grid)
    idxs, ds = _ball_lists(grid, I, ball, active_only, mt, scratch)
    adj = _IndexSpaceAdjacency(sz, _periodic_flags(grid), _stencil_offsets(Val{N}(), _reach_stencil(reach)))
    seed = _linidx(sz, I...)
    order = _component(idxs, seed, adj)
    ci = CartesianIndices(sz)
    return _emit_component(f, init, idxs, ds, order, seed, self, lin -> Tuple(@inbounds ci[lin]))
end

function _connected_fold(
    f::F, init, grid::Grids.UnstructuredGrid, idx::Int, ball,
    active_only::Bool, self::Bool, mt::MetricTopology, scratch, reach::Connected,
) where {F}
    reach isa Connected{Nothing} || throw(ArgumentError(
        "an `UnstructuredGrid` has no index space for a stencil to mean anything in; its adjacency is " *
        "the neighbour lists it stores, so use `Connected()`",
    ))
    idxs, ds = _ball_lists(grid, idx, ball, active_only, mt, scratch)
    adj = _StoredAdjacency(Grids.neighbor_nbrs(grid), Grids.neighbor_ptr(grid))
    order = _component(idxs, idx, adj)
    return _emit_component(f, init, idxs, ds, order, idx, self, identity)
end

# The one place reach is selected. Dispatching on the singleton keeps the choice at compile time, so
# `Unrestricted` pays nothing for `Connected` existing.
@inline _route_fold(f::F, init, ::Unrestricted, args...) where {F} = _fold_within(f, init, args...)
@inline _route_fold(f::F, init, r::Connected, args...) where {F} = _connected_fold(f, init, args..., r)

# ---------------------------------------------------------------------------
# Queries seeded by a point rather than a cell
# ---------------------------------------------------------------------------
#
# Observation data does not arrive on the grid: a float, a station, a ship track is a coordinate. Every
# query above is seeded by a cell index, so relating such data to a grid was not expressible at all on a
# curvilinear or node grid, where there are no axes to compose `locate` along either.
#
# The gate is the same `distance ≤ r` under the geometry's own metric; only the enumeration differs,
# because a point has no cell whose window to take.

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
    f::F, init, grid::Grids.StructuredGrid{G,T,N}, p::NTuple{N,Real};
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

function fold_at(
    f::F, init, grid::Grids.CurvilinearGrid{T,G,N}, p::NTuple{N,Real};
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
) where {F,T,G,N}
    return _fold_at_scattered(f, init, grid, ntuple(d -> T(p[d]), Val(N)), ball, active_only,
                              topology, scratch, Grids.size_tuple(grid))
end

function fold_at(
    f::F, init, grid::Grids.UnstructuredGrid{T,G,N}, p::NTuple{N,Real};
    ball, active_only::Bool = true, topology = MetricTopology(grid), scratch = nothing,
) where {F,T,G,N}
    return _fold_at_scattered(f, init, grid, ntuple(d -> T(p[d]), Val(N)), ball, active_only,
                              topology, scratch, nothing)
end

function _fold_at_scattered(
    f::F, init, grid::Grids.AbstractGrid{G,T}, p0::NTuple{D,T}, ball, active_only::Bool,
    mt::MetricTopology, scratch, sz,
) where {F,G,T,D}
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

# Anything else goes through the index's own point query, which raises if it has none — rather than
# quietly scanning every cell behind a topology the caller passed in order to avoid exactly that.
@inline _fold_at_index(f::F, acc, index, grid, p0, r, scratch) where {F} =
    Grids.fold_candidates_at(f, acc, index, Grids.embed_point(grid, p0), r, scratch)

"""
    neighbors_within(grid, p; ball, …) -> Vector{Int}
    nneighbors_within(grid, p; ball, …) -> Int

Cells within `ball` of the **point** `p`, and how many. A coordinate rather than the cell indices the
other methods take, which is what distinguishes them; it may be a `Tuple`, `NamedTuple`,
`AbstractVector` or `SVector`, as everywhere else a point is accepted.
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
    r = _knn_seed_radius_at(grid, p, kk)
    rmax = _knn_radius_ceiling(grid)
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
# widens as `k_nearest` does; tracking one best rather than a heap keeps it free of any buffer.
function Grids.locate(
    grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, p::NTuple{D,Real};
    active_only::Bool = false, topology = MetricTopology(grid), scratch = nothing,
) where {T,D}
    sz = grid isa Grids.UnstructuredGrid ? nothing : Grids.size_tuple(grid)
    r = _knn_seed_radius_at(grid, p, 1)
    rmax = _knn_radius_ceiling(grid)
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
@inline function _knn_seed_radius_at(grid::Grids.StructuredGrid{G,T,N}, p, k::Int) where {G,T,N}
    sz = Grids.size_tuple(grid)
    raw = Grids.locate(grid, ntuple(d -> T(p[d]), Val(N)))
    return _knn_seed_radius(grid, ntuple(d -> clamp(raw[d], 1, sz[d]), Val(N)), k)
end

# Elsewhere there are no axes to locate along, so the scale comes from the mean cell rather than the
# one the point fell in. Only the starting radius depends on it; the doubling reaches the rest.
@inline function _knn_seed_radius_at(grid::Grids.AbstractGrid{G,T}, p, k::Int) where {G,T}
    D = length(Grids.coordinates(grid))
    ncell = length(Grids.mask(grid))
    vol = one(T)
    for d in 1:D
        e = T(Grids.extent(grid, d))
        e > 0 && (vol *= e)
    end
    cell = ncell > 0 ? (vol / ncell)^(one(T) / D) : one(T)
    return T(1.5) * cell * T(max(k, 1))^(one(T) / D)
end

# ---------------------------------------------------------------------------
# k nearest
# ---------------------------------------------------------------------------

# A bounded max-heap over (distance, index), held in the caller's two buffers. Keeping the k smallest
# needs no sort of the candidate set and no allocation, and it is the same code for every architecture
# because it consumes `fold_within`.
# Ordered by (distance, index), so an equal-distance tie resolves the same way whatever order the
# traversal happened to visit in — which is what makes an indexed query and a scan agree exactly.
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

# A radius to start the search from: the cell's own size scaled to hold roughly `k` of them. Too small
# only costs a doubling, and the doubling is what makes the result independent of the guess.
@inline function _knn_seed_radius(grid::Grids.AbstractGrid{G,T}, I, k::Int) where {G,T}
    N = length(Grids.size_tuple(grid))
    m = T(Grids.measure(grid, I...))
    cell = m > 0 ? m^(one(T) / N) : one(T)
    return T(1.5) * cell * T(max(k, 1))^(one(T) / N)
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
    r = _knn_seed_radius(grid, Iraw, kk)
    rmax = _knn_radius_ceiling(grid)
    n = 0
    while true
        n = _cell_fold(0, grid, Ii, r, NearestImage(), active_only, false, topology, scratch, reach) do m, J, d
            return _heap_offer!(dist, idx, m, kk, d, _sweep_linear(grid, J))
        end
        (n ≥ kk || r ≥ rmax) && break
        r = min(r * 2, rmax)
    end
    _heap_sort!(dist, idx, n)
    return n
end

# One pair of methods per architecture, each with the arity in the signature: a `Vararg{Integer}` with
# no length parameter is not specialized on arity and allocates on every call.
for (GT, NP) in ((:(Grids.StructuredGrid{G,T,N}), true), (:(Grids.CurvilinearGrid{T,G,N}), true))
    @eval begin
        function k_nearest!(
            idx::AbstractVector{<:Integer}, dist::AbstractVector, grid::$GT, I::Vararg{Integer,N};
            k::Integer, active_only::Bool = true, topology = MetricTopology(grid),
            scratch = nothing, reach::AbstractReach = Unrestricted(),
        ) where {G,T,N}
            Ii = map(Int, I)
            return _k_nearest!(idx, dist, grid, Ii, Ii, Int(k), active_only, topology, scratch, reach)
        end
        function k_nearest(grid::$GT, I::Vararg{Integer,N}; k::Integer, kwargs...) where {G,T,N}
            kk = Int(k)
            idx = Vector{Int}(undef, kk)
            dist = Vector{T}(undef, kk)
            n = k_nearest!(idx, dist, grid, I...; k = kk, kwargs...)
            return resize!(idx, n), resize!(dist, n)
        end
    end
end

function k_nearest!(
    idx::AbstractVector{<:Integer}, dist::AbstractVector, grid::Grids.UnstructuredGrid, i::Integer;
    k::Integer, active_only::Bool = true, topology = MetricTopology(grid),
    scratch = nothing, reach::AbstractReach = Unrestricted(),
)
    ii = Int(i)
    return _k_nearest!(idx, dist, grid, ii, (ii,), Int(k), active_only, topology, scratch, reach)
end

function k_nearest(grid::Grids.UnstructuredGrid{T}, i::Integer; k::Integer, kwargs...) where {T}
    kk = Int(k)
    idx = Vector{Int}(undef, kk)
    dist = Vector{T}(undef, kk)
    n = k_nearest!(idx, dist, grid, i; k = kk, kwargs...)
    return resize!(idx, n), resize!(dist, n)
end

# Every cell is within this of every other, so the widening always terminates.
@inline function _knn_radius_ceiling(grid::Grids.AbstractGrid{G,T}) where {G<:Geometry.AbstractSphericalGeometry,T}
    return T(π) * T(Geometry.radius(Grids.grid_geometry(grid)))
end
@inline function _knn_radius_ceiling(grid::Grids.AbstractGrid{G,T}) where {G<:Geometry.AbstractEllipsoidalGeometry,T}
    return T(π) * T(Geometry.semimajor_axis(Grids.grid_geometry(grid)))
end
# Over the COORDINATE directions, which on a node set is not the index dimension: `size_tuple` there
# counts nodes, so using it would leave every direction but the first out of the diagonal.
@inline function _knn_radius_ceiling(grid::Grids.AbstractGrid{G,T}) where {G,T}
    D = length(Grids.coordinates(grid))
    s = zero(T)
    for d in 1:D
        e = T(Grids.extent(grid, d))
        s += e * e
    end
    return sqrt(s) + one(T)
end

# A node set indexes by one integer; the others by a tuple, including in one dimension.
@inline _sweep_linear(grid::Union{Grids.StructuredGrid,Grids.CurvilinearGrid}, J) =
    _linidx(Grids.size_tuple(grid), J...)
@inline _sweep_linear(::Grids.UnstructuredGrid, k::Integer) = Int(k)


# ---------------------------------------------------------------------------
# Sweeps — every cell's ball, with the per-grid work done once
# ---------------------------------------------------------------------------

@inline _sweep_cells(grid::Union{Grids.StructuredGrid,Grids.CurvilinearGrid}) =
    CartesianIndices(Grids.size_tuple(grid))
@inline _sweep_cells(grid::Grids.UnstructuredGrid) = Base.OneTo(length(Grids.mask(grid)))

@inline _sweep_index(::Union{Grids.StructuredGrid,Grids.CurvilinearGrid}, ci::CartesianIndex) = Tuple(ci)
@inline _sweep_index(::Grids.UnstructuredGrid, k::Integer) = Int(k)

# The per-cell traversal, with each architecture's own argument order.
@inline _cell_fold(f::F, init, grid::Grids.StructuredGrid, I, ball, images, active_only, self,
                   mt, _scratch, reach) where {F} =
    _route_fold(f, init, reach, grid, I, ball, images, active_only, self, mt)
@inline _cell_fold(f::F, init, grid::Grids.CurvilinearGrid, I, ball, images, active_only, self,
                   mt, scratch, reach) where {F} =
    _route_fold(f, init, reach, grid, I, ball, active_only, self, mt, scratch)
@inline _cell_fold(f::F, init, grid::Grids.UnstructuredGrid, I, ball, images, active_only, self,
                   mt, scratch, reach) where {F} =
    _route_fold(f, init, reach, grid, I, ball, active_only, self, mt, scratch)

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
    self::Bool = false, topology = default_sweep_topology(grid, ball),
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
            acc = _cell_fold(acc, grid, I, ball, images, active_only, self, topology, s, reach) do a, J, d
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
    self::Bool = false, topology = default_sweep_topology(grid, ball),
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
        _cell_fold(nothing, grid, I, ball, images, active_only, self, topology, nothing, reach) do _, J, d
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
            _cell_fold(nothing, grid, I, ball, images, active_only, self, topology, s, reach) do _, J, d
                f(I, J, d)
                return nothing
            end
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# build_connectivity
# ---------------------------------------------------------------------------

"""
    build_connectivity(grid; stencil=Axial(1), active_only=true) -> CSRConnectivity

Materialize CSR adjacency. Unstructured wraps existing buffers without re-validation.

For adjacency by physical distance rather than by stencil, see [`build_connectivity_within`](@ref).
"""
function build_connectivity end

build_connectivity(grid::Grids.UnstructuredGrid; _...) =
    csr_connectivity(Grids.neighbor_nbrs(grid), Grids.neighbor_ptr(grid); validate = false)

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
    grid::Grids.StructuredGrid{G,T,N}; ball, active_only::Bool = true, backend = nothing,
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
            deg[k] = _within_scan_curvilinear(nothing, grid, Tuple(ci[k]), ball, active_only, topology, s)
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
            _within_scan_curvilinear(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid, Tuple(ci[k]), ball,
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
            deg[k] = _within_scan_unstructured(nothing, grid, k, ball, active_only, topology, s)
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
            _within_scan_unstructured(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid, k, ball,
                                      active_only, topology, s)
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

# ---------------------------------------------------------------------------
# Dense adjacency — bang first; grid path fills from stencil (no CSR)
# ---------------------------------------------------------------------------

"""
    adjacency_matrix!(A, conn) -> A
    adjacency_matrix!(A, grid; stencil=Axial(1), active_only=true) -> A

Fill preallocated `N×N` `A`. Grid overload uses the stencil directly (no CSR alloc).
"""
function adjacency_matrix!(A::AbstractMatrix{Bool}, conn::CSRConnectivity)
    n = nnodes(conn)
    size(A) == (n, n) || throw(DimensionMismatch(
        "adjacency buffer size $(size(A)) != ($n, $n) = nnodes×nnodes",
    ))
    fill!(A, false)
    @inbounds for i in 1:n
        for j in Grids.neighbors(conn, i)
            A[i, j] = true
        end
    end
    return A
end

# Structured and curvilinear read only `(size, periodic, mask)` here, so they share one N-generic
# implementation through the topology, exactly as the neighbour queries above do.
adjacency_matrix!(
    A::AbstractMatrix{Bool}, grid::Union{Grids.StructuredGrid,Grids.CurvilinearGrid};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) = adjacency_matrix!(A, IndexTopology(grid); stencil = stencil, active_only = active_only)

function adjacency_matrix!(
    A::AbstractMatrix{Bool}, t::IndexTopology{N};
    stencil = Stencils.Axial(1), active_only::Bool = true,
) where {N}
    n = prod(t.size)
    size(A) == (n, n) || throw(DimensionMismatch(
        "adjacency buffer size $(size(A)) != ($n, $n)",
    ))
    fill!(A, false)
    sz = t.size
    per = t.periodic
    offs = _stencil_offsets(Val{N}(), _stencil_val(stencil))
    @inbounds for ci in CartesianIndices(sz)
        I = Tuple(ci)
        active_only && !_active(t, I...) && continue
        row = _linidx(sz, I...)
        for δ in offs
            J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
            any(==(0), J) && continue
            active_only && !_active(t, J...) && continue
            A[row, _linidx(sz, J...)] = true
        end
    end
    return A
end

function adjacency_matrix!(A::AbstractMatrix{Bool}, grid::Grids.UnstructuredGrid; _...)
    return adjacency_matrix!(A, build_connectivity(grid))
end

"""
    adjacency_matrix(grid_or_conn; kwargs...) -> Matrix{Bool}

Dense `n × n` adjacency over the `n` nodes.

**This allocates n² bytes**, which is quadratic in the node count and therefore quartic in the side
of a 2-D grid: a 1000×1000 grid has 10⁶ nodes and so needs ~10¹² bytes. Dense adjacency is for small
node counts and for testing. For anything else use [`sparse_adjacency_matrix`](@ref), which stores
`nedges` entries instead of `n²`, or [`neighbors!`](@ref), which answers neighbour queries from the
index stencil with no graph storage at all.
"""
adjacency_matrix(conn::CSRConnectivity) =
    adjacency_matrix!(Matrix{Bool}(undef, nnodes(conn), nnodes(conn)), conn)

adjacency_matrix(grid::Grids.AbstractGrid; kwargs...) =
    adjacency_matrix!(Matrix{Bool}(undef, length(Grids.mask(grid)), length(Grids.mask(grid))),
                      grid; kwargs...)

# ---------------------------------------------------------------------------
# SparseMatrixCSC — extension; COO bang in core for reuse
# ---------------------------------------------------------------------------

"""
    sparse_adjacency_coo!(I, J, conn) -> ne
    sparse_adjacency_coo!(I, J, V, conn) -> ne

Fill preallocated COO buffers (`length ≥ nedges(conn)`). `V`, if given, is set to `true`.
"""
function sparse_adjacency_coo!(I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}, conn::CSRConnectivity)
    ne = nedges(conn)
    length(I) ≥ ne && length(J) ≥ ne || throw(DimensionMismatch(
        "COO buffers need length ≥ nedges=$ne (got $(length(I)), $(length(J)))",
    ))
    k = 0
    n = nnodes(conn)
    @inbounds for i in 1:n
        for j in Grids.neighbors(conn, i)
            k += 1
            I[k] = i
            J[k] = j
        end
    end
    return k
end

function sparse_adjacency_coo!(
    I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}, V::AbstractVector{Bool}, conn::CSRConnectivity,
)
    ne = sparse_adjacency_coo!(I, J, conn)
    length(V) ≥ ne || throw(DimensionMismatch("V length must be ≥ nedges=$ne"))
    @inbounds for k in 1:ne
        V[k] = true
    end
    return ne
end

"""
    sort_neighbors!(conn) -> conn

Sort each node's neighbor block ascending, in place. Rows are stencil-short, so an insertion sort
per row is `O(nedges)` overall and needs no scratch.
"""
function sort_neighbors!(conn::CSRConnectivity)
    ptr, nbrs = conn.ptr, conn.nbrs
    @inbounds for i in 1:nnodes(conn)
        lo, hi = ptr[i], ptr[i + 1] - 1
        for a in (lo + 1):hi
            v = nbrs[a]
            b = a - 1
            while b ≥ lo && nbrs[b] > v
                nbrs[b + 1] = nbrs[b]
                b -= 1
            end
            nbrs[b + 1] = v
        end
    end
    return conn
end

"""
    is_symmetric_adjacency(conn) -> Bool

Whether `j ∈ N(i)` implies `i ∈ N(j)` throughout. `O(nedges + nnodes)`, by comparing the graph with its
transpose rather than searching a row per edge; guards the shortcut that reads a CSR as a CSC.
"""
function is_symmetric_adjacency(conn::CSRConnectivity)
    ptr, nbrs = conn.ptr, conn.nbrs
    n = nnodes(conn)
    n == 0 && return true
    # Transpose by counting sort, then compare each row against it. Searching row `j` for `i` per edge
    # instead is `O(nedges·degree)`, which a stencil graph hides (degree 6) and a ball graph does not
    # (degree in the hundreds).
    deg = zeros(Int, n)
    @inbounds for k in eachindex(nbrs)
        j = Int(nbrs[k])
        (1 ≤ j ≤ n) || return false
        deg[j] += 1
    end
    tptr = Vector{Int}(undef, n + 1)
    tptr[1] = 1
    @inbounds for i in 1:n
        tptr[i + 1] = tptr[i] + deg[i]
    end
    cursor = copy(tptr)
    tnbrs = Vector{Int}(undef, length(nbrs))
    @inbounds for i in 1:n
        for k in ptr[i]:(ptr[i + 1] - 1)
            j = Int(nbrs[k])
            tnbrs[cursor[j]] = i
            cursor[j] += 1
        end
    end
    # Row `i` of the transpose is exactly the set of nodes naming `i`. Equal degrees plus every
    # transpose entry present in row `i` means the two rows agree, which is the symmetry claim.
    stamp = zeros(Int, n)
    @inbounds for i in 1:n
        (ptr[i + 1] - ptr[i]) == (tptr[i + 1] - tptr[i]) || return false
        for k in ptr[i]:(ptr[i + 1] - 1)
            stamp[Int(nbrs[k])] = i
        end
        for k in tptr[i]:(tptr[i + 1] - 1)
            stamp[tnbrs[k]] == i || return false
        end
    end
    return true
end

"""
    sparse_adjacency_csc!(colptr, rowval, conn) -> nedges

Fill caller-owned CSC structure arrays (`length(colptr) ≥ nnodes+1`, `length(rowval) ≥ nedges`) for
the adjacency of `conn`, so that entry `(i, j)` is set iff `j` is a neighbor of `i`.

This is the direct route to a sparse matrix: CSR and CSC are the same layout transposed, so the
structure is obtained by one counting pass and one placement pass over the existing neighbor list —
no coordinate triples are materialized, nothing is sorted, and no permutation vector is built. Row
indices come out ascending within each column for free, because the placement pass walks nodes in
order. The running cursors live in `colptr` itself and are shifted back at the end, so this needs no
scratch beyond the two output arrays.
"""
function sparse_adjacency_csc!(
    colptr::AbstractVector{<:Integer}, rowval::AbstractVector{<:Integer}, conn::CSRConnectivity,
)
    n = nnodes(conn)
    ne = nedges(conn)
    length(colptr) ≥ n + 1 || throw(DimensionMismatch("colptr needs length ≥ nnodes+1 = $(n + 1)"))
    length(rowval) ≥ ne || throw(DimensionMismatch("rowval needs length ≥ nedges = $ne"))
    ptr, nbrs = conn.ptr, conn.nbrs

    @inbounds for j in 1:(n + 1)
        colptr[j] = 0
    end
    @inbounds for k in 1:ne
        colptr[nbrs[k] + 1] += 1
    end
    @inbounds colptr[1] = 1
    @inbounds for j in 1:n
        colptr[j + 1] += colptr[j]
    end
    # Place each (i → j) as row i of column j, advancing that column's cursor in place.
    @inbounds for i in 1:n
        for k in ptr[i]:(ptr[i + 1] - 1)
            j = nbrs[k]
            rowval[colptr[j]] = i
            colptr[j] += 1
        end
    end
    # Cursors now sit one past each column; shift them back into proper offsets.
    @inbounds for j in n:-1:1
        colptr[j + 1] = colptr[j]
    end
    @inbounds colptr[1] = 1
    return ne
end

"""
    sparse_adjacency_matrix(grid_or_conn; kwargs...) -> SparseMatrixCSC

Requires `using SparseArrays` (extension). Prefer [`sparse_adjacency_coo!`](@ref)
when reusing COO buffers.
"""
function sparse_adjacency_matrix end

"""
    sparse_adjacency_matrix!(colptr, rowval, nzval, conn) -> SparseMatrixCSC

Assemble the adjacency matrix into caller-owned buffers and wrap them without copying, so a repeated
build reuses storage instead of allocating a new matrix each time. Requires `using SparseArrays`.

The returned matrix ALIASES the three buffers, so reusing them for a later call invalidates any
matrix built from them earlier. Buffers longer than needed are trimmed to fit (`nnodes+1` and
`nedges`), since a matrix must own arrays of exactly the right length.
"""
function sparse_adjacency_matrix! end

# ---------------------------------------------------------------------------
# Mask topology
# ---------------------------------------------------------------------------

"""
    interior(grid; stencil = Stencils.Axial(1)) -> Array{Bool}

Which active cells have their whole stencil active and in range. `false` at a domain edge that does
not wrap, and beside any masked-out cell.
"""
function interior(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1))
    return _interior(IndexTopology(grid), _stencil_val(stencil))
end

function _interior(t::IndexTopology{N}, sten::Stencils.AbstractStencil) where {N}
    sz = t.size
    per = t.periodic
    offs = _stencil_offsets(Val{N}(), sten)
    out = fill(false, sz)
    @inbounds for ci in CartesianIndices(sz)
        I = Tuple(ci)
        _active(t, I...) || continue
        ok = true
        for δ in offs
            J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
            if any(==(0), J) || !_active(t, J...)
                ok = false
                break
            end
        end
        out[ci] = ok
    end
    return out
end

"""
    boundary_cells(grid; stencil = Stencils.Axial(1)) -> Array{Bool}

Which active cells are NOT [`interior`](@ref): the active cells that touch an edge or a masked-out
neighbour.
"""
function boundary_cells(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1))
    t = IndexTopology(grid)
    int = _interior(t, _stencil_val(stencil))
    out = similar(int)
    @inbounds for ci in CartesianIndices(t.size)
        out[ci] = _active(t, Tuple(ci)...) && !int[ci]
    end
    return out
end

"""
    connected_components(grid; stencil = Stencils.Axial(1), active = true) -> (labels, ncomponents)

Label the connected components of the active region (or of the inactive region with
`active = false`), by flood fill honouring the grid's own wrapping. `labels` is `0` off the region and
`1:ncomponents` on it.
"""
function connected_components(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1), active::Bool = true)
    return _connected_components(IndexTopology(grid), _stencil_val(stencil), active)
end

function _connected_components(
    t::IndexTopology{N}, sten::Stencils.AbstractStencil, want::Bool,
) where {N}
    sz = t.size
    per = t.periodic
    offs = _stencil_offsets(Val{N}(), sten)
    labels = zeros(Int, sz)
    ncomp = 0
    stack = CartesianIndex{N}[]
    @inbounds for seed in CartesianIndices(sz)
        (_active(t, Tuple(seed)...) == want && labels[seed] == 0) || continue
        ncomp += 1
        empty!(stack)
        push!(stack, seed)
        labels[seed] = ncomp
        while !isempty(stack)
            I = Tuple(pop!(stack))
            for δ in offs
                J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
                any(==(0), J) && continue
                cj = CartesianIndex(J)
                (labels[cj] == 0 && _active(t, J...) == want) || continue
                labels[cj] = ncomp
                push!(stack, cj)
            end
        end
    end
    return labels, ncomp
end

"""
    count_holes(grid; stencil = Stencils.Axial(1)) -> Int

How many connected inactive regions are fully enclosed by active cells — the number of holes in the
active region, and so an estimate of its first Betti number.

A region that reaches a non-wrapping edge is outside rather than enclosed. Along a wrapping direction
there is no edge to reach, so enclosure there is decided by the fill alone.
"""
function count_holes(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1))
    t = IndexTopology(grid)
    labels, ncomp = _connected_components(t, _stencil_val(stencil), false)
    ncomp == 0 && return 0
    sz = t.size
    per = t.periodic
    N = length(sz)
    touches = falses(ncomp)
    @inbounds for ci in CartesianIndices(sz)
        l = labels[ci]
        l == 0 && continue
        I = Tuple(ci)
        for d in 1:N
            per[d] && continue
            (I[d] == 1 || I[d] == sz[d]) && (touches[l] = true)
        end
    end
    return count(!, touches)
end

include("ConnectivitySpherical.jl")

end # module
