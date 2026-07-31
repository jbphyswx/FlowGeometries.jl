module Connectivity

using ..Execution: Execution
using ..Geometry: Geometry
using ..Stencils: Stencils
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
@inline function _min_step(grid::Grids.StructuredGrid{G,T}, d::Int) where {G,T}
    s = T(Grids.minimum_spacing(grid, d))
    if Grids.isperiodic(grid, d)
        x = Grids.coordinates(grid, d)
        p = T(Grids.period(grid, d))
        if length(x) ≥ 2 && p > 0
            seam = p - abs(T(last(x)) - T(first(x)))
            seam > 0 && (s = min(s, seam))
        end
    end
    return s
end

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
where the direction wraps — which is one number on a uniform axis and an `O(N)` scan on a stretched
one. On a spherical or ellipsoidal grid the longitude cut additionally walks the latitude window for
its smallest `cosφ`, so it costs the window it returns.

This is a *bound*, not the answer: it is the window [`neighbors_within!`](@ref) scans before filtering on
the geometry's own `distance`. It never under-covers, which is why it is geometry-specific. On a
spherical grid one longitude step spans `R·cosφ·Δλ`, so the λ half-width is taken at the latitude in the
window nearest a pole rather than at the cell's own latitude — at a polar cell every longitude is in
range, and the window says so.
"""
function metric_window end

function metric_window(
    grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Integer}, ball,
) where {G<:Geometry.AbstractCartesianGeometry, T, N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    return ntuple(d -> _steps(r, _min_step(grid, d), sz[d]), Val(N))
end

# Spherical `(λ, φ, r, …)`, distance the great-circle arc (2-D) or the chord (3-D). Both are bounded
# below by the chord, so one angular cut serves both. Solved in dependency order: the radial window
# fixes the smallest radius `ρ` the chords are taken at (chord ≥ |Δr|, so cells beyond the radial
# window are already excluded); the latitude cut follows (central angle ≥ |Δφ|); and the λ cut uses the
# smallest `cosφ` in the latitude window (`sin(σ/2) ≥ cosφ·sin(|Δλ|/2)` with both endpoint latitudes in
# that window).
function metric_window(
    grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Integer}, ball,
) where {G<:Geometry.AbstractSphericalGeometry, T, N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    wrest = ntuple(d -> _steps(r, _min_step(grid, d + 2), sz[d + 2]), Val(max(N - 2, 0)))
    ρ = if N ≥ 3
        _window_min(abs, Grids.coordinates(grid, 3), Int(I[3]), wrest[1], per[3])
    else
        T(Geometry.radius(Grids.grid_geometry(grid)))
    end
    wφ = N ≥ 2 ? _angle_steps(r, 2ρ, _min_step(grid, 2), sz[2]) : 0
    cosmin = N ≥ 2 ?
        _window_min(φ -> abs(cos(φ)), Grids.coordinates(grid, 2), Int(I[2]), wφ, per[2]) : one(T)
    wλ = _angle_steps(r, 2ρ * cosmin, _min_step(grid, 1), sz[1])
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
    grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Integer}, ball,
) where {G<:Geometry.AbstractEllipsoidalGeometry, T, N}
    geo = Grids.grid_geometry(grid)
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    per = _periodic_flags(grid)
    a = T(Geometry.semimajor_axis(geo))
    M0 = a * (one(T) - T(Geometry.eccentricity²(geo)))   # M(0) = b²/a, the smallest curvature radius
    wrest = ntuple(d -> _steps(r, _min_step(grid, d + 2), sz[d + 2]), Val(max(N - 2, 0)))
    if N ≥ 3
        hmin = minimum(Grids.coordinates(grid, 3))
        wφ = _steps(r, (M0 + hmin - r) * _min_step(grid, 2), sz[2])
        cosmin = _window_min(φ -> abs(cos(φ)), Grids.coordinates(grid, 2), Int(I[2]), wφ, per[2])
        wλ = _angle_steps(r, 2 * (M0 + min(hmin, zero(T))) * cosmin, _min_step(grid, 1), sz[1])
        return ntuple(d -> d == 1 ? wλ : d == 2 ? wφ : wrest[d - 2], Val(N))
    end
    wφ = N ≥ 2 ? _steps(r, M0 * _min_step(grid, 2), sz[2]) : 0
    cosreach = if N ≥ 2
        φ0 = T(@inbounds Grids.coordinates(grid, 2)[I[2]])
        reach = r / M0
        lo, hi = max(φ0 - reach, -T(π) / 2), min(φ0 + reach, T(π) / 2)
        lo ≤ 0 ≤ hi ? min(cos(lo), cos(hi)) : min(abs(cos(lo)), abs(cos(hi)))
    else
        one(T)
    end
    wλ = _steps(r, a * cosreach * _min_step(grid, 1), sz[1])
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
@inline function _image_step(grid::Grids.StructuredGrid{G,T}, d::Int) where {G,T}
    s = _min_step(grid, d)
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
    grid::Grids.StructuredGrid{G,T,N}, ball,
) where {G<:Geometry.AbstractCartesianGeometry, T, N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    return ntuple(Val(N)) do d
        s = _image_step(grid, d)
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

@inline _ball_window(grid, I, ball, ::NearestImage) = metric_window(grid, I, ball)
@inline _ball_window(grid, _I, ball, ::AllImages) = _image_window(grid, ball)

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
    neighbors_within!(out, grid, I...; ball, active_only=true) -> n_written

Write the linear indices of every cell whose centre lies within `ball` of cell `I`. `ball` is a
[`Stencils.MetricBall`](@ref) or a bare radius in the geometry's length units.

Distance is the geometry's own — great-circle on a sphere, Vincenty on a spheroid, the chord where a
third direction is present — so the neighbourhood is a genuine metric ball, not a box. The cell itself
is excluded, matching stencil semantics where the zero offset is not a neighbour. A periodic direction
wraps, each cell appears at most once, and its coordinate is taken by minimum image, so the seam
neither shortens nor lengthens a distance.

Cost is [`metric_window`](@ref) — `O(1)` per direction on a uniform axis — times one distance evaluation
per candidate. Size the buffer with [`nneighbors_within`](@ref); there is no fixed count, since how many
cells fall within a fixed distance varies from cell to cell on any non-uniform or curved grid.
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
"""
function fold_within end

@inline function _fold_within(
    f::F, init, grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Int}, ball,
    images::AbstractImageConvention, active_only::Bool, self::Bool,
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
    w = _ball_window(grid, I, ball, images)
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
fold_within(
    f::F, init, grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    ball, images::AbstractImageConvention = NearestImage(),
    active_only::Bool = true, self::Bool = false,
) where {F,G,T,N} = _fold_within(f, init, grid, map(Int, I), ball, images, active_only, self)

# The counting and writing forms are the one-image fold with a counting/appending step: one traversal, so
# the window bound and the distance gate cannot drift between conventions.
@inline function _within_scan(
    out, grid::Grids.StructuredGrid{G,T,N}, I::NTuple{N,Int}, ball, active_only::Bool,
) where {G,T,N}
    sz = Grids.size_tuple(grid)
    return _fold_within(0, grid, I, ball, NearestImage(), active_only, false) do n, J, _
        m = n + 1
        out === nothing || _keep!(out, m, _linidx(sz, J...))
        return m
    end
end

neighbors_within!(
    out::AbstractVector{<:Integer}, grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true,
) where {G,T,N} = _within_scan(out, grid, map(Int, I), ball, active_only)

nneighbors_within(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N}; ball, active_only::Bool = true,
) where {G,T,N} = _within_scan(nothing, grid, map(Int, I), ball, active_only)

function neighbors_within(
    grid::Grids.StructuredGrid{G,T,N}, I::Vararg{Integer,N}; ball, active_only::Bool = true,
) where {G,T,N}
    Ii = map(Int, I)
    out = Vector{Int}(undef, _within_scan(nothing, grid, Ii, ball, active_only))
    _within_scan(out, grid, Ii, ball, active_only)
    return out
end

# ---- Curvilinear and unstructured -------------------------------------------
# Neither has separable axes, so no index window bounds the ball and the scan is over every cell. Exact,
# and `O(n)` per query: to query many cells of a large grid, materialize `build_connectivity_within`
# once instead.

@inline function _keep!(out, n::Int, lin::Int)
    n ≤ length(out) ||
        throw(ArgumentError("out too short for this ball (need ≥ $n; size it with nneighbors_within)"))
    @inbounds out[n] = lin
    return nothing
end

function _within_scan_curvilinear(
    out, grid::Grids.CurvilinearGrid{T,G,N}, I::NTuple{N,Int}, ball, active_only::Bool,
) where {T,G,N}
    sz = Grids.size_tuple(grid)
    @inbounds for d in 1:N
        (1 ≤ I[d] ≤ sz[d]) || throw(BoundsError(Grids.mask(grid), I))
    end
    (active_only && !Grids.isactive(grid, I...)) && return 0
    geo = Grids.grid_geometry(grid)
    r = _ball_radius(ball)
    p0 = Grids._raw_coords(grid, I...)
    prd = map(x -> oftype(first(p0), x), _wrap_lengths(grid, Val(N)))
    msk = Grids.mask(grid)
    n = 0
    @inbounds for ci in CartesianIndices(sz)
        J = Tuple(ci)
        J == I && continue
        active_only && !msk[J...] && continue
        q = _min_image(p0, Grids._raw_coords(grid, J...), prd)
        Geometry.distance(geo, p0, q) ≤ r || continue
        n += 1
        out === nothing || _keep!(out, n, _linidx(sz, J...))
    end
    return n
end

function _within_scan_unstructured(
    out, grid::Grids.UnstructuredGrid{T,G,N}, idx::Int, ball, active_only::Bool,
) where {T,G,N}
    msk = Grids.mask(grid)
    1 ≤ idx ≤ length(msk) || throw(BoundsError(msk, idx))
    (active_only && !Grids.isactive(grid, idx)) && return 0
    geo = Grids.grid_geometry(grid)
    r = _ball_radius(ball)
    p0 = Grids._raw_coords(grid, idx)
    prd = map(x -> oftype(first(p0), x), _wrap_lengths(grid, Val(N)))
    n = 0
    @inbounds for k in eachindex(msk)
        k == idx && continue
        active_only && !msk[k] && continue
        q = _min_image(p0, Grids._raw_coords(grid, k), prd)
        Geometry.distance(geo, p0, q) ≤ r || continue
        n += 1
        out === nothing || _keep!(out, n, k)
    end
    return n
end

nneighbors_within(
    grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N}; ball, active_only::Bool = true,
) where {T,G,N} = _within_scan_curvilinear(nothing, grid, map(Int, I), ball, active_only)
neighbors_within!(
    out::AbstractVector{<:Integer}, grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N};
    ball, active_only::Bool = true,
) where {T,G,N} = _within_scan_curvilinear(out, grid, map(Int, I), ball, active_only)
function neighbors_within(
    grid::Grids.CurvilinearGrid{T,G,N}, I::Vararg{Integer,N}; ball, active_only::Bool = true,
) where {T,G,N}
    Ii = map(Int, I)
    out = Vector{Int}(undef, _within_scan_curvilinear(nothing, grid, Ii, ball, active_only))
    _within_scan_curvilinear(out, grid, Ii, ball, active_only)
    return out
end

nneighbors_within(grid::Grids.UnstructuredGrid, idx::Integer; ball, active_only::Bool = true) =
    _within_scan_unstructured(nothing, grid, Int(idx), ball, active_only)
neighbors_within!(
    out::AbstractVector{<:Integer}, grid::Grids.UnstructuredGrid, idx::Integer;
    ball, active_only::Bool = true,
) = _within_scan_unstructured(out, grid, Int(idx), ball, active_only)
function neighbors_within(grid::Grids.UnstructuredGrid, idx::Integer; ball, active_only::Bool = true)
    out = Vector{Int}(undef, nneighbors_within(grid, idx; ball, active_only))
    _within_scan_unstructured(out, grid, Int(idx), ball, active_only)
    return out
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

Symmetric by construction, since the metric is. On scattered points use the k-d-tree
`unstructured_grid` constructor's `radius`, which builds exactly this.
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
    Execution.run_chunks(n, backend) do rng
        @inbounds for k in rng
            deg[k] = _within_scan(nothing, grid, Tuple(ci[k]), ball, active_only)
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
            _within_scan(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid, Tuple(ci[k]), ball, active_only)
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
    Execution.run_chunks(n, backend) do rng
        @inbounds for k in rng
            I = Tuple(ci[k])
            active_only && !_active(t, I...) && continue
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
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    Execution.run_chunks(n, backend) do rng
        @inbounds for k in rng
            I = Tuple(ci[k])
            active_only && !_active(t, I...) && continue
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

# No index window bounds a ball on curvilinear coordinates, so each row is a full scan and the build is
# O(n²) — done once, which is the point of materializing it.
function build_connectivity_within(
    grid::Grids.CurvilinearGrid; ball, active_only::Bool = true, backend = nothing,
)
    sz = Grids.size_tuple(grid)
    n = prod(sz)
    ci = CartesianIndices(sz)
    deg = zeros(Int, n)
    Execution.run_chunks(n, backend) do rng
        @inbounds for k in rng
            deg[k] = _within_scan_curvilinear(nothing, grid, Tuple(ci[k]), ball, active_only)
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
            _within_scan_curvilinear(view(nbrs, ptr[k]:(ptr[k + 1] - 1)), grid, Tuple(ci[k]), ball,
                                     active_only)
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

Whether `j ∈ N(i)` implies `i ∈ N(j)` throughout. `O(nedges·degree)`; guards the shortcut that reads
a CSR as a CSC.
"""
function is_symmetric_adjacency(conn::CSRConnectivity)
    ptr, nbrs = conn.ptr, conn.nbrs
    @inbounds for i in 1:nnodes(conn)
        for k in ptr[i]:(ptr[i + 1] - 1)
            j = nbrs[k]
            found = false
            for q in ptr[j]:(ptr[j + 1] - 1)
                if nbrs[q] == i
                    found = true
                    break
                end
            end
            found || return false
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
