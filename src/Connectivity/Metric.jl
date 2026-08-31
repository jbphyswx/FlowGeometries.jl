# ---------------------------------------------------------------------------
# Metric neighbourhoods — queries by physical distance
# ---------------------------------------------------------------------------
#
# A distance query needs coordinates and the geometry's own `distance`, so these take the grid; a
# stencil query reads only `(size, periodic, mask)`. A [`Stencils.MetricBall`](@ref) has no fixed
# offset set: how many cells lie within a given distance varies from cell to cell.

@inline _ball_radius(b::Stencils.MetricBall) = Stencils.radius(b)
@inline _ball_radius(r::Real) = r ≥ 0 ? r : throw(ArgumentError("a search radius must be ≥ 0, got $r"))

# Index half-width covering physical distance `r` when one index step spans at least `s`: a cell more
# than `w` steps out is then at least `w·s ≥ r` away. Clamped to the axis. A degenerate direction —
# one sample, or no positive spacing — opens to the full axis.
@inline function _steps(r::T, s::T, n::Int) where {T<:AbstractFloat}
    (s > 0 && isfinite(s)) || return n
    w = ceil(r / s)
    return w ≥ n ? n : Int(w)
end

# The chord between points at radii ≥ ρ separated by central angle σ is ≥ 2ρ·c·sin(σ/2), with `c` the
# `cosφ` attenuation (1 where none applies). Inverted, that is the angle beyond which every cell is
# farther than `r` in the chord metric, and so in the arc metric too since arc ≥ chord. Returns `n`
# where no angle is far enough, as when `r` reaches the antipode.
@inline function _angle_steps(r::T, scale::T, Δ::T, n::Int) where {T<:AbstractFloat}
    (scale > 0 && isfinite(scale)) || return n
    x = r / scale
    x < 1 || return n
    return _steps(one(T), Δ / (2 * asin(x)), n)   # w = ceil(σ_cut / Δ), through the same clamp
end

# The smallest interior gap of direction `d`, including the seam gap where it wraps: a periodic axis's
# seam can be its narrowest gap, and a window bound omitting it under-covers across the seam.
#
# `Grids.minimum_spacing` is an `O(N)` scan on a stretched axis, so this is computed once into a
# `MetricTopology` and read from there by every query.
@inline function _min_step_scan(grid::Grids.AbstractGrid{G,T}, d::Int) where {G,T}
    s = T(Grids.minimum_spacing(grid, d))
    if Grids.isperiodic(grid, d)
        p = T(Grids.period(grid, d))
        if @inbounds(Grids.size_tuple(grid)[d]) ≥ 2 && p > 0
            seam = p - T(Grids.extent(grid, d))
            seam > 0 && (s = min(s, seam))
        end
    end
    return s
end

"""
    MetricTopology(grid; index = nothing)

What a distance query reads that depends on the grid alone: the tightest per-direction step bound,
which sizes the search window, the coordinate span, and a spatial index where the layout has no
separable axes to bound with. [`IndexTopology`](@ref) is its counterpart for a stencil query.

Constructing one is `O(1)` on a rectilinear grid, so passing `topology` to a query is optional. The
**index** is worth hoisting: building a k-d tree for a single query costs more than the scan it
replaces, so it is left out by default. [`foreach_within`](@ref) and [`mapreduce_within`](@ref) hoist
one for a sweep, and [`indexed`](@ref) builds one explicitly.

Grid types are immutable and the `Adapt` extension reconstructs them field by field for a device, so
this is a separate value and never a cache field on the grid.
"""
struct MetricTopology{N,T,S}
    steps::NTuple{N,T}   # smallest index step per direction, seam included
    min3::T              # global minimum of direction 3 (geodetic height); zero below 3 directions
    span::NTuple{N,T}    # coordinate extent per direction; see `_span_of`
    index::S
end

# The per-direction extent, reduced once here and read by every query. On a rectilinear grid it is two
# endpoint reads; off one the coordinates are per-cell fields with no order, so it is a scan, and a
# `k`-nearest query needs it to size its opening radius.
@inline _span_of(grid::Grids.AbstractGrid{G,T}, ::Val{N}) where {G,T,N} =
    ntuple(d -> T(Grids.extent(grid, d)), Val(N))

MetricTopology(grid::Grids.AbstractGrid; index = nothing) =
    MetricTopology(grid, index, Grids.candidate_source(grid))

function MetricTopology(grid::Grids.AbstractGrid{G,T}, index, ::Grids.SeparableWindow) where {G,T}
    N = Grids.ncoordinates(grid)
    steps = ntuple(d -> _min_step_scan(grid, d), Val(N))
    m3 = N ≥ 3 ? T(Grids.bounds(grid, 3)[1]) : zero(T)
    return MetricTopology{N,T,typeof(index)}(steps, m3, _span_of(grid, Val(N)), index)
end

# With no separable axes there is no step bound to carry, and the topology holds only the index.
function MetricTopology(grid::Grids.AbstractGrid{G,T}, index, ::Grids.IndexedCandidates) where {G,T}
    N = Grids.ncoordinates(grid)
    return MetricTopology{N,T,typeof(index)}(ntuple(_ -> zero(T), Val(N)), zero(T),
                                             _span_of(grid, Val(N)), index)
end

@inline _min_step(mt::MetricTopology, d::Int) = @inbounds mt.steps[d]

"""
    _buffered_candidates(index) -> Bool

Whether an index has to materialize a candidate list to be queried.

A cell list folds its bins directly and a bare scan has nothing to buffer, so both answer a query with
no per-task storage, and a sweep over them runs per index, on a device included. A tree deduplicates
the periodic images it searches over, so it needs a buffer, and its sweep needs chunks to give each
task one.
"""
@inline _buffered_candidates(::Nothing) = false
@inline _buffered_candidates(::Grids.CellListIndex) = false
@inline _buffered_candidates(_index) = true

# An index built over the active region alone is a superset of the active ball only. A query for
# masked cells needs one built over every cell, and raises where it has the wrong one.
@inline _check_index_covers(_index, _active_only::Bool) = nothing
@inline function _check_index_covers(ix::Grids.CellListIndex, active_only::Bool)
    (active_only || !ix.active_only) || throw(ArgumentError(
        "this index was built with `active_only = true`, so it holds no masked cell and cannot answer " *
        "a query for one. Build it with `cell_list(grid; ball, active_only = false)`.",
    ))
    return nothing
end

"""
    indexed(grid) -> MetricTopology

A [`MetricTopology`](@ref) carrying a spatial index, which brings a ball query to `O(log n + m)`.

Requires `NearestNeighbors`; [`Grids.spatial_index`](@ref) raises without it, and an unindexed
topology answers the same queries in `O(n)`.
"""
indexed(grid::Grids.AbstractGrid) = MetricTopology(grid; index = Grids.spatial_index(grid))

# Candidate enumeration. Without an index every cell is a candidate; with one, the index returns a
# superset of the ball and the caller's exact gate does the rest.
#
# `scratch` is the index's candidate buffer and belongs to the caller. One `MetricTopology` is shared
# by every task in a threaded sweep, so it stays read-only.
@inline _candidates(mt::MetricTopology, grid, I, r, scratch) =
    _index_candidates(mt.index, grid, I, r, scratch)

@inline _index_candidates(::Nothing, grid, _I, _r, _scratch) = Base.OneTo(length(Grids.mask(grid)))

# `scratch === nothing` is a type test, so the branch resolves at compile time.
@inline _index_candidates(index, grid, I, r, scratch) =
    scratch === nothing ? Grids.index_within(index, grid, I, r) :
                          Grids.index_within!(scratch, index, grid, I, r)

# Threads the accumulator through the candidates as they are enumerated. An index that can enumerate
# without materializing defines `Grids.fold_candidates`; the traversal then holds no buffer and runs
# inside a kernel.
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

A candidate buffer to hand to repeated ball queries through their `scratch` argument. One buffer per
task; an indexed query then holds one allocation across any number of calls.
"""
ball_scratch() = Int[]

# The smallest `f(x[j])` over the index window, bounding a scale factor that varies across it. Walks
# the clamped window; a periodic direction can reach every sample, so it walks all of them.
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

The window is a bound: [`neighbors_within!`](@ref) scans it and then filters on the geometry's own
`distance`. Covering the ball for every geometry is what makes it geometry-specific. On a spherical grid
one longitude step spans `R·cosφ·Δλ`, so the λ half-width is taken at the latitude in the window nearest
a pole; at a polar cell every longitude is in range, and the window says so.
"""
function metric_window end

metric_window(grid::Grids.AbstractStructuredGrid, I::NTuple, ball) =
    metric_window(grid, I, ball, MetricTopology(grid))

# A radius, however it is spelled. The grid-level forms below take this type to keep them apart from
# the per-cell forms, whose second argument is the cell index.
const _BallLike = Union{Real,Stencils.MetricBall}

"""
    metric_window(grid, ball) -> NTuple{N,Int}
    metric_window(grid, ball, topology) -> NTuple{N,Int}

The per-cell window maximised over every cell of `grid`, for sizing a cache or a footprint table.

`O(1)`: [`Grids.minimum_spacing`](@ref) gives the smallest gap per axis, and the extreme `|cos φ|` over
a latitude axis is at one of its two ends, since `|cos|` on `[-π/2, π/2]` peaks in the middle. No `cos`
per row, and no scan.

The result is the per-cell window at the worst cell, so it covers the ball at every cell.
"""
metric_window(grid::Grids.AbstractStructuredGrid, ball::_BallLike) =
    metric_window(grid, ball, MetricTopology(grid))

function metric_window(
    grid::Grids.AbstractStructuredGrid{G,T}, ball::_BallLike, mt::MetricTopology{N,T},
) where {G<:Geometry.AbstractCartesianGeometry,T,N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    return ntuple(d -> _steps(r, _min_step(mt, d), sz[d]), Val(N))   # already point-independent
end

function metric_window(
    grid::Grids.AbstractStructuredGrid{G,T}, ball::_BallLike, mt::MetricTopology{N,T},
) where {G<:Geometry.AbstractSphericalGeometry,T,N}
    r = T(_ball_radius(ball))
    sz = Grids.size_tuple(grid)
    wrest = ntuple(d -> _steps(r, _min_step(mt, d + 2), sz[d + 2]), Val(max(N - 2, 0)))
    # The smallest radius anywhere, which is where a given arc spans the least distance.
    ρ = if N ≥ 3
        lo, hi = Grids.bounds(grid, 3)
        min(abs(T(lo)), abs(T(hi)))
    else
        T(Geometry.radius(Grids.grid_geometry(grid)))
    end
    wφ = N ≥ 2 ? _angle_steps(r, 2ρ, _min_step(mt, 2), sz[2]) : 0
    wλ = _angle_steps(r, 2ρ * _cos_extreme(grid, Val(N)), _min_step(mt, 1), sz[1])
    return ntuple(d -> d == 1 ? wλ : d == 2 ? wφ : wrest[d - 2], Val(N))
end

function metric_window(
    grid::Grids.AbstractStructuredGrid{G,T}, ball::_BallLike, mt::MetricTopology{N,T},
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
@inline function _cos_extreme(grid::Grids.AbstractStructuredGrid{G,T}, ::Val{N}) where {G,T,N}
    N ≥ 2 || return one(T)
    lo, hi = Grids.bounds(grid, 2)
    return min(abs(cos(T(lo))), abs(cos(T(hi))))
end


"""
    metric_band(grid, dim, coord_t, coord_n, ball) -> T

The **exact** half-width along direction `dim` of the part of the row at `coord_n` that lies within
`ball` of a point at `coord_t`, in that direction's own coordinate units. `coord_t` and `coord_n` are
coordinates on the *other* direction of a two-direction grid.

[`metric_window`](@ref) returns a bounding box, which suits a query that then filters on distance. A
**separable sweep** — a prefix sum along a row, a row-by-row convolution — has no filtering step and
needs the exact extent. This is the same geodesic solve, resolved per row.

Returns a negative number where the row is out of reach entirely, so `band < 0` is the empty test. A
row the ball covers completely gives the half-width of the whole direction (`π` in longitude).

On a sphere, for `dim = 1`, this inverts the spherical law of cosines:

```math
|Δλ| ≤ \\arccos\\left(\\frac{\\cos(r/R) - \\sin φ_t \\sin φ_n}{\\cos φ_t \\cos φ_n}\\right)
```

The empty band, the full circle and a pole at either end all fall out of that one expression. At a pole
the denominator vanishes and the separation stops depending on `λ`; this returns the full half-width
there, so no caller handles it.
"""
function metric_band end

function metric_band(
    grid::Grids.AbstractStructuredGrid{G,T}, dim::Integer, coord_t::Real, coord_n::Real, ball,
) where {G<:Geometry.AbstractCartesianGeometry,T}
    r = T(_ball_radius(ball))
    d = abs(T(coord_n) - T(coord_t))
    d > r && return -one(T)                       # the row is farther than the ball reaches
    return sqrt(r * r - d * d)                    # a circle's half-chord at that offset, exactly
end

function metric_band(
    grid::Grids.AbstractStructuredGrid{G,T}, dim::Integer, coord_t::Real, coord_n::Real, ball,
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
    grid::Grids.AbstractStructuredGrid{G,T}, I::NTuple{N,Integer}, ball, mt::MetricTopology,
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
    grid::Grids.AbstractStructuredGrid{G,T}, I::NTuple{N,Integer}, ball, mt::MetricTopology,
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
    grid::Grids.AbstractStructuredGrid{G,T}, I::NTuple{N,Integer}, ball, mt::MetricTopology,
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

Singleton types, like a stencil. The traversal branches on this per candidate; as a type the branch and
the coordinate expression behind it resolve at compile time, and the walk allocates nothing.
"""
abstract type AbstractImageConvention end

"""
    NearestImage()

Visit each cell at most once, at its nearest image. The neighbour-set convention, and the default.
"""
struct NearestImage <: AbstractImageConvention end

"""
    AllImages()

Visit every image of a cell that lands inside the ball, each carrying its own displacement. This is what
a periodic convolution sums over. See [`fold_within`](@ref).
"""
struct AllImages <: AbstractImageConvention end

"""
    AbstractReach

Which of the cells within `ball` a query returns: [`Unrestricted`](@ref), every one of them, or
[`Connected`](@ref), those reachable from the seed without leaving the ball.

A type, like [`AbstractImageConvention`](@ref). The two are different sets computed by different
algorithms, so the choice is visible in the call and fixed at compile time.
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
farther than `r` stops at `Q` and never reaches `R`, though `R` is inside the ball. Such a walk computes
`Connected`; the ball is a different set, reached only by a spatial query.
"""
struct Connected{S} <: AbstractReach
    adjacency::S
end
Connected() = Connected(nothing)

@inline _reach_stencil(::Connected{Nothing}) = Stencils.Axial(1)
@inline _reach_stencil(c::Connected) = _stencil_val(c.adjacency)

# Per-direction candidate offsets. Bounded: the window, clipped at the walls. Periodic under
# `NearestImage`: the window clamped to one full turn, since offsets congruent mod `n` are the same cell
# and minimum image fixes the coordinate it is seen at. Periodic under `AllImages`: no clamp, since each
# offset is then a distinct position of that cell.
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
@inline function _image_step(grid::Grids.AbstractStructuredGrid{G,T}, d::Int, mt::MetricTopology) where {G,T}
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
    grid::Grids.AbstractStructuredGrid{G,T}, ball, mt::MetricTopology{N,T},
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

# Summing images treats a periodic direction as a translation of the domain, as it is on a Cartesian
# torus: `x` and `x+L` are distinct positions of the same cell, and a convolution counts each. An angular
# direction is an identification: `λ` and `λ+2π` are one point, the geometry's distance is already
# `2π`-periodic in it, and its images coincide. Summing them counts one cell repeatedly, so it raises.
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
# `x[mod1(raw, n)] + (periods wrapped)·L`, the same construction `Discretization.axis_stencils` uses for
# a wrapped stencil node. Minimum image is not applied to it.
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

using ..Grids: _min_image, _wrap_lengths

"""
    neighbors_within!(out, grid, I...; ball, active_only=true, topology, scratch, reach) -> n_written

Write the linear indices of every cell whose centre lies within `ball` of cell `I`. `ball` is a
[`Stencils.MetricBall`](@ref) or a bare radius in the geometry's length units.

Distance is the geometry's own — great-circle on a sphere, Vincenty on a spheroid, the chord where a
third direction is present — so the neighbourhood is a metric ball in that distance. The cell itself is
excluded, matching stencil semantics where the zero offset is no neighbour. A periodic direction
wraps, each cell appears at most once, and its coordinate is taken by minimum image, so the seam
neither shortens nor lengthens a distance.

Cost is [`metric_window`](@ref) — `O(1)` per direction on any separable axis, uniform or stretched,
given the per-direction minimum steps that `topology` carries — times one distance evaluation per
candidate. Size the buffer with [`nneighbors_within`](@ref): on a non-uniform or curved grid the number
of cells within a fixed distance varies from cell to cell.

Three arguments matter for anything beyond a single query:

  * `topology` — a [`MetricTopology`](@ref), the grid invariants a ball query reads. The default is `O(1)`
    and allocation-free. On a curvilinear or node grid, pass [`indexed`](@ref) to bring each query to
    `O(log n + m)`, or use [`foreach_within`](@ref), which builds the index once for a whole sweep.
  * `scratch` — a candidate buffer from [`ball_scratch`](@ref), one per task. Accepted on every grid type
    and used where a query goes through a spatial index, i.e. on a curvilinear or node grid; a separable
    window has no candidate list to buffer. With one, an indexed query allocates nothing; without one it
    allocates its candidate list per call.
  * `reach` — [`Unrestricted`](@ref) (the ball, and the default) or [`Connected`](@ref) (the part of it
    reachable from `I` without leaving it). A ball is not a connected patch: with a mask or a concave
    domain it can contain cells reachable from the seed only by going outside `ball`.
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
traversal every distance query here is built on; the accumulator is threaded through as a value, so
nothing is captured and nothing is boxed.

`images` selects how a periodic direction is treated — [`NearestImage`](@ref) (the default, each cell
once, the convention [`neighbors_within!`](@ref) exposes) or [`AllImages`](@ref), which visits every
image of a cell that lands inside the ball, each carrying its own displacement.

`AllImages` is what a periodic convolution needs: on a torus of period `L`,
`f̄(x) = Σₖ ∫ K(x − y − kL) f(y) dy`, so where the kernel support exceeds `L/2` one cell contributes
through several images at different displacements, and keeping only the nearest drops the rest. Below
`L/2` the two conventions coincide exactly. It also widens the search: the window becomes the uncapped
`ceil(r/s)` per periodic direction, in place of [`metric_window`](@ref)'s one-turn cap. It raises where
a periodic direction is an angular identification.

`self = true` also folds the centre cell, at distance zero. The default excludes it, matching a
neighbour set; a convolution needs it, and it carries the kernel's largest weight.

`reach` selects the ball ([`Unrestricted`](@ref)) or the part of it reachable from the seed without
leaving it ([`Connected`](@ref)); `topology` and `scratch` are as in [`neighbors_within!`](@ref).
"""
function fold_within end

# `_fold_within` takes the same arguments on every layout — `images` and `scratch` included, each ignored
# where it has no meaning — so `_route_fold` passes them through unexamined. Which body runs is
# `Grids.candidate_source`.
@inline _fold_within(
    f::F, init, grid::Grids.AbstractGrid, I, ball, images::AbstractImageConvention,
    active_only::Bool, self::Bool, mt::MetricTopology, scratch = nothing,
) where {F} =
    _fold_within(f, init, grid, I, ball, images, active_only, self, mt, scratch,
                 Grids.candidate_source(grid))

# `N` is bound by the cell tuple: a layout with separable axes names its cells by an index tuple, so the
# tuple's own length fixes every `Val(N)` below at compile time.
@inline function _fold_within(
    f::F, init, grid::Grids.AbstractGrid{G,T}, I::NTuple{N,Int}, ball,
    images::AbstractImageConvention, active_only::Bool, self::Bool, mt::MetricTopology, _scratch,
    ::Grids.SeparableWindow,
) where {F,G,T,N}
    sz = Grids.size_tuple(grid)
    Grids._cell_checkbounds(grid, I)
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

# `f::F` forces a specialization: Julia does not specialize on a function-typed argument that is only
# passed through, and the inner call then goes dynamic and boxes the grid, the index tuple and the radius.
# Every entry point takes the same arguments on every layout — a separable window enumerates candidates
# without a buffer, so `scratch` reaches it and goes unused.
fold_within(
    f::F, init, grid::Grids.AbstractGrid, I::Vararg{Integer,NI};
    ball, images::AbstractImageConvention = NearestImage(),
    active_only::Bool = true, self::Bool = false, topology = MetricTopology(grid),
    reach::AbstractReach = Unrestricted(), scratch = nothing,
) where {F,NI} =
    _route_fold(f, init, reach, grid, _seed_cell(grid, I), ball, images, active_only, self,
                topology, scratch)

# The counting and writing forms are the one-image fold with a counting/appending step: one traversal, so
# the window bound and the distance gate cannot drift between them. `out === nothing` counts.
@inline function _within_scan(
    out, grid::Grids.AbstractGrid, I, ball, active_only::Bool,
    mt::MetricTopology = MetricTopology(grid), scratch = nothing,
    reach::AbstractReach = Unrestricted(),
)
    return _route_fold(0, reach, grid, I, ball, NearestImage(), active_only, false, mt,
                       scratch) do n, J, _
        m = n + 1
        out === nothing || _keep!(out, m, _sweep_linear(grid, J))
        return m
    end
end

# Appending fold, for the allocating `neighbors_within` where a counting pass costs a second index query
# or a second full scan.
@inline function _within_push(
    grid::Grids.AbstractGrid, I, ball, active_only::Bool, mt::MetricTopology, scratch,
    reach::AbstractReach,
)
    return _route_fold(Int[], reach, grid, I, ball, NearestImage(), active_only, false, mt,
                       scratch) do v, J, _
        push!(v, _sweep_linear(grid, J))
        return v
    end
end

# The arity stays a type parameter on every entry point below: a `Vararg{Integer}` with no length is not
# specialized on arity and allocates on every call.
@inline _seed_cell(grid::Grids.AbstractGrid, I::Tuple{Vararg{Integer}}) = Grids._cell_named_by(grid, I)

neighbors_within!(
    out::AbstractVector{<:Integer}, grid::Grids.AbstractGrid, I::Vararg{Integer,NI};
    ball, active_only::Bool = true, topology = MetricTopology(grid),
    reach::AbstractReach = Unrestricted(), scratch = nothing,
) where {NI} =
    _within_scan(out, grid, _seed_cell(grid, I), ball, active_only, topology, scratch, reach)

nneighbors_within(
    grid::Grids.AbstractGrid, I::Vararg{Integer,NI};
    ball, active_only::Bool = true, topology = MetricTopology(grid),
    reach::AbstractReach = Unrestricted(), scratch = nothing,
) where {NI} =
    _within_scan(nothing, grid, _seed_cell(grid, I), ball, active_only, topology, scratch, reach)

function neighbors_within(
    grid::Grids.AbstractGrid, I::Vararg{Integer,NI};
    ball, active_only::Bool = true, topology = MetricTopology(grid),
    reach::AbstractReach = Unrestricted(), scratch = nothing,
) where {NI}
    cell = _seed_cell(grid, I)
    # `Connected` materializes its component to walk it, and an indexed query pays a second range query,
    # so both append in one pass. A separable window is cheap to count, so it sizes the output exactly.
    if reach isa Connected || Grids.candidate_source(grid) isa Grids.IndexedCandidates
        return _within_push(grid, cell, ball, active_only, topology, scratch, reach)
    end
    n = _within_scan(nothing, grid, cell, ball, active_only, topology, scratch, reach)
    out = Vector{Int}(undef, n)
    _within_scan(out, grid, cell, ball, active_only, topology, scratch, reach)
    return out
end

# ---- Curvilinear and unstructured -------------------------------------------
# Neither has separable axes, so `metric_window` has nothing to bound with. Two ways to enumerate
# candidates, and the fold below is written so they cannot disagree:
#
#   * no index — every cell, `O(n)` per query.
#   * a spatial index — a range query, `O(log n + m)`.
#
# The index only has to return a superset of the ball. Membership is decided by the exact `distance ≤ r`
# gate below, which is the same code on both paths, so the indexed and unindexed results agree.

@inline function _keep!(out, n::Int, lin::Int)
    n ≤ length(out) ||
        throw(ArgumentError("out too short for this ball (need ≥ $n; size it with nneighbors_within)"))
    @inbounds out[n] = lin
    return nothing
end

# A neighbour list is a set of cells in whatever order enumerated them: a window walks index order, a
# tree walks tree order, a cell list walks bin order. No entry point sorts, which would put an
# `O(m log m)` pass on an `O(m)` query. `sort_neighbors!` sorts on request.

# One body for every layout with no separable axes to bound a window with — a curvilinear mesh and a node
# set alike, which differ only in how a cell is named (`Grids.cell_address`) — so their enumeration and
# distance gate are the same code.
#
# A non-nearest image convention raises here: summing images needs a periodic direction to be a
# translation of the domain, and these layouts visit each cell once.
@inline function _fold_within(
    f::F, init, grid::Grids.AbstractGrid, I, ball, images::AbstractImageConvention,
    active_only::Bool, self::Bool, mt::MetricTopology, scratch, ::Grids.IndexedCandidates,
) where {F}
    images isa NearestImage || throw(ArgumentError(
        "`images = $(images)` needs a separable periodic direction; $(nameof(typeof(grid))) visits " *
        "each cell once, so use NearestImage()",
    ))
    _check_index_covers(mt.index, active_only)
    Grids._cell_checkbounds(grid, I)
    (active_only && !Grids._cell_active(grid, I)) && return init
    geo = Grids.grid_geometry(grid)
    r = _ball_radius(ball)
    p0 = Grids._cell_coords(grid, I)
    prd = map(x -> oftype(first(p0), x), _wrap_lengths(grid, Val(length(p0))))
    acc = init
    self && (acc = f(acc, I, zero(eltype(p0))))
    return _fold_candidates(acc, mt, grid, I, r, scratch) do a, lin
        J = Grids._cell_from_linear(grid, lin)
        J == I && return a
        active_only && !Grids._cell_active(grid, J) && return a
        q = _min_image(p0, Grids._cell_coords(grid, J), prd)
        dist = Geometry.distance(geo, p0, q)
        dist ≤ r || return a
        return f(a, J, dist)
    end
end
