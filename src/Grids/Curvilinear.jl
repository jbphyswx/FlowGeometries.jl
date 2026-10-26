# ---------------------------------------------------------------------------
# Curvilinear Grid
# ---------------------------------------------------------------------------

"""
    CurvilinearGrid{T, G, N, TP, C, KC, MA, B}

Curvilinear grid whose cell-center coordinates are `N`-dimensional arrays (e.g. an orthogonal
curvilinear mesh). `coordinates` holds one `N`-D cell-center array per direction.

At `N = 2` the cell `measure` is the exact quadrilateral area through the cell's four vertices. Those
vertex arrays are what `corners` holds, one per direction and one larger in every direction. They are
retained when the caller supplies them or asks for them with `keep_corners = true`, and are otherwise
construction input to the area kernel alone — so `corners` holds empty arrays on a grid built from
centres, and [`has_corners`](@ref) reports which. In any dimension other than 2 the measure is the
caller's to supply: the corner-area kernel is a 2-D algorithm with no N-D generalization here.

# Type parameters
- `T`: coordinate float type. `G<:AbstractGeometry{T}` is tied to it, so a mismatched-eltype geometry
  raises a type error. `T` therefore precedes `G`: Julia binds type parameters left to right, and
  `G<:AbstractGeometry{T}` requires `T` already bound.
- `N`: number of coordinate directions.
- `C`: tuple type of the center coordinate arrays.
- `KC`: tuple type of the cell-vertex arrays, empty where they were not retained.
- `MA`: array type of the derived `measure` field — independent of `C`, since it is a computed field
  with no reason to match the coordinate arrays' storage type.
- `B`: array type of the active `mask`.
"""
struct CurvilinearGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractGeometry{T},
    N,
    TP<:NTuple{N,AbstractTopology},
    C<:NTuple{N,AbstractArray{T,N}},
    KC<:NTuple{N,AbstractArray{T,N}},
    MA<:AbstractArray{T,N},
    B<:AbstractArray{Bool,N},
} <: AbstractCurvilinearGrid{G, T}
    geometry::G
    coordinates::C            # cell-center coordinate array per direction
    corners::KC               # cell-vertex arrays, one larger in each direction, or empty
    measure::MA               # cell measure
    mask::B                   # active mask (true = active/included)
    topology::TP              # per-direction closure (singletons: no storage)
    period::NTuple{N,T}       # wrap length per direction; meaningless where Bounded
    bounds::NTuple{N,Tuple{T,T}}   # (lo, hi) of each coordinate array, taken at construction
    total_measure::T          # Σ measure, taken at construction
    spans::NTuple{N,T}        # each direction's vertex span (see `_vertex_span`), taken at construction
end

# `bounds`, `total_measure` and `spans` travel as values: a rebuild serves a storage change, which leaves
# every coordinate and the measure unchanged.
@inline _from_fields(
    ::Type{<:CurvilinearGrid},
    geometry::G, coordinates::C, corners::KC, measure::MA, mask::B, topology::TP,
    period::NTuple{N,T}, bounds::NTuple{N,Tuple{T,T}}, total_measure::T, spans::NTuple{N,T},
) where {T,G<:Geometry.AbstractGeometry{T},N,TP,C,KC,MA,B} =
    CurvilinearGrid{T,G,N,TP,C,KC,MA,B}(geometry, coordinates, corners, measure, mask, topology,
                                        period, bounds, total_measure, spans)

@inline topology(grid::CurvilinearGrid) = getfield(grid, :topology)
@inline period(grid::CurvilinearGrid, d::Integer) =
    @inbounds getfield(grid, :period)[_checked_direction(getfield(grid, :period), d)]
@inline bounds(grid::CurvilinearGrid, d::Integer) =
    @inbounds getfield(grid, :bounds)[_checked_direction(getfield(grid, :bounds), d)]
@inline _total_measure(grid::CurvilinearGrid) = getfield(grid, :total_measure)

@inline function _raw_coords(
    grid::CurvilinearGrid{T,G,N}, I::Vararg{Integer,N},
) where {T,G,N}
    c = coordinates(grid)
    @boundscheck checkbounds(c[1], I...)
    return ntuple(d -> @inbounds(c[d][I...]), Val(N))
end

"""
    corners(grid::CurvilinearGrid) -> NTuple{N,AbstractArray}
    corners(grid::CurvilinearGrid, d::Integer) -> AbstractArray

The cell-vertex coordinate arrays — one larger than [`coordinates`](@ref) in every direction, and in
the same direction order.

Available on a grid that was given them or built with `keep_corners = true`; see
[`has_corners`](@ref).
"""
@inline function corners(grid::CurvilinearGrid)
    has_corners(grid) || _no_corners_error()
    return getfield(grid, :corners)
end
@inline corners(grid::CurvilinearGrid, d::Integer) = @inbounds corners(grid)[d]

@noinline _no_corners_error() = throw(ArgumentError(
    "this CurvilinearGrid holds no cell-vertex arrays: they were construction input to the cell-area " *
    "kernel. Rebuild it with `keep_corners = true`, or pass `corners` explicitly.",
))

"""
    has_corners(grid) -> Bool

Whether `grid` retained its cell-vertex arrays, and so can answer [`corners`](@ref) and
[`corner_coords`](@ref).
"""
@inline has_corners(grid::CurvilinearGrid) = !isempty(@inbounds getfield(grid, :corners)[1])
@inline has_corners(::AbstractGrid) = false

"""
    corner_coords(grid::CurvilinearGrid, I...) -> NamedTuple
    corner_coords(S, grid::CurvilinearGrid, I...) -> S

Vertex `I` of the cell-vertex array, named by the geometry exactly as [`coords`](@ref) names
cell centers.
"""
@inline function corner_coords(
    grid::CurvilinearGrid{T,G,N}, I::Vararg{Integer,N},
) where {T,G,N}
    return Geometry.named_point(grid_geometry(grid), _raw_corner_coords(grid, I...))
end

@inline function corner_coords(
    ::Type{S}, grid::CurvilinearGrid{T,G,N}, I::Vararg{Integer,N},
) where {S,T,G,N}
    return Geometry.build_point(S, coordinate_names(grid), _raw_corner_coords(grid, I...))
end

@inline function _raw_corner_coords(
    grid::CurvilinearGrid{T,G,N}, I::Vararg{Integer,N},
) where {T,G,N}
    k = corners(grid)
    @boundscheck checkbounds(k[1], I...)
    return ntuple(d -> @inbounds(k[d][I...]), Val(N))
end

function _bounded_domain_length(grid::CurvilinearGrid, d::Integer)
    L = @inbounds getfield(grid, :spans)[_checked_direction(getfield(grid, :spans), d)]
    isnan(L) && throw(ArgumentError(
        "this CurvilinearGrid has fewer than 2 centers across some direction, so no cell vertices to " *
        "span; build it with explicit `corners`",
    ))
    return L
end

"""
    _vertex_span(C, K) -> T

`maximum - minimum` over a direction's cell vertices: the corner array `K` where it is held, and otherwise
the vertices [`_centers_to_corners`](@ref) reconstructs from the centres `C`, reduced as each is formed.
`NaN` where `K` is empty and `C` has fewer than 2 centers across some direction.
"""
function _vertex_span(C::AbstractArray{T,N}, K::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    if !isempty(K)
        lo, hi = extrema(K)
        return hi - lo
    end
    all(≥(2), size(C)) || return T(NaN)
    lo, hi = T(Inf), T(-Inf)
    @inbounds for ci in CartesianIndices(size(C) .+ 1)
        v = _corner_value(C, Tuple(ci))
        lo = min(lo, v)
        hi = max(hi, v)
    end
    return hi - lo
end

# ---------------------------------------------------------------------------
# Curvilinear grid construction: corner-based exact quadrilateral cell areas
# ---------------------------------------------------------------------------

# Adapt a coordinate matrix to element type `T`, preserving the concrete array type (`similar`, not
# `Matrix{T}`) and copying only when the eltype actually differs.
_to_arr(::Type{T}, A::AbstractArray{T}) where {T<:AbstractFloat} = A
_to_arr(::Type{T}, A::AbstractArray) where {T<:AbstractFloat} = copyto!(similar(A, T), A)

"""
    _centers_to_corners(C) -> K

Reconstruct a cell-vertex array one larger in every direction from the `N`-D cell-center array `C`, by
averaging the (up to `2^N`) surrounding centers with a linearly-extrapolated one-cell ghost ring, so
the true domain-boundary vertices land a half-cell outside the outermost centers. Used only when the
caller does not supply explicit corner arrays; requires at least 2 centers across every direction.

Unlike the corner-area kernel this is dimension-generic: it is a multilinear midpoint, and the ghost
ring is a per-direction linear extension of it.
"""
function _centers_to_corners(C::AbstractArray{T,N}) where {T<:AbstractFloat, N}
    all(≥(2), size(C)) || throw(ArgumentError(
        "auto-deriving curvilinear cell corners needs at least 2 centers across every direction " *
        "(got $(size(C))); supply the `corners` arrays explicitly for a smaller grid",
    ))
    # `_ghosted` returns the value a padded copy holds at an index, so the ghost ring stays implicit:
    # the vertex pass reads straight from `C` in one pass and allocates only the result.
    K = similar(C, T, (size(C) .+ 1)...)
    @inbounds for ci in CartesianIndices(K)
        K[ci] = _corner_value(C, Tuple(ci))
    end
    return K
end

# Vertex `I` of the reconstruction: the mean of the `2^N` ghosted centers around it.
@inline function _corner_value(C::AbstractArray{T,N}, I::NTuple{N,Int}) where {T,N}
    acc = zero(T)
    @inbounds for s in CartesianIndices(ntuple(_ -> 0:1, Val(N)))
        acc += _ghosted(C, ntuple(d -> I[d] - 1 + s[d], Val(N)))
    end
    return acc * (one(T) / T(2^N))
end

# Value of the one-cell linearly-extrapolated ghost ring around `C` at the (possibly out-of-range)
# center index `I`. Each out-of-range direction contributes its own linear extrapolation `2·edge − inner`
# taken with every other direction clamped, and where several are out at once those contributions add
# with the shared clamped value counted once — the standard halo fill, exact for a field linear in each
# direction. The flat loop keeps it inlinable; a recursion over directions does not inline, and its
# intermediate index tuples box.
@inline function _ghosted(C::AbstractArray{T,N}, I::NTuple{N,Int}) where {T,N}
    sz = size(C)
    cl = ntuple(d -> clamp(I[d], 1, sz[d]), Val(N))
    nout = 0
    acc = zero(T)
    @inbounds begin
        for d in 1:N
            (1 ≤ I[d] ≤ sz[d]) && continue
            nout += 1
            inner = I[d] < 1 ? 2 : sz[d] - 1
            acc += T(2) * C[cl...] - C[ntuple(k -> k == d ? inner : cl[k], Val(N))...]
        end
        nout == 0 && return C[cl...]
        return acc - T(nout - 1) * C[cl...]
    end
end



# Exact planar quadrilateral area (shoelace) over the (Nx+1)×(Ny+1) corner arrays.
function _corner_areas(
    ::G, xc::AbstractMatrix{T}, yc::AbstractMatrix{T}, Nx::Integer, Ny::Integer;
    execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {T<:AbstractFloat, G<:Geometry.AbstractCartesianGeometry{T}}
    areas = similar(xc, T, Nx, Ny)
    Execution.run_chunks(Int(Ny), execution) do rows
    @inbounds for j in rows, i in 1:Nx
        # Cell (i,j) has vertices (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1) (counter-clockwise in index space).
        x1 = xc[i, j];         y1 = yc[i, j]
        x2 = xc[i+1, j];       y2 = yc[i+1, j]
        x3 = xc[i+1, j+1];     y3 = yc[i+1, j+1]
        x4 = xc[i, j+1];       y4 = yc[i, j+1]
        areas[i, j] = T(0.5) * abs(x1 * (y2 - y4) + x2 * (y3 - y1) + x3 * (y4 - y2) + x4 * (y1 - y3))
    end
    end
    return areas
end

function _fill_dir_row!(
    buf::AbstractVector{NTuple{3,T}}, λc::AbstractMatrix{T}, φc::AbstractMatrix{T}, j::Int, nk::Int,
) where {T}
    @inbounds for i in 1:nk
        sinλ, cosλ = sincos(λc[i, j])
        sinφ, cosφ = sincos(φc[i, j])
        buf[i] = (cosφ * cosλ, cosφ * sinλ, sinφ)
    end
    return buf
end

# Exact spherical quadrilateral area, as the two triangles (p1,p2,p3) and (p1,p3,p4).
#
# Each corner's unit vector is computed once and reused: every interior vertex is shared by four cells
# and, within a cell, by both triangles. Two rows are live at a time — row j is finished as soon as
# row j+1's cells are done — so the buffer holds two rows, `O(Nx)`. The three edges from `p1` are formed
# from the corner coordinates, so a cell's area keeps their relative precision whatever its size.
function _corner_areas(
    geometry::G, λc::AbstractMatrix{T}, φc::AbstractMatrix{T}, Nx::Integer, Ny::Integer;
    execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {T<:AbstractFloat, G<:Geometry.AbstractSphericalGeometry{T}}
    nk = Nx + 1
    R2 = Geometry.radius(geometry)^2
    areas = similar(λc, T, Nx, Ny)
    # Chunked over rows, each chunk with its own two-row buffer. A chunk re-derives its first row, at
    # one extra row of trig per chunk, which keeps the chunks independent.
    Execution.run_chunks(Int(Ny), execution) do rows
        lo = Vector{NTuple{3,T}}(undef, nk)
        hi = Vector{NTuple{3,T}}(undef, nk)
        _fill_dir_row!(lo, λc, φc, first(rows), nk)
        @inbounds for j in rows
            _fill_dir_row!(hi, λc, φc, j + 1, nk)
            for i in 1:Nx
                d1 = lo[i]; d2 = lo[i + 1]; d3 = hi[i + 1]; d4 = hi[i]
                p1 = (λc[i, j], φc[i, j])
                e2 = Geometry._lonlat_chord(p1, (λc[i + 1, j], φc[i + 1, j]))
                e3 = Geometry._lonlat_chord(p1, (λc[i + 1, j + 1], φc[i + 1, j + 1]))
                e4 = Geometry._lonlat_chord(p1, (λc[i, j + 1], φc[i, j + 1]))
                areas[i, j] = R2 * (Geometry._excess(d1, d2, d3, e2, e3) + Geometry._excess(d1, d3, d4, e3, e4))
            end
            lo, hi = hi, lo      # row j+1 becomes the next cell row's lower edge
        end
    end
    return areas
end

"""
    CurvilinearGrid(geometry, coords..., mask; measure=nothing, corners=nothing, …)
    CurvilinearGrid(geometry, coords..., measure, mask; corners=nothing, …)

Build a curvilinear grid in **any** number of directions from one `N`-D cell-center coordinate array
per direction. `mask` is the trailing array; a `measure` array before it is used verbatim (common when
a dataset ships its own cell areas), and may equally be given as the `measure` keyword.

With no measure supplied, one is computed from the cell-vertex arrays — **at `N = 2` only**, as the
exact quadrilateral cell area. Spherical cells use the exact spherical-quadrilateral area, the
spherical excess of the two triangles through the cell's four corner directions (see
[`Geometry.spherical_excess`](@ref)); Cartesian cells the exact planar shoelace area. That kernel is a
2-D algorithm with no N-D generalization here, so in any other dimension the measure must be given.

Pass `corners` (a tuple of arrays, each one larger than the centers in every direction) for exact
cell vertices, e.g. from the source mesh's own vertex grid; otherwise they are reconstructed from the
centers per direction (see [`_centers_to_corners`](@ref)), which requires at least 2 cells across.

Corners supplied this way are kept on the grid, and reconstructed ones are input to the area kernel —
pass `keep_corners = true` to keep those too, at one array per direction. Reconstruction happens only
where something needs it, so a grid given its own `measure` derives none.

`topology` is per-direction closure, as for [`StructuredGrid`](@ref): [`Periodic`](@ref)/[`Bounded`](@ref)
instances, a tuple of them, a `Bool`, or a tuple of `Bool`s, a single value or a short tuple applying to
the leading directions. When omitted, every direction is bounded.
"""
function CurvilinearGrid(geometry::Geometry.AbstractGeometry, args...; kwargs...)
    coords, measure, mask = _split_curvilinear_args(args)
    return _curvilinear_grid(geometry, coords, measure, mask; kwargs...)
end

# Optional trailing `mask`, optionally preceded by a `measure`; everything before them is a coordinate
# array. The mask is recognised by its element type, so omitting it is unambiguous. A fully active grid
# omits it: `AllActive` costs one size tuple, where a dense all-true mask costs a load and a branch per
# cell.
function _split_curvilinear_args(args::Tuple)
    hasmask = !isempty(args) && last(args) isa AbstractArray{Bool}
    mask = hasmask ? last(args) : nothing
    rest = hasmask ? Base.front(args) : args
    isempty(rest) && throw(ArgumentError("a CurvilinearGrid needs at least one coordinate array"))
    # A trailing real array is the measure when there is one more array than each is dimensional:
    # `N` coordinates plus it. Otherwise it is the last coordinate.
    if length(rest) ≥ 2 && last(rest) isa AbstractArray{<:Real} &&
       !(last(rest) isa AbstractArray{Bool}) && length(rest) - 1 == ndims(last(rest))
        return (Base.front(rest), last(rest), mask)
    end
    return (rest, nothing, mask)
end

# No mask means every cell participates, which `AllActive` expresses as a size. This resolves in its own
# method: a `nothing` branch inside the one below leaves `mask` a small union and costs the whole
# constructor its specialization.
_curvilinear_grid(
    geometry::Geometry.AbstractGeometry, coords::NTuple{N,AbstractArray}, measure_pos, ::Nothing;
    kwargs...,
) where {N} = _curvilinear_grid(geometry, coords, measure_pos, AllActive(size(first(coords)));
                                kwargs...)

function _curvilinear_grid(
    geometry::G, coords::NTuple{N,AbstractArray}, measure_pos, mask::AbstractArray{Bool,N};
    corners = nothing, measure = nothing, keep_corners::Bool = false,
    topology = nothing, period = nothing,
    execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {N, T<:AbstractFloat, G<:Geometry.AbstractGeometry{T}}
    N == ndims(mask) || throw(ArgumentError(
        "got $N coordinate arrays for a $(ndims(mask))-dimensional mask",
    ))
    all(c -> ndims(c) == N, coords) || throw(ArgumentError(
        "every coordinate array must be $N-dimensional, got $(map(ndims, coords))",
    ))
    all(c -> size(c) == size(mask), coords) || throw(ArgumentError(
        "coordinate arrays and mask must have the same size; got $(map(size, coords)) and $(size(mask))",
    ))
    centers = ntuple(d -> _to_arr(T, coords[d]), Val(N))
    # The vertex, area and period builders walk the centres element by element, so they read host
    # copies; what they derive is held in the centres' own storage.
    hc = map(Axes._host_array, centers)

    m = measure_pos === nothing ? measure : measure_pos
    # Vertex arrays are derived only where something asks for them: the area kernel, or the caller.
    kc = corners === nothing ? nothing : _given_corners(T, corners, size(mask))
    kh = kc === nothing ? _derived_corners(hc, keep_corners || (m === nothing && N == 2)) :
                          map(Axes._host_array, kc)

    meas = if m !== nothing
        size(m) == size(mask) || throw(ArgumentError(
            "measure size $(size(m)) does not match the coordinate arrays' $(size(mask))",
        ))
        _to_arr(T, m)
    elseif N == 2
        Axes._like(centers[1], _corner_areas(geometry, kh[1], kh[2], size(mask, 1), size(mask, 2);
                                             execution = execution))
    else
        throw(ArgumentError(
            "a $N-dimensional CurvilinearGrid has no measure to derive: the corner-area kernel is a " *
            "2-D algorithm (exact quadrilateral area) with no N-D form here. Pass the cell measure " *
            "explicitly.",
        ))
    end

    tp = topology === nothing ? ntuple(_ -> Bounded(), Val(N)) : _as_topology(Val(N), topology)
    prd = _curvilinear_periods(geometry, hc, tp, period)
    # The caller's own vertex arrays are theirs to keep; ones derived here are the area kernel's input,
    # and are retained on request.
    held = kc !== nothing ? kc : keep_corners ? map(Axes._like, centers, kh) : _no_corners(centers)
    return CurvilinearGrid{T, G, N, typeof(tp), typeof(centers), typeof(held), typeof(meas),
                           typeof(mask)}(
        geometry, centers, held, meas, mask, tp, prd, map(_lohi, hc), T(sum(meas)),
        map(_vertex_span, hc, kh),
    )
end

# Vertex arrays reconstructed from the centres, or empty arrays of the same type.
_derived_corners(centers::Tuple, derive::Bool) = derive ? map(_centers_to_corners, centers) : _no_corners(centers)
_no_corners(centers::Tuple) = map(c -> similar(c, eltype(c), map(_ -> 0, size(c))), centers)

function _given_corners(::Type{T}, corners, sz::NTuple{N,Int}) where {T,N}
    length(corners) == N || throw(ArgumentError("expected $N corner arrays, got $(length(corners))"))
    want = sz .+ 1
    return ntuple(Val(N)) do d
        k = _to_arr(T, corners[d])
        size(k) == want || throw(ArgumentError(
            "corner array $d must be $(want) (one larger than the centers in every direction); got $(size(k))",
        ))
        k
    end
end

@inline _lohi(x::AbstractArray{T}) where {T} = isempty(x) ? (T(Inf), T(-Inf)) : extrema(x)

"""
    rotate(grid::StructuredGrid, rot) -> RotatedGrid
    unrotate(grid::StructuredGrid, rot) -> RotatedGrid

The same mesh with its coordinates expressed in the other frame of [`Geometry.PoleRotation`](@ref)
`rot` — `unrotate` being the usual direction, taking a rotated-pole grid's rectilinear `(λ′, φ′)` axes
to the geographic coordinates of each cell.

A rotated lat–lon mesh is logically rectangular and geometrically warped, and only its own frame's axes
are separable. The warping is a formula, so the result stores the mesh and the rotation and evaluates a
cell's position where it is asked for; see [`RotatedGrid`](@ref). Rotating a grid therefore costs one
`PoleRotation`, against two centre arrays, two corner arrays and a dense measure to materialize it.

Two things carry over. The **cell measure** is exact under a rotation, an isometry of the sphere, so it
is shared; recomputing it from rotated corners adds round-off. The **index topology** carries over too:
the same mesh has the same neighbours, so a direction that wrapped still wraps, and longitude remains
an angle mod `2π` in either frame.

To difference along the mesh's own axes, ask [`base_grid`](@ref) for them: the frame changes where a
cell is reported, leaving the lattice as it is.
"""
rotate(grid::AbstractStructuredGrid, rot::Geometry.PoleRotation) =
    RotatedGrid(grid, rot, Geometry.rotate)

"""
    unrotate(grid::StructuredGrid, rot) -> RotatedGrid

[`rotate`](@ref) the other way: the mesh's axes are the rotated frame's, and each cell is reported at
its geographic position. The usual direction for a rotated-pole grid.
"""
unrotate(grid::AbstractStructuredGrid, rot::Geometry.PoleRotation) =
    RotatedGrid(grid, rot, Geometry.unrotate)
