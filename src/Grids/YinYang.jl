# ---------------------------------------------------------------------------
# Yin–Yang
# ---------------------------------------------------------------------------

"""
    YinYangGrid(geometry, nlon, nlat; mask = nothing)

The Kageyama–Sato Yin–Yang grid: two `nlon × nlat` lat–lon panels covering
`[-3π/4, 3π/4] × [-π/4, π/4]` in their own frames, `2·nlon·nlat` cells in all.

It stores `nlon`, `nlat`, the geometry, the mask and the `nlon × nlat` table of cell measures, which
both panels share. In its own frame each panel is a separable lat–lon patch, and yang is yin rigidly
rotated, so a cell's coordinates are the panel formula on yin and the formula composed with
[`SphericalSampling._yin_yang_rotate`](@ref) on yang, longitude in `[0, 2π)` on both.

Cells are numbered yin first with `i` fastest, then yang the same way, matching
[`SphericalSampling.spherical_points`](@ref)`(YinYangSampling(), nlon, nlat)`. [`panel_cell`](@ref) and
[`cell_id`](@ref) convert between a cell id and its `(panel, i, j)`.

!!! note "The panels overlap"
    The two panels cover a band of the sphere twice, 6.07% of it at every resolution, so their cell
    areas sum to `3√2πR²`. A cell's [`measure`](@ref) is its area less half the part of it the other
    panel also covers, so the measures sum to `4πR²` and `Σ f·measure` integrates over the sphere,
    every point counted once. The panels are not cross-linked: a cell's neighbours are its own panel's,
    the standard Yin–Yang discrete topology, the panels coupling through interpolation.
"""
struct YinYangGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    B<:AbstractVector{Bool},
    M<:AbstractMatrix{T},
} <: AbstractGrid{G,T}
    geometry::G
    nlon::Int
    nlat::Int
    mask::B
    cell_measure::M    # `nlon × nlat`, the same on both panels; see `measure`
end

function YinYangGrid(
    geometry::Geometry.AbstractSphericalGeometry{T}, nlon::Integer, nlat::Integer; mask = nothing,
) where {T<:AbstractFloat}
    nl = Int(nlon)
    nt = Int(nlat)
    (nl ≥ 1 && nt ≥ 1) || throw(ArgumentError("Yin–Yang nlon/nlat must be ≥ 1, got $nl and $nt"))
    ncell = 2 * nl * nt
    m = mask === nothing ? AllActive((ncell,)) : mask
    length(m) == ncell || throw(DimensionMismatch(
        "mask holds $(length(m)) entries for a grid of $ncell cells",
    ))
    cm = _yin_yang_measures(T(Geometry.radius(geometry)), nl, nt)
    return YinYangGrid{T,typeof(geometry),typeof(m),typeof(cm)}(geometry, nl, nt, m, cm)
end

YinYangGrid(nlon::Integer, nlat::Integer; kwargs...) =
    YinYangGrid(Geometry.SphericalGeometry(), nlon, nlat; kwargs...)

@inline _from_fields(
    ::Type{<:YinYangGrid},
    geometry::G, nlon::Int, nlat::Int, mask::B, cell_measure::M,
) where {T,G<:Geometry.AbstractSphericalGeometry{T},B,M} =
    YinYangGrid{T,G,B,M}(geometry, nlon, nlat, mask, cell_measure)

@inline npanels(::YinYangGrid) = 2

@inline sampling(::YinYangGrid) = SphericalSampling.YinYangSampling()

"""
    panel_shape(grid::YinYangGrid) -> (nlon, nlat)

The cell counts across one panel.
"""
@inline panel_shape(grid::YinYangGrid) = (getfield(grid, :nlon), getfield(grid, :nlat))

@inline _panel_cells(grid::YinYangGrid) = getfield(grid, :nlon) * getfield(grid, :nlat)

@inline function panel_cell(grid::YinYangGrid, k::Integer)
    nlon, _ = panel_shape(grid)
    np = _panel_cells(grid)
    q, r = divrem(Int(k) - 1, np)
    j, i = divrem(r, nlon)
    return (q + 1, i + 1, j + 1)
end

@inline function cell_id(grid::YinYangGrid, panel::Integer, i::Integer, j::Integer)
    nlon, _ = panel_shape(grid)
    return (Int(panel) - 1) * _panel_cells(grid) + (Int(j) - 1) * nlon + Int(i)
end

# ---- the three traits -------------------------------------------------------

@inline cell_address(::YinYangGrid) = FlatCells()
@inline adjacency_source(::YinYangGrid) = FormulaNeighbors()

# Panel-local axial offsets, and the panels are not cross-linked.
@inline has_symmetric_adjacency(::YinYangGrid) = true
@inline candidate_source(::YinYangGrid) = IndexedCandidates()

# ---- coordinates ------------------------------------------------------------

@inline ncoordinates(::YinYangGrid) = 2

@inline function _raw_coords(grid::YinYangGrid{T}, k::Integer) where {T}
    nlon, nlat = panel_shape(grid)
    p, i, j = panel_cell(grid, k)
    λ, φ = SphericalSampling._yin_yang_panel_coords(T, nlon, nlat, i, j)
    return p == 1 ? (Geometry.wrap_longitude(λ), φ) : SphericalSampling._yin_yang_rotate(λ, φ)
end

coordinates(grid::YinYangGrid) = throw(ArgumentError(
    "a YinYangGrid stores no coordinate arrays — a cell's position is its panel-local lat–lon, " *
    "rotated onto the sphere for yang. Use `coords(grid, k)` for one cell, or " *
    "`Grids.materialize(grid)` for the whole cloud.",
))

# ---- topology and span ------------------------------------------------------

# Yin's own frame spans three quarters of the circle, so no direction of the cell numbering wraps: the
# two panels are separate patches and neither closes on itself.
@inline topology(::YinYangGrid) = (Bounded(), Bounded())
@inline period(grid::YinYangGrid{T}, d::Integer) where {T} =
    (_checked_direction((1, 2), d); zero(T))

# The union of the two panels covers the sphere, and yang reaches both poles.
@inline bounds(grid::YinYangGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? (zero(T), T(2π)) : (-T(π) / 2, T(π) / 2)

@inline origin(grid::YinYangGrid, d::Integer) = bounds(grid, d)[1]

# ---- measure ----------------------------------------------------------------

@inline function measure(grid::YinYangGrid, k::Integer)
    _, i, j = panel_cell(grid, k)
    return getfield(grid, :cell_measure)[i, j]
end

@inline measure(grid::YinYangGrid) = GridMeasure(grid)

# The panels' union is the sphere, and the measures count the shared band once.
_total_measure(grid::YinYangGrid{T}) where {T} = 4 * T(π) * T(Geometry.radius(grid_geometry(grid)))^2

"""
    _yin_yang_measures(R, nlon, nlat) -> Matrix

Every panel cell's measure, `nlon × nlat`, the same on both panels. A cell `[λa, λb] × [φa, φb]` in its
panel's frame has area `R²·Δλ·2 sin(Δφ/2)·cos φ`, and its measure is that less half its overlap with the
other panel, which [`_yin_yang_overlap`](@ref) integrates exactly. The overlap is symmetric under
`λ → −λ` and `φ → −φ`, so a quarter of the cells is computed and the rest mirrored.

An overlap is a difference of antiderivatives of size one, accurate to a few `eps` on the unit sphere
whatever the cell's size. The table is built once on the host at no less than `Float64` and stored in
`T`, so each cell holds `T`'s own precision.
"""
function _yin_yang_measures(R::T, nlon::Int, nlat::Int) where {T<:AbstractFloat}
    TW = promote_type(T, Float64) # TODO :: Make this work in native res, on devices etc
    Δλ = (3 * TW(π) / 2) / TW(nlon)
    Δφ = (TW(π) / 2) / TW(nlat)
    m = Matrix{T}(undef, nlon, nlat)
    for j in 1:cld(nlat, 2), i in 1:cld(nlon, 2)
        λa = -3 * TW(π) / 4 + TW(i - 1) * Δλ
        φa = -TW(π) / 4 + TW(j - 1) * Δφ
        _, φc = SphericalSampling._yin_yang_panel_coords(TW, nlon, nlat, i, j)
        area = Δλ * 2 * sin(Δφ / 2) * cos(φc)
        v = T(TW(R)^2 * (area - _yin_yang_overlap(λa, λa + Δλ, φa, φa + Δφ) / 2))
        m[i, j] = v
        m[nlon + 1 - i, j] = v
        m[i, nlat + 1 - j] = v
        m[nlon + 1 - i, nlat + 1 - j] = v
    end
    return m
end

# Seen from either panel, the other covers `|y| ≤ 1/√2` less the wedge `x > 0, |z| < x`, the map
# `(x, y, z) ↦ (−x, z, y)` being an involution. At `|φ| = u ≤ π/4` that is the longitudes
# `[α, β] ∪ [π − β, 3π/4]` and their mirror images, `α(u) = acos(tan u)`, `β(u) = asin(1/(√2 cos u))`, the
# first interval empty below `u = π/6`. Each interval end is `σ·base(u) + κ`, `base` one of `0`, `α`, `β`.
#
# With `r = √(cos 2u)`, `α = atan(r, sin u)` and `β = atan(1, r)`. Every angle below is a two-argument
# `atan` of this kind, whose arguments vanish together nowhere on `[0, π/4]`, so at the panel edge
# `u = π/4`, where `r → 0` and the antiderivatives flatten as `(π/4 − u)^{3/2}`, they keep absolute
# precision.
struct _OverlapEdge{T}
    kind::Int8      # 0: the constant `κ`; 1: `σα(u) + κ`; 2: `σβ(u) + κ`
    σ::T
    κ::T
end

@inline _ov_r(u::T) where {T} = sqrt(max(zero(T), cos(2u)))
@inline _ov_α(u::T) where {T} = atan(_ov_r(u), sin(u))
@inline _ov_β(u::T) where {T} = atan(one(T), _ov_r(u))

# `∫ cos u · α(u) du` and `∫ cos u · β(u) du`, by parts.
@inline function _ov_Fα(u::T) where {T}
    s, r = sin(u), _ov_r(u)
    return s * atan(r, s) - atan(r)
end

@inline function _ov_Fβ(u::T) where {T}
    s, r = sin(u), _ov_r(u)
    return s * atan(one(T), r) - atan(s, r) + atan(sqrt(T(2)) * s, r) / sqrt(T(2))
end

@inline _ov_value(e::_OverlapEdge{T}, u::T) where {T} =
    e.kind == 0 ? e.κ : e.σ * (e.kind == 1 ? _ov_α(u) : _ov_β(u)) + e.κ

# `∫ cos u · e(u) du`.
@inline _ov_antideriv(e::_OverlapEdge{T}, u::T) where {T} =
    e.κ * sin(u) + (e.kind == 0 ? zero(T) : e.σ * (e.kind == 1 ? _ov_Fα(u) : _ov_Fβ(u)))

# The `u ∈ [0, π/4]` where `e(u) = λ`, `NaN` where there is none.
@inline function _ov_crossing(e::_OverlapEdge{T}, λ::T) where {T}
    e.kind == 0 && return T(NaN)
    c = (λ - e.κ) / e.σ
    e.kind == 1 && return zero(T) ≤ c ≤ T(π) / 2 ? atan(cos(c)) : T(NaN)
    return T(π) / 4 ≤ c ≤ T(π) / 2 ? atan(sqrt(max(zero(T), -cos(2c)))) : T(NaN)
end

# `∫_{u1}^{u2} cos u · |[λa, λb] ∩ [lo(u), hi(u)]| du`. Between consecutive crossings of the ends with
# `λa`, `λb` and each other the clipped interval keeps its ends, so each piece is a difference of
# antiderivatives.
function _ov_strip(lo::_OverlapEdge{T}, hi::_OverlapEdge{T}, λa::T, λb::T, u1::T, u2::T) where {T}
    cuts = T[u1, u2]
    for c in (T(π) / 6, _ov_crossing(lo, λa), _ov_crossing(lo, λb), _ov_crossing(hi, λa),
              _ov_crossing(hi, λb))
        u1 < c < u2 && push!(cuts, c)
    end
    sort!(cuts)
    acc = zero(T)
    for k in 1:(length(cuts) - 1)
        p, q = cuts[k], cuts[k + 1]
        q > p || continue
        mid = (p + q) / 2
        L = λa ≥ _ov_value(lo, mid) ? _OverlapEdge{T}(0, one(T), λa) : lo
        H = λb ≤ _ov_value(hi, mid) ? _OverlapEdge{T}(0, one(T), λb) : hi
        _ov_value(H, mid) > _ov_value(L, mid) || continue
        acc += (_ov_antideriv(H, q) - _ov_antideriv(H, p)) - (_ov_antideriv(L, q) - _ov_antideriv(L, p))
    end
    return acc
end

# The other panel is symmetric in `φ`, so a band is folded onto `u = |φ|`.
function _ov_band(lo::_OverlapEdge{T}, hi::_OverlapEdge{T}, λa::T, λb::T, φa::T, φb::T) where {T}
    φb ≤ 0 && return _ov_strip(lo, hi, λa, λb, -φb, -φa)
    φa ≥ 0 && return _ov_strip(lo, hi, λa, λb, φa, φb)
    return _ov_strip(lo, hi, λa, λb, zero(T), -φa) + _ov_strip(lo, hi, λa, λb, zero(T), φb)
end

"""
    _yin_yang_overlap(λa, λb, φa, φb) -> T

The area on the unit sphere of the panel cell `[λa, λb] × [φa, φb]`, in its own panel's frame, that the
other panel also covers: `∫ cos φ · |[λa, λb] ∩ D(|φ|)| dφ` over the cell, `D(u)` the other panel's
longitudes at `|φ| = u`, in closed form.
"""
function _yin_yang_overlap(λa::T, λb::T, φa::T, φb::T) where {T<:AbstractFloat}
    q = 3 * T(π) / 4
    edges = (
        (_OverlapEdge{T}(1, one(T), zero(T)), _OverlapEdge{T}(2, one(T), zero(T))),      # [α, β]
        (_OverlapEdge{T}(2, -one(T), T(π)), _OverlapEdge{T}(0, one(T), q)),              # [π − β, 3π/4]
        (_OverlapEdge{T}(2, -one(T), zero(T)), _OverlapEdge{T}(1, -one(T), zero(T))),    # [−β, −α]
        (_OverlapEdge{T}(0, one(T), -q), _OverlapEdge{T}(2, one(T), -T(π))),             # [−3π/4, β − π]
    )
    s = zero(T)
    for (lo, hi) in edges
        s += _ov_band(lo, hi, λa, λb, φa, φb)
    end
    return s
end

# ---- display ----------------------------------------------------------------

Base.show(io::IO, grid::YinYangGrid{T}) where {T} =
    print(io, "YinYangGrid{", T, "}(", panel_shape(grid)[1], "×", panel_shape(grid)[2],
          " per panel, ", length(grid), " cells)")

function Base.show(io::IO, ::MIME"text/plain", grid::YinYangGrid{T}) where {T}
    nlon, nlat = panel_shape(grid)
    println(io, "YinYangGrid{", T, "} 2 × ", nlon, " × ", nlat, " (", count(mask(grid)), "/",
            length(grid), " active)")
    println(io, "  geometry:  ", grid_geometry(grid))
    print(io, "  measure:   ", sum(measure(grid)), " total")
end
