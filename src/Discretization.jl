module Discretization

using ..Axes: Axes
using ..Execution: Execution
using ..Geometry: Geometry

# The geometric inputs a numerical method needs from a grid: where a point falls, the weights that
# interpolate to it, where a cell's faces are, the metric factors of the coordinate system, and the
# finite-difference weights of an arbitrary stencil.
#
# Nearly all of it is a function of coordinates alone. The one exception is `apply_stencil!`, which
# applies a weight set along ONE direction leaving the result where the input was — a case in which
# every convention is already fixed (no staggering to pick, `fd_weights`' inward shift at a bounded end,
# wrapping on a periodic one, hence no halo). Operators that genuinely do need a result location and a
# boundary-condition policy — a staggered difference, a divergence, a curl — are the caller's to
# assemble, from these weights and the metric factors below.

# ---------------------------------------------------------------------------
# Staggering
# ---------------------------------------------------------------------------

"""
    AbstractLocation

Where a value sits within a cell: [`Center`](@ref) or [`Face`](@ref).
"""
abstract type AbstractLocation end

"""
    Center()

Cell centres — `N` of them along a direction of `N` cells.
"""
struct Center <: AbstractLocation end

"""
    Face()

Cell boundaries — `N+1` of them along a direction of `N` cells.

Centres do not determine faces: `N` centres leave the `N+1` faces underdetermined by one. The
convention here is that faces sit midway between neighbouring centres, with the two outermost placed
by linear extrapolation, which is the same rule the curvilinear corner reconstruction uses.
"""
struct Face <: AbstractLocation end

"""
    faces(x) -> AbstractVector

The `N+1` cell boundaries of an axis of `N` cell centres: midpoints of neighbouring centres, with the
outermost two extrapolated a half-cell beyond the end centres.

A uniform axis stays uniform — its faces are another [`Axes.UniformAxis`](@ref), offset by half a
cell — so nothing about the axis's spacing guarantee is lost.
"""
function faces(x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && return similar(x, T, 0)
    f = similar(x, T, n + 1)
    if n == 1
        # A single cell has no neighbour to halve the distance to; it is given unit width.
        @inbounds f[1] = x[1] - one(T) / 2
        @inbounds f[2] = x[1] + one(T) / 2
        return f
    end
    @inbounds for i in 2:n
        f[i] = (x[i-1] + x[i]) / T(2)
    end
    @inbounds f[1] = x[1] - (x[2] - x[1]) / T(2)
    @inbounds f[n+1] = x[n] + (x[n] - x[n-1]) / T(2)
    return f
end

function faces(a::AbstractRange{T}) where {T<:AbstractFloat}
    n = length(a)
    Δ = T(step(a))
    return Axes.UniformAxis{T}(n == 0 ? T(first(a)) : T(first(a)) - Δ / T(2), Δ, n + 1)
end

"""
    centers(f) -> AbstractVector

The `N` cell centres of an axis of `N+1` cell boundaries: the midpoint of each pair.

It inverts [`faces`](@ref) exactly on a uniform axis. On a stretched one it does not: `faces` places a
boundary midway between two centres, and re-midpointing those boundaries averages neighbouring cell
widths rather than recovering the centre. The two are inverse only where the widths are constant.
"""
function centers(f::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(f) - 1
    n ≥ 0 || throw(ArgumentError("an axis of faces needs at least one entry"))
    c = similar(f, T, n)
    @inbounds for i in 1:n
        c[i] = (f[i] + f[i+1]) / T(2)
    end
    return c
end

function centers(a::AbstractRange{T}) where {T<:AbstractFloat}
    Δ = T(step(a))
    return Axes.UniformAxis{T}(T(first(a)) + Δ / T(2), Δ, max(length(a) - 1, 0))
end

"""
    nodes(x, loc) -> AbstractVector

An axis's sample positions at location `loc`: `x` itself at [`Center`](@ref), and its
[`faces`](@ref) at [`Face`](@ref).
"""
@inline nodes(x::AbstractVector, ::Center) = x
@inline nodes(x::AbstractVector, ::Face) = faces(x)

# ---------------------------------------------------------------------------
# Gaps and cell widths
# ---------------------------------------------------------------------------

"""
    local_spacing(x, i, period=nothing) -> (h_m, h_p)

The one-sided coordinate gaps around index `i` of an axis `x`: `h_m = x[i]-x[i-1]` and
`h_p = x[i+1]-x[i]`.

This is the primitive a finite-difference operator assembled at the call site is built from — the
`h_m`, `h_p` of [`Geometry.nonuniform_first_derivative`](@ref), and what a staggered difference, a
divergence or a curl needs from the grid. Always a scalar subtraction of two already-stored elements,
never a heap allocation, so it is safe to call per grid point in a hot loop. On an axis whose spacing
is known from its type there is not even a subtraction.

The gaps are **signed**, so a descending axis reports negative gaps and a derivative stencil keeps the
index-versus-coordinate direction: composed with `nonuniform_first_derivative` this gives the same
derivative with respect to the coordinate whichever way the axis is stored. [`cell_width`](@ref) is
where the sign is dropped, being a length rather than a difference.

`period`, if given (e.g. `2π` for a periodic longitude axis), makes the boundary gaps *wrap* instead
of vanishing: at `i == 1`, `h_m` is the gap to the unwrapped previous point `x[n]-period`; at
`i == n`, `h_p` is the gap to the unwrapped next point `x[1]+period`. Pass `nothing` (the default) for
a non-periodic axis, where a boundary gap is zero and the caller falls back to a one-sided stencil.

`Grids.local_spacing` is this on a grid direction, taking the period from the grid itself.
"""
@inline function local_spacing(
    x::AbstractVector{T}, i::Integer, period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    if period === nothing
        h_m = i > 1 ? @inbounds(x[i] - x[i-1]) : zero(T)
        h_p = i < n ? @inbounds(x[i+1] - x[i]) : zero(T)
    else
        # The wrapped neighbour is one period away along the INDEX direction, which is not the
        # coordinate direction on a descending axis, so the offset carries the orientation.
        p = Axes.wrap_sign(x) * T(period)
        h_m = i > 1 ? @inbounds(x[i] - x[i-1]) : @inbounds(x[1] - (x[n] - p))
        h_p = i < n ? @inbounds(x[i+1] - x[i]) : @inbounds((x[1] + p) - x[n])
    end
    return h_m, h_p
end

# On ANY range the interior gap is `step`, so it is returned rather than recovered by differencing two
# coordinates — which makes it exactly constant, where differencing varies by an ulp. This is why a
# caller's own range does not need converting to get the fast path.
@inline function local_spacing(
    x::AbstractRange{T}, i::Integer, period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    Δ = T(step(x))
    if period === nothing
        h_m = i > 1 ? Δ : zero(T)
        h_p = i < n ? Δ : zero(T)
    else
        p = Axes.wrap_sign(x) * T(period)
        h_m = i > 1 ? Δ : first(x) - (last(x) - p)
        h_p = i < n ? Δ : (first(x) + p) - last(x)
    end
    return h_m, h_p
end

"""
    cell_width(x, i, period=nothing) -> width

The coordinate width of cell `i` of an axis of cell centres `x`: the centred width
`(|h_m| + |h_p|)/2` at an interior cell — and, given a `period`, at the wrapped boundary too — the
one-sided gap to the single neighbour at a genuinely non-periodic boundary, and `1` for a length-1
axis. On a uniform axis every width is the constant step.

Equivalently `abs(faces(x)[i+1] - faces(x)[i])`, which is what it means: [`faces`](@ref) places a
boundary midway between neighbouring centres, so the width between them is the average of the two
adjacent gaps. This form is the one to call per cell, since `faces` materializes the whole axis.

A width is a physical measure and so is non-negative however the axis is stored — increasing or
decreasing, as a dataset holding latitude, depth or pressure levels top-down would. This is the one
place that turns a spacing into a length/area/volume contribution, so it is where the `abs` belongs;
[`local_spacing`](@ref) itself keeps the sign.

A length-1 axis contributes the multiplicative **identity**, not zero, to a measure that is a product
of per-axis widths (Cartesian `Δx·Δy`), so a degenerate direction reduces an area to a length rather
than collapsing the product. The spherical `R²cosφ·Δλ·Δφ` measure is not a plain product and handles
its own singleton case, in the `Grids.StructuredGrid` constructor.

`Grids.cell_width` is this on a grid direction, and `Grids.cell_widths` the whole axis at once.
"""
@inline function cell_width(
    x::AbstractVector{T}, i::Integer, period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    n == 1 && return one(T)
    h_m, h_p = local_spacing(x, i, period)
    if period === nothing
        i == 1 && return abs(h_p)
        i == n && return abs(h_m)
    end
    return (abs(h_m) + abs(h_p)) / T(2)
end

"""
    cell_widths(x, period=nothing) -> AbstractVector

[`cell_width`](@ref) at every index of an axis at once, for a caller that wants the whole profile
rather than one cell — the coordinate widths a separable measure or a flux divergence is weighted by.

A uniform axis gets an [`Axes.ConstantVector`](@ref): one number and a length, since every one of its
cells has the same width, so nothing is materialized. Anything else is built densely into the same
kind of storage as `x`, by broadcasts over views rather than a scalar loop, so a device-resident axis
is widened in place.

`Grids.cell_widths` is this on a grid direction, taking the period from the grid itself.
"""
function cell_widths(x::AbstractVector, period::Union{Nothing,Real} = nothing)
    return _cell_widths_dense(x, period)
end

# Every cell is `|Δ|` wide, boundaries included, whenever the period is the axis's own closure. A
# regional axis declared periodic against a larger period has a genuinely wider seam, so it falls
# through to the dense path.
function cell_widths(
    x::AbstractRange{T}, period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    # A singleton axis contributes a multiplicative identity (see `cell_width`).
    n == 1 && return Axes.ConstantVector(one(T), 1)
    n == 0 && return Axes.ConstantVector(zero(T), 0)
    w = abs(T(step(x)))
    period === nothing && return Axes.ConstantVector(w, n)
    _, h_seam = local_spacing(x, n, period)
    isapprox(abs(h_seam), w; rtol = 8 * eps(T)) && return Axes.ConstantVector(w, n)
    return _cell_widths_dense(x, period)
end

function _cell_widths_dense(x::AbstractVector{T}, period::Union{Nothing,Real} = nothing) where {T<:AbstractFloat}
    n = length(x)
    w = similar(x, T, n)
    if n == 1
        # A singleton axis contributes a multiplicative identity, so a measure that is a product of
        # per-axis widths degenerates (area → length) instead of collapsing to zero.
        fill!(w, one(T))
        return w
    end
    # Broadcasts over views, never a scalar loop, so a device-resident axis is widened in place.
    d = abs.(@view(x[2:n]) .- @view(x[1:(n - 1)]))
    @views w[2:(n - 1)] .= (d[1:(n - 2)] .+ d[2:(n - 1)]) ./ T(2)
    if period === nothing
        # A genuine boundary sees only the one gap it has.
        @views w[1:1] .= d[1:1]
        @views w[n:n] .= d[(n - 1):(n - 1)]
    else
        # Both ends additionally see the seam gap, written from magnitudes so it is the same seam in
        # either storage order.
        g = abs(T(period) - abs(@inbounds(x[n]) - @inbounds(x[1])))
        @views w[1:1] .= (g .+ d[1:1]) ./ T(2)
        @views w[n:n] .= (d[(n - 1):(n - 1)] .+ g) ./ T(2)
    end
    return w
end

# ---------------------------------------------------------------------------
# Point location
# ---------------------------------------------------------------------------

"""
    locate(x, v) -> Int

The index of the cell of axis `x` that contains coordinate `v`, or `0` when `v` lies outside.

Cells are the intervals between the axis's [`faces`](@ref), so cell `i` holds
`f[i] ≤ v < f[i+1]` (the last cell includes its far face). `O(1)` on a uniform axis, from the closed
form; `O(log n)` on a stretched one, by bisection. Both storage orders work.
"""
function locate(x::AbstractVector{T}, v::Real) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && return 0
    vT = T(v)
    @inbounds ascending = n == 1 || x[n] ≥ x[1]
    f1 = _face_at(x, 1, n)
    fe = _face_at(x, n + 1, n)
    if ascending
        (vT < f1 || vT > fe) && return 0
    else
        (vT > f1 || vT < fe) && return 0
    end
    lo, hi = 1, n + 1
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        fm = _face_at(x, mid, n)
        if ascending ? (fm ≤ vT) : (fm ≥ vT)
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

# The bisection above needs `log₂ n` of the `n + 1` faces, so they are evaluated one at a time from
# the centres rather than materialized — `faces(x)` would allocate the whole axis on every query.
# Same rule as `faces`, so the two never disagree about where a boundary is.
@inline function _face_at(x::AbstractVector{T}, j::Int, n::Int) where {T}
    @inbounds begin
        n == 1 && return j == 1 ? x[1] - one(T) / 2 : x[1] + one(T) / 2
        j == 1 && return x[1] - (x[2] - x[1]) / T(2)
        j == n + 1 && return x[n] + (x[n] - x[n-1]) / T(2)
        return (x[j-1] + x[j]) / T(2)
    end
end

function locate(a::AbstractRange{T}, v::Real) where {T<:AbstractFloat}
    n = length(a)
    n == 0 && return 0
    Δ = T(step(a))
    iszero(Δ) && return 1
    vT = T(v)
    # Faces start half a cell before the first centre, so the cell index is the number of whole cells
    # from that first face — no search, and no faces materialized.
    f0 = T(first(a)) - Δ / T(2)
    fe = f0 + T(n) * Δ
    ascending = Δ > 0
    if ascending
        (vT < f0 || vT > fe) && return 0
    else
        (vT > f0 || vT < fe) && return 0
    end
    i = clamp(Int(floor((vT - f0) / Δ)) + 1, 1, n)
    # That division is not exact, so it can land an ulp the wrong side of a face — and `f[i] ≤ v <
    # f[i+1]` is decided exactly there. One step against the face values themselves settles it; the
    # seed is never off by more than a cell, so this is still O(1).
    @inbounds if i > 1 && (ascending ? vT < f0 + T(i - 1) * Δ : vT > f0 + T(i - 1) * Δ)
        i -= 1
    elseif i < n && (ascending ? vT ≥ f0 + T(i) * Δ : vT ≤ f0 + T(i) * Δ)
        i += 1
    end
    return i
end

"""
    nearest_index(x, v) -> Int

The index of the axis sample closest to `v`, clamped into range. Unlike [`locate`](@ref) this always
returns a valid index, since a nearest sample exists for any `v`.

Exact ties go to the LOWER index, and the uniform closed form and the general bisection agree on
that, so the two paths never disagree. `O(1)` on a uniform axis, `O(log n)` on a stretched one.
"""
function nearest_index(x::AbstractVector{T}, v::Real) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && throw(ArgumentError("an empty axis has no nearest sample"))
    n == 1 && return 1
    vT = T(v)
    # The nearest sample is one of the two that bracket `v`, so the same bisection that
    # `interpolation_weights` uses answers this in `O(log n)` rather than by scanning the axis.
    i = _bracket(x, vT, n)
    @inbounds return abs(x[i+1] - vT) < abs(x[i] - vT) ? i + 1 : i   # strict `<` keeps a tie low
end

function nearest_index(a::AbstractRange{T}, v::Real) where {T<:AbstractFloat}
    n = length(a)
    n == 0 && throw(ArgumentError("an empty axis has no nearest sample"))
    n == 1 && return 1
    vT = T(v)
    # Deliberately the same bracket-then-compare as the general path, rather than a closed-form
    # round-to-nearest: the samples of a stretched and a uniform axis are the same numbers, so the
    # answers must match, and `(v - first)/Δ` can read as an exact tie where the two representable
    # samples are not in fact equidistant from `v`.
    i = _bracket(a, vT, n)
    @inbounds return abs(a[i+1] - vT) < abs(a[i] - vT) ? i + 1 : i   # strict `<` keeps a tie low
end

# ---------------------------------------------------------------------------
# Interpolation weights
# ---------------------------------------------------------------------------

"""
    interpolation_weights(x, v) -> (i, w)

Linear interpolation on axis `x` at coordinate `v`, as the left sample index `i` and the weight pair
`w = (w_i, w_{i+1})` with `w_i + w_{i+1} == 1`.

Weights only: applying them to a field is the caller's loop, and the field's layout is not this
module's business. Outside the axis the nearest end is used with weight 1, so the result is a clamp
rather than an extrapolation.
"""
function interpolation_weights(x::AbstractVector{T}, v::Real) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && throw(ArgumentError("cannot interpolate on an empty axis"))
    n == 1 && return (1, (one(T), zero(T)))
    vT = T(v)
    i = _bracket(x, vT, n)
    @inbounds x0, x1 = x[i], x[i+1]
    h = x1 - x0
    iszero(h) && return (i, (one(T), zero(T)))
    t = clamp((vT - x0) / h, zero(T), one(T))
    return (i, (one(T) - t, t))
end

# The left index of the pair of samples that brackets `v`, clamped to the ends.
function _bracket(x::AbstractVector{T}, v::T, n::Int) where {T}
    @inbounds ascending = x[n] ≥ x[1]
    lo, hi = 1, n
    @inbounds while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if ascending ? (x[mid] ≤ v) : (x[mid] ≥ v)
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

function _bracket(a::AbstractRange{T}, v::T, n::Int) where {T<:AbstractFloat}
    Δ = T(step(a))
    iszero(Δ) && return 1
    return clamp(Int(floor((v - T(first(a))) / Δ)) + 1, 1, n - 1)
end

"""
    lagrange_weights(x, v, nodes) -> (indices, weights)

Lagrange interpolation weights of `length(nodes)` points on axis `x` at coordinate `v`, exact for
polynomials up to degree `nodes-1` and valid on an arbitrarily spaced axis.

The stencil is centred on `v` as far as the axis allows and shifted inward at the ends, so the node
count is honoured everywhere rather than degrading near a boundary.
"""
function lagrange_weights(x::AbstractVector{T}, v::Real, nodes::Integer) where {T<:AbstractFloat}
    n = length(x)
    k = Int(nodes)
    k ≥ 1 || throw(ArgumentError("need at least one node, got $k"))
    k ≤ n || throw(ArgumentError("cannot use $k nodes on an axis of $n samples"))
    i0 = clamp(nearest_index(x, v) - (k - 1) ÷ 2, 1, n - k + 1)
    idx = i0:(i0 + k - 1)
    w = Vector{T}(undef, k)
    vT = T(v)
    @inbounds for a in 1:k
        num = one(T)
        den = one(T)
        xa = x[idx[a]]
        for b in 1:k
            b == a && continue
            xb = x[idx[b]]
            num *= (vT - xb)
            den *= (xa - xb)
        end
        w[a] = num / den
    end
    return (idx, w)
end

# ---------------------------------------------------------------------------
# Finite-difference weights (Fornberg 1988)
# ---------------------------------------------------------------------------

"""
    fd_weights(nodes, x₀, order) -> Vector

Finite-difference weights approximating the `order`-th derivative at `x₀` from the values at `nodes`,
by the recursion of Fornberg (1988), *Math. Comp.* **51**, 699–706.

One recursion covers every case: any derivative order, any node count (hence any order of accuracy),
any evaluation point — inside the node set or outside it — and arbitrarily spaced nodes. With `m`
nodes the result is exact for polynomials of degree `m-1`, so accuracy order `m - order`.

`sum(w .* f.(nodes))` is then the derivative estimate. This returns the weights only; applying them
to a field is the caller's.

    fd_weights([0.0, 1.0, 2.0], 1.0, 1)   # ≈ [-0.5, 0.0, 0.5], the centred first difference
"""
function fd_weights(nodes::AbstractVector{T}, x₀::Real, order::Integer) where {T<:AbstractFloat}
    n = length(nodes)
    w = Vector{T}(undef, n)
    # `max(…, 0)` so a negative `order` reaches `fd_weights!`'s own message rather than an
    # array-dimension error from sizing the table.
    c = Matrix{T}(undef, n, max(Int(order) + 1, 0))
    fd_weights!(w, c, nodes, x₀, order)
    return w
end

"""
    fd_weights!(w, c, nodes, x₀, order) -> w

[`fd_weights`](@ref) into caller buffers: `w` holds the `length(nodes)` weights and `c` is the
`length(nodes) × (order+1)` recursion table. Both are overwritten.

The allocating form is one of these per call, and a stencil is built once per sample of an axis, so a
4096-sample axis costs ~8000 allocations without this. The degrade path in [`apply_stencil!`](@ref) needs
one per cell near a mask edge, which is the reason it exists.
"""
@inline fd_weights!(
    w::AbstractVector{T}, c::AbstractMatrix{T}, nodes::AbstractVector{T}, x₀::Real, order::Integer,
) where {T<:AbstractFloat} = _fd_weights!(w, c, nodes, length(nodes), x₀, order)

# The node count is separate from `length(nodes)` so a caller holding an oversized buffer can use its
# first `n` entries without a `view` — which allocates 48 bytes per call, and the degrade path calls
# this once per cell it rebuilds.
function _fd_weights!(
    w::AbstractVector{T}, c::AbstractMatrix{T}, nodes::AbstractVector{T}, n::Int, x₀::Real,
    order::Integer,
) where {T<:AbstractFloat}
    m = Int(order)
    m ≥ 0 || throw(ArgumentError("derivative order must be ≥ 0, got $m"))
    length(nodes) ≥ n || throw(DimensionMismatch("asked for $n nodes from a buffer of $(length(nodes))"))
    n ≥ m + 1 || throw(ArgumentError(
        "a degree-$m derivative needs at least $(m + 1) nodes, got $n",
    ))
    (length(w) ≥ n && size(c, 1) ≥ n && size(c, 2) ≥ m + 1) || throw(DimensionMismatch(
        "fd_weights! needs w of length ≥ $n and c of size ≥ ($n, $(m + 1))",
    ))
    z = T(x₀)
    # c[i, k+1] holds the weight of node i for the k-th derivative, built up over the nodes.
    @inbounds for k in 1:(m + 1), i in 1:n
        c[i, k] = zero(T)
    end
    c1 = one(T)
    c4 = @inbounds(nodes[1]) - z
    @inbounds c[1, 1] = one(T)
    @inbounds for i in 2:n
        mn = min(i, m + 1)
        c2 = one(T)
        c5 = c4
        c4 = nodes[i] - z
        for j in 1:(i - 1)
            c3 = nodes[i] - nodes[j]
            iszero(c3) && throw(ArgumentError("fd_weights needs distinct nodes; $(nodes[i]) repeats"))
            c2 *= c3
            if j == i - 1
                for k in mn:-1:2
                    c[i, k] = c1 * (T(k - 1) * c[i-1, k-1] - c5 * c[i-1, k]) / c2
                end
                c[i, 1] = -c1 * c5 * c[i-1, 1] / c2
            end
            for k in mn:-1:2
                c[j, k] = (c4 * c[j, k] - T(k - 1) * c[j, k-1]) / c3
            end
            c[j, 1] = c4 * c[j, 1] / c3
        end
        c1 = c2
    end
    @inbounds for i in 1:n
        w[i] = c[i, m + 1]
    end
    return w
end

"""
    fd_weights(x, i, order, nodes) -> (indices, weights)

Weights for the `order`-th derivative at sample `i` of axis `x`, using `nodes` of its samples.

The stencil is centred on `i` where the axis allows and shifted inward at a boundary, so the accuracy
order is the same everywhere — a clipped stencil would silently drop to first order at the two ends.
Built on the arbitrary-node form above, so a stretched axis costs nothing extra.
"""
function fd_weights(
    x::AbstractVector{T}, i::Integer, order::Integer, nodes::Integer,
) where {T<:AbstractFloat}
    n = length(x)
    k = Int(nodes)
    k ≤ n || throw(ArgumentError("cannot use $k nodes on an axis of $n samples"))
    1 ≤ i ≤ n || throw(BoundsError(x, i))
    i0 = _window_start(Int(i), k, 1, n)
    idx = i0:(i0 + k - 1)
    # A view, not a copy: `fd_weights` only reads its nodes, and this runs once per sample of the axis.
    return (idx, fd_weights(@view(x[idx]), @inbounds(x[i]), order))
end

# ---------------------------------------------------------------------------
# Metric factors
# ---------------------------------------------------------------------------

"""
    scale_factors(geometry, point) -> NTuple

The metric scale factors `hᵈ` at `point`: the physical length of a unit step in each coordinate
direction. Cartesian gives `1` in every direction; spherical gives `(R cosφ, R)` on the surface and
`(r cosφ, r, 1)` with a radius direction.

These are what turns a coordinate derivative into a physical one — `∂/∂sᵈ = (1/hᵈ)·∂/∂ξᵈ` — so a
divergence or a curl is assembled from these plus [`fd_weights`](@ref) without this module having to
choose a staggering or a boundary condition.
"""
@inline scale_factors(geometry::Geometry.AbstractGeometry, point) =
    Geometry.scale_factors(geometry, point)

"""
    jacobian(geometry, point) -> Real

`∏ hᵈ` from [`scale_factors`](@ref): the volume element per unit coordinate volume at `point`.
"""
@inline jacobian(geometry::Geometry.AbstractGeometry, point) =
    Geometry.jacobian(geometry, point)

# ---------------------------------------------------------------------------
# Applying a weight set along one direction
# ---------------------------------------------------------------------------

"""
    AbstractMaskPolicy

What [`apply_stencil!`](@ref) does at the edge of the active region: [`BlankMasked`](@ref),
[`ShiftWithinRun`](@ref) or [`ReduceInRun`](@ref).

A **type**, like the image and reach conventions in `Connectivity`: which cells carry a number and which
carry `masked` is a property of the result, so it belongs in the call rather than in a runtime tag.
"""
abstract type AbstractMaskPolicy end

"""
    BlankMasked()

Write `masked` at a cell that is inactive **or** whose stencil reads an inactive cell. The default, and
the only policy that never invents a value: where the stencil cannot be formed from active data, there
is no derivative.

Its cost is a dead band. Every active cell within `nodes - 1` of a masked cell is blanked, so a
five-point derivative loses two cells either side of every coastline.
"""
struct BlankMasked <: AbstractMaskPolicy end

"""
    ShiftWithinRun()

Shift the stencil to fit inside the run of active samples containing the cell, keeping the full node
count — the same thing the stencil already does at the end of a bounded axis, with the end of the active
run as the boundary. `masked` only where the run is shorter than `nodes`.

The accuracy order is therefore the same everywhere a value is written, which is the property
[`fd_weights`](@ref) exists to preserve. On a run of at least `nodes` active samples the weights are
**identical** to the unmasked ones, so the interior of an active region is bit-for-bit unchanged.
"""
struct ShiftWithinRun <: AbstractMaskPolicy end

"""
    ReduceInRun()

[`ShiftWithinRun`](@ref), and where the run cannot hold `nodes`, use the largest window it can, down to
`order + 1` samples. `masked` below that, where no derivative of that order exists.

This trades accuracy order for coverage — a five-point scheme becomes three-point in a strait three
cells wide — so it is named rather than reached by fallback. Ask for it when a value everywhere matters
more than a uniform order.

Under this policy `nodes` is a **ceiling**, not a demand, and that applies to the end of the axis as
well as the end of a run: an axis with fewer than `nodes` samples uses as many as it has instead of
raising, and one with fewer than `order + 1` is `masked` throughout. A single-latitude strip, a
two-level column and a one-cell-wide channel are ordinary grids, and asking for "second order where the
axis allows it" should not require the caller to clamp `nodes` themselves. The other two policies keep
the error, since neither claims to degrade.
"""
struct ReduceInRun <: AbstractMaskPolicy end

"""
    _window_start(i, k, lo, hi) -> Int

First index of a `k`-node window centred on `i` and shifted to fit inside `[lo, hi]`. The whole-axis
case is `lo = 1, hi = n`; the masked case is the same expression with the bounds of the active run,
which is why both share this.
"""
@inline _window_start(i::Int, k::Int, lo::Int, hi::Int) = clamp(i - (k - 1) ÷ 2, lo, hi - k + 1)

"""
    axis_stencils(x, order, nodes; period=nothing) -> (indices, weights)

The `order`-th derivative's [`fd_weights`](@ref) at **every** sample of axis `x`, as two `n × nodes`
matrices: the axis indices each sample reads, and the weight on each.

One row per sample, so a stretched axis costs nothing extra downstream — the varying weights are
already here. Built once and reused by [`apply_stencil!`](@ref).

`period === nothing` shifts the stencil inward at the two ends, exactly as the single-sample
[`fd_weights`](@ref) does. Given a period the stencil stays centred everywhere and wraps, with the
wrapped samples' coordinates carried across the seam so the spacing there is the true one.
"""
function axis_stencils(
    x::AbstractVector{T}, order::Integer, nodes::Integer; period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    k = Int(nodes)
    ord = Int(order)
    k ≥ ord + 1 || throw(ArgumentError(
        "an order-$ord derivative needs at least $(ord + 1) nodes, got $k",
    ))
    k ≤ n || throw(ArgumentError("cannot use $k nodes on an axis of $n samples"))
    idx = Matrix{Int}(undef, n, k)
    wts = Matrix{T}(undef, n, k)
    half = (k - 1) ÷ 2
    buf = Vector{T}(undef, k)
    # Reused across every sample: the allocating `fd_weights` would be two per sample, ~8000 on a
    # 4096-sample axis, for a table whose size never changes.
    wbuf = Vector{T}(undef, k)
    cbuf = Matrix{T}(undef, k, ord + 1)
    P = period === nothing ? zero(T) : T(period) * Axes.wrap_sign(x)
    @inbounds for i in 1:n
        if period === nothing
            i0 = _window_start(i, k, 1, n)
            for q in 1:k
                idx[i, q] = i0 + q - 1
                buf[q] = x[i0 + q - 1]
            end
        else
            for q in 1:k
                raw = i - half + q - 1
                idx[i, q] = mod1(raw, n)
                buf[q] = x[mod1(raw, n)] + T(fld(raw - 1, n)) * P
            end
        end
        fd_weights!(wbuf, cbuf, buf, x[i], ord)
        for q in 1:k
            wts[i, q] = wbuf[q]
        end
    end
    return idx, wts
end

"""
    apply_stencil!(out, field, x, dim; order=1, nodes=order+1, period=nothing,
                   mask=nothing, masked=zero) -> out
    apply_stencil!(out, field, indices, weights, dim; mask=nothing, masked=zero) -> out

Apply a weight set along direction `dim` of `field`, writing
`out[I] = Σ_q weights[I[dim], q] · field[…, indices[I[dim], q], …]`.

This is the one field-touching operation here, and it is here because every convention it needs is
already settled elsewhere in the package rather than being the caller's to choose: the result sits at
the same location as the input, so there is no staggering decision; the stencil shifts inward at a
bounded end and wraps on a periodic one, which is [`fd_weights`](@ref)'s stated boundary behaviour and
removes any need for a halo. What is *not* here is anything that does need those choices — a staggered
difference, or a multi-direction operator like a divergence or a curl, which additionally needs a
result location and a boundary-condition policy.

Pass the axis and an order to have the weights built for you, or precomputed `indices`/`weights` from
[`axis_stencils`](@ref) to reuse them across many fields.

With a `mask`, a cell is written as `masked` when it is inactive **or when its stencil reads an
inactive cell** — the derivative there is not determined by the active data, so it is not invented.
`out` and `field` may not alias.
"""
function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, x::AbstractVector{<:AbstractFloat},
    dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
    period::Union{Nothing,Real} = nothing, mask = nothing, masked = zero(S), backend = nothing,
    policy::AbstractMaskPolicy = BlankMasked(), scratch = nothing,
) where {S,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    size(field, dim) == length(x) || throw(DimensionMismatch(
        "axis has $(length(x)) samples but direction $dim of the field has $(size(field, dim))",
    ))
    ord = Int(order)
    k = Int(nodes)
    if policy isa ReduceInRun
        # Under this policy `nodes` is a CEILING, not a demand. The end of the axis bounds a window
        # exactly as the end of an active run does — the policy already says a run too short for
        # `nodes` uses the largest window it can hold — so the two are made to behave the same way
        # rather than one degrading and the other erroring. A single-latitude strip, a two-level
        # column and a one-cell channel are ordinary grids, not mistakes.
        if length(x) < ord + 1
            fill!(out, masked)      # no derivative of this order exists anywhere on such an axis
            return out
        end
        k = min(k, length(x))
    end
    idx, wts = axis_stencils(x, ord, k; period = period)
    return apply_stencil!(out, field, x, idx, wts, dim; order = ord, period = period, mask = mask,
                          masked = masked, backend = backend, policy = policy, scratch = scratch)
end

"""
    apply_stencil!(out, field, x, indices, weights, dim; order=1, period=nothing, mask=nothing,
                   masked=zero, policy=BlankMasked(), backend=nothing) -> out

Apply a table built by [`axis_stencils`](@ref) **and** keep the axis, so any mask policy works.

The table depends on the axis and not on the field, so a caller differencing many fields along one
direction should build it once. The bare `(indices, weights)` form cannot degrade at a mask edge —
that needs the axis to rebuild a window from — so it accepts only [`BlankMasked`](@ref); this form
takes both and serves every policy.

The split is the one the degrade path already makes internally: the precomputed row is used wherever
the window is intact, which is every cell away from a mask, and the axis is touched only where a
window is actually rebuilt.

Building the table is `O(n)` against an `O(n²)` apply, so holding it matters most on small grids —
2.4–13× at `n = 48`, 10–40% at `n = 256`, amortized away by `n = 1024`. The allocation it avoids is
there at every size: 49 600 bytes per call at `n = 1024`.
"""
function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, x::AbstractVector{<:AbstractFloat},
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix, dim::Integer;
    order::Integer = 1, period::Union{Nothing,Real} = nothing, mask = nothing, masked = zero(S),
    backend = nothing, policy::AbstractMaskPolicy = BlankMasked(), scratch = nothing,
) where {S,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    size(field, dim) == length(x) || throw(DimensionMismatch(
        "axis has $(length(x)) samples but direction $dim of the field has $(size(field, dim))",
    ))
    size(indices, 1) == length(x) || throw(DimensionMismatch(
        "got $(size(indices, 1)) stencil rows for an axis of $(length(x)) samples",
    ))
    # The precomputed rows are the whole answer under `BlankMasked`, and they stay the answer in the
    # interior of every active run under the others — a degraded row is only built where one is needed.
    if policy isa BlankMasked || mask === nothing
        return apply_stencil!(out, field, indices, weights, dim; mask = mask, masked = masked,
                              backend = backend)
    end
    return _apply_stencil_degrade!(out, field, x, indices, weights, Int(dim), mask, masked,
                                   Int(order), size(indices, 2), period, policy, backend, scratch)
end

# Rebuilding a stencil needs the axis and a scratch table, so it is a chunked host loop: a launch has
# nowhere to put the per-cell Fornberg table. `BlankMasked` above keeps the index-parallel path.
"""
    StencilScratch{T}

The working buffers a degrading [`apply_stencil!`](@ref) needs to rebuild a window at a mask edge:
the Fornberg table and the node list. Build one with [`stencil_scratch`](@ref).

**One per task**, exactly as [`Connectivity.ball_scratch`](@ref) is — the buffers are written per cell,
so chunks cannot share them. A threaded `backend` therefore allocates its own set per chunk and ignores
one passed here.
"""
struct StencilScratch{T<:AbstractFloat}
    w::Vector{T}      # the weights of one rebuilt row
    c::Matrix{T}      # the Fornberg recursion table
    n::Vector{T}      # the row's node coordinates, unwrapped across a seam
end

"""
    stencil_scratch(order, nodes; T = Float64) -> StencilScratch

Buffers for the degrade path, so a caller taking many derivatives on a masked grid does not allocate
them per call. Without one a degrading call allocates a few hundred bytes each time — `O(1)` in the
grid, but per *call*, so a flux computation taking nine derivatives pays it nine times.

Only the degrading policies need it. An unmasked grid, and any grid under [`BlankMasked`](@ref), never
rebuilds a window and allocates nothing regardless.
"""
function stencil_scratch(order::Integer, nodes::Integer; T::Type{<:AbstractFloat} = Float64)
    k = Int(nodes)
    m = Int(order)
    k ≥ 1 || throw(ArgumentError("stencil_scratch needs nodes ≥ 1, got $k"))
    m ≥ 0 || throw(ArgumentError("stencil_scratch needs order ≥ 0, got $m"))
    return StencilScratch{T}(Vector{T}(undef, k), Matrix{T}(undef, k, m + 1), Vector{T}(undef, k))
end

@inline function _fits(s::StencilScratch, k::Int, ord::Int)
    return length(s.w) ≥ k && length(s.n) ≥ k && size(s.c, 1) ≥ k && size(s.c, 2) ≥ ord + 1
end

function _apply_stencil_degrade!(
    out::AbstractArray{S,N}, field, x::AbstractVector{T}, idx, wts, dim::Int, mask, masked,
    ord::Int, k::Int, period, policy::AbstractMaskPolicy, backend, scratch,
) where {S,N,T}
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    size(mask) == size(field) || throw(DimensionMismatch(
        "mask $(size(mask)) and field $(size(field)) must have the same size",
    ))
    sz = size(field)
    ci = CartesianIndices(sz)
    n = sz[dim]
    P = period === nothing ? zero(T) : T(period) * Axes.wrap_sign(x)
    wrap = period !== nothing
    # A caller's buffers are usable only where there is one chunk: they are written per cell, so
    # concurrent chunks would race on them. The threaded path allocates its own set per chunk.
    if backend === nothing && scratch isa StencilScratch{T}
        _fits(scratch, k, ord) || throw(DimensionMismatch(
            "scratch holds $(length(scratch.w)) nodes × $(size(scratch.c, 2)) orders; this call " *
            "needs $k × $(ord + 1) — build it with `stencil_scratch($ord, $k)`",
        ))
        @inbounds for lin in Base.OneTo(length(ci))
            c = ci[lin]
            _stencil_cell_degrade!(out, field, x, idx, wts, dim, mask, masked, k, ord, n, P, wrap,
                                   Tuple(c), c, policy, scratch.w, scratch.c, scratch.n)
        end
        return out
    end
    Execution.run_chunks(length(ci), backend) do rng
        wbuf = Vector{T}(undef, k)
        cbuf = Matrix{T}(undef, k, ord + 1)
        nbuf = Vector{T}(undef, k)
        @inbounds for lin in rng
            c = ci[lin]
            _stencil_cell_degrade!(out, field, x, idx, wts, dim, mask, masked, k, ord, n, P, wrap,
                                   Tuple(c), c, policy, wbuf, cbuf, nbuf)
        end
    end
    return out
end

# The cell index with direction `dim` replaced. `j` arrives as an ARGUMENT rather than being captured:
# a local that is both reassigned and closed over is boxed by Julia, and the loops below reassign
# theirs every iteration — measured at 288 bytes per `_run_reach` call before this was split out.
@inline _at_dim(I::NTuple{N,Int}, dim::Int, j::Int) where {N} =
    ntuple(d -> d == dim ? j : I[d], Val(N))

# How far the active run containing `i` reaches, walked at most `k` steps: past that the run is longer
# than any window, which is all the clamp needs to know. Bounding the walk is what keeps this `O(k)`
# rather than `O(run length)`.
@inline function _run_reach(mask, I::NTuple{N,Int}, dim::Int, i::Int, n::Int, k::Int, wrap::Bool) where {N}
    back = 0
    @inbounds while back < k
        j = i - back - 1
        j < 1 && (wrap ? (j = n) : break)
        mask[_at_dim(I, dim, j)...] || break
        back += 1
        back ≥ n && break
    end
    fwd = 0
    @inbounds while fwd < k
        j = i + fwd + 1
        j > n && (wrap ? (j = 1) : break)
        mask[_at_dim(I, dim, j)...] || break
        fwd += 1
        back + fwd + 1 ≥ n && break
    end
    return back, fwd
end

@inline function _stencil_cell_degrade!(
    out::AbstractArray{S,N}, field, x::AbstractVector{T}, idx, wts, dim::Int, mask, masked,
    k::Int, ord::Int, n::Int, P::T, wrap::Bool, I::NTuple{N,Int}, ci, policy, wbuf, cbuf, nbuf,
) where {S,N,T}
    @inbounds begin
        if !mask[ci]
            out[ci] = masked
            return nothing
        end
        i = I[dim]
        # The precomputed window, if every node it reads is active. Contiguous and all-active means it
        # lies in this cell's run, and the run-clamped window is then the same window — so this branch
        # is bit-for-bit the unmasked result, and it is the one the interior of a region takes.
        # Accumulated while checking, in the same order: a window that turns out to be intact has then
        # been walked once rather than twice, and this is the branch every cell away from a mask takes.
        intact = true
        acc = zero(S)
        for q in 1:k
            J = ntuple(d -> d == dim ? Int(idx[i, q]) : I[d], Val(N))
            if !mask[J...]
                intact = false
                break
            end
            acc += S(wts[i, q]) * S(field[J...])
        end
        if intact
            out[ci] = acc
            return nothing
        end

        back, fwd = _run_reach(mask, I, dim, i, n, k, wrap)
        len = back + fwd + 1
        kk = policy isa ReduceInRun ? min(k, len) : k
        if len < kk || kk < ord + 1
            out[ci] = masked
            return nothing
        end
        # Run-local coordinates, so one expression covers a wrapping run and a bounded one.
        pos = back                        # 0-based offset of `i` from the run's first sample
        s = clamp(pos - (kk - 1) ÷ 2, 0, len - kk)
        base = i - back                   # may be ≤ 0 when the run wraps; `mod1` puts it back
        for q in 1:kk
            raw = base + s + q - 1
            j = mod1(raw, n)
            nbuf[q] = x[j] + T(fld(raw - 1, n)) * P
        end
        _fd_weights!(wbuf, cbuf, nbuf, kk, x[i], ord)
        acc = zero(S)
        for q in 1:kk
            raw = base + s + q - 1
            J = _at_dim(I, dim, mod1(raw, n))
            acc += S(wbuf[q]) * S(field[J...])
        end
        out[ci] = acc
    end
    return nothing
end

function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N},
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix, dim::Integer;
    mask = nothing, masked = zero(S), backend = nothing,
    policy::AbstractMaskPolicy = BlankMasked(),
) where {S,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    # Degrading means rebuilding a stencil, which needs the axis this form was not given.
    policy isa BlankMasked || throw(ArgumentError(
        "$(policy) needs the axis to rebuild a stencil from; call the `(out, field, x, dim)` form",
    ))
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    size(indices) == size(weights) || throw(DimensionMismatch(
        "indices $(size(indices)) and weights $(size(weights)) must have the same size",
    ))
    size(indices, 1) == size(field, dim) || throw(DimensionMismatch(
        "got $(size(indices, 1)) stencil rows for direction $dim of length $(size(field, dim))",
    ))
    mask === nothing || size(mask) == size(field) || throw(DimensionMismatch(
        "mask $(size(mask)) and field $(size(field)) must have the same size",
    ))
    k = size(indices, 2)
    if backend === nothing
        # On the host the loop shape is ours to choose, and the index-parallel one is the wrong shape:
        # see `_stencil_sweep_host!`. Both paths are the same arithmetic in the same order, so they
        # agree bit for bit.
        return _dispatch_dim(Int(dim), Val(N)) do vdim
            _dispatch_nodes(k) do vk
                _stencil_sweep_host!(out, field, indices, weights, mask, masked, vdim, vk, Val(N))
            end
        end
    end
    sz = size(field)
    ci = CartesianIndices(sz)
    Execution.run_indices(length(ci), backend) do lin
        _stencil_cell!(out, field, indices, weights, Int(dim), mask, masked, k,
                       Tuple(@inbounds ci[lin]), @inbounds ci[lin])
    end
    return out
end

# Two runtime values are wanted in the type: the differenced direction, so the loop nest can be split
# around it, and the node count, so the innermost loop has a known trip count. Both are resolved ONCE
# per sweep, here, rather than per cell — a `Val` built deeper costs more than it saves. Specialization
# stays bounded: directions by `N`, node counts by the cap below, above which the runtime loop stands.
@inline _dispatch_dim(f::F, dim::Int, ::Val{N}) where {F,N} = _dim_switch(f, dim, Val(N))
# `dim` is validated into `1:N` by the caller, so walking down from `N` always lands.
@inline _dim_switch(f::F, ::Int, ::Val{1}) where {F} = f(Val(1))
@inline _dim_switch(f::F, dim::Int, ::Val{M}) where {F,M} =
    dim == M ? f(Val(M)) : _dim_switch(f, dim, Val(M - 1))

@inline function _dispatch_nodes(f::F, k::Int) where {F}
    k == 2 && return f(Val(2))
    k == 3 && return f(Val(3))
    k == 4 && return f(Val(4))
    k == 5 && return f(Val(5))
    k == 6 && return f(Val(6))
    k == 7 && return f(Val(7))
    k == 9 && return f(Val(9))
    return f(k)
end

"""
    _stencil_sweep_host!(out, field, indices, weights, mask, masked, Val(dim), nodes, Val(N)) -> out

The host sweep. The index-parallel form exists so one body serves a device launch; on the host it is
the wrong shape, and three things it cannot express are worth ~6.6× together:

- iterate the Cartesian range **directly**, rather than recovering an index per cell from a linear one;
- **split the nest at `dim`**, so the stencil row — which depends only on the index along `dim` — is
  hoisted out of the contiguous inner loop whenever the differenced direction is not the fastest
  varying one;
- carry the **node count in the type**, so the innermost loop has a known trip count, unrolls, and the
  weights reach registers instead of being re-loaded per node.

The arithmetic and its order are identical to `_stencil_cell!`, so the two paths agree bit for bit.
"""
function _stencil_sweep_host!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, ::Val{dim}, nodes, ::Val{N},
) where {S,N,dim}
    sz = size(field)
    if _linear_layout(out, field, mask)
        # Column-major and one-based, so the whole nest is address arithmetic: cells that differ only
        # before `dim` are `1` apart, and a node is a fixed offset of `stride` per index step along
        # `dim`. The innermost span is then contiguous and vectorizes.
        stride = prod(ntuple(d -> sz[d], Val(dim - 1)))
        npost = prod(ntuple(d -> sz[dim + d], Val(N - dim)))
        outer = stride * sz[dim]
        if dim == 1
            # The differenced direction is itself the contiguous one, so there is no span to hoist a
            # row out of — every cell has its own row, used once. Hoisting it into a tuple would be
            # pure overhead; the loop over `j` is the contiguous one instead.
            for p in 0:(npost - 1)
                _stencil_first_linear!(out, field, indices, weights, mask, masked, p * outer,
                                       sz[1], nodes)
            end
            return out
        end
        for p in 0:(npost - 1), j in 1:sz[dim]
            _stencil_row_linear!(out, field, indices, weights, mask, masked, j, p * outer, stride,
                                 nodes)
        end
        return out
    end
    pre = CartesianIndices(ntuple(d -> sz[d], Val(dim - 1)))
    post = CartesianIndices(ntuple(d -> sz[dim + d], Val(N - dim)))
    @inbounds for Ipost in post, j in 1:sz[dim]
        _stencil_row!(out, field, indices, weights, mask, masked, j, Tuple(Ipost), pre, nodes,
                      Val(dim), Val(N))
    end
    return out
end

# The address arithmetic above assumes linear indexing over one-based axes. Anything else — an offset
# array, a view with a non-trivial stride — takes the Cartesian nest, which asks the array for its own
# indexing and is correct for any of them.
@inline _linear_layout(out, field, mask) =
    IndexStyle(out) === IndexLinear() && IndexStyle(field) === IndexLinear() &&
    (mask === nothing || IndexStyle(mask) === IndexLinear()) &&
    !Base.has_offset_axes(out, field) && (mask === nothing || !Base.has_offset_axes(mask))

# `dim == 1`: the slab is one contiguous run along the differenced direction, so the stencil row
# changes every step and is read straight out of the table rather than hoisted.
@inline function _stencil_first_linear!(
    out::AbstractArray{S}, field, indices, weights, mask, masked, off::Int, n::Int, ::Val{k},
) where {S,k}
    @inbounds for j in 1:n
        if mask !== nothing && !mask[off + j]
            out[off + j] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            m = off + Int(indices[j, q])
            if mask !== nothing && !mask[m]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[m])
        end
        out[off + j] = blocked ? masked : acc
    end
    return nothing
end

@inline function _stencil_first_linear!(
    out::AbstractArray{S}, field, indices, weights, mask, masked, off::Int, n::Int, k::Int,
) where {S}
    @inbounds for j in 1:n
        if mask !== nothing && !mask[off + j]
            out[off + j] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            m = off + Int(indices[j, q])
            if mask !== nothing && !mask[m]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[m])
        end
        out[off + j] = blocked ? masked : acc
    end
    return nothing
end

# One row, as offsets. `off` is the start of this slab, `stride` the distance between consecutive
# indices along `dim`, so `base + t` walks the contiguous span and `js[q] + t` reads node `q` of it.
@inline function _stencil_row_linear!(
    out::AbstractArray{S}, field, indices, weights, mask, masked, j::Int, off::Int, stride::Int,
    ::Val{k},
) where {S,k}
    @inbounds js = ntuple(q -> off + (Int(indices[j, q]) - 1) * stride, Val(k))
    @inbounds ws = ntuple(q -> S(weights[j, q]), Val(k))
    base = off + (j - 1) * stride
    if mask === nothing
        @inbounds for t in 1:stride
            acc = zero(S)
            for q in 1:k
                acc += ws[q] * S(field[js[q] + t])
            end
            out[base + t] = acc
        end
    else
        @inbounds for t in 1:stride
            if !mask[base + t]
                out[base + t] = masked
                continue
            end
            acc = zero(S)
            blocked = false
            for q in 1:k
                m = js[q] + t
                if !mask[m]
                    blocked = true
                    break
                end
                acc += ws[q] * S(field[m])
            end
            out[base + t] = blocked ? masked : acc
        end
    end
    return nothing
end

# Above the specialization cap the row cannot become a tuple, so it is read per cell.
@inline function _stencil_row_linear!(
    out::AbstractArray{S}, field, indices, weights, mask, masked, j::Int, off::Int, stride::Int,
    k::Int,
) where {S}
    base = off + (j - 1) * stride
    @inbounds for t in 1:stride
        if mask !== nothing && !mask[base + t]
            out[base + t] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            m = off + (Int(indices[j, q]) - 1) * stride + t
            if mask !== nothing && !mask[m]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[m])
        end
        out[base + t] = blocked ? masked : acc
    end
    return nothing
end

# One row: every cell whose index along `dim` is `j`. The stencil is the same for all of them, so it is
# read once here rather than once per cell.
@inline function _stencil_row!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, j::Int, Ipost::Tuple, pre,
    ::Val{k}, ::Val{dim}, ::Val{N},
) where {S,N,dim,k}
    # The row, read once. With `k` in the type these are stack tuples, so the inner loop unrolls over
    # registers instead of re-reading two matrix columns per cell.
    @inbounds js = ntuple(q -> Int(indices[j, q]), Val(k))
    @inbounds ws = ntuple(q -> S(weights[j, q]), Val(k))
    @inbounds for Ipre in pre
        I = (Tuple(Ipre)..., j, Ipost...)
        if mask !== nothing && !mask[I...]
            out[I...] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            J = ntuple(d -> d == dim ? js[q] : I[d], Val(N))
            if mask !== nothing && !mask[J...]
                blocked = true
                break
            end
            acc += ws[q] * S(field[J...])
        end
        out[I...] = blocked ? masked : acc
    end
    return nothing
end

# Above the specialization cap the node count stays a runtime value: the row cannot become a tuple, so
# it is read per cell as before. Correct, and the shape a very wide stencil would not benefit from
# unrolling anyway.
@inline function _stencil_row!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, j::Int, Ipost::Tuple, pre,
    k::Int, ::Val{dim}, ::Val{N},
) where {S,N,dim}
    @inbounds for Ipre in pre
        I = (Tuple(Ipre)..., j, Ipost...)
        if mask !== nothing && !mask[I...]
            out[I...] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            J = ntuple(d -> d == dim ? Int(indices[j, q]) : I[d], Val(N))
            if mask !== nothing && !mask[J...]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[J...])
        end
        out[I...] = blocked ? masked : acc
    end
    return nothing
end

# One output cell, written from its own inputs only, so the loop above is index-parallel and the same
# body serves a host loop and a device launch.
@inline function _stencil_cell!(
    out::AbstractArray{S,N}, field, indices, weights, dim::Int, mask, masked, k::Int,
    I::NTuple{N,Int}, ci,
) where {S,N}
    @inbounds begin
        if mask !== nothing && !mask[ci]
            out[ci] = masked
            return nothing
        end
        j = I[dim]
        acc = zero(S)
        blocked = false
        for q in 1:k
            J = ntuple(d -> d == dim ? Int(indices[j, q]) : I[d], Val(N))
            if mask !== nothing && !mask[J...]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[J...])
        end
        out[ci] = blocked ? masked : acc
    end
    return nothing
end

# ---------------------------------------------------------------------------
# The physical derivative
# ---------------------------------------------------------------------------

"""
    derivative!(out, field, grid, dim; order=1, nodes=order+1, policy=BlankMasked(),
                masked=zero, active_only=true, backend=nothing) -> out

The derivative with respect to **distance** along direction `dim`, rather than with respect to the
coordinate: [`apply_stencil!`](@ref) divided by the metric factor,

    ∂f/∂sᵈ = (1/hᵈ) · ∂f/∂ξᵈ,    hᵈ = Geometry.scale_factors(geo, p)[d]

On a Cartesian metric every `hᵈ` is `1` and this is `apply_stencil!` exactly, at no cost. Anywhere else
it is the derivative a physical law is written in, and assembling it from the parts was the one thing
every geometry-aware caller had to add — including the coordinate singularity below, which is not
theirs to get right.

**Where the metric degenerates the derivative does not exist**, and `masked` is written rather than a
number invented. Longitude at a pole is the case: `h_λ = R cos φ → 0`, so `1/h_λ` diverges. The test is
relative to the geometry's own size and to the precision, `|h| ≤ L·√eps(T)` — an absolute threshold
cannot be right for both, since `1e-12` is below `eps(Float32)` and in `Float32`
`cos(Float32(π/2)) ≈ -4.4e-8`, so a pole row would quietly receive a large finite number instead.

No scale factor in this package depends on **longitude**, so `hᵈ` is constant along the first axis
whichever direction is differenced. The scaling is applied once per remaining index and swept along
that contiguous axis. (It is *not* generally constant along the differenced direction — on a spheroid
`h_φ = M(φ)` varies with `φ` — so it is not hoisted that way.)

A divergence or a curl is still the caller's to assemble, needing a result location and a boundary
policy this does not choose. Note the flux form when doing so: on a sphere

    ∇·u = (1/(R cos φ)) [ ∂u_λ/∂λ + ∂(u_φ cos φ)/∂φ ]

so the second term differentiates `u_φ cos φ`, not `u_φ`; taking two physical derivatives and adding
them is a different, wrong expression.
"""
function derivative! end

"""
    metric_floor(geometry) -> T

The magnitude below which a scale factor is treated as degenerate: `L·√eps(T)` for a curved geometry of
size `L`, and `0` for a Cartesian one, whose metric never degenerates.

Relative to both the geometry's size and the element type on purpose — see [`derivative!`](@ref) for
what an absolute constant does in `Float32`.
"""
function metric_floor end

"""
    GradientPlan{T}

The geometry of a least-squares gradient, separated from any field: for each cell, the coefficient on
each neighbour's *difference* from it. Built by `Connectivity.gradient_plan`, applied by
[`gradient!`](@ref).

`apply_stencil!` covers the separable case. A `CurvilinearGrid` has no separable axis to build a scalar
stencil along, and a node set's neighbours come from connectivity rather than an index offset, so
neither has a gradient without this.

The construction, with tangent-plane displacements `Δrₖ` from
[`Geometry.project_to_tangent_plane`](@ref), differences `Δfₖ = fₖ - f₀` and weights `wₖ = 1/|Δrₖ|²`:
minimising `Σₖ wₖ (∇f·Δrₖ - Δfₖ)²` gives `A ∇f = b` with `A = Σₖ wₖ Δrₖ⊗Δrₖ` and
`b = Σₖ wₖ Δrₖ Δfₖ`. `A` depends only on the geometry, so it is inverted once here and the per-neighbour
coefficients `A⁻¹ wₖ Δrₖ` are what the plan stores; each apply is then one dot product per cell and
allocates nothing.

What this buys over inverting an index-space Jacobian:

- **exact for a linear field on any stencil**, however skewed — the least-squares combination cancels
  the leading truncation term, which a 2×2 Jacobian inverse does not;
- second order on locally symmetric stencils, degrading toward first on strongly skewed cells;
- it reduces to the centred difference where the stencil is separable and orthogonal (`A` diagonal), so
  it agrees with `apply_stencil!` where both apply.

Where `A` is rank deficient — a tangent direction with no data, at a boundary or beside a mask — that
component is **zeroed rather than invented**, by the pseudo-inverse. Same rule `apply_stencil!` states
for a mask: not determined by the active data, so not produced.
"""
struct GradientPlan{T,VI<:AbstractVector{Int},VT<:AbstractVector{T}}
    ptr::VI          # CSR row pointers into `nbr`/`c1`/`c2`, length n+1
    nbr::VI          # neighbour cell, as a linear index
    c1::VT           # coefficient on this neighbour's difference, first tangent component
    c2::VT           # …and second
    names::NTuple{2,Symbol}   # what those two components are, from the geometry
end

Base.length(p::GradientPlan) = length(p.ptr) - 1

function Base.show(io::IO, p::GradientPlan{T}) where {T}
    return print(io, "GradientPlan{", T, "}(", length(p), " cells, ", length(p.nbr),
                 " neighbour coefficients, ", p.names, ")")
end

"""
    gradient!(g1, g2, field, plan) -> (g1, g2)

Apply a [`GradientPlan`](@ref): the two tangent components of `∇field` at every cell, written into
`g1` and `g2`. One dot product per cell over its neighbours, allocating nothing.

`field`, `g1` and `g2` are indexed linearly, so an `N`-D array of the grid's shape works as is. The
components are named by `plan.names` — `(:λ, :φ)` on a sphere, `(:x, :y)` on a plane — and are per unit
**distance**, the tangent plane being metric already.
"""
function gradient!(
    g1::AbstractArray{S}, g2::AbstractArray{S}, field::AbstractArray, plan::GradientPlan,
) where {S}
    n = length(plan)
    (length(g1) == n && length(g2) == n && length(field) == n) || throw(DimensionMismatch(
        "plan is for $n cells; got $(length(field)) field, $(length(g1))/$(length(g2)) output",
    ))
    # Indexed linearly rather than through `vec`, which would build three array wrappers per call —
    # 192 bytes on an operation that otherwise allocates nothing. An `N`-D array of the grid's shape
    # indexes linearly as it stands.
    @inbounds for i in 1:n
        f0 = field[i]
        s1 = zero(S)
        s2 = zero(S)
        for t in plan.ptr[i]:(plan.ptr[i + 1] - 1)
            df = S(field[plan.nbr[t]] - f0)
            s1 += S(plan.c1[t]) * df
            s2 += S(plan.c2[t]) * df
        end
        g1[i] = s1
        g2[i] = s2
    end
    return (g1, g2)
end

"""
    _sympinv2(a, b, c, tol) -> (p11, p12, p22)

Pseudo-inverse of the symmetric 2×2 `[a b; b c]`, dropping any eigendirection whose eigenvalue is below
`tol`. Closed form: a 2×2 symmetric eigenproblem has one.

Dropping rather than regularising is the point. A stencil that carries no information along some
tangent direction — every neighbour on one line, which happens at a boundary and beside a mask — leaves
`A` singular in that direction, and the gradient there is not determined by the data. Inverting a
nudged matrix would answer anyway, with a number governed by the nudge.
"""
@inline function _sympinv2(a::T, b::T, c::T, tol::T) where {T}
    τ = a + c
    disc = sqrt(max((a - c)^2 + 4b^2, zero(T)))
    λ1 = (τ + disc) / 2                      # ≥ λ2, both ≥ 0 for a Gram matrix
    λ2 = (τ - disc) / 2
    λ1 ≤ tol && return (zero(T), zero(T), zero(T))          # no direction resolved at all
    if λ2 > tol
        δ = a * c - b * b
        return (c / δ, -b / δ, a / δ)                        # full rank: the ordinary inverse
    end
    # Rank one: keep the resolved direction only, `A⁺ = v vᵀ / λ1`.
    v1, v2 = abs(b) > eps(T) * max(abs(a), abs(c), one(T)) ? (λ1 - c, b) :
             (a ≥ c ? (one(T), zero(T)) : (zero(T), one(T)))
    nrm = sqrt(v1 * v1 + v2 * v2)
    nrm ≤ zero(T) && return (zero(T), zero(T), zero(T))
    u1, u2 = v1 / nrm, v2 / nrm
    return (u1 * u1 / λ1, u1 * u2 / λ1, u2 * u2 / λ1)
end

"""
    interpolate(field, grid, p; policy=BlankMasked(), masked=NaN, …) -> value

The value of `field` at the coordinate `p`, which is the question observational data asks: a station, a
float or a ship track has a coordinate, not a cell index.

`interpolation_weights` gives this along **one** axis, and nothing composed them, so a caller had to
build the tensor product themselves on a rectilinear grid and had nothing at all on the others.

- `StructuredGrid` — multilinear, the tensor product of the per-axis weights. A periodic direction
  interpolates *across* its seam rather than clamping at the last sample.
- `CurvilinearGrid`, `UnstructuredGrid` — a weighted least-squares plane fitted to the `k` nearest
  cells in the tangent plane at `p`, which is **exact for a linear field** and reproduces a cell's own
  value at its centre. Falls back to the weighted mean where the fit is rank deficient, that being the
  part of it the data still determines.

`p` may be written any way a point is accepted elsewhere.

The mask policies say what an inactive contributor means, as they do for a stencil:
[`BlankMasked`](@ref) — the default — returns `masked` if any contributor is inactive, and
[`ReduceInRun`](@ref) renormalizes over the active ones. [`ShiftWithinRun`](@ref) has no meaning here,
there being no window to shift, and says so.
"""
function interpolate end

@inline function _interp_mask_error(policy)
    return throw(ArgumentError(
        "$(policy) has no meaning when interpolating — there is no window to shift. Use " *
        "`ReduceInRun()` to renormalize over the active contributors, or `BlankMasked()`.",
    ))
end

@inline metric_floor(::Geometry.AbstractCartesianGeometry{T}) where {T} = zero(T)
@inline metric_floor(g::Geometry.AbstractSphericalGeometry{T}) where {T} =
    Geometry.radius(g) * sqrt(eps(T))
@inline metric_floor(g::Geometry.AbstractEllipsoidalGeometry{T}) where {T} =
    Geometry.semimajor_axis(g) * sqrt(eps(T))

end # module Discretization
