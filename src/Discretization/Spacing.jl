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
