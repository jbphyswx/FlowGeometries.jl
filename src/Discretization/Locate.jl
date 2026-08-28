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
