
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
    return faces!(similar(x, T, n + 1), x)
end

"""
    faces!(out, x) -> out

[`faces`](@ref) into a caller-owned vector of length `N+1`.

The form to use when staggering repeatedly — a moving mesh, a column solver stepping in time — where the
`N+1` faces are otherwise allocated per call.
"""
function faces!(out::AbstractVector{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(x)
    # An axis of `n` centres has `n+1` faces, and an empty one has none, matching what `faces` returns
    # for it.
    want = iszero(n) ? 0 : n + 1
    length(out) == want || throw(DimensionMismatch(
        "out holds $(length(out)) faces for an axis of $n centres, which has $want",
    ))
    n == 0 && return out
    return _faces!(out, x, n)
end

function _faces!(f::AbstractVector{T}, x::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
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
widths. The two are inverse only where the widths are constant.
"""
function centers(f::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(f) - 1
    n ≥ 0 || throw(ArgumentError("an axis of faces needs at least one entry"))
    return centers!(similar(f, T, n), f)
end

"""
    centers!(out, f) -> out

[`centers`](@ref) into a caller-owned vector of length `N`, for an axis of `N+1` faces.
"""
function centers!(out::AbstractVector{T}, f::AbstractVector{T}) where {T<:AbstractFloat}
    n = length(f) - 1
    n ≥ 0 || throw(ArgumentError("an axis of faces needs at least one entry"))
    length(out) == n || throw(DimensionMismatch(
        "out holds $(length(out)) centres for $(length(f)) faces, which have $n",
    ))
    @inbounds for i in 1:n
        out[i] = (f[i] + f[i+1]) / T(2)
    end
    return out
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
