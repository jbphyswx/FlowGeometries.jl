module Discretization

using ..Axes: Axes
using ..Geometry: Geometry

# The geometric inputs a numerical method needs from a grid: where a point falls, the weights that
# interpolate to it, where a cell's faces are, the metric factors of the coordinate system, and the
# finite-difference weights of an arbitrary stencil.
#
# All of it is a function of coordinates alone. Applying weights to a FIELD is not here: that needs a
# result location, a boundary-condition policy and a halo convention, which are the caller's to choose.

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

faces(a::Axes.UniformAxis{T}) where {T} =
    Axes.UniformAxis{T}(a.n == 0 ? a.origin : a.origin - a.Δ / T(2), a.Δ, a.n + 1)

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

centers(a::Axes.UniformAxis{T}) where {T} =
    Axes.UniformAxis{T}(a.origin + a.Δ / T(2), a.Δ, max(a.n - 1, 0))

"""
    nodes(x, loc) -> AbstractVector

An axis's sample positions at location `loc`: `x` itself at [`Center`](@ref), and its
[`faces`](@ref) at [`Face`](@ref).
"""
@inline nodes(x::AbstractVector, ::Center) = x
@inline nodes(x::AbstractVector, ::Face) = faces(x)

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
    f = faces(x)
    return _locate_in_faces(f, T(v), n)
end

function locate(a::Axes.UniformAxis{T}, v::Real) where {T<:AbstractFloat}
    n = a.n
    n == 0 && return 0
    iszero(a.Δ) && return 1
    # Faces start half a cell before the first centre, so the cell index is the number of whole cells
    # from that first face — no search, and no faces materialized.
    t = (T(v) - (a.origin - a.Δ / T(2))) / a.Δ
    i = Int(floor(t)) + 1
    return 1 ≤ i ≤ n ? i : 0
end

function _locate_in_faces(f::AbstractVector{T}, v::T, n::Int) where {T}
    @inbounds ascending = f[n+1] ≥ f[1]
    if ascending
        @inbounds (v < f[1] || v > f[n+1]) && return 0
        lo, hi = 1, n + 1
        @inbounds while hi - lo > 1
            mid = (lo + hi) ÷ 2
            f[mid] ≤ v ? (lo = mid) : (hi = mid)
        end
        return lo
    else
        @inbounds (v > f[1] || v < f[n+1]) && return 0
        lo, hi = 1, n + 1
        @inbounds while hi - lo > 1
            mid = (lo + hi) ÷ 2
            f[mid] ≥ v ? (lo = mid) : (hi = mid)
        end
        return lo
    end
end

"""
    nearest_index(x, v) -> Int

The index of the axis sample closest to `v`, clamped into range. Unlike [`locate`](@ref) this always
returns a valid index, since a nearest sample exists for any `v`.

Exact ties go to the LOWER index, and the uniform closed form and the general scan agree on that, so
the two paths never disagree.
"""
function nearest_index(x::AbstractVector{T}, v::Real) where {T<:AbstractFloat}
    n = length(x)
    n == 0 && throw(ArgumentError("an empty axis has no nearest sample"))
    vT = T(v)
    best = 1
    @inbounds bd = abs(x[1] - vT)
    @inbounds for i in 2:n
        d = abs(x[i] - vT)
        d < bd && (bd = d; best = i)      # strict `<` keeps a tie on the lower index
    end
    return best
end

function nearest_index(a::Axes.UniformAxis{T}, v::Real) where {T<:AbstractFloat}
    a.n == 0 && throw(ArgumentError("an empty axis has no nearest sample"))
    iszero(a.Δ) && return 1
    t = (T(v) - a.origin) / a.Δ
    # `ceil(t - 1/2)` rounds to nearest and sends an exact tie DOWN in index for either sign of Δ,
    # matching the scan above; `round` would send ties to even and disagree.
    i = Int(ceil(t - one(T) / 2)) + 1
    return clamp(i, 1, a.n)
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

function _bracket(a::Axes.UniformAxis{T}, v::T, n::Int) where {T}
    iszero(a.Δ) && return 1
    return clamp(Int(floor((v - a.origin) / a.Δ)) + 1, 1, n - 1)
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
    m = Int(order)
    m ≥ 0 || throw(ArgumentError("derivative order must be ≥ 0, got $m"))
    n = length(nodes)
    n ≥ m + 1 || throw(ArgumentError(
        "a degree-$m derivative needs at least $(m + 1) nodes, got $n",
    ))
    z = T(x₀)
    # c[i, k+1] holds the weight of node i for the k-th derivative, built up over the nodes.
    c = zeros(T, n, m + 1)
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
    return c[:, m + 1]
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
    i0 = clamp(Int(i) - (k - 1) ÷ 2, 1, n - k + 1)
    idx = i0:(i0 + k - 1)
    return (idx, fd_weights(collect(@view x[idx]), @inbounds(x[i]), order))
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

end # module Discretization
