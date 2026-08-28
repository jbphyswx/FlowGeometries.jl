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
