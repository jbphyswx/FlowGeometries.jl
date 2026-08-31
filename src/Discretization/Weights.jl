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
    # `max(…, 0)` lets a negative `order` reach `fd_weights!`'s own message; sizing the table with it
    # raises an array-dimension error first.
    c = Matrix{T}(undef, n, max(Int(order) + 1, 0))
    fd_weights!(w, c, nodes, x₀, order)
    return w
end

"""
    fd_weights!(w, c, nodes, x₀, order) -> w

[`fd_weights`](@ref) into caller buffers: `w` holds the `length(nodes)` weights and `c` is the
`length(nodes) × (order+1)` recursion table. Both are overwritten.

The allocating form allocates two arrays per call, and a stencil is built once per sample of an axis.
The degrade path in [`apply_stencil!`](@ref FlowGeometries.Operators.apply_stencil!) rebuilds one per
cell near a mask edge, and holds these buffers across all of them.
"""
@inline fd_weights!(
    w::AbstractVector{T}, c::AbstractMatrix{T}, nodes::AbstractVector{T}, x₀::Real, order::Integer,
) where {T<:AbstractFloat} = _fd_weights!(w, c, nodes, length(nodes), x₀, order)

# The node count is separate from `length(nodes)` so a caller holding an oversized buffer can use its
# first `n` entries without a `view`, which the degrade path would allocate once per cell it rebuilds.
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
    # A view: `fd_weights` only reads its nodes, and this runs once per sample of the axis.
    return (idx, fd_weights(@view(x[idx]), @inbounds(x[i]), order))
end

# ---------------------------------------------------------------------------
# Metric factors
# ---------------------------------------------------------------------------

"""
    scale_factors(geometry, point) -> NTuple
    scale_factors(geometry, point, Val(N)) -> NTuple{N}

The metric scale factors `hᵈ` at `point`: the physical length of a unit step in each coordinate
direction. Cartesian gives `1` in every direction; spherical gives `(R cosφ, R)` on the surface and
`(r cosφ, r, 1)` with a radius direction.

They turn a coordinate derivative into a physical one, `∂/∂sᵈ = (1/hᵈ)·∂/∂ξᵈ`, so a divergence or a
curl is assembled from these plus [`fd_weights`](@ref) at the call site, where the staggering and the
boundary condition are chosen.

`Val(N)` names how many components `point` has, giving a concrete `NTuple{N}` from a point whose
length is a runtime value.
"""
@inline scale_factors(geometry::Geometry.AbstractGeometry, point) =
    Geometry.scale_factors(geometry, point)

@inline scale_factors(geometry::Geometry.AbstractGeometry, point, v::Val) =
    Geometry.scale_factors(geometry, point, v)

"""
    jacobian(geometry, point) -> Real

`∏ hᵈ` from [`scale_factors`](@ref): the volume element per unit coordinate volume at `point`.
"""
@inline jacobian(geometry::Geometry.AbstractGeometry, point) =
    Geometry.jacobian(geometry, point)


"""
    StencilScratch{T}

The working buffers a degrading [`apply_stencil!`](@ref FlowGeometries.Operators.apply_stencil!) needs to rebuild a window at a mask edge:
the Fornberg table and the node list. Build one with [`stencil_scratch`](@ref).

**One per task**, exactly as `Connectivity.ball_scratch` is — the buffers are written per cell,
so chunks cannot share them. A threaded `backend` therefore allocates its own set per chunk and ignores
one passed here.
"""
struct StencilScratch{T<:AbstractFloat,VT<:AbstractVector{T},MT<:AbstractMatrix{T}}
    w::VT             # the weights of one rebuilt row
    c::MT             # the Fornberg recursion table
    n::VT             # the row's node coordinates, unwrapped across a seam
end

"""
    stencil_scratch([T = Float64], order, nodes) -> StencilScratch

Buffers for the degrade path, so a caller taking many derivatives on a masked grid does not allocate
them per call. Without one a degrading call allocates a few hundred bytes each time — `O(1)` in the
grid, but per *call*, so a flux computation taking nine derivatives pays it nine times.

Only the degrading policies need it. An unmasked grid, and any grid under [`BlankMasked`](@ref FlowGeometries.Operators.BlankMasked), never
rebuilds a window and allocates nothing regardless.
"""
stencil_scratch(order::Integer, nodes::Integer) = stencil_scratch(Float64, order, nodes)

function stencil_scratch(::Type{T}, order::Integer, nodes::Integer) where {T<:AbstractFloat}
    k = Int(nodes)
    m = Int(order)
    k ≥ 1 || throw(ArgumentError("stencil_scratch needs nodes ≥ 1, got $k"))
    m ≥ 0 || throw(ArgumentError("stencil_scratch needs order ≥ 0, got $m"))
    w = Vector{T}(undef, k)
    c = Matrix{T}(undef, k, m + 1)
    return StencilScratch{T,typeof(w),typeof(c)}(w, c, Vector{T}(undef, k))
end

"""
    _window_start(i, k, lo, hi) -> Int

First index of a `k`-node window centred on `i` and shifted to fit inside `[lo, hi]`. The whole-axis
case is `lo = 1, hi = n`; the masked case passes the bounds of the active run.
"""
@inline _window_start(i::Int, k::Int, lo::Int, hi::Int) = clamp(i - (k - 1) ÷ 2, lo, hi - k + 1)

"""
    axis_stencils(x, order, nodes; period=nothing) -> (indices, weights)

The `order`-th derivative's [`fd_weights`](@ref) at **every** sample of axis `x`, as two `n × nodes`
matrices: the axis indices each sample reads, and the weight on each.

One row per sample, so a stretched axis costs nothing extra downstream — the varying weights are
already here. Built once and reused by [`apply_stencil!`](@ref FlowGeometries.Operators.apply_stencil!).

`period === nothing` shifts the stencil inward at the two ends, exactly as the single-sample
[`fd_weights`](@ref) does. Given a period the stencil stays centred everywhere and wraps, with the
wrapped samples' coordinates carried across the seam so the spacing there is the true one.
"""
function axis_stencils(
    x::AbstractVector{T}, order::Integer, nodes::Integer; period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    k = Int(nodes)
    _check_stencil_shape(n, k, Int(order))
    return axis_stencils!(Matrix{Int}(undef, n, k), Matrix{T}(undef, n, k), x, order, nodes;
                          period = period)
end

@inline function _check_stencil_shape(n::Int, k::Int, ord::Int)
    k ≥ ord + 1 || throw(ArgumentError(
        "an order-$ord derivative needs at least $(ord + 1) nodes, got $k",
    ))
    k ≤ n || throw(ArgumentError("cannot use $k nodes on an axis of $n samples"))
    return nothing
end

"""
    axis_stencils!(indices, weights, x, order, nodes; period=nothing, scratch=nothing) -> (indices, weights)

[`axis_stencils`](@ref) into two caller-owned `n × nodes` matrices.

The table is `O(n·k)`, so a caller rebuilding it — at each step of a stretching mesh, say — owns the
matrices and passes them in. Pass a [`stencil_scratch`](@ref) as well and the call allocates nothing:
the remaining buffers are the Fornberg table and the node list, `O(k)` each, allocated per call
without one.
"""
function axis_stencils!(
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix{T},
    x::AbstractVector{T}, order::Integer, nodes::Integer;
    period::Union{Nothing,Real} = nothing, scratch = nothing,
) where {T<:AbstractFloat}
    n = length(x)
    k = Int(nodes)
    ord = Int(order)
    _check_stencil_shape(n, k, ord)
    size(indices) == (n, k) || throw(DimensionMismatch(
        "indices is $(size(indices)) but the table is ($n, $k)",
    ))
    size(weights) == (n, k) || throw(DimensionMismatch(
        "weights is $(size(weights)) but the table is ($n, $k)",
    ))
    # The table's size is the same at every sample, so one set of buffers serves the whole axis.
    buf, wbuf, cbuf = _weight_bufs(scratch, T, k, ord)
    half = (k - 1) ÷ 2
    P = period === nothing ? zero(T) : T(period) * Axes.wrap_sign(x)
    @inbounds for i in 1:n
        if period === nothing
            i0 = _window_start(i, k, 1, n)
            for q in 1:k
                indices[i, q] = i0 + q - 1
                buf[q] = x[i0 + q - 1]
            end
        else
            for q in 1:k
                raw = i - half + q - 1
                indices[i, q] = mod1(raw, n)
                buf[q] = x[mod1(raw, n)] + T(fld(raw - 1, n)) * P
            end
        end
        fd_weights!(wbuf, cbuf, buf, x[i], ord)
        for q in 1:k
            weights[i, q] = wbuf[q]
        end
    end
    return indices, weights
end

# The node list, the row's weights and the Fornberg table, from a caller's scratch or freshly.
@inline _weight_bufs(::Nothing, ::Type{T}, k::Int, ord::Int) where {T} =
    (Vector{T}(undef, k), Vector{T}(undef, k), Matrix{T}(undef, k, ord + 1))

@inline function _weight_bufs(s::StencilScratch{T}, ::Type{T}, k::Int, ord::Int) where {T}
    (length(s.n) ≥ k && length(s.w) ≥ k && size(s.c, 1) ≥ k && size(s.c, 2) ≥ ord + 1) ||
        throw(ArgumentError(
            "this scratch holds $(length(s.w)) × $(size(s.c, 2)) and the table needs $k × $(ord + 1) " *
            "— build it with `stencil_scratch($ord, $k)`",
        ))
    return (s.n, s.w, s.c)
end


"""
    metric_floor(geometry) -> T

The magnitude below which a scale factor is treated as degenerate: `L·√eps(T)` for a curved geometry of
size `L`, and `0` for a Cartesian one, whose metric never degenerates.

It scales with both the geometry's size and the element type. At `Float32` on Earth's radius,
`cos(Float32(π/2)) ≈ -4.4e-8` puts `h_λ` at the pole around 0.28 m, above any fixed small constant.
"""
function metric_floor end


@inline metric_floor(::Geometry.AbstractCartesianGeometry{T}) where {T} = zero(T)
@inline metric_floor(g::Geometry.AbstractSphericalGeometry{T}) where {T} =
    Geometry.radius(g) * sqrt(eps(T))
@inline metric_floor(g::Geometry.AbstractEllipsoidalGeometry{T}) where {T} =
    Geometry.semimajor_axis(g) * sqrt(eps(T))
