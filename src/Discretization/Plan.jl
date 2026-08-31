# ---------------------------------------------------------------------------
# Stencil plans
# ---------------------------------------------------------------------------
#
# A weight set for one axis, held so a caller differencing many fields along it builds it once. Which
# form it takes is the axis's own spacing: on a uniform axis the exact weights are translation-invariant,
# so `O(K²)` numbers describe every one of the `n` rows; on a stretched axis the rows genuinely differ
# and the `n × K` table is what there is.

"""
    AbstractStencilPlan{T}

A finite-difference weight set for one axis. [`UniformStencilPlan`](@ref) or
[`TabulatedStencilPlan`](@ref), chosen by the axis's [`Axes.spacing_trait`](@ref).

Build one with [`stencil_plan`](@ref) and apply it with `Operators.apply_stencil!`.
"""
abstract type AbstractStencilPlan{T<:AbstractFloat} end

"""
    UniformStencilPlan{T,K,MI,MW}

The weight set of a uniformly spaced axis: one centred row of `K` weights, plus the rows at the two ends
where a bounded window shifts inward.

Every interior sample reads `field[j - left + q - 1]` weighted by `weights[q]`, the same `K` numbers
throughout, so they reach the sweep as a tuple in registers with no per-cell gather. Storage is `O(K²)`,
independent of the axis length. A wrapping axis has no shifted rows at all, every sample being centred.

`left` and `right` count the nodes before and after the sample and are equal only for odd `K`: a 4-node
window reads one sample back and two forward, so two rows shift at the top end and one at the bottom.

The weights are built from the spacing. Mathematically they are the row [`axis_stencils`](@ref) holds,
and the table carries per-row round-off from reconstructing each window's coordinates, at `~1e-14`
relative on a 5-node first derivative; one exact row applied everywhere carries none.
"""
struct UniformStencilPlan{
    T<:AbstractFloat, K, MI<:AbstractMatrix{<:Integer}, MW<:AbstractMatrix{T},
} <: AbstractStencilPlan{T}
    weights::NTuple{K,T}      # the centred row, shared by every interior sample
    left::Int                 # nodes before the sample, (K - 1) ÷ 2
    right::Int                # nodes after it, K - 1 - left
    n::Int                    # samples on the axis this describes
    order::Int
    wrap::Bool
    edge_idx::MI              # left + right rows, the shifted ones; empty when wrapping
    edge_wts::MW
end

"""
    TabulatedStencilPlan{T,K,MI,MW}

The weight set of a stretched axis: one row of node indices and weights per sample, as the `n × K` pair
of matrices [`axis_stencils`](@ref) returns.
"""
struct TabulatedStencilPlan{
    T<:AbstractFloat, K, MI<:AbstractMatrix{<:Integer}, MW<:AbstractMatrix{T},
} <: AbstractStencilPlan{T}
    indices::MI
    weights::MW
    order::Int
end

# `K` is the node count, carried in the type so that `plan_row`'s `K`-tuple has a width the compiler
# knows and a concrete return type. Read from `size`, that width is a runtime value.
function TabulatedStencilPlan(
    idx::AbstractMatrix{<:Integer}, w::AbstractMatrix{T}, order::Integer,
) where {T}
    size(idx) == size(w) || throw(DimensionMismatch(
        "indices $(size(idx)) and weights $(size(w)) must have the same size",
    ))
    return TabulatedStencilPlan{T,size(w, 2),typeof(idx),typeof(w)}(idx, w, Int(order))
end

"""
    nnodes(plan) -> Int

How many samples each row of `plan` reads.
"""
@inline nnodes(::UniformStencilPlan{T,K}) where {T,K} = K
@inline nnodes(::TabulatedStencilPlan{T,K}) where {T,K} = K

"""
    derivative_order(plan) -> Int

Which derivative `plan`'s weights compute.
"""
@inline derivative_order(p::AbstractStencilPlan) = p.order

"""
    axis_length(plan) -> Int

The number of samples on the axis `plan` describes. A field differenced with it has to have that many
along the differenced direction.
"""
@inline axis_length(p::UniformStencilPlan) = p.n
@inline axis_length(p::TabulatedStencilPlan) = size(p.weights, 1)

"""
    stencil_plan(x, order, nodes; period = nothing) -> AbstractStencilPlan

The `order`-th derivative's weights along axis `x`, held for reuse.

A uniform axis gives a [`UniformStencilPlan`](@ref) and a stretched one a
[`TabulatedStencilPlan`](@ref); which it is comes from the axis's type through
[`Axes.spacing_trait`](@ref), never from inspecting its values.

A wrapping uniform axis takes the uniform form only when the `period` given is the one the axis's own
spacing implies, `n·h`. This checks the argument against the axis type, and reads no coordinate: any
other period puts a seam gap other than `h` at the wrap, which the tabulated form describes.
"""
function stencil_plan(
    x::AbstractVector{T}, order::Integer, nodes::Integer; period::Union{Nothing,Real} = nothing,
) where {T<:AbstractFloat}
    ord = Int(order)
    k = Int(nodes)
    n = length(x)
    k ≥ ord + 1 || throw(ArgumentError(
        "an order-$ord derivative needs at least $(ord + 1) nodes, got $k",
    ))
    k ≤ n || throw(ArgumentError("cannot use $k nodes on an axis of $n samples"))
    return _stencil_plan(Axes.spacing_trait(x), x, ord, Val(k), period)
end

_stencil_plan(::Axes.NonuniformSpacing, x::AbstractVector, ord::Int, ::Val{K}, period) where {K} =
    TabulatedStencilPlan(axis_stencils(x, ord, K; period = period)..., ord)

function _stencil_plan(
    ::Axes.UniformSpacing, x::AbstractVector{T}, ord::Int, ::Val{K}, period,
) where {T,K}
    n = length(x)
    h = T(Axes.spacing(x))
    left = (K - 1) ÷ 2
    right = K - 1 - left
    if period !== nothing
        # The wrapped step is `period/n`, which has to be the axis's own `h` for the seam to be uniform.
        P = T(period) * Axes.wrap_sign(x)
        isapprox(P, T(n) * h; rtol = 8 * eps(T)) ||
            return TabulatedStencilPlan(axis_stencils(x, ord, K; period = period)..., ord)
        return UniformStencilPlan{T,K,Matrix{Int},Matrix{T}}(
            _uniform_row(T, h, -left, Val(K), ord), left, right, n, ord, true,
            Matrix{Int}(undef, 0, K), Matrix{T}(undef, 0, K),
        )
    end
    # A bounded window shifts inward at the ends: rows `1:left` all read the first `K` samples and rows
    # `n-right+1:n` the last `K`, each from its own sample's point of view.
    rows = vcat(1:left, (n - right + 1):n)
    edge_idx = Matrix{Int}(undef, length(rows), K)
    edge_wts = Matrix{T}(undef, length(rows), K)
    @inbounds for (r, j) in enumerate(rows)
        i0 = _window_start(j, K, 1, n)
        row = _uniform_row(T, h, i0 - j, Val(K), ord)
        for q in 1:K
            edge_idx[r, q] = i0 + q - 1
            edge_wts[r, q] = row[q]
        end
    end
    return UniformStencilPlan{T,K,Matrix{Int},Matrix{T}}(
        _uniform_row(T, h, -left, Val(K), ord), left, right, n, ord, false, edge_idx, edge_wts,
    )
end

"""
    _uniform_row(T, h, s, Val(K), order) -> NTuple{K,T}

The weights of a row whose first node sits `s` steps of size `h` from the sample it is evaluated at.

`s = -left` is the centred row. Only the offsets enter, so this is the exact weight set for every sample
sharing that offset — on a uniform axis, all of them away from a shifted end.
"""
function _uniform_row(::Type{T}, h::T, s::Int, ::Val{K}, ord::Int) where {T,K}
    nodes = ntuple(q -> T(s + q - 1) * h, Val(K))
    w = Vector{T}(undef, K)
    c = Matrix{T}(undef, K, ord + 1)
    fd_weights!(w, c, collect(nodes), zero(T), ord)
    return ntuple(q -> @inbounds(w[q]), Val(K))
end

"""
    plan_row(plan, j) -> (nodes, weights)

Row `j` of `plan`, as two `K`-tuples: the axis indices that sample reads and the weight on each.

The generic accessor, for a path that wants a row whatever form the plan took. The uniform sweep does
not go through it — the whole point there is that the interior row is not fetched per cell.
"""
@inline function plan_row(p::UniformStencilPlan{T,K}, j::Int) where {T,K}
    left = p.left
    p.wrap && return (ntuple(q -> mod1(j - left + q - 1, p.n), Val(K)), p.weights)
    if j ≤ left
        return (ntuple(q -> @inbounds(Int(p.edge_idx[j, q])), Val(K)),
                ntuple(q -> @inbounds(p.edge_wts[j, q]), Val(K)))
    elseif j > p.n - p.right
        r = left + (j - (p.n - p.right))
        return (ntuple(q -> @inbounds(Int(p.edge_idx[r, q])), Val(K)),
                ntuple(q -> @inbounds(p.edge_wts[r, q]), Val(K)))
    end
    return (ntuple(q -> j - left + q - 1, Val(K)), p.weights)
end

@inline plan_row(p::TabulatedStencilPlan{T,K}, j::Int) where {T,K} =
    (ntuple(q -> Int(@inbounds p.indices[j, q]), Val(K)),
     ntuple(q -> @inbounds(p.weights[j, q]), Val(K)))

Base.show(io::IO, p::UniformStencilPlan{T,K}) where {T,K} = print(
    io, "UniformStencilPlan{", T, "}(order=", p.order, ", nodes=", K, ", n=", p.n,
    p.wrap ? ", wrapping)" : ")",
)

Base.show(io::IO, p::TabulatedStencilPlan{T}) where {T} = print(
    io, "TabulatedStencilPlan{", T, "}(order=", p.order, ", nodes=", nnodes(p),
    ", n=", axis_length(p), ")",
)
