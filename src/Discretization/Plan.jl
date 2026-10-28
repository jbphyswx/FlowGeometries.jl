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
    UniformStencilPlan{T,V,MI,MW}

The weight set of a uniformly spaced axis: one centred row of `K` weights, plus the rows at the two ends
where a bounded window shifts inward.

Every interior sample reads `field[j - left + q - 1]` weighted by `weights[q]`, the same `K` numbers
throughout, so they reach the sweep as a tuple in registers with no per-cell gather. Storage is `O(K²)`,
independent of the axis length. A wrapping axis has no shifted rows at all, every sample being centred.
A plan of no rows, `K = 0`, describes a direction too short for the derivative, which is written
`masked` everywhere.

`left` and `right` count the nodes before and after the sample and are equal only for odd `K`: a 4-node
window reads one sample back and two forward, so two rows shift at the top end and one at the bottom.

The weights are built from the spacing. Mathematically they are the row [`axis_stencils`](@ref) holds,
and the table carries per-row round-off from reconstructing each window's coordinates, at `~1e-14`
relative on a 5-node first derivative; one exact row applied everywhere carries none.
"""
struct UniformStencilPlan{
    T<:AbstractFloat, V<:AbstractVector{T}, MI<:AbstractMatrix{<:Integer}, MW<:AbstractMatrix{T},
} <: AbstractStencilPlan{T}
    weights::V                # the centred row, shared by every interior sample
    left::Int                 # nodes before the sample, (K - 1) ÷ 2
    right::Int                # nodes after it, K - 1 - left
    n::Int                    # samples on the axis this describes
    order::Int
    wrap::Bool
    edge_idx::MI              # left + right rows, the shifted ones; empty when wrapping
    edge_wts::MW
end

"""
    TabulatedStencilPlan{T,MI,MW}

The weight set of a stretched axis: one row of node indices and weights per sample, the `n × K` pair of
matrices [`axis_stencils`](@ref) returns. Rows `left+1:n-right` read samples `j-left:j+right`, each with
its own weights; the rows at the two ends shift inward on a bounded axis and wrap on a periodic one, as
the uniform plan's do. `K = 0` describes a direction too short for the derivative.
"""
struct TabulatedStencilPlan{T<:AbstractFloat, MI<:AbstractMatrix{<:Integer}, MW<:AbstractMatrix{T}} <:
       AbstractStencilPlan{T}
    indices::MI
    weights::MW
    left::Int                 # nodes before the sample in an interior row, (K - 1) ÷ 2
    right::Int                # nodes after it, K - 1 - left
    order::Int
end

# `idx` and `w` are a table `axis_stencils` built, whose windows place the interior rows.
function TabulatedStencilPlan(
    idx::AbstractMatrix{<:Integer}, w::AbstractMatrix{T}, order::Integer,
) where {T}
    size(idx) == size(w) || throw(DimensionMismatch(
        "indices $(size(idx)) and weights $(size(w)) must have the same size",
    ))
    k = size(w, 2)
    left = max(k - 1, 0) ÷ 2
    return TabulatedStencilPlan{T,typeof(idx),typeof(w)}(idx, w, left, max(k - 1 - left, 0), Int(order))
end

"""
    nnodes(plan) -> Int

How many samples each row of `plan` reads; `0` where its direction carries no derivative.
"""
@inline nnodes(p::UniformStencilPlan) = length(p.weights)
@inline nnodes(p::TabulatedStencilPlan) = size(p.weights, 2)

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

With a `period`, an [`Axes.PeriodicUniformAxis`](@ref) gives the uniform form, its seam being one step wide. Any
other axis given a period gives the tabulated form, which describes whatever seam gap the period leaves.
"""
stencil_plan(
    x::AbstractVector{<:AbstractFloat}, order::Integer, nodes::Integer; period::Union{Nothing,Real} = nothing,
) = _axis_plan(x, Int(order), Int(nodes), period)

function _axis_plan(x::AbstractVector, ord::Int, k::Int, period)
    n = length(x)
    k ≥ ord + 1 || throw(ArgumentError(
        "an order-$ord derivative needs at least $(ord + 1) nodes, got $k",
    ))
    k ≤ n || throw(ArgumentError("cannot use $k nodes on an axis of $n samples"))
    return _stencil_plan(Axes.spacing_trait(x), x, ord, k, period)
end

# `k = 0` gives the plan of no rows, of the type the axis's plans take.
_stencil_plan(::Axes.NonuniformSpacing, x::AbstractVector, ord::Int, k::Int, period) =
    _tabulated_plan(x, ord, k, period)
_stencil_plan(::Axes.UniformSpacing, x::AbstractVector, ord::Int, k::Int, period::Real) =
    _tabulated_plan(x, ord, k, period)

function _tabulated_plan(x::AbstractVector{T}, ord::Int, k::Int, period) where {T}
    k == 0 && return TabulatedStencilPlan(Matrix{Int}(undef, length(x), 0), Matrix{T}(undef, length(x), 0), ord)
    return TabulatedStencilPlan(axis_stencils(x, ord, k; period = period)..., ord)
end

function _stencil_plan(
    ::Axes.UniformSpacing, x::Axes.PeriodicUniformAxis{T}, ord::Int, k::Int, ::Real,
) where {T}
    left = max(k - 1, 0) ÷ 2
    w = k == 0 ? T[] : _uniform_row(T, step(x), -left, k, ord)
    return UniformStencilPlan{T,Vector{T},Matrix{Int},Matrix{T}}(
        w, left, max(k - 1 - left, 0), length(x), ord, true, Matrix{Int}(undef, 0, k), Matrix{T}(undef, 0, k),
    )
end

function _stencil_plan(
    ::Axes.UniformSpacing, x::AbstractVector{T}, ord::Int, k::Int, ::Nothing,
) where {T}
    n = length(x)
    k == 0 && return UniformStencilPlan{T,Vector{T},Matrix{Int},Matrix{T}}(
        T[], 0, 0, n, ord, false, Matrix{Int}(undef, 0, 0), Matrix{T}(undef, 0, 0))
    h = T(Axes.spacing(x))
    left = (k - 1) ÷ 2
    right = k - 1 - left
    # A bounded window shifts inward at the ends: rows `1:left` all read the first `K` samples and rows
    # `n-right+1:n` the last `K`, each from its own sample's point of view.
    rows = vcat(1:left, (n - right + 1):n)
    edge_idx = Matrix{Int}(undef, length(rows), k)
    edge_wts = Matrix{T}(undef, length(rows), k)
    @inbounds for (r, j) in enumerate(rows)
        i0 = _window_start(j, k, 1, n)
        row = _uniform_row(T, h, i0 - j, k, ord)
        for q in 1:k
            edge_idx[r, q] = i0 + q - 1
            edge_wts[r, q] = row[q]
        end
    end
    return UniformStencilPlan{T,Vector{T},Matrix{Int},Matrix{T}}(
        _uniform_row(T, h, -left, k, ord), left, right, n, ord, false, edge_idx, edge_wts,
    )
end

"""
    _uniform_row(T, h, s, k, order) -> Vector{T}

The `k` weights of a row whose first node sits `s` steps of size `h` from the sample it is evaluated at.

`s = -left` is the centred row. Only the offsets enter, so this is the exact weight set for every sample
sharing that offset — on a uniform axis, all of them away from a shifted end.
"""
function _uniform_row(::Type{T}, h::T, s::Int, k::Int, ord::Int) where {T}
    w = Vector{T}(undef, k)
    fd_weights!(w, Matrix{T}(undef, k, ord + 1), T[T(s + q - 1) * h for q in 1:k], zero(T), ord)
    return w
end

"""
    plan_row(plan, j, nodes) -> (nodes, weights)

Row `j` of `plan`: the axis indices that sample reads and the weight on each, [`nnodes`](@ref) of them.
With `nodes = Val(K)` they are two `K`-tuples on the stack; with `nodes` an `Int` they are vectors
reading the values where the plan holds them.
"""
@inline function plan_row(p::UniformStencilPlan, j::Int, nodes::Val)
    left = p.left
    p.wrap && return (ntuple(q -> mod1(j - left + q - 1, p.n), nodes), _centred_row(p, nodes))
    if j ≤ left
        return (ntuple(q -> @inbounds(Int(p.edge_idx[j, q])), nodes),
                ntuple(q -> @inbounds(p.edge_wts[j, q]), nodes))
    elseif j > p.n - p.right
        r = left + (j - (p.n - p.right))
        return (ntuple(q -> @inbounds(Int(p.edge_idx[r, q])), nodes),
                ntuple(q -> @inbounds(p.edge_wts[r, q]), nodes))
    end
    return (ntuple(q -> j - left + q - 1, nodes), _centred_row(p, nodes))
end

@inline function plan_row(p::UniformStencilPlan, j::Int, ::Int)
    r = _end_row(p, j)
    return (_RowNodes(p, j, r), _RowWeights(p, r))
end

@inline plan_row(p::TabulatedStencilPlan, j::Int, nodes::Val) =
    (ntuple(q -> Int(@inbounds p.indices[j, q]), nodes),
     ntuple(q -> @inbounds(p.weights[j, q]), nodes))

@inline plan_row(p::TabulatedStencilPlan, j::Int, ::Int) = (view(p.indices, j, :), view(p.weights, j, :))

# Which row of the end tables sample `j` reads, or `0` where it reads the centred row.
@inline function _end_row(p::UniformStencilPlan, j::Int)
    p.wrap && return 0
    j ≤ p.left && return j
    j > p.n - p.right && return p.left + (j - (p.n - p.right))
    return 0
end

# Row `j` of a uniform plan read in place, `r` its end-table row or `0`.
struct _RowNodes{P<:UniformStencilPlan} <: AbstractVector{Int}
    plan::P
    j::Int
    r::Int
end

struct _RowWeights{T,P<:UniformStencilPlan{T}} <: AbstractVector{T}
    plan::P
    r::Int
end

Base.size(x::Union{_RowNodes,_RowWeights}) = (nnodes(x.plan),)

@inline function Base.getindex(x::_RowNodes, q::Int)
    @boundscheck checkbounds(x, q)
    p = x.plan
    x.r > 0 && return Int(@inbounds p.edge_idx[x.r, q])
    i = x.j - p.left + q - 1
    return p.wrap ? mod1(i, p.n) : i
end

@inline function Base.getindex(x::_RowWeights, q::Int)
    @boundscheck checkbounds(x, q)
    return x.r > 0 ? (@inbounds x.plan.edge_wts[x.r, q]) : (@inbounds x.plan.weights[q])
end

"""
    _centred_row(plan, nodes)

A uniform plan's interior weights: a tuple, the form the sweep holds in registers, for `nodes = Val(K)`;
the held vector for an `Int`.
"""
@inline _centred_row(p::UniformStencilPlan, nodes::Val) = ntuple(q -> @inbounds(p.weights[q]), nodes)
@inline _centred_row(p::UniformStencilPlan, ::Int) = p.weights

Base.show(io::IO, p::UniformStencilPlan{T}) where {T} = print(
    io, "UniformStencilPlan{", T, "}(order=", p.order, ", nodes=", nnodes(p), ", n=", p.n,
    p.wrap ? ", wrapping)" : ")",
)

Base.show(io::IO, p::TabulatedStencilPlan{T}) where {T} = print(
    io, "TabulatedStencilPlan{", T, "}(order=", p.order, ", nodes=", nnodes(p),
    ", n=", axis_length(p), ")",
)
