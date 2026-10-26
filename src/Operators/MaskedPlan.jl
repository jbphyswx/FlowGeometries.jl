# ---------------------------------------------------------------------------
# A direction's weights on a masked grid, resolved per cell once
# ---------------------------------------------------------------------------

"""
    MaskedStencilPlan{R,I,W}

The `order`-th derivative weights of one direction of a masked grid, resolved per cell. `row[J]`, over
the mask's own axes, names the row of `(indices, weights)` cell `J` reads: the axis table's row `J[dim]`
where its window holds only active cells, a row rebuilt over the cell's active run where the policy
degrades it, and `0` where the cell is written `masked`. Rebuilt rows follow the table's `n` rows, each
padded with zero weights to the table's width.

`distance` marks weights solved over the window's distances, per line, with no table rows: the form of an
`order ≥ 2` derivative along a direction whose own scale factor varies along it. `mask` is the mask the
rows were resolved for; [`derivative!`](@ref) refuses a grid with another.

The apply is one gather per cell, with no mask test and no weight solve, so it runs the same on the host
and in a kernel. Build one with [`masked_stencil_plan`](@ref).
"""
struct MaskedStencilPlan{R<:AbstractArray{Int32}, I<:AbstractMatrix{Int}, W<:AbstractMatrix{<:AbstractFloat},
                         K<:AbstractArray{Bool}}
    row::R
    indices::I
    weights::W
    dim::Int
    order::Int
    distance::Bool
    mask::K
end

Base.show(io::IO, p::MaskedStencilPlan) = print(
    io, "MaskedStencilPlan(dim=", p.dim, ", order=", p.order, ", nodes=", size(p.weights, 2), ", rows=",
    size(p.weights, 1), p.distance ? ", in distance)" : ")",
)

"""
    masked_stencil_plan(x, mask, dim, order, nodes; period=nothing, policy=ReduceInRun()) -> MaskedStencilPlan
    masked_stencil_plan(grid, dim; order=1, nodes=order+1, policy=ReduceInRun()) -> MaskedStencilPlan

The `order`-th derivative's weights along direction `dim` of a field masked by `mask`, under `policy`,
for every cell: [`apply_stencil!`](@ref)'s answer for that mask, held. The axis `x`, its `period` and
the mask fix it, so a caller differencing many fields builds it once. Built on the host; `Adapt` moves it
to a device.

The grid form solves the weights over distance where [`derivative!`](@ref) needs them so: at
`order ≥ 2` along a direction outside [`Geometry.scale_invariant_directions`](@ref).
"""
masked_stencil_plan(
    x::AbstractVector{<:AbstractFloat}, mask::AbstractArray{Bool}, dim::Integer, order::Integer, nodes::Integer;
    period::Union{Nothing,Real} = nothing, policy::AbstractMaskPolicy = ReduceInRun(),
) = _masked_axis_plan(x, mask, Int(dim), Int(order), Int(nodes), period, policy)

function masked_stencil_plan(
    grid::Grids.StructuredGrid{T,G,N}, dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
    policy::AbstractMaskPolicy = ReduceInRun(),
) where {T,G,N}
    d = Int(dim)
    1 ≤ d ≤ N || throw(ArgumentError("direction $d is outside 1:$N"))
    msk = _host_mask(Grids.mask(grid))
    _along_arc(grid, d, Int(order)) && return _distance_stencil_plan(grid, msk, d, Int(order), Int(nodes), policy)
    return _dispatch_dim(d, Val(N)) do vd
        _masked_axis_plan(Axes._host_array(Grids.coordinates(grid)[_val(vd)]), msk, _val(vd), Int(order), Int(nodes),
                          Grids._wrap_period(grid, _val(vd)), policy)
    end
end

function _masked_axis_plan(
    x::AbstractVector{T}, mask::AbstractArray{Bool,M}, d::Int, ord::Int, nodes::Int, period, policy,
) where {T<:AbstractFloat,M}
    1 ≤ d ≤ M || throw(ArgumentError("direction $d is outside the mask's 1:$M"))
    size(mask, d) == length(x) || throw(DimensionMismatch(
        "axis has $(length(x)) samples but direction $d of the mask has $(size(mask, d))",
    ))
    n = length(x)
    k = policy isa ReduceInRun ? min(nodes, n) : nodes
    if policy isa ReduceInRun && n < ord + 1
        return _empty_masked_plan(T, mask, d, ord, k, false)
    end
    idx, wts = Discretization.axis_stencils(x, ord, k; period = period)
    return _masked_stencil_plan(x, mask, d, ord, k, period, policy, idx, wts)
end

# No derivative of this order exists anywhere on an axis this short.
_empty_masked_plan(::Type{T}, mask, d::Int, ord::Int, k::Int, distance::Bool) where {T} =
    MaskedStencilPlan(zeros(Int32, size(mask)), Matrix{Int}(undef, 0, k), Matrix{T}(undef, 0, k), d, ord, distance,
                      mask)

# A grid with no inactive cell holds its mask as a size alone, readable anywhere.
@inline _host_mask(m::Grids.AllActive) = m
@inline _host_mask(m::AbstractArray{Bool}) = Axes._host_array(m)

# Each active cell takes the table's row where every node of it is active. Elsewhere the window is the
# largest the policy allows inside the cell's active run, shifted to fit, and its weights are solved
# over the run's own coordinates; a window is fixed by its sample, first node and length, so cells
# sharing all three share a row.
function _masked_stencil_plan(
    x::AbstractVector{T}, mask::AbstractArray{Bool,M}, d::Int, ord::Int, k::Int, period, policy,
    idx::AbstractMatrix{<:Integer}, wts::AbstractMatrix,
) where {T,M}
    n = length(x)
    P = period === nothing ? zero(T) : T(period) * Axes.wrap_sign(x)
    wrap = period !== nothing
    row = zeros(Int32, size(mask))
    seen = Dict{NTuple{3,Int},Int32}()
    extra_idx = Int[]
    extra_w = T[]
    wbuf = Vector{T}(undef, k)
    cbuf = Matrix{T}(undef, k, ord + 1)
    nbuf = Vector{T}(undef, k)
    @inbounds for c in CartesianIndices(mask)
        mask[c] || continue
        J = Tuple(c)
        i = J[d]
        if all(q -> mask[_at_dim(J, d, Int(idx[i, q]))...], 1:k)
            row[c] = Int32(i)
            continue
        end
        policy isa BlankMasked && continue
        kk, start = _degraded_window(mask, J, d, i, n, k, ord, wrap, policy)
        iszero(kk) && continue
        r = get(seen, (i, start, kk), Int32(0))
        if iszero(r)
            for q in 1:kk
                raw = start + q - 1
                nbuf[q] = x[mod1(raw, n)] + T(fld(raw - 1, n)) * P
            end
            Discretization._fd_weights!(wbuf, cbuf, nbuf, kk, T(x[i]), ord)
            for q in 1:k
                push!(extra_idx, q ≤ kk ? mod1(start + q - 1, n) : i)
                push!(extra_w, q ≤ kk ? wbuf[q] : zero(T))
            end
            r = Int32(n + length(seen) + 1)
            seen[(i, start, kk)] = r
        end
        row[c] = r
    end
    ne = length(seen)
    indices = vcat(Matrix{Int}(idx), permutedims(reshape(extra_idx, k, ne)))
    weights = vcat(Matrix{T}(wts), permutedims(reshape(extra_w, k, ne)))
    return MaskedStencilPlan(row, indices, weights, d, ord, false, mask)
end

# Every cell's weights solved over distance: node `q` of the window sits at
# `Geometry.line_distance(geo, point, d, ξᵢ, ξ_q)` from the cell along its own line, so the weights give
# the derivative with respect to distance itself. The window is the axis table's where it holds only
# active cells and the policy's degraded one elsewhere. A line's distances depend on the coordinates its
# scale factor reads, so cells sharing a window and every coordinate outside the metric-invariant
# directions share a row. A cell whose own scale factor is zero, as at a pole, is written `masked`, as in
# the factor form.
_distance_stencil_plan(
    grid::Grids.StructuredGrid{T,G,N}, mask::AbstractArray{Bool,N}, d::Int, ord::Int, nodes::Int, policy,
) where {T,G,N} = _dispatch_dim(d, Val(N)) do vd
    _distance_stencil_plan(grid, mask, vd, ord, nodes, policy)
end

function _distance_stencil_plan(
    grid::Grids.StructuredGrid{T,G,N}, mask::AbstractArray{Bool,N}, ::Val{d}, ord::Int, nodes::Int, policy,
) where {T,G,N,d}
    x = Axes._host_array(Grids.coordinates(grid)[d])
    n = length(x)
    k = policy isa ReduceInRun ? min(nodes, n) : nodes
    policy isa ReduceInRun && n < ord + 1 && return _empty_masked_plan(T, mask, d, ord, k, true)
    Discretization._check_stencil_shape(n, k, ord)
    geo = Grids.grid_geometry(grid)
    axes = map(Axes._host_array, Grids.coordinates(grid))
    invariant = Geometry.metric_invariant_directions(geo)
    wrap = Grids.isperiodic(grid, d)
    P = wrap ? T(Grids.period(grid, d)) * Axes.wrap_sign(x) : zero(T)
    half = (k - 1) ÷ 2
    row = zeros(Int32, size(mask))
    seen = Dict{NTuple{N + 2,Int},Int32}()
    out_idx = Int[]
    out_w = T[]
    wbuf = Vector{T}(undef, k)
    cbuf = Matrix{T}(undef, k, ord + 1)
    sbuf = Vector{T}(undef, k)
    @inbounds for c in CartesianIndices(mask)
        mask[c] || continue
        I = Tuple(c)
        i = I[d]
        start = wrap ? i - half : Discretization._window_start(i, k, 1, n)
        kk = k
        if !_window_active(mask, I, d, start, k, n)
            policy isa BlankMasked && continue
            kk, start = _degraded_window(mask, I, d, i, n, k, ord, wrap, policy)
            iszero(kk) && continue
        end
        pt = map((a, j) -> T(a[j]), axes, I)
        iszero(Geometry.scale_factors(geo, pt)[d]) && continue
        key = (start, kk, ntuple(e -> e in invariant ? 0 : I[e], Val(N))...)
        r = get(seen, key, Int32(0))
        if iszero(r)
            for q in 1:kk
                raw = start + q - 1
                sbuf[q] = Geometry.line_distance(geo, pt, d, T(x[i]), x[mod1(raw, n)] + T(fld(raw - 1, n)) * P)
            end
            Discretization._fd_weights!(wbuf, cbuf, sbuf, kk, zero(T), ord)
            for q in 1:k
                push!(out_idx, q ≤ kk ? mod1(start + q - 1, n) : i)
                push!(out_w, q ≤ kk ? wbuf[q] : zero(T))
            end
            r = Int32(length(seen) + 1)
            seen[key] = r
        end
        row[c] = r
    end
    ne = length(seen)
    return MaskedStencilPlan(
        row, permutedims(reshape(out_idx, k, ne)), permutedims(reshape(out_w, k, ne)), d, ord, true, mask,
    )
end

# Whether every node of the `k`-node window from `start` along `d` through cell `I` is active.
@inline function _window_active(mask, I::NTuple{N,Int}, d::Int, start::Int, k::Int, n::Int) where {N}
    @inbounds for q in 1:k
        mask[_at_dim(I, d, mod1(start + q - 1, n))...] || return false
    end
    return true
end

"""
    apply_stencil!(out, field, plan::MaskedStencilPlan; masked=zero, execution=SerialBackend()) -> out

Apply a held [`MaskedStencilPlan`](@ref) to `field`, which may carry trailing batch axes past the mask's.
The plan is read in `execution`'s memory: one held on the host is moved there for the call, and one moved
there once with `Execution.on_backend` is read as it is.
"""
function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, plan::MaskedStencilPlan;
    masked = zero(S), execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {S,N}
    return _apply_masked_plan!(out, field, plan, masked, execution, nothing)
end

"""
    derivative!(out, field, grid, plan::MaskedStencilPlan; masked=zero, execution=SerialBackend()) -> out

[`derivative!`](@ref) along `plan.dim` from a held [`MaskedStencilPlan`](@ref) of `grid`, built for
`grid`'s mask. Where the derivative needs weights over distance, the plan must be the grid form's,
`masked_stencil_plan(grid, dim; order, nodes, policy)`.
"""
function derivative!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T,G,N},
    plan::MaskedStencilPlan; masked = zero(S), execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {S,G,T,N,NA}
    _check_batched(out, field, grid, Val(N), plan.dim)
    b = Execution.resolve(execution)
    p = Execution.on_backend(b, plan)
    _same_mask(Grids.mask(grid), p.mask) || throw(ArgumentError(
        "this plan's rows were resolved for another mask than the grid's; build it on the grid it differences",
    ))
    p.distance && return _apply_held_plan!(out, field, p, masked, b, nothing)
    _along_arc(grid, p.dim, p.order) && throw(ArgumentError(
        "an order-$(p.order) derivative along direction $(p.dim) of this grid needs weights over " *
        "distance, and this plan holds them over the coordinate; build it with " *
        "`masked_stencil_plan(grid, $(p.dim); order = $(p.order), …)`",
    ))
    return _apply_held_plan!(out, field, p, masked, b, _derivative_factors(grid, p.dim, b, p.order))
end

# The plan's own mask object costs nothing to check. Two masks of one array type are in one memory, the
# plan having been moved to the backend first, and `==` runs there; masks of two types are compared on
# the host.
@inline _same_mask(a::A, b::A) where {A<:AbstractArray{Bool}} = a === b || a == b
@inline _same_mask(a::AbstractArray{Bool}, b::AbstractArray{Bool}) = _host_mask(a) == _host_mask(b)

function _apply_masked_plan!(out, field, plan::MaskedStencilPlan, masked, backend, fac)
    b = Execution.resolve(backend)
    return _apply_held_plan!(out, field, Execution.on_backend(b, plan), masked, b, fac)
end

# `p` is already in the memory `b`'s loops read.
function _apply_held_plan!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, p::MaskedStencilPlan, masked, b, fac,
) where {S,N}
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    vm = _plan_extent(p.row, field)
    return _dispatch_dim(p.dim, vm) do vdim
        _dispatch_nodes(size(p.weights, 2)) do vk
            _masked_plan_sweep!(out, field, p.row, p.indices, p.weights, vdim, masked, vk, vm, b, fac)
        end
    end
end

# The plan's rank `M`, after checking it covers the field's leading `M` axes.
function _plan_extent(row::AbstractArray{<:Any,M}, field::AbstractArray{<:Any,N}) where {M,N}
    (M ≤ N && size(row) == ntuple(d -> size(field, d), Val(M))) || throw(DimensionMismatch(
        "the plan covers $(size(row)) but the field has size $(size(field))",
    ))
    return Val(M)
end

# The host walks the Cartesian range itself; a launch recovers each cell from its linear index.
function _masked_plan_sweep!(
    out::AbstractArray{S,N}, field, row, indices, weights, vdim::Val, masked, nodes, vm::Val, b, fac,
) where {S,N}
    ci = CartesianIndices(size(field))
    if b isa CB.AbstractSerialBackend
        @inbounds for c in ci
            _masked_plan_cell!(out, field, row, indices, weights, vdim, masked, nodes, Tuple(c), c, vm, fac)
        end
        return out
    end
    Execution.run_indices(length(ci), b, Execution.Written(out)) do lin
        c = @inbounds ci[lin]
        _masked_plan_cell!(out, field, row, indices, weights, vdim, masked, nodes, Tuple(c), c, vm, fac)
    end
    return out
end

@inline function _masked_plan_cell!(
    out::AbstractArray{S,N}, field, row, indices, weights, ::Val{dim}, masked, nodes,
    I::NTuple{N,Int}, ci, ::Val{M}, fac,
) where {S,N,dim,M}
    @inbounds begin
        r = Int(row[_spatial(I, Val(M))...])
        if iszero(r)
            out[ci] = masked
            return nothing
        end
        acc = zero(S)
        for q in 1:_nodecount(nodes)
            J = ntuple(d -> d == dim ? Int(indices[r, q]) : I[d], Val(N))
            acc += S(weights[r, q]) * S(field[J...])
        end
        out[ci] = _scaled(acc, _cell_factor(fac, I), masked)
    end
    return nothing
end
