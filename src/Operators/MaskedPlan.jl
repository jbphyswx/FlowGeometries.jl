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
`order ≥ 2` derivative along a direction whose own scale factor varies along it.

The apply is one gather per cell, with no mask test and no weight solve, so it runs the same on the host
and in a kernel. Build one with [`masked_stencil_plan`](@ref).
"""
struct MaskedStencilPlan{R<:AbstractArray{Int32}, I<:AbstractMatrix{Int}, W<:AbstractMatrix{<:AbstractFloat}}
    row::R
    indices::I
    weights::W
    dim::Int
    order::Int
    distance::Bool
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
function masked_stencil_plan(
    x::AbstractVector{T}, mask::AbstractArray{Bool,M}, dim::Integer, order::Integer, nodes::Integer;
    period::Union{Nothing,Real} = nothing, policy::AbstractMaskPolicy = ReduceInRun(),
) where {T<:AbstractFloat,M}
    d = Int(dim)
    ord = Int(order)
    1 ≤ d ≤ M || throw(ArgumentError("direction $d is outside the mask's 1:$M"))
    size(mask, d) == length(x) || throw(DimensionMismatch(
        "axis has $(length(x)) samples but direction $d of the mask has $(size(mask, d))",
    ))
    n = length(x)
    k = policy isa ReduceInRun ? min(Int(nodes), n) : Int(nodes)
    if policy isa ReduceInRun && n < ord + 1
        return _empty_masked_plan(T, mask, d, ord, k, false)
    end
    idx, wts = Discretization.axis_stencils(x, ord, k; period = period)
    return _masked_stencil_plan(x, mask, d, ord, k, period, policy, idx, wts)
end

function masked_stencil_plan(
    grid::Grids.StructuredGrid{T,G,N}, dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
    policy::AbstractMaskPolicy = ReduceInRun(),
) where {T,G,N}
    d = Int(dim)
    1 ≤ d ≤ N || throw(ArgumentError("direction $d is outside 1:$N"))
    _along_arc(grid, d, Int(order)) &&
        return _distance_stencil_plan(grid, _host_mask(Grids.mask(grid)), d, Int(order), Int(nodes), policy)
    return masked_stencil_plan(
        _host_array(Grids.coordinates(grid, d)), _host_mask(Grids.mask(grid)), d, order, nodes;
        period = Grids.isperiodic(grid, d) ? Grids.period(grid, d) : nothing, policy = policy,
    )
end

# No derivative of this order exists anywhere on an axis this short.
_empty_masked_plan(::Type{T}, mask, d::Int, ord::Int, k::Int, distance::Bool) where {T} =
    MaskedStencilPlan(zeros(Int32, size(mask)), Matrix{Int}(undef, 0, k), Matrix{T}(undef, 0, k), d, ord, distance)

# An array as host data, which the build walks element by element: a device array is copied over.
@inline _host_array(a::Array) = a
@inline _host_array(a::BitArray) = a
@inline _host_array(a::AbstractRange) = a
@inline _host_array(a::AbstractArray) = Array(a)

# A grid with no inactive cell holds its mask as a size alone, readable anywhere.
@inline _host_mask(m::Grids.AllActive) = m
@inline _host_mask(m::AbstractArray{Bool}) = _host_array(m)

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
    return MaskedStencilPlan(row, indices, weights, d, ord, false)
end

# Every cell's weights solved over distance: node `q` of the window sits at
# `Geometry.line_distance(geo, point, d, ξᵢ, ξ_q)` from the cell along its own line, so the weights give
# the derivative with respect to distance itself. The window is the axis table's where it holds only
# active cells and the policy's degraded one elsewhere. A line's distances depend on the coordinates its
# scale factor reads, so cells sharing a window and every coordinate outside the metric-invariant
# directions share a row. A cell whose own scale factor is at the geometry's floor is written `masked`,
# as in the factor form.
function _distance_stencil_plan(
    grid::Grids.StructuredGrid{T,G,N}, mask::AbstractArray{Bool,N}, d::Int, ord::Int, nodes::Int, policy,
) where {T,G,N}
    x = _host_array(Grids.coordinates(grid, d))
    n = length(x)
    k = policy isa ReduceInRun ? min(nodes, n) : nodes
    policy isa ReduceInRun && n < ord + 1 && return _empty_masked_plan(T, mask, d, ord, k, true)
    Discretization._check_stencil_shape(n, k, ord)
    geo = Grids.grid_geometry(grid)
    axes = map(_host_array, Grids.coordinates(grid))
    floor_ = Discretization.metric_floor(geo, d)
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
        if !all(q -> mask[_at_dim(I, d, mod1(start + q - 1, n))...], 1:k)
            policy isa BlankMasked && continue
            kk, start = _degraded_window(mask, I, d, i, n, k, ord, wrap, policy)
            iszero(kk) && continue
        end
        pt = map((a, j) -> T(a[j]), axes, I)
        abs(Geometry.scale_factors(geo, pt)[d]) ≤ floor_ && continue
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
        row, permutedims(reshape(out_idx, k, ne)), permutedims(reshape(out_w, k, ne)), d, ord, true,
    )
end

"""
    apply_stencil!(out, field, plan::MaskedStencilPlan; masked=zero, backend=SerialBackend()) -> out

Apply a held [`MaskedStencilPlan`](@ref) to `field`, which may carry trailing batch axes past the mask's.
The plan's arrays live where `backend` runs.
"""
function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, plan::MaskedStencilPlan;
    masked = zero(S), backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {S,N}
    return _apply_masked_plan!(out, field, plan, masked, backend, nothing)
end

"""
    derivative!(out, field, grid, plan::MaskedStencilPlan; masked=zero, backend=SerialBackend()) -> out

[`derivative!`](@ref) along `plan.dim` from a held [`MaskedStencilPlan`](@ref) of `grid`. Where the
derivative needs weights over distance, the plan must be the grid form's,
`masked_stencil_plan(grid, dim; order, nodes, policy)`.
"""
function derivative!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T,G,N},
    plan::MaskedStencilPlan; masked = zero(S), backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {S,G,T,N,NA}
    _check_batched(out, field, grid, Val(N), plan.dim)
    plan.distance && return _apply_masked_plan!(out, field, plan, masked, backend, nothing)
    _along_arc(grid, plan.dim, plan.order) && throw(ArgumentError(
        "an order-$(plan.order) derivative along direction $(plan.dim) of this grid needs weights over " *
        "distance, and this plan holds them over the coordinate; build it with " *
        "`masked_stencil_plan(grid, $(plan.dim); order = $(plan.order), …)`",
    ))
    return _apply_masked_plan!(out, field, plan, masked, backend,
                               _derivative_factors(grid, plan.dim, backend, plan.order))
end

function _apply_masked_plan!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, plan::MaskedStencilPlan, masked, backend, fac,
) where {S,N}
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    vm = _plan_extent(plan.row, field)
    b = Execution.resolve(backend)
    return _dispatch_dim(plan.dim, vm) do vdim
        _dispatch_nodes(size(plan.weights, 2)) do vk
            _masked_plan_sweep!(out, field, plan.row, plan.indices, plan.weights, vdim, masked, vk, vm, b, fac)
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
