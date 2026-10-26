# ---------------------------------------------------------------------------
# Applying a held stencil plan
# ---------------------------------------------------------------------------
#
# A uniform axis's interior weights are the same `K` numbers at every sample, so they reach the sweep as
# a tuple and the innermost loop has constant coefficients. That shape pays where the differenced
# direction is the contiguous one; where it is a slower-varying one the row is hoisted out of the inner
# loop either way.
#
# A plan reads the same samples as the table — the node indices are identical — and its weights are the
# exact translation-invariant ones, where the table recomputes each row from its own window's coordinates
# and so carries per-row round-off.

"""
    apply_stencil!(out, field, plan, dim; mask=nothing, masked=zero, execution=SerialBackend()) -> out

Apply a held [`Discretization.stencil_plan`](@ref) along direction `dim` of `field`.

The primary form: the weights are built once, so a caller differencing many fields along one direction
pays for them once, and nothing is allocated per call. The `(out, field, x, dim; order, nodes)` forms
build a plan and call this, which is where their allocation comes from.

`mask` and `masked` behave as they do for the table forms: a cell whose stencil reads an inactive cell
is written `masked`.
"""
function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N},
    plan::Discretization.AbstractStencilPlan, dim::Integer;
    mask = nothing, masked = zero(S), execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {S,N}
    return _apply_plan_checked!(out, field, plan, Int(dim), mask, masked, execution, nothing)
end

# The plan form, with `fac` the metric factor a derivative writes each value with — see
# [`_RowFactors`](@ref) — or `nothing` for the plain sum.
function _apply_plan_checked!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, plan::Discretization.AbstractStencilPlan,
    d::Int, mask, masked, backend, fac,
) where {S,N}
    1 ≤ d ≤ N || throw(ArgumentError("direction $d is outside 1:$N"))
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    size(field, d) == Discretization.axis_length(plan) || throw(DimensionMismatch(
        "the plan describes an axis of $(Discretization.axis_length(plan)) samples but direction $d " *
        "of the field has $(size(field, d))",
    ))
    _check_mask_extent(mask, size(field), d)
    Discretization.nnodes(plan) == 0 && return fill!(out, masked)
    return _apply_plan!(out, field, plan, d, mask, masked, backend, fac)
end

# A tabulated plan holds exactly the two matrices the table form takes, so it goes straight there and
# shares that path's specializations; its matrices go where the backend reads.
function _apply_plan!(
    out, field, plan::Discretization.TabulatedStencilPlan, d::Int, mask, masked, backend, fac,
)
    b = Execution.resolve(backend)
    p = Execution.on_backend(b, plan)
    return _apply_table!(out, field, p.indices, p.weights, d, mask, masked, b, fac)
end

function _apply_plan!(
    out::AbstractArray{S,N}, field, plan::Discretization.UniformStencilPlan, d::Int,
    mask, masked, backend, fac,
) where {S,N}
    vm = _mask_rank(mask, Val(N))
    b = Execution.resolve(backend)
    # On a device each work item reads its own row from the plan: the centred row and the `O(K²)` end
    # rows, held where the backend reads them, as the mask is.
    if !_sweeps_on_host(b, out, field, mask, fac)
        p = Execution.on_backend(b, plan)
        m = Execution.on_backend(b, mask)
        return _dispatch_dim(d, vm) do vdim
            _dispatch_nodes(Discretization.nnodes(plan)) do vk
                _launch_plan!(out, field, p, m, masked, vdim, Val(N), vm, b, fac, vk)
            end
        end
    end
    return _dispatch_dim(d, vm) do vdim
        _dispatch_nodes(Discretization.nnodes(plan)) do vk
            _plan_sweep_host!(out, field, plan, mask, masked, vdim, Val(N), vm, fac, b, vk)
        end
    end
end

function _launch_plan!(
    out::AbstractArray{S,N}, field, plan, mask, masked, ::Val{dim}, ::Val{N}, ::Val{M}, backend, fac, vk,
) where {S,N,dim,M}
    ci = CartesianIndices(size(field))
    Execution.run_indices(length(ci), backend, Execution.Written(out)) do lin
        c = @inbounds ci[lin]
        I = Tuple(c)
        nodes, wts = Discretization.plan_row(plan, I[dim], vk)
        _plan_cell!(out, field, nodes, wts, mask, masked, I, c, Val(dim), Val(N), Val(M), fac)
    end
    return out
end

# One cell from its row, written from its own inputs alone — `_stencil_cell!` with the row in hand.
# `nodes` and `wts` are tuples for a node count in the type, vectors over the plan otherwise.
@inline function _plan_cell!(
    out::AbstractArray{S,N}, field, nodes, wts, mask, masked,
    I::NTuple{N,Int}, ci, ::Val{dim}, ::Val{N}, ::Val{M}, fac,
) where {S,N,dim,M}
    @inbounds begin
        if mask !== nothing && !mask[_spatial(I, Val(M))...]
            out[ci] = masked
            return nothing
        end
        acc = zero(S)
        for q in eachindex(nodes)
            J = ntuple(d -> d == dim ? Int(nodes[q]) : I[d], Val(N))
            if mask !== nothing && !mask[_spatial(J, Val(M))...]
                out[ci] = masked
                return nothing
            end
            acc += S(wts[q]) * S(field[J...])
        end
        out[ci] = _scaled(acc, _cell_factor(fac, I), masked)
    end
    return nothing
end

"""
    _plan_sweep_host!(out, field, plan, mask, masked, Val(dim), Val(N), Val(M), fac, backend, nodes) -> out

The host sweep for a plan. Same nest as [`_stencil_sweep_host!`](@ref) — Cartesian range walked
directly, nest split at `dim`, `nodes` the node count as `_dispatch_nodes` gives it — with a
uniform plan's interior row a tuple in registers and the shifted end rows read from its own `O(K²)`
table.

`fac` is the metric factor each value is written with — see [`_RowFactors`](@ref) — or `nothing` for
the plain sweep. Under `backend` a chunk is a contiguous range of cells, as for the table sweep.
"""
function _plan_sweep_host!(
    out::AbstractArray{S,N}, field, plan::Discretization.UniformStencilPlan,
    mask, masked, ::Val{dim}, ::Val{N}, ::Val{M}, fac, backend, vk,
) where {S,N,dim,M}
    sz = size(field)
    n = sz[dim]
    if _run_addressable(out, field, mask, fac)
        stride = prod(ntuple(d -> sz[d], Val(dim - 1)))
        npost = prod(ntuple(d -> sz[dim + d], Val(N - dim)))
        mpost = prod(ntuple(d -> sz[dim + d], Val(M - dim)))
        # The span written covers `stride ÷ sz[1]` direction-1 runs, one factor each. Its first run,
        # counted over directions `2:N` in column-major order, is `nruns · ((j−1) + n·p)`.
        nruns = stride ÷ sz[1]
        runlen = fac === nothing ? stride : sz[1]
        Execution.run_chunks(npost * stride * n, backend, Execution.Written(out)) do rng
            _plan_span_linear!(out, field, plan, mask, masked, Val(dim), first(rng) - 1, last(rng), n,
                               stride, mpost, nruns, runlen, fac, vk)
        end
        return out
    end
    # Anything that does not index linearly asks the array for its own indexing.
    pre = CartesianIndices(ntuple(d -> sz[d], Val(dim - 1)))
    post = CartesianIndices(ntuple(d -> sz[dim + d], Val(N - dim)))
    Execution.run_chunks(length(post) * n, backend) do rng
        @inbounds for u in rng
            q, jm1 = divrem(u - 1, n)
            nodes, wts = Discretization.plan_row(plan, jm1 + 1, vk)
            _plan_row!(out, field, nodes, wts, mask, masked, jm1 + 1, Tuple(post[q + 1]), pre,
                       Val(dim), Val(N), Val(M), fac)
        end
    end
    return out
end

# The cells with 0-based linear offsets `c0 : c1-1`, slab by slab (`dim == 1`) or row by row, as
# `_stencil_span_linear!` walks them. A slab is one direction-1 run, and `p` counts the batch axes too —
# the batch is the slowest, so slab `p` takes mask slab `p % mpost` and its factor from run `p`.
@inline function _plan_span_linear!(
    out, field, plan::Discretization.UniformStencilPlan, mask, masked, ::Val{dim}, c0::Int,
    c1::Int, n::Int, stride::Int, mpost::Int, nruns::Int, runlen::Int, fac, vk,
) where {dim}
    outer = stride * n
    c = c0
    while c < c1
        if dim == 1
            p = c ÷ outer
            off = p * outer
            jhi = min(outer, c1 - off)
            _plan_first_linear!(out, field, plan, mask, masked, off, (p % mpost) * outer, n, c - off + 1,
                                jhi, vk, _slab_factor(fac, p))
            c = off + jhi
        else
            u = c ÷ stride
            p, jm1 = divrem(u, n)
            base = u * stride
            thi = min(stride, c1 - base)
            nodes, wts = Discretization.plan_row(plan, jm1 + 1, vk)
            _plan_row_linear!(out, field, nodes, wts, mask, masked, jm1 + 1, p * outer,
                              (p % mpost) * outer, stride, fac, nruns * (jm1 + n * p), runlen,
                              c - base + 1, thi)
            c = base + thi
        end
    end
    return nothing
end

# `dim == 1`: the differenced direction is the contiguous one. The interior span carries the weights in
# registers and reads `field` at a fixed offset per node. Cells `jlo:jhi` of the slab are written; `h` is
# the run's metric factor.
@inline function _plan_first_linear!(
    out::AbstractArray{S}, field, plan::Discretization.UniformStencilPlan,
    mask, masked, off::Int, moff::Int, n::Int, jlo::Int, jhi::Int, vk, h,
) where {S}
    left = plan.left
    right = plan.right
    w = Discretization._centred_row(plan, vk)
    K = length(w)
    # The two ends, whichever way the axis closes: `plan_row` gives a shifted window on a bounded axis
    # and a wrapped one on a periodic axis. Only these rows differ between the two — a wrapping axis's
    # interior does not wrap either, so it takes the same constant-coefficient span below.
    @inbounds for j in jlo:min(left, jhi)
        _plan_cell_linear!(out, field, Discretization.plan_row(plan, j, vk)..., mask, masked,
                           off, moff, j, h)
    end
    # The interior: constant coefficients, nodes at a fixed offset. This is the span the plan exists for.
    @inbounds for j in max(left + 1, jlo):min(n - right, jhi)
        if mask !== nothing && !mask[moff + j]
            out[off + j] = masked
            continue
        end
        blocked = false
        if mask !== nothing
            for q in 1:K
                mask[moff + j - left + q - 1] || (blocked = true; break)
            end
        end
        if blocked
            out[off + j] = masked
            continue
        end
        acc = zero(S)
        for q in 1:K
            acc += S(w[q]) * S(field[off + j - left + q - 1])
        end
        out[off + j] = _scaled(acc, h, masked)
    end
    @inbounds for j in max(n - right + 1, jlo):jhi
        _plan_cell_linear!(out, field, Discretization.plan_row(plan, j, vk)..., mask, masked,
                           off, moff, j, h)
    end
    return nothing
end

# One cell from an explicit row, linearly addressed. The end rows and the wrapping case use it.
@inline function _plan_cell_linear!(
    out::AbstractArray{S}, field, nodes, wts, mask, masked, off::Int, moff::Int, j::Int, h,
) where {S}
    @inbounds begin
        if mask !== nothing && !mask[moff + j]
            out[off + j] = masked
            return nothing
        end
        acc = zero(S)
        for q in eachindex(nodes)
            ix = nodes[q]
            if mask !== nothing && !mask[moff + ix]
                out[off + j] = masked
                return nothing
            end
            acc += S(wts[q]) * S(field[off + ix])
        end
        out[off + j] = _scaled(acc, h, masked)
    end
    return nothing
end

# `dim != 1`: the row is hoisted out of the contiguous span, so it is read once per row on both plan
# shapes, and this one runs level with the table form. Cells `tlo:thi` of the span are written, through
# the runs of `runlen` cells they meet, run `r` with factor `_run_factor(fac, rbase, r)`.
@inline function _plan_row_linear!(
    out::AbstractArray{S}, field, nodes, wts, mask, masked, j::Int, off::Int, moff::Int, stride::Int,
    fac, rbase::Int, runlen::Int, tlo::Int, thi::Int,
) where {S}
    K = length(nodes)
    base = off + (j - 1) * stride
    mbase = moff + (j - 1) * stride
    @inbounds for r in ((tlo - 1) ÷ runlen):((thi - 1) ÷ runlen)
        h = _run_factor(fac, rbase, r)
        for i in max(tlo, r * runlen + 1):min(thi, (r + 1) * runlen)
            if mask !== nothing && !mask[mbase + i]
                out[base + i] = masked
                continue
            end
            acc = zero(S)
            blocked = false
            for q in 1:K
                src = off + (nodes[q] - 1) * stride + i
                msrc = moff + (nodes[q] - 1) * stride + i
                if mask !== nothing && !mask[msrc]
                    blocked = true
                    break
                end
                acc += S(wts[q]) * S(field[src])
            end
            out[base + i] = blocked ? masked : _scaled(acc, h, masked)
        end
    end
    return nothing
end

# The Cartesian fallback: correct for an offset array or a strided view, where the address arithmetic
# above does not apply.
@inline function _plan_row!(
    out::AbstractArray{S,N}, field, nodes, wts, mask, masked, j::Int, tpost::Tuple, pre, ::Val{dim},
    ::Val{N}, ::Val{M}, fac,
) where {S,N,dim,M}
    K = length(nodes)
    @inbounds for Ipre in pre
        tpre = Tuple(Ipre)
        I = (tpre..., j, tpost...)
        if mask !== nothing && !mask[_spatial(I, Val(M))...]
            out[I...] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:K
            J = (tpre..., nodes[q], tpost...)
            if mask !== nothing && !mask[_spatial(J, Val(M))...]
                blocked = true
                break
            end
            acc += S(wts[q]) * S(field[J...])
        end
        out[I...] = blocked ? masked : _scaled(acc, _cell_factor(fac, I), masked)
    end
    return nothing
end
