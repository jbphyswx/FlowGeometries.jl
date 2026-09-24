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
    apply_stencil!(out, field, plan, dim; mask=nothing, masked=zero, backend=nothing) -> out

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
    mask = nothing, masked = zero(S), backend = nothing,
) where {S,N}
    return _apply_plan_checked!(out, field, plan, Int(dim), mask, masked, backend, nothing)
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
    return _apply_plan!(out, field, plan, d, mask, masked, backend, fac)
end

# A tabulated plan holds exactly the two matrices the table form takes, so it goes straight there and
# shares that path's specializations.
@inline _apply_plan!(
    out, field, plan::Discretization.TabulatedStencilPlan, d::Int, mask, masked, backend, fac,
) = _apply_table!(out, field, plan.indices, plan.weights, d, mask, masked, backend, fac)

function _apply_plan!(
    out::AbstractArray{S,N}, field, plan::Discretization.UniformStencilPlan{T,K}, d::Int,
    mask, masked, backend, fac,
) where {S,N,T,K}
    # The device path takes the per-cell body, which reads a row whatever the plan's form — the constant
    # coefficients are a host loop-shape win and a launch has its own.
    if backend !== nothing
        idx, w = _plan_tables(plan)
        return _apply_table!(out, field, idx, w, d, mask, masked, backend, fac)
    end
    vm = _mask_rank(mask, Val(N))
    return _dispatch_dim(d, vm) do vdim
        _plan_sweep_host!(out, field, plan, mask, masked, vdim, Val(N), vm, fac)
    end
end

# The full `n × K` tables a uniform plan describes, materialized. Only the device path asks for them:
# a launch reads one row per work item, so there is nothing for constant coefficients to save.
function _plan_tables(plan::Discretization.UniformStencilPlan{T,K}) where {T,K}
    n = plan.n
    idx = Matrix{Int}(undef, n, K)
    w = Matrix{T}(undef, n, K)
    @inbounds for j in 1:n
        nodes, wts = Discretization.plan_row(plan, j)
        for q in 1:K
            idx[j, q] = nodes[q]
            w[j, q] = wts[q]
        end
    end
    return idx, w
end

"""
    _plan_sweep_host!(out, field, plan, mask, masked, Val(dim), Val(N), Val(M), fac) -> out

The host sweep for a plan. Same nest as [`_stencil_sweep_host!`](@ref) — Cartesian range walked
directly, nest split at `dim`, node count in the type — with a uniform plan's interior row a tuple in
registers and the shifted end rows read from its own `O(K²)` table.

`fac` is the metric factor each value is written with — see [`_RowFactors`](@ref) — or `nothing` for
the plain sweep.
"""
function _plan_sweep_host!(
    out::AbstractArray{S,N}, field, plan::Discretization.UniformStencilPlan{T,K},
    mask, masked, ::Val{dim}, ::Val{N}, ::Val{M}, fac = nothing,
) where {S,N,T,K,dim,M}
    sz = size(field)
    n = sz[dim]
    if _run_addressable(out, field, mask, fac)
        stride = prod(ntuple(d -> sz[d], Val(dim - 1)))
        npost = prod(ntuple(d -> sz[dim + d], Val(N - dim)))
        outer = stride * sz[dim]
        mpost = prod(ntuple(d -> sz[dim + d], Val(M - dim)))
        if dim == 1
            # The slab written is one direction-1 run, and `p` counts the batch axes too — the batch is
            # the slowest, so slab `p` takes mask slab `p % mpost` and its factor from run `p`.
            for p in 0:(npost - 1)
                _plan_first_linear!(out, field, plan, mask, masked, p * outer,
                                    (p % mpost) * outer, n, Val(K), _slab_factor(fac, p))
            end
            return out
        end
        # The span written covers `stride ÷ sz[1]` direction-1 runs, one factor each. Its first run,
        # counted over directions `2:N` in column-major order, is `nruns · ((j−1) + n·p)`.
        nruns = stride ÷ sz[1]
        runlen = fac === nothing ? stride : sz[1]
        for p in 0:(npost - 1), j in 1:n
            nodes, wts = Discretization.plan_row(plan, j)
            _plan_row_linear!(out, field, nodes, wts, mask, masked, j, p * outer,
                              (p % mpost) * outer, stride, Val(K), fac, nruns * ((j - 1) + n * p),
                              runlen)
        end
        return out
    end
    # Anything that does not index linearly asks the array for its own indexing.
    pre = CartesianIndices(ntuple(d -> sz[d], Val(dim - 1)))
    post = CartesianIndices(ntuple(d -> sz[dim + d], Val(N - dim)))
    @inbounds for Ipost in post, j in 1:n
        nodes, wts = Discretization.plan_row(plan, j)
        _plan_row!(out, field, nodes, wts, mask, masked, j, Tuple(Ipost), pre, Val(K),
                   Val(dim), Val(N), Val(M), fac)
    end
    return out
end

# A tabulated plan holds the two matrices the table sweep takes, and that sweep has the same nest and
# the same span boundaries. `K` is in the plan's type, so the node count needs no runtime switch.
@inline _plan_sweep_host!(
    out::AbstractArray{S,N}, field, plan::Discretization.TabulatedStencilPlan{T,K},
    mask, masked, vdim::Val, ::Val{N}, vm::Val, fac = nothing,
) where {S,N,T,K} =
    _stencil_sweep_host!(out, field, plan.indices, plan.weights, mask, masked, vdim, Val(K),
                         Val(N), vm, fac)

# `dim == 1`: the differenced direction is the contiguous one. The interior span carries the weights in
# registers and reads `field` at a fixed offset per node. `h` is the run's metric factor.
@inline function _plan_first_linear!(
    out::AbstractArray{S}, field, plan::Discretization.UniformStencilPlan{T,K},
    mask, masked, off::Int, moff::Int, n::Int, ::Val{K}, h,
) where {S,T,K}
    left = plan.left
    right = plan.right
    w = plan.weights
    # The two ends, whichever way the axis closes: `plan_row` gives a shifted window on a bounded axis
    # and a wrapped one on a periodic axis. Only these rows differ between the two — a wrapping axis's
    # interior does not wrap either, so it takes the same constant-coefficient span below.
    @inbounds for j in 1:left
        _plan_cell_linear!(out, field, Discretization.plan_row(plan, j)..., mask, masked,
                           off, moff, j, Val(K), h)
    end
    # The interior: constant coefficients, nodes at a fixed offset. This is the span the plan exists for.
    @inbounds for j in (left + 1):(n - right)
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
    @inbounds for j in (n - right + 1):n
        _plan_cell_linear!(out, field, Discretization.plan_row(plan, j)..., mask, masked,
                           off, moff, j, Val(K), h)
    end
    return nothing
end

# One cell from an explicit row, linearly addressed. The end rows and the wrapping case use it.
@inline function _plan_cell_linear!(
    out::AbstractArray{S}, field, nodes::NTuple{K,Int}, wts::NTuple{K,Real},
    mask, masked, off::Int, moff::Int, j::Int, ::Val{K}, h,
) where {S,K}
    @inbounds begin
        if mask !== nothing && !mask[moff + j]
            out[off + j] = masked
            return nothing
        end
        acc = zero(S)
        for q in 1:K
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
# shapes, and this one runs level with the table form. The span is walked as `stride ÷ runlen` runs,
# run `r` written with factor `_run_factor(fac, rbase, r)`.
@inline function _plan_row_linear!(
    out::AbstractArray{S}, field, nodes::NTuple{K,Int}, wts::NTuple{K,Real},
    mask, masked, j::Int, off::Int, moff::Int, stride::Int, ::Val{K}, fac, rbase::Int, runlen::Int,
) where {S,K}
    base = off + (j - 1) * stride
    mbase = moff + (j - 1) * stride
    @inbounds for r in 0:((stride ÷ runlen) - 1)
        h = _run_factor(fac, rbase, r)
        for t in 1:runlen
            i = r * runlen + t
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
    out::AbstractArray{S,N}, field, nodes::NTuple{K,Int}, wts::NTuple{K,Real},
    mask, masked, j::Int, tpost::Tuple, pre, ::Val{K}, ::Val{dim}, ::Val{N}, ::Val{M}, fac,
) where {S,N,K,dim,M}
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
