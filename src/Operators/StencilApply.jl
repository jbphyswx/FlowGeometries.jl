"""
    apply_stencil!(out, field, x, dim; order=1, nodes=order+1, period=nothing,
                   mask=nothing, masked=zero) -> out
    apply_stencil!(out, field, indices, weights, dim; mask=nothing, masked=zero) -> out

Apply a weight set along direction `dim` of `field`, writing
`out[I] = Σ_q weights[I[dim], q] · field[…, indices[I[dim], q], …]`.

This is the one field-touching operation here, and it is here because every convention it needs is
already settled elsewhere in the package rather than being the caller's to choose: the result sits at
the same location as the input, so there is no staggering decision; the stencil shifts inward at a
bounded end and wraps on a periodic one, which is [`fd_weights`](@ref)'s stated boundary behaviour and
removes any need for a halo. What is *not* here is anything that does need those choices — a staggered
difference, or a multi-direction operator like a divergence or a curl, which additionally needs a
result location and a boundary-condition policy.

Pass the axis and an order to have the weights built for you, or precomputed `indices`/`weights` from
[`axis_stencils`](@ref) to reuse them across many fields.

With a `mask`, a cell is written as `masked` when it is inactive **or when its stencil reads an
inactive cell** — the derivative there is not determined by the active data, so it is not invented.
`out` and `field` may not alias.
"""
function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, x::AbstractVector{<:AbstractFloat},
    dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
    period::Union{Nothing,Real} = nothing, mask = nothing, masked = zero(S), backend = nothing,
    policy::AbstractMaskPolicy = BlankMasked(), scratch = nothing,
) where {S,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    size(field, dim) == length(x) || throw(DimensionMismatch(
        "axis has $(length(x)) samples but direction $dim of the field has $(size(field, dim))",
    ))
    ord = Int(order)
    k = Int(nodes)
    if policy isa ReduceInRun
        # Under this policy `nodes` is a CEILING, not a demand. The end of the axis bounds a window
        # exactly as the end of an active run does — the policy already says a run too short for
        # `nodes` uses the largest window it can hold — so the two are made to behave the same way
        # rather than one degrading and the other erroring. A single-latitude strip, a two-level
        # column and a one-cell channel are ordinary grids, not mistakes.
        if length(x) < ord + 1
            fill!(out, masked)      # no derivative of this order exists anywhere on such an axis
            return out
        end
        k = min(k, length(x))
    end
    # A plan under the blanking policy, which is what the register-resident uniform weights need; the
    # degrading policies rebuild a window from the axis and so want the table.
    if policy isa BlankMasked || mask === nothing
        return apply_stencil!(out, field, Discretization.stencil_plan(x, ord, k; period = period),
                              dim; mask = mask, masked = masked, backend = backend)
    end
    idx, wts = Discretization.axis_stencils(x, ord, k; period = period)
    return apply_stencil!(out, field, x, idx, wts, dim; order = ord, period = period, mask = mask,
                          masked = masked, backend = backend, policy = policy, scratch = scratch)
end

"""
    apply_stencil!(out, field, x, indices, weights, dim; order=1, period=nothing, mask=nothing,
                   masked=zero, policy=BlankMasked(), backend=nothing) -> out

Apply a table built by [`axis_stencils`](@ref) **and** keep the axis, so any mask policy works.

The table depends on the axis and not on the field, so a caller differencing many fields along one
direction should build it once. The bare `(indices, weights)` form cannot degrade at a mask edge —
that needs the axis to rebuild a window from — so it accepts only [`BlankMasked`](@ref); this form
takes both and serves every policy.

The split is the one the degrade path already makes internally: the precomputed row is used wherever
the window is intact, which is every cell away from a mask, and the axis is touched only where a
window is actually rebuilt.

Building the table is `O(n)` against an `O(n²)` apply, so holding it matters most on small grids —
2.4–13× at `n = 48`, 10–40% at `n = 256`, amortized away by `n = 1024`. The allocation it avoids is
there at every size: 49 600 bytes per call at `n = 1024`.
"""
function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N}, x::AbstractVector{<:AbstractFloat},
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix, dim::Integer;
    order::Integer = 1, period::Union{Nothing,Real} = nothing, mask = nothing, masked = zero(S),
    backend = nothing, policy::AbstractMaskPolicy = BlankMasked(), scratch = nothing,
) where {S,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    size(field, dim) == length(x) || throw(DimensionMismatch(
        "axis has $(length(x)) samples but direction $dim of the field has $(size(field, dim))",
    ))
    size(indices, 1) == length(x) || throw(DimensionMismatch(
        "got $(size(indices, 1)) stencil rows for an axis of $(length(x)) samples",
    ))
    # The precomputed rows are the whole answer under `BlankMasked`, and they stay the answer in the
    # interior of every active run under the others — a degraded row is only built where one is needed.
    if policy isa BlankMasked || mask === nothing
        return apply_stencil!(out, field, indices, weights, dim; mask = mask, masked = masked,
                              backend = backend)
    end
    return _apply_stencil_degrade!(out, field, x, indices, weights, Int(dim), mask, masked,
                                   Int(order), size(indices, 2), period, policy, backend, scratch)
end

# Rebuilding a stencil needs the axis and a scratch table, so it is a chunked host loop: a launch has
# nowhere to put the per-cell Fornberg table. `BlankMasked` above keeps the index-parallel path.
@inline function _fits(s::Discretization.StencilScratch, k::Int, ord::Int)
    return length(s.w) ≥ k && length(s.n) ≥ k && size(s.c, 1) ≥ k && size(s.c, 2) ≥ ord + 1
end

function _apply_stencil_degrade!(
    out::AbstractArray{S,N}, field, x::AbstractVector{T}, idx, wts, dim::Int, mask, masked,
    ord::Int, k::Int, period, policy::AbstractMaskPolicy, backend, scratch,
) where {S,N,T}
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    _check_mask_extent(mask, size(field), dim)
    vm = _mask_rank(mask, Val(N))
    sz = size(field)
    ci = CartesianIndices(sz)
    n = sz[dim]
    P = period === nothing ? zero(T) : T(period) * Axes.wrap_sign(x)
    wrap = period !== nothing
    # A caller's buffers are usable only where there is one chunk: they are written per cell, so
    # concurrent chunks would race on them. The threaded path allocates its own set per chunk.
    if backend === nothing && scratch isa Discretization.StencilScratch{T}
        _fits(scratch, k, ord) || throw(DimensionMismatch(
            "scratch holds $(length(scratch.w)) nodes × $(size(scratch.c, 2)) orders; this call " *
            "needs $k × $(ord + 1) — build it with `stencil_scratch($ord, $k)`",
        ))
        @inbounds for lin in Base.OneTo(length(ci))
            c = ci[lin]
            _stencil_cell_degrade!(out, field, x, idx, wts, dim, mask, masked, k, ord, n, P, wrap,
                                   Tuple(c), c, vm, policy, scratch.w, scratch.c, scratch.n)
        end
        return out
    end
    Execution.run_chunks(length(ci), backend) do rng
        wbuf = Vector{T}(undef, k)
        cbuf = Matrix{T}(undef, k, ord + 1)
        nbuf = Vector{T}(undef, k)
        @inbounds for lin in rng
            c = ci[lin]
            _stencil_cell_degrade!(out, field, x, idx, wts, dim, mask, masked, k, ord, n, P, wrap,
                                   Tuple(c), c, vm, policy, wbuf, cbuf, nbuf)
        end
    end
    return out
end

# The cell index with direction `dim` replaced. `j` arrives as an ARGUMENT rather than being captured:
# a local that is both reassigned and closed over is boxed by Julia, and the loops below reassign
# theirs every iteration — measured at 288 bytes per `_run_reach` call before this was split out.
@inline _at_dim(I::NTuple{N,Int}, dim::Int, j::Int) where {N} =
    ntuple(d -> d == dim ? j : I[d], Val(N))

# How far the active run containing `i` reaches, walked at most `k` steps: past that the run is longer
# than any window, which is all the clamp needs to know. Bounding the walk is what keeps this `O(k)`
# rather than `O(run length)`.
@inline function _run_reach(mask, I::NTuple{N,Int}, dim::Int, i::Int, n::Int, k::Int, wrap::Bool) where {N}
    back = 0
    @inbounds while back < k
        j = i - back - 1
        j < 1 && (wrap ? (j = n) : break)
        mask[_at_dim(I, dim, j)...] || break
        back += 1
        back ≥ n && break
    end
    fwd = 0
    @inbounds while fwd < k
        j = i + fwd + 1
        j > n && (wrap ? (j = 1) : break)
        mask[_at_dim(I, dim, j)...] || break
        fwd += 1
        back + fwd + 1 ≥ n && break
    end
    return back, fwd
end

@inline function _stencil_cell_degrade!(
    out::AbstractArray{S,N}, field, x::AbstractVector{T}, idx, wts, dim::Int, mask, masked,
    k::Int, ord::Int, n::Int, P::T, wrap::Bool, I::NTuple{N,Int}, ci, ::Val{M}, policy,
    wbuf, cbuf, nbuf,
) where {S,N,T,M}
    Is = _spatial(I, Val(M))
    @inbounds begin
        if !mask[Is...]
            out[ci] = masked
            return nothing
        end
        i = I[dim]
        # The precomputed window, if every node it reads is active. Contiguous and all-active means it
        # lies in this cell's run, and the run-clamped window is then the same window — so this branch
        # is bit-for-bit the unmasked result, and it is the one the interior of a region takes.
        # Accumulated while checking, in the same order: a window that turns out to be intact has then
        # been walked once rather than twice, and this is the branch every cell away from a mask takes.
        intact = true
        acc = zero(S)
        for q in 1:k
            J = ntuple(d -> d == dim ? Int(idx[i, q]) : I[d], Val(N))
            if !mask[_spatial(J, Val(M))...]
                intact = false
                break
            end
            acc += S(wts[i, q]) * S(field[J...])
        end
        if intact
            out[ci] = acc
            return nothing
        end

        # The run this cell sits in is a spatial property, so it is walked in spatial coordinates and
        # the same run serves every batch element.
        back, fwd = _run_reach(mask, Is, dim, i, n, k, wrap)
        len = back + fwd + 1
        kk = policy isa ReduceInRun ? min(k, len) : k
        if len < kk || kk < ord + 1
            out[ci] = masked
            return nothing
        end
        # Run-local coordinates, so one expression covers a wrapping run and a bounded one.
        pos = back                        # 0-based offset of `i` from the run's first sample
        s = clamp(pos - (kk - 1) ÷ 2, 0, len - kk)
        base = i - back                   # may be ≤ 0 when the run wraps; `mod1` puts it back
        for q in 1:kk
            raw = base + s + q - 1
            j = mod1(raw, n)
            nbuf[q] = x[j] + T(fld(raw - 1, n)) * P
        end
        Discretization._fd_weights!(wbuf, cbuf, nbuf, kk, x[i], ord)
        acc = zero(S)
        for q in 1:kk
            raw = base + s + q - 1
            J = _at_dim(I, dim, mod1(raw, n))
            acc += S(wbuf[q]) * S(field[J...])
        end
        out[ci] = acc
    end
    return nothing
end

function apply_stencil!(
    out::AbstractArray{S,N}, field::AbstractArray{<:Any,N},
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix, dim::Integer;
    mask = nothing, masked = zero(S), backend = nothing,
    policy::AbstractMaskPolicy = BlankMasked(),
) where {S,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    # Degrading means rebuilding a stencil, which needs the axis this form was not given.
    policy isa BlankMasked || throw(ArgumentError(
        "$(policy) needs the axis to rebuild a stencil from; call the `(out, field, x, dim)` form",
    ))
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    size(indices) == size(weights) || throw(DimensionMismatch(
        "indices $(size(indices)) and weights $(size(weights)) must have the same size",
    ))
    size(indices, 1) == size(field, dim) || throw(DimensionMismatch(
        "got $(size(indices, 1)) stencil rows for direction $dim of length $(size(field, dim))",
    ))
    _check_mask_extent(mask, size(field), Int(dim))
    k = size(indices, 2)
    if backend === nothing
        # On the host the loop shape is ours to choose, and the index-parallel one is the wrong shape:
        # see `_stencil_sweep_host!`. Both paths are the same arithmetic in the same order, so they
        # agree bit for bit. The switch walks the SPATIAL rank, since `dim` is one of those, so a
        # trailing batch axis adds no specializations.
        return _dispatch_dim(Int(dim), _mask_rank(mask, Val(N))) do vdim
            _dispatch_nodes(k) do vk
                _stencil_sweep_host!(out, field, indices, weights, mask, masked, vdim, vk, Val(N),
                                     _mask_rank(mask, Val(N)))
            end
        end
    end
    # The differenced direction and the node count are resolved ONCE, before the launch, for the same
    # reason the host sweep resolves them once per sweep: they are properties of the weight set, not of
    # the cell, and a `Val` built inside the body would cost more than it saves.
    vm = _mask_rank(mask, Val(N))
    return _dispatch_dim(Int(dim), vm) do vdim
        _dispatch_nodes(k) do vk
            _launch_stencil!(out, field, indices, weights, mask, masked, vdim, vk, Val(N), vm,
                             backend)
        end
    end
end

# Over the WHOLE field, batch axes included: that is what makes a batched sweep one launch over
# `prod(spatial) * prod(batch)` work items rather than one launch per slice.
function _launch_stencil!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, ::Val{dim}, nodes, ::Val{N},
    ::Val{M}, backend,
) where {S,N,dim,M}
    ci = CartesianIndices(size(field))
    Execution.run_indices(length(ci), backend) do lin
        _stencil_cell!(out, field, indices, weights, Val(dim), mask, masked, nodes,
                       Tuple(@inbounds ci[lin]), (@inbounds ci[lin]), Val(M))
    end
    return out
end

# Two runtime values are wanted in the type: the differenced direction, so the loop nest can be split
# around it, and the node count, so the innermost loop has a known trip count. Both are resolved ONCE
# per sweep, here, rather than per cell — a `Val` built deeper costs more than it saves. Specialization
# stays bounded: directions by `N`, node counts by the cap below, above which the runtime loop stands.
@inline _dispatch_dim(f::F, dim::Int, ::Val{N}) where {F,N} = _dim_switch(f, dim, Val(N))
# `dim` is validated into `1:N` by the caller, so walking down from `N` always lands.
@inline _dim_switch(f::F, ::Int, ::Val{1}) where {F} = f(Val(1))
@inline _dim_switch(f::F, dim::Int, ::Val{M}) where {F,M} =
    dim == M ? f(Val(M)) : _dim_switch(f, dim, Val(M - 1))

@inline function _dispatch_nodes(f::F, k::Int) where {F}
    k == 2 && return f(Val(2))
    k == 3 && return f(Val(3))
    k == 4 && return f(Val(4))
    k == 5 && return f(Val(5))
    k == 6 && return f(Val(6))
    k == 7 && return f(Val(7))
    k == 8 && return f(Val(8))
    k == 9 && return f(Val(9))
    return f(k)
end

"""
    _stencil_sweep_host!(out, field, indices, weights, mask, masked, Val(dim), nodes, Val(N)) -> out

The host sweep. The index-parallel form exists so one body serves a device launch; on the host it is
the wrong shape, and three things it cannot express are worth ~6.6× together:

- iterate the Cartesian range **directly**, rather than recovering an index per cell from a linear one;
- **split the nest at `dim`**, so the stencil row — which depends only on the index along `dim` — is
  hoisted out of the contiguous inner loop whenever the differenced direction is not the fastest
  varying one;
- carry the **node count in the type**, so the innermost loop has a known trip count, unrolls, and the
  weights reach registers instead of being re-loaded per node.

The arithmetic and its order are identical to `_stencil_cell!`, so the two paths agree bit for bit.
"""
function _stencil_sweep_host!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, ::Val{dim}, nodes, ::Val{N},
    ::Val{M},
) where {S,N,dim,M}
    sz = size(field)
    if _linear_layout(out, field, mask)
        # Column-major and one-based, so the whole nest is address arithmetic: cells that differ only
        # before `dim` are `1` apart, and a node is a fixed offset of `stride` per index step along
        # `dim`. The innermost span is then contiguous and vectorizes.
        stride = prod(ntuple(d -> sz[d], Val(dim - 1)))
        npost = prod(ntuple(d -> sz[dim + d], Val(N - dim)))
        outer = stride * sz[dim]
        # The mask spans the leading `M` axes only, so it has fewer slabs than the field: the batch axes
        # are the slowest, so field slab `p` reads mask slab `p % mpost`. One remainder per SLAB, not per
        # cell. With no batch `mpost == npost` and the remainder is the identity.
        mpost = prod(ntuple(d -> sz[dim + d], Val(M - dim)))
        if dim == 1
            # The differenced direction is itself the contiguous one, so there is no span to hoist a
            # row out of — every cell has its own row, used once. Hoisting it into a tuple would be
            # pure overhead; the loop over `j` is the contiguous one instead.
            for p in 0:(npost - 1)
                _stencil_first_linear!(out, field, indices, weights, mask, masked, p * outer,
                                       (p % mpost) * outer, sz[1], nodes)
            end
            return out
        end
        for p in 0:(npost - 1), j in 1:sz[dim]
            _stencil_row_linear!(out, field, indices, weights, mask, masked, j, p * outer,
                                 (p % mpost) * outer, stride, nodes)
        end
        return out
    end
    pre = CartesianIndices(ntuple(d -> sz[d], Val(dim - 1)))
    post = CartesianIndices(ntuple(d -> sz[dim + d], Val(N - dim)))
    @inbounds for Ipost in post, j in 1:sz[dim]
        _stencil_row!(out, field, indices, weights, mask, masked, j, Tuple(Ipost), pre, nodes,
                      Val(dim), Val(N), Val(M))
    end
    return out
end

# A field may carry trailing BATCH axes beyond the ones the mask spans: `(nx, ny, nb)` differenced
# against a 2-D grid, where the same mask applies to every slice. The mask's own rank is therefore the
# spatial rank — nothing else has to be declared — and a cell's mask index is the leading `M`
# components of its index. With no mask, or a mask of the field's own rank, `M == N` and every
# expression below is what it was.
@inline _mask_rank(::Nothing, ::Val{N}) where {N} = Val(N)
@inline _mask_rank(::AbstractArray{Bool,M}, ::Val{N}) where {M,N} = Val(M)

@inline _spatial(I::NTuple{N,Int}, ::Val{M}) where {N,M} = ntuple(d -> I[d], Val(M))

# The mask fixes how many leading axes are spatial; the rest of the field is batch. A mask must match
# the field over exactly those axes — a disagreement there is a real mistake and still raises — and the
# differenced direction has to be one of them, since the stencil table describes a grid axis and not a
# batch.
@inline _check_mask_extent(::Nothing, ::Tuple, ::Int) = nothing
@inline function _check_mask_extent(mask::AbstractArray{Bool,M}, sz::NTuple{N,Int}, dim::Int) where {M,N}
    M ≤ N || throw(DimensionMismatch(
        "mask has $M axes but the field has only $N",
    ))
    size(mask) == ntuple(d -> sz[d], Val(M)) || throw(DimensionMismatch(
        "mask $(size(mask)) must match the field's leading $M axes $(ntuple(d -> sz[d], Val(M)))",
    ))
    dim ≤ M || throw(ArgumentError(
        "direction $dim is a batch axis; only 1:$M can be differenced, the axes the mask spans",
    ))
    return nothing
end

# The address arithmetic above assumes linear indexing over one-based axes. Anything else — an offset
# array, a view with a non-trivial stride — takes the Cartesian nest, which asks the array for its own
# indexing and is correct for any of them.
@inline _linear_layout(out, field, mask) =
    IndexStyle(out) === IndexLinear() && IndexStyle(field) === IndexLinear() &&
    (mask === nothing || IndexStyle(mask) === IndexLinear()) &&
    !Base.has_offset_axes(out, field) && (mask === nothing || !Base.has_offset_axes(mask))

# `dim == 1`: the slab is one contiguous run along the differenced direction, so the stencil row
# changes every step and is read straight out of the table rather than hoisted.
@inline function _stencil_first_linear!(
    out::AbstractArray{S}, field, indices, weights, mask, masked, off::Int, moff::Int, n::Int,
    ::Val{k},
) where {S,k}
    @inbounds for j in 1:n
        if mask !== nothing && !mask[moff + j]
            out[off + j] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            ix = Int(indices[j, q])
            if mask !== nothing && !mask[moff + ix]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[off + ix])
        end
        out[off + j] = blocked ? masked : acc
    end
    return nothing
end

@inline function _stencil_first_linear!(
    out::AbstractArray{S}, field, indices, weights, mask, masked, off::Int, moff::Int, n::Int,
    k::Int,
) where {S}
    @inbounds for j in 1:n
        if mask !== nothing && !mask[moff + j]
            out[off + j] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            ix = Int(indices[j, q])
            if mask !== nothing && !mask[moff + ix]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[off + ix])
        end
        out[off + j] = blocked ? masked : acc
    end
    return nothing
end

# One row, as offsets. `off` is the start of this slab, `stride` the distance between consecutive
# indices along `dim`, so `base + t` walks the contiguous span and `js[q] + t` reads node `q` of it.
@inline function _stencil_row_linear!(
    out::AbstractArray{S}, field, indices, weights, mask, masked, j::Int, off::Int, moff::Int,
    stride::Int, ::Val{k},
) where {S,k}
    @inbounds js = ntuple(q -> off + (Int(indices[j, q]) - 1) * stride, Val(k))
    @inbounds ws = ntuple(q -> S(weights[j, q]), Val(k))
    base = off + (j - 1) * stride
    if mask === nothing
        @inbounds for t in 1:stride
            acc = zero(S)
            for q in 1:k
                acc += ws[q] * S(field[js[q] + t])
            end
            out[base + t] = acc
        end
    else
        # The mask spans only the spatial axes, so it has its own slab base and its own node
        # addresses; `moff == off` whenever the field carries no batch and this is the same arithmetic.
        @inbounds ms = ntuple(q -> moff + (Int(indices[j, q]) - 1) * stride, Val(k))
        mbase = moff + (j - 1) * stride
        @inbounds for t in 1:stride
            if !mask[mbase + t]
                out[base + t] = masked
                continue
            end
            acc = zero(S)
            blocked = false
            for q in 1:k
                if !mask[ms[q] + t]
                    blocked = true
                    break
                end
                acc += ws[q] * S(field[js[q] + t])
            end
            out[base + t] = blocked ? masked : acc
        end
    end
    return nothing
end

# Above the specialization cap the row cannot become a tuple, so it is read per cell.
@inline function _stencil_row_linear!(
    out::AbstractArray{S}, field, indices, weights, mask, masked, j::Int, off::Int, moff::Int,
    stride::Int, k::Int,
) where {S}
    base = off + (j - 1) * stride
    mbase = moff + (j - 1) * stride
    @inbounds for t in 1:stride
        if mask !== nothing && !mask[mbase + t]
            out[base + t] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            ix = (Int(indices[j, q]) - 1) * stride + t
            if mask !== nothing && !mask[moff + ix]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[off + ix])
        end
        out[base + t] = blocked ? masked : acc
    end
    return nothing
end

# One row: every cell whose index along `dim` is `j`. The stencil is the same for all of them, so it is
# read once here rather than once per cell.
@inline function _stencil_row!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, j::Int, Ipost::Tuple, pre,
    ::Val{k}, ::Val{dim}, ::Val{N}, ::Val{M},
) where {S,N,dim,k,M}
    # The row, read once. With `k` in the type these are stack tuples, so the inner loop unrolls over
    # registers instead of re-reading two matrix columns per cell.
    @inbounds js = ntuple(q -> Int(indices[j, q]), Val(k))
    @inbounds ws = ntuple(q -> S(weights[j, q]), Val(k))
    @inbounds for Ipre in pre
        I = (Tuple(Ipre)..., j, Ipost...)
        if mask !== nothing && !mask[_spatial(I, Val(M))...]
            out[I...] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            J = ntuple(d -> d == dim ? js[q] : I[d], Val(N))
            if mask !== nothing && !mask[_spatial(J, Val(M))...]
                blocked = true
                break
            end
            acc += ws[q] * S(field[J...])
        end
        out[I...] = blocked ? masked : acc
    end
    return nothing
end

# Above the specialization cap the node count stays a runtime value: the row cannot become a tuple, so
# it is read per cell as before. Correct, and the shape a very wide stencil would not benefit from
# unrolling anyway.
@inline function _stencil_row!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, j::Int, Ipost::Tuple, pre,
    k::Int, ::Val{dim}, ::Val{N}, ::Val{M},
) where {S,N,dim,M}
    @inbounds for Ipre in pre
        I = (Tuple(Ipre)..., j, Ipost...)
        if mask !== nothing && !mask[_spatial(I, Val(M))...]
            out[I...] = masked
            continue
        end
        acc = zero(S)
        blocked = false
        for q in 1:k
            J = ntuple(d -> d == dim ? Int(indices[j, q]) : I[d], Val(N))
            if mask !== nothing && !mask[_spatial(J, Val(M))...]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[J...])
        end
        out[I...] = blocked ? masked : acc
    end
    return nothing
end

# One output cell, written from its own inputs only, so the loop above is index-parallel and the same
# body serves a host loop and a device launch.
"""
    _nodecount(nodes) -> Int

The node count, from either a `Val` or a plain `Int`.

One body then serves both: with a `Val` the trip count is a literal, so the loop unrolls and the
weights reach registers; with an `Int` it is an ordinary loop, which is what a node count above the
specialized set gets. Writing the loop twice would be two copies of the same arithmetic to keep in step.
"""
@inline _nodecount(::Val{k}) where {k} = k
@inline _nodecount(k::Int) = k

# One cell, by its own index and nothing else — the body a launch runs. `dim` and the node count arrive
# as type parameters, so the index construction folds and the node loop unrolls: resolved as runtime
# values, `d == dim` is a comparison per node per cell and the loop has no known trip count.
@inline function _stencil_cell!(
    out::AbstractArray{S,N}, field, indices, weights, ::Val{dim}, mask, masked, nodes,
    I::NTuple{N,Int}, ci, ::Val{M},
) where {S,N,M,dim}
    @inbounds begin
        if mask !== nothing && !mask[_spatial(I, Val(M))...]
            out[ci] = masked
            return nothing
        end
        j = I[dim]
        acc = zero(S)
        blocked = false
        for q in 1:_nodecount(nodes)
            J = ntuple(d -> d == dim ? Int(indices[j, q]) : I[d], Val(N))
            if mask !== nothing && !mask[_spatial(J, Val(M))...]
                blocked = true
                break
            end
            acc += S(weights[j, q]) * S(field[J...])
        end
        out[ci] = blocked ? masked : acc
    end
    return nothing
end
