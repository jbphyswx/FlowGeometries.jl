"""
    apply_stencil!(out, field, x, dim; order=1, nodes=order+1, period=nothing,
                   mask=nothing, masked=zero) -> out
    apply_stencil!(out, field, indices, weights, dim; mask=nothing, masked=zero) -> out

Apply a weight set along direction `dim` of `field`, writing
`out[I] = Σ_q weights[I[dim], q] · field[…, indices[I[dim], q], …]`.

Every convention this needs is fixed by the package: the result sits at the same location as the input,
so there is no staggering decision, and the stencil shifts inward at a bounded end and wraps on a
periodic one, which is [`Discretization.fd_weights`](@ref)'s stated boundary behaviour and removes the
need for a halo. Operations that do take those choices live elsewhere — a staggered difference, or a
multi-direction operator like a divergence or a curl, each of which needs a result location and a
boundary-condition policy.

Pass the axis and an order to have the weights built for you, or precomputed `indices`/`weights` from
[`Discretization.axis_stencils`](@ref) to reuse them across many fields.

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
        # Under this policy `nodes` is a ceiling. The end of the axis bounds a window exactly as the end
        # of an active run does, and the policy already gives a run too short for `nodes` the largest
        # window it holds, so a short axis degrades the same way. A single-latitude strip, a two-level
        # column and a one-cell channel are ordinary grids.
        if length(x) < ord + 1
            fill!(out, masked)      # no derivative of this order exists anywhere on such an axis
            return out
        end
        k = min(k, length(x))
    end
    # The blanking policy takes a plan, which holds uniform weights in registers. The degrading policies
    # rebuild a window from the axis, so they take the table.
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

Apply a table built by [`Discretization.axis_stencils`](@ref) **and** keep the axis, so any mask policy works.

The table depends on the axis alone, so a caller differencing many fields along one direction builds it
once. Degrading at a mask edge needs the axis to rebuild a window from, so the bare `(indices, weights)`
form accepts only [`BlankMasked`](@ref); this form takes both and serves every policy.

The split is the one the degrade path makes internally: the precomputed row is used wherever the window
is intact, which is every cell away from a mask, and the axis is touched only where a window is rebuilt.

Building the table is `O(n)` against an `O(n²)` apply, so holding it across fields matters most on small
grids. It also removes an `O(n · nodes)` allocation from every call, at any size.
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
    # A caller's buffers are written per cell, so they hold only under a single chunk. The threaded
    # path allocates its own set per chunk.
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

# The cell index with direction `dim` replaced. `j` is an argument: Julia boxes a local that is both
# reassigned and closed over, and the loops below reassign theirs every iteration.
@inline _at_dim(I::NTuple{N,Int}, dim::Int, j::Int) where {N} =
    ntuple(d -> d == dim ? j : I[d], Val(N))

# How far the active run containing `i` reaches, walked at most `k` steps: past that the run is longer
# than any window, which is all the clamp needs to know. The bound on the walk makes this `O(k)`.
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
        # lies in this cell's run, and the run-clamped window is then the same window, so this branch
        # gives bit-for-bit the unmasked result. Accumulation happens while checking, in the same order,
        # so an intact window is walked once — the branch every cell away from a mask takes.
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
    # The switch walks the spatial rank, since `dim` is one of those, so a trailing batch axis adds no
    # specializations.
    vm = _mask_rank(mask, Val(N))
    if backend === nothing
        # On the host the loop shape is ours to choose, and the index-parallel one is the wrong shape:
        # see `_stencil_sweep_host!`. Both paths are the same arithmetic in the same order, so they
        # agree bit for bit.
        return _dispatch_dim(Int(dim), vm) do vdim
            _dispatch_nodes(k) do vk
                _stencil_sweep_host!(out, field, indices, weights, mask, masked, vdim, vk, Val(N), vm)
            end
        end
    end
    # The differenced direction and the node count are properties of the weight set, so they resolve to
    # types once before the launch, as they do once per sweep on the host.
    return _dispatch_dim(Int(dim), vm) do vdim
        _dispatch_nodes(k) do vk
            _launch_stencil!(out, field, indices, weights, mask, masked, vdim, vk, Val(N), vm,
                             backend)
        end
    end
end

# Over the entire field, batch axes included, so a batched sweep is a single launch over
# `prod(spatial) * prod(batch)` work items.
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

# Two runtime values are lifted into the type: the differenced direction, so the loop nest splits around
# it, and the node count, so the innermost loop has a known trip count. Both resolve once per sweep,
# here. Specialization stays bounded: directions by `N`, node counts by the cap below, above which the
# runtime loop stands.
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
    _stencil_sweep_host!(out, field, indices, weights, mask, masked, Val(dim), nodes, Val(N),
                         Val(M), invh, R) -> out

The host sweep. The index-parallel form serves a device launch; this one adds three things a
per-work-item body cannot express:

- iterate the Cartesian range directly, with no index recovered per cell from a linear one;
- split the nest at `dim`, hoisting the stencil row — which depends only on the index along `dim` — out
  of the contiguous inner loop wherever the differenced direction is a slower-varying one;
- carry the node count in the type, so the innermost loop has a known trip count, unrolls, and holds
  the weights in registers.

The arithmetic and its order are identical to `_stencil_cell!`, so the two paths agree bit for bit.

`invh` fuses the metric factor into the sweep — see [`_fuse_scale!`](@ref). Pass `nothing` for the
plain sweep.
"""
function _stencil_sweep_host!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, ::Val{dim}, nodes, ::Val{N},
    ::Val{M}, invh = nothing, R::Int = 0,
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
        # are the slowest, so field slab `p` reads mask slab `p % mpost`. One remainder per slab. With no
        # batch `mpost == npost` and the remainder is the identity.
        mpost = prod(ntuple(d -> sz[dim + d], Val(M - dim)))
        if dim == 1
            # The differenced direction is itself the contiguous one, so every cell has its own row,
            # read once, and there is no span to hoist it out of. The loop over `j` is contiguous.
            for p in 0:(npost - 1)
                _stencil_first_linear!(out, field, indices, weights, mask, masked, p * outer,
                                       (p % mpost) * outer, sz[1], nodes)
                invh === nothing || _fuse_scale!(out, invh, R, p * outer + 1, 1, sz[1], p, masked)
            end
            return out
        end
        nruns = stride ÷ sz[1]
        for p in 0:(npost - 1), j in 1:sz[dim]
            _stencil_row_linear!(out, field, indices, weights, mask, masked, j, p * outer,
                                 (p % mpost) * outer, stride, nodes)
            invh === nothing ||
                _fuse_scale!(out, invh, R, p * outer + (j - 1) * stride + 1, nruns, sz[1],
                             nruns * ((j - 1) + sz[dim] * p), masked)
        end
        return out
    end
    invh === nothing || throw(ArgumentError(
        "a fused metric factor needs the linear layout; scale as a separate pass for this array type",
    ))
    pre = CartesianIndices(ntuple(d -> sz[d], Val(dim - 1)))
    post = CartesianIndices(ntuple(d -> sz[dim + d], Val(N - dim)))
    @inbounds for Ipost in post, j in 1:sz[dim]
        _stencil_row!(out, field, indices, weights, mask, masked, j, Tuple(Ipost), pre, nodes,
                      Val(dim), Val(N), Val(M))
    end
    return out
end

# A field may carry trailing batch axes beyond the ones the mask spans: `(nx, ny, nb)` differenced
# against a 2-D grid, where the same mask applies to every slice. The mask's own rank is therefore the
# spatial rank — nothing else has to be declared — and a cell's mask index is the leading `M`
# components of its index. With no mask, or a mask of the field's own rank, `M == N`.
@inline _mask_rank(::Nothing, ::Val{N}) where {N} = Val(N)
@inline _mask_rank(::AbstractArray{Bool,M}, ::Val{N}) where {M,N} = Val(M)

@inline _spatial(I::NTuple{N,Int}, ::Val{M}) where {N,M} = ntuple(d -> I[d], Val(M))

# The mask fixes how many leading axes are spatial; the rest of the field is batch. A mask must match
# the field over exactly those axes, and the differenced direction must be one of them, since the
# stencil table describes a grid axis.
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

# `dim == 1`: the slab is one contiguous run along the differenced direction, so the stencil row changes
# every step and is read straight out of the table.
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

# The same body above the specialization cap, where `k` stays a runtime value: the node loop keeps its
# literal trip count in the `Val` method and unrolls there, and runs as an ordinary loop here.
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

# One row: every cell whose index along `dim` is `j`. The stencil is the same for all of them, so the
# row is read once here.
@inline function _stencil_row!(
    out::AbstractArray{S,N}, field, indices, weights, mask, masked, j::Int, Ipost::Tuple, pre,
    ::Val{k}, ::Val{dim}, ::Val{N}, ::Val{M},
) where {S,N,dim,k,M}
    # The row, read once. With `k` in the type these are stack tuples, so the inner loop unrolls and
    # keeps them in registers across the cells of the row.
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

# Above the specialization cap the node count stays a runtime value, so the row stays in the table and
# is read per cell.
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

"""
    _nodecount(nodes) -> Int

The node count, from either a `Val` or a plain `Int`.

One body then serves both. With a `Val` the trip count is a literal, so the loop unrolls and the weights
reach registers; with an `Int`, the form a node count above the specialized set takes, it is an ordinary
loop.
"""
@inline _nodecount(::Val{k}) where {k} = k
@inline _nodecount(k::Int) = k

# One output cell, written from its own inputs alone, so the loop over cells is index-parallel and this
# body serves a host loop and a device launch. `dim` and the node count arrive as type parameters, so
# the index construction folds and the node loop unrolls; as runtime values `d == dim` costs a
# comparison per node per cell and the loop has no known trip count.
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
