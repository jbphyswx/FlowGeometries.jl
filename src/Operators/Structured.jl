
"""
    apply_stencil!(out, field, grid, dim; order=1, nodes=order+1, active_only=true, masked=zero) -> out

[`apply_stencil!`](@ref) with the axis, wrap period and mask taken from `grid`, so a
periodic direction wraps and an inactive cell is honoured without restating any of it.

Only a rectilinear direction has a 1-D axis to difference along, so this is a `StructuredGrid` method.
"""
function apply_stencil!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T, G,N},
    dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
    active_only::Bool = true, masked = zero(S), backend = nothing,
    policy::AbstractMaskPolicy = BlankMasked(), scratch = nothing,
) where {S,G,T,N,NA}
    _check_batched(out, field, grid, Val(N), Int(dim))
    msk = active_only && !(Grids.mask(grid) isa Grids.AllActive) ? Grids.mask(grid) : nothing
    return apply_stencil!(
        out, field, Grids.coordinates(grid, dim), dim;
        order = order, nodes = nodes,
        period = Grids.isperiodic(grid, dim) ? Grids.period(grid, dim) : nothing,
        mask = msk, masked = masked, backend = backend, policy = policy, scratch = scratch,
    )
end

# The two samples of direction `d` that bracket `v`, with their weights. A periodic direction wraps:
# past the last sample the pair is `(n, 1)` across the seam, where `interpolation_weights` alone would
# clamp and return the endpoint value.
@inline function _interp_pair(grid::Grids.StructuredGrid{T, G,N}, d::Int, v::T) where {G,T,N}
    x = Grids.coordinates(grid, d)
    n = length(x)
    n == 1 && return (1, 1, one(T), zero(T))
    if Grids.isperiodic(grid, d)
        L = T(Grids.period(grid, d))
        if L > 0
            lo = T(Grids.bounds(grid, d)[1])
            v = lo + mod(v - lo, L)
            @inbounds x1, xn = T(x[1]), T(x[n])
            asc = xn ≥ x1
            beyond = asc ? v > xn : v < xn
            if beyond
                h = asc ? (x1 + L) - xn : (x1 - L) - xn
                t = iszero(h) ? zero(T) : (v - xn) / h
                return (n, 1, one(T) - t, t)
            end
        end
    end
    i, w = Discretization.interpolation_weights(x, v)
    return (Int(i), Int(i) + 1, w[1], w[2])
end

function interpolate(
    field::AbstractArray{S,N}, grid::Grids.StructuredGrid{T, G,N}, p::NTuple{N,Real};
    active_only::Bool = true, masked = S(NaN),
    policy::AbstractMaskPolicy = BlankMasked(),
) where {S,G,T,N}
    _check_interp(field, grid, Val(N), policy)
    return _interp_at(field, grid, p, 0, active_only, masked, policy)
end

@inline function _check_interp(
    field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T, G,N}, ::Val{N}, policy,
) where {NA,G,T,N}
    policy isa ShiftWithinRun && _interp_mask_error(policy)
    NA ≥ N || throw(DimensionMismatch("field has $NA axes but the grid has $N"))
    ntuple(d -> size(field, d), Val(N)) == Grids.size_tuple(grid) || throw(DimensionMismatch(
        "field's leading $N axes $(ntuple(d -> size(field, d), Val(N))) do not match the grid " *
        "$(Grids.size_tuple(grid))",
    ))
    return nothing
end

# One batch element, at linear offset `off` into the field. The bracketing cell and its weights are a
# property of the POINT and the grid, so a batched call solves them once and calls this per element.
function _interp_at(
    field::AbstractArray{S}, grid::Grids.StructuredGrid{T, G,N}, p::NTuple{N,Real}, off::Int,
    active_only::Bool, masked, policy,
) where {S,G,T,N}
    prs = ntuple(d -> _interp_pair(grid, d, T(p[d])), Val(N))
    msk = active_only && !(Grids.mask(grid) isa Grids.AllActive) ? Grids.mask(grid) : nothing
    sz = Grids.size_tuple(grid)
    acc = zero(S)
    wsum = zero(T)
    # The `2^N` corners of the bracketing cell, each weighted by the product of its per-axis weights.
    @inbounds for c in CartesianIndices(ntuple(_ -> 2, Val(N)))
        w = prod(ntuple(d -> c[d] == 1 ? prs[d][3] : prs[d][4], Val(N)))
        iszero(w) && continue
        I = ntuple(d -> c[d] == 1 ? prs[d][1] : prs[d][2], Val(N))
        if msk !== nothing && !msk[I...]
            policy isa BlankMasked && return masked
            continue                                   # `ReduceInRun`: drop it and renormalize
        end
        # Linear, so one expression serves the unbatched field and a slice of a batched one.
        lin = I[N]
        for d in (N - 1):-1:1
            lin = (lin - 1) * sz[d] + I[d]
        end
        acc += S(w) * S(field[off + lin])
        wsum += w
    end
    return wsum > 0 ? acc / S(wsum) : masked
end

@inline interpolate(
    field::AbstractArray, grid::Grids.StructuredGrid, p::Geometry.PointLike; kwargs...,
) =
    interpolate(field, grid, Geometry.as_ntuple(p); kwargs...)

"""
    interpolate!(out, field, grid, p; active_only=true, masked=NaN, policy=BlankMasked()) -> out

[`interpolate`](@ref) at one coordinate for a field carrying trailing batch axes: `out` receives one
value per batch element, in the order those axes are laid out.

The bracketing cell and its `2^N` corner weights depend on the point and the grid alone, so they are
solved once here and applied to every element.
"""
function interpolate!(
    out::AbstractVector{S}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T, G,N},
    p::NTuple{N,Real}; active_only::Bool = true, masked = S(NaN),
    policy::AbstractMaskPolicy = BlankMasked(),
) where {S,G,T,N,NA}
    _check_interp(field, grid, Val(N), policy)
    n = prod(Grids.size_tuple(grid))
    nb = length(field) ÷ n
    length(out) == nb || throw(DimensionMismatch(
        "out holds $(length(out)) values but the field carries $nb batch elements",
    ))
    @inbounds for b in 1:nb
        out[b] = _interp_at(field, grid, p, (b - 1) * n, active_only, masked, policy)
    end
    return out
end

@inline interpolate!(
    out::AbstractVector, field::AbstractArray, grid::Grids.StructuredGrid, p::Geometry.PointLike; kwargs...,
) = interpolate!(out, field, grid, Geometry.as_ntuple(p); kwargs...)

# The allocating form, as everywhere else in the package: `spherical_points!`/`spherical_points`,
# `latitude_weights!`/`latitude_weights`.
function interpolate(
    field::AbstractArray{S,NA}, grid::Grids.StructuredGrid{T, G,N}, p::NTuple{N,Real}; kwargs...,
) where {S,G,T,N,NA}
    n = prod(Grids.size_tuple(grid))
    return interpolate!(Vector{S}(undef, length(field) ÷ n), field, grid, p; kwargs...)
end

function derivative!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T, G,N},
    dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
    active_only::Bool = true, masked = zero(S), backend = nothing,
    policy::AbstractMaskPolicy = BlankMasked(), scratch = nothing,
) where {S,G,T,N,NA}
    d = Int(dim)
    geo = Grids.grid_geometry(grid)
    # The blanking policy on the host is the path a plan serves; the degrading policies rebuild a window
    # from the axis and a device launch has its own loop shape.
    if backend === nothing && policy isa BlankMasked
        _check_batched(out, field, grid, Val(N), d)
        plan = Discretization.stencil_plan(grid, d; order = order, nodes = nodes)
        msk = active_only && !(Grids.mask(grid) isa Grids.AllActive) ? Grids.mask(grid) : nothing
        if _fusable(out, field, geo, msk)
            vm = _mask_rank(msk, Val(NA))
            # The factors are built inside the switch, so the differenced direction is a type
            # parameter there: `scale_factors(geo, pt)[dim]` folds to one component only then.
            buf = _metric_scratch(T, _nfactors(grid))
            _dispatch_dim(d, vm) do vdim
                R = _metric_row_factors!(buf, grid, vdim, Val(N))
                _plan_sweep_host!(out, field, plan, msk, masked, vdim, Val(NA), vm, buf, R)
            end
            return out
        end
        apply_stencil!(out, field, plan, d; mask = msk, masked = masked)
        return _scale_by_metric!(out, grid, d, masked)
    end
    apply_stencil!(out, field, grid, dim; order = order, nodes = nodes,
                                  active_only = active_only, masked = masked, backend = backend,
                                  policy = policy, scratch = scratch)
    return _scale_by_metric!(out, grid, d, masked)
end

"""
    _fusable(out, field, geo, mask) -> Bool

Whether the metric factor can be applied inside the sweep, which needs it to be constant across each
contiguous direction-1 run the sweep writes.

For a geometry that declares direction 1 metric-invariant the factor depends on directions `2:N`, so it
is constant on every such run, whichever direction is differenced — the sweep scales run by run. A
Cartesian metric is the identity and has nothing to apply, and the run addressing needs the linear
layout.
"""
@inline function _fusable(out, field, geo, msk)
    geo isa Geometry.AbstractCartesianGeometry && return false
    1 in Geometry.metric_invariant_directions(geo) || return false
    return _linear_layout(out, field, msk)
end

"""
    derivative!(out, field, grid, plan, dim; active_only=true, masked=zero) -> out

[`derivative!`](@ref) from a held [`Discretization.stencil_plan`](@ref).

The form to use in a loop: the weights and each row's metric factor depend on the grid alone, so both
are built once here. Where the factor is constant across the span the sweep writes — see
[`_fusable`](@ref) — it is applied to each row as that row is written, while it is still in cache,
saving a second pass over `out`.
"""
function derivative!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T, G,N},
    plan::Discretization.AbstractStencilPlan, dim::Integer;
    active_only::Bool = true, masked = zero(S),
) where {S,G,T,N,NA}
    d = Int(dim)
    _check_batched(out, field, grid, Val(N), d)
    geo = Grids.grid_geometry(grid)
    msk = active_only && !(Grids.mask(grid) isa Grids.AllActive) ? Grids.mask(grid) : nothing
    if _fusable(out, field, geo, msk)
        vm = _mask_rank(msk, Val(NA))
        buf = _metric_scratch(T, _nfactors(grid))
        _dispatch_dim(d, vm) do vdim
            R = _metric_row_factors!(buf, grid, vdim, Val(N))
            _plan_sweep_host!(out, field, plan, msk, masked, vdim, Val(NA), vm, buf, R)
        end
        return out
    end
    apply_stencil!(out, field, plan, d; mask = msk, masked = masked)
    return _scale_by_metric!(out, grid, d, masked)
end

"""
    _metric_row_factors!(buf, grid, Val(dim), Val(N)) -> Int

Write the inverse scale factor of every direction-1 run into `buf`, in column-major order over
directions `2:N` — the order the sweep writes them in, for any differenced direction. Returns how many.

Direction 1 is metric-invariant here (see [`_fusable`](@ref)), so one factor covers a whole run and any
axis-1 coordinate serves as the point's first component.

A degenerate factor — one at or below the geometry's [`Discretization.metric_floor`](@ref) — is stored
as zero, and [`_scale_span!`](@ref) blanks a run whose factor is zero.
"""
function _metric_row_factors!(
    buf::AbstractVector{T}, grid::Grids.StructuredGrid{T,G,N}, ::Val{dim}, ::Val{N},
) where {G,T,N,dim}
    geo = Grids.grid_geometry(grid)
    floor_ = Discretization.metric_floor(geo)
    sz = Grids.size_tuple(grid)
    @inbounds x1 = T(Grids.coordinates(grid, 1)[1])
    rest = CartesianIndices(ntuple(d -> sz[d + 1], Val(N - 1)))
    @inbounds for (r, Ir) in enumerate(rest)
        pt = (x1, ntuple(d -> T(Grids.coordinates(grid, d + 1)[Ir[d]]), Val(N - 1))...)
        h = Geometry.scale_factors(geo, pt)[dim]
        buf[r] = abs(h) ≤ floor_ ? zero(T) : inv(h)
    end
    return length(rest)
end

"""
    _metric_scratch(T, n) -> Vector{T}

A task-held buffer of at least `n` factors, as
[`Connectivity.ball_scratch`](@ref FlowGeometries.Connectivity.ball_scratch) is for a ball query.

The fused sweeps are serial and call no user code, so one buffer per task is enough; it grows to the
largest grid the task has swept and is refilled per call.
"""
function _metric_scratch(::Type{T}, n::Int) where {T}
    v = get!(() -> T[], task_local_storage(), (:flowgeometries_metric_factors, T))::Vector{T}
    length(v) < n && resize!(v, n)
    return v
end

@inline _nfactors(grid::Grids.StructuredGrid{T,G,N}) where {T,G,N} =
    prod(ntuple(d -> @inbounds(Grids.size_tuple(grid)[d + 1]), Val(N - 1)))

# `out`/`field` may carry trailing batch axes beyond the grid's own: a `(Nx, Ny, Nb)` field against a
# 2-D grid is `Nb` fields differenced together in one pass. The grid fixes how many leading axes are
# spatial, and a disagreement over those raises; only extra trailing axes are admitted.
@inline function _check_batched(
    out::AbstractArray{<:Any,NA}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T, G,N},
    ::Val{N}, dim::Int,
) where {NA,G,T,N}
    NA ≥ N || throw(DimensionMismatch(
        "field has $NA axes but the grid has $N",
    ))
    1 ≤ dim ≤ N || throw(ArgumentError(
        "direction $dim is outside the grid's 1:$N" *
        (dim ≤ NA ? " (it is a batch axis, which carries no stencil)" : ""),
    ))
    size(out) == size(field) || throw(DimensionMismatch(
        "out $(size(out)) and field $(size(field)) must have the same size",
    ))
    ntuple(d -> size(field, d), Val(N)) == Grids.size_tuple(grid) || throw(DimensionMismatch(
        "field's leading $N axes $(ntuple(d -> size(field, d), Val(N))) do not match the grid " *
        "$(Grids.size_tuple(grid))",
    ))
    return nothing
end

# A Cartesian metric is the identity, so the derivative with respect to distance is already the one
# `apply_stencil!` wrote and there is nothing to divide by.
@inline _scale_by_metric!(
    out::AbstractArray{S,NA}, ::Grids.StructuredGrid{T, G,N}, ::Int, _masked,
) where {S,G<:Geometry.AbstractCartesianGeometry,T,N,NA} = out

function _scale_by_metric!(
    out::AbstractArray{S,NA}, grid::Grids.StructuredGrid{T, G,N}, dim::Int, masked,
) where {S,G,T,N,NA}
    geo = Grids.grid_geometry(grid)
    # Whether the factor can be solved once per row is the geometry's to say: a geometry defined outside
    # this package writes its own `scale_factors`, and hoisting one that varies along direction 1 gives
    # a wrong derivative.
    1 in Geometry.metric_invariant_directions(geo) ||
        return _scale_by_metric_percell!(out, grid, dim, masked)
    floor_ = Discretization.metric_floor(geo)
    sz = Grids.size_tuple(grid)
    # The factor is constant along axis 1 whichever direction is differenced: computed once per remaining
    # index, then swept along the contiguous axis. Any axis-1 coordinate serves for the point it is
    # evaluated at, so the first one is used.
    @inbounds x1 = first(Grids.coordinates(grid, 1))
    rest = CartesianIndices(ntuple(d -> sz[d + 1], Val(N - 1)))
    # No scale factor depends on the batch either, so `h` is solved once per spatial index and reused
    # across the batch.
    #
    # Addressed linearly: `rest` walks the spatial slabs in column-major order, so slab `p` starts at
    # `(p-1)*sz[1]` and batch element `b` a whole grid further on. `out[i, tr..., Tuple(Ib)...]` costs
    # a nested splat the compiler does not see through.
    ncell = prod(sz)
    nb = length(out) ÷ ncell
    if IndexStyle(out) === IndexLinear() && !Base.has_offset_axes(out)
        @inbounds for (p, Ir) in enumerate(rest)
            pt = (x1, ntuple(d -> T(Grids.coordinates(grid, d + 1)[Ir[d]]), Val(N - 1))...)
            h = Geometry.scale_factors(geo, pt)[dim]
            sbase = (p - 1) * sz[1]
            if abs(h) ≤ floor_
                for b in 0:(nb - 1), i in 1:sz[1]
                    out[b * ncell + sbase + i] = masked
                end
            else
                inv_h = inv(h)
                for b in 0:(nb - 1), i in 1:sz[1]
                    out[b * ncell + sbase + i] *= inv_h
                end
            end
        end
        return out
    end
    # Anything that does not index linearly — an offset array, a strided view — asks the array for its
    # own indexing instead.
    batch = CartesianIndices(ntuple(d -> size(out, N + d), Val(NA - N)))
    @inbounds for Ir in rest
        pt = (x1, ntuple(d -> T(Grids.coordinates(grid, d + 1)[Ir[d]]), Val(N - 1))...)
        h = Geometry.scale_factors(geo, pt)[dim]
        tr = Tuple(Ir)
        for Ib in batch
            tb = Tuple(Ib)
            if abs(h) ≤ floor_
                for i in 1:sz[1]
                    out[i, tr..., tb...] = masked
                end
            else
                inv_h = inv(h)
                for i in 1:sz[1]
                    out[i, tr..., tb...] *= inv_h
                end
            end
        end
    end
    return out
end

# The general path: the scale factor is evaluated at each cell's own point, direction 1 included. Taken
# by a geometry that does not declare direction 1 metric-invariant, which is the safe default.
function _scale_by_metric_percell!(
    out::AbstractArray{S,NA}, grid::Grids.StructuredGrid{T, G,N}, dim::Int, masked,
) where {S,G,T,N,NA}
    geo = Grids.grid_geometry(grid)
    floor_ = Discretization.metric_floor(geo)
    sz = Grids.size_tuple(grid)
    ncell = prod(sz)
    nb = length(out) ÷ ncell
    spatial = CartesianIndices(sz)
    linear = IndexStyle(out) === IndexLinear() && !Base.has_offset_axes(out)
    batch = CartesianIndices(ntuple(d -> size(out, N + d), Val(NA - N)))
    @inbounds for (k, I) in enumerate(spatial)
        pt = ntuple(d -> T(Grids.coordinates(grid, d)[I[d]]), Val(N))
        h = Geometry.scale_factors(geo, pt)[dim]
        degenerate = abs(h) ≤ floor_
        inv_h = degenerate ? zero(T) : inv(h)
        if linear
            for b in 0:(nb - 1)
                j = b * ncell + k
                out[j] = degenerate ? masked : out[j] * inv_h
            end
        else
            ti = Tuple(I)
            for Ib in batch
                J = (ti..., Tuple(Ib)...)
                out[J...] = degenerate ? masked : out[J...] * inv_h
            end
        end
    end
    return out
end

"""
    Discretization.axis_stencils(grid, dim; order=1, nodes=order+1) -> (indices, weights)

[`Discretization.axis_stencils`](@ref) for direction `dim` of `grid`, taking that direction's axis and
wrap period from the grid.

The table depends on the grid alone, so a caller differencing many fields along the same direction
builds it once and hands it to the `(out, field, grid, indices, weights, dim)` form. The
`(out, field, grid, dim)` form above rebuilds it on every call.
"""
function Discretization.axis_stencils(
    grid::Grids.StructuredGrid{T, G,N}, dim::Integer; order::Integer = 1, nodes::Integer = Int(order) + 1,
) where {G,T,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    return Discretization.axis_stencils(
        Grids.coordinates(grid, dim), order, nodes;
        period = Grids.isperiodic(grid, dim) ? Grids.period(grid, dim) : nothing,
    )
end

"""
    Discretization.stencil_plan(grid, dim; order=1, nodes=order+1) -> AbstractStencilPlan

[`Discretization.stencil_plan`](@ref) for direction `dim` of `grid`, taking that direction's axis and
wrap period from the grid.

The form to hold in a loop over fields. Which plan it is follows the axis's own spacing, so a uniform
direction gets the register-resident weights and a stretched one the table.
"""
function Discretization.stencil_plan(
    grid::Grids.StructuredGrid{T, G,N}, dim::Integer;
    order::Integer = 1, nodes::Integer = Int(order) + 1,
) where {G,T,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    return Discretization.stencil_plan(
        Grids.coordinates(grid, dim), order, nodes;
        period = Grids.isperiodic(grid, dim) ? Grids.period(grid, dim) : nothing,
    )
end

"""
    apply_stencil!(out, field, grid, indices, weights, dim; order=1, active_only=true,
                   masked=zero, policy=BlankMasked(), backend=nothing) -> out

Apply a stencil table built by [`Discretization.axis_stencils`](@ref). The mask, the wrap period and
the axis all come from `grid`.

The axis coming too, **any mask policy works here**. The bare `(indices, weights)` form has no axis to
rebuild a window from at a mask edge, so it accepts only `BlankMasked`.

This is the form to use in a loop over fields: the table is the same for all of them, and it is the one
part of the work that depends on the grid alone.
"""
function apply_stencil!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T, G,N},
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix, dim::Integer;
    order::Integer = 1, active_only::Bool = true, masked = zero(S), backend = nothing,
    policy::AbstractMaskPolicy = BlankMasked(), scratch = nothing,
) where {S,G,T,N,NA}
    _check_batched(out, field, grid, Val(N), Int(dim))
    # Resolved with a ternary the mask and the period are `Union{Nothing, …}`, and a small union
    # crossing a keyword boundary boxes. Branching leaves every leaf concretely typed.
    msk = Grids.mask(grid)
    if active_only && !(msk isa Grids.AllActive)
        return _apply_tbl!(out, field, grid, indices, weights, Int(dim), Int(order), msk, masked,
                           backend, policy, scratch)
    end
    return _apply_tbl!(out, field, grid, indices, weights, Int(dim), Int(order), nothing, masked,
                       backend, policy, scratch)
end

@inline function _apply_tbl!(
    out, field, grid::Grids.StructuredGrid, indices, weights, dim::Int, order::Int, msk, masked, backend,
    policy, scratch,
)
    x = Grids.coordinates(grid, dim)
    return Grids.isperiodic(grid, dim) ?
        apply_stencil!(out, field, x, indices, weights, dim; order = order,
                                      period = Grids.period(grid, dim), mask = msk, masked = masked,
                                      backend = backend, policy = policy, scratch = scratch) :
        apply_stencil!(out, field, x, indices, weights, dim; order = order,
                                      period = nothing, mask = msk, masked = masked,
                                      backend = backend, policy = policy, scratch = scratch)
end

"""
    derivative!(out, field, grid, indices, weights, dim; order=1, active_only=true, masked=zero,
                policy=BlankMasked(), backend=nothing) -> out

[`derivative!`](@ref) from a table the caller holds — the same reuse as the `apply_stencil!` form
above, for the entry point a geometry-aware caller actually uses.

The metric fuses into the sweep here too, on the terms [`_fusable`](@ref) states.
"""
function derivative!(
    out::AbstractArray{S,NA}, field::AbstractArray{<:Any,NA}, grid::Grids.StructuredGrid{T, G,N},
    indices::AbstractMatrix{<:Integer}, weights::AbstractMatrix, dim::Integer;
    order::Integer = 1, active_only::Bool = true, masked = zero(S), backend = nothing,
    policy::AbstractMaskPolicy = BlankMasked(), scratch = nothing,
) where {S,G,T,N,NA}
    d = Int(dim)
    if backend === nothing && policy isa BlankMasked
        _check_batched(out, field, grid, Val(N), d)
        size(indices) == size(weights) || throw(DimensionMismatch(
            "indices $(size(indices)) and weights $(size(weights)) must have the same size",
        ))
        size(indices, 1) == size(field, d) || throw(DimensionMismatch(
            "got $(size(indices, 1)) stencil rows for direction $d of length $(size(field, d))",
        ))
        msk = active_only && !(Grids.mask(grid) isa Grids.AllActive) ? Grids.mask(grid) : nothing
        if _fusable(out, field, Grids.grid_geometry(grid), msk)
            _check_mask_extent(msk, size(field), d)
            vm = _mask_rank(msk, Val(NA))
            buf = _metric_scratch(T, _nfactors(grid))
            _dispatch_dim(d, vm) do vdim
                R = _metric_row_factors!(buf, grid, vdim, Val(N))
                _dispatch_nodes(size(indices, 2)) do vk
                    _stencil_sweep_host!(out, field, indices, weights, msk, masked, vdim, vk,
                                         Val(NA), vm, buf, R)
                end
            end
            return out
        end
    end
    apply_stencil!(out, field, grid, indices, weights, dim; order = order,
                                  active_only = active_only, masked = masked, backend = backend,
                                  policy = policy, scratch = scratch)
    return _scale_by_metric!(out, grid, d, masked)
end
