# ---------------------------------------------------------------------------
# Mask topology
# ---------------------------------------------------------------------------

"""
    _stencil_closed(t, stencil, I) -> Bool

Whether cell `I` is active and every offset from it lands in range on an active cell — the one question
both [`interior`](@ref) and [`boundary_cells`](@ref) ask, so they cannot answer it differently.

The offset walk is unrolled at compile time through [`Stencils.fold_offsets`](@ref)
"""
@inline function _stencil_closed(
    t::IndexTopology{N}, sten::Stencils.AbstractStencil, I::NTuple{N,Int},
) where {N}
    _active(t, I...) || return false
    sz = t.size
    per = t.periodic
    return Stencils.fold_offsets(true, sten, Val(N)) do ok, δ
        ok || return false
        J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
        return !any(==(0), J) && _active(t, J...)
    end
end

"""
    interior(grid; stencil = Stencils.Axial(1), backend = nothing) -> Array{Bool}

Which active cells have their whole stencil active and in range. `false` at a domain edge that does
not wrap, and beside any masked-out cell.

Each cell's answer depends on its own neighbourhood alone, so `backend` runs the cells concurrently.
"""
function interior(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1), backend = nothing)
    return _interior(IndexTopology(grid), _stencil_val(stencil); backend = backend)
end

function _interior(
    t::IndexTopology{N}, sten::Stencils.AbstractStencil; backend = nothing,
) where {N}
    sz = t.size
    out = fill(false, sz)
    ci = CartesianIndices(sz)
    Execution.run_indices(prod(sz), backend) do k
        @inbounds out[k] = _stencil_closed(t, sten, Tuple(ci[k]))
    end
    return out
end

"""
    boundary_cells(grid; stencil = Stencils.Axial(1), backend = nothing) -> Array{Bool}

Which active cells are NOT [`interior`](@ref): the active cells that touch an edge or a masked-out
neighbour.

One pass and one array: a cell is a boundary cell exactly when it is active and its stencil is not
closed, which is the same predicate `interior` reads.
"""
function boundary_cells(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1), backend = nothing)
    return _boundary_cells(IndexTopology(grid), _stencil_val(stencil); backend = backend)
end

function _boundary_cells(
    t::IndexTopology{N}, sten::Stencils.AbstractStencil; backend = nothing,
) where {N}
    sz = t.size
    out = fill(false, sz)
    ci = CartesianIndices(sz)
    Execution.run_indices(prod(sz), backend) do k
        I = Tuple(@inbounds ci[k])
        @inbounds out[k] = _active(t, I...) && !_stencil_closed(t, sten, I)
    end
    return out
end

"""
    connected_components(grid; stencil = Stencils.Axial(1), active = true) -> (labels, ncomponents)

Label the connected components of the active region (or of the inactive region with
`active = false`), by flood fill honouring the grid's own wrapping. `labels` is `0` off the region and
`1:ncomponents` on it.

There is no `backend`: a fill's next cell is decided by the cells already labelled, so the traversal is
sequential. [`interior`](@ref) and [`boundary_cells`](@ref) are the per-cell questions, and both take one.
"""
function connected_components(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1), active::Bool = true)
    return _connected_components(IndexTopology(grid), _stencil_val(stencil), active)
end

function _connected_components(
    t::IndexTopology{N}, sten::Stencils.AbstractStencil, want::Bool,
) where {N}
    sz = t.size
    per = t.periodic
    offs = _stencil_offsets(Val{N}(), sten)
    labels = zeros(Int, sz)
    ncomp = 0
    stack = CartesianIndex{N}[]
    @inbounds for seed in CartesianIndices(sz)
        (_active(t, Tuple(seed)...) == want && labels[seed] == 0) || continue
        ncomp += 1
        empty!(stack)
        push!(stack, seed)
        labels[seed] = ncomp
        while !isempty(stack)
            I = Tuple(pop!(stack))
            for δ in offs
                J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
                any(==(0), J) && continue
                cj = CartesianIndex(J)
                (labels[cj] == 0 && _active(t, J...) == want) || continue
                labels[cj] = ncomp
                push!(stack, cj)
            end
        end
    end
    return labels, ncomp
end

"""
    count_holes(grid; stencil = Stencils.Axial(1)) -> Int

How many connected inactive regions are fully enclosed by active cells — the number of holes in the
active region, and so an estimate of its first Betti number.

A region that reaches a non-wrapping edge counts as outside. Along a wrapping direction there is no
edge to reach, so enclosure there is decided by the fill alone.
"""
function count_holes(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1))
    t = IndexTopology(grid)
    labels, ncomp = _connected_components(t, _stencil_val(stencil), false)
    ncomp == 0 && return 0
    sz = t.size
    per = t.periodic
    N = length(sz)
    touches = falses(ncomp)
    @inbounds for ci in CartesianIndices(sz)
        l = labels[ci]
        l == 0 && continue
        I = Tuple(ci)
        for d in 1:N
            per[d] && continue
            (I[d] == 1 || I[d] == sz[d]) && (touches[l] = true)
        end
    end
    return count(!, touches)
end
