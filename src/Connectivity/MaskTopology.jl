# ---------------------------------------------------------------------------
# Mask topology
# ---------------------------------------------------------------------------

"""
    interior(grid; stencil = Stencils.Axial(1)) -> Array{Bool}

Which active cells have their whole stencil active and in range. `false` at a domain edge that does
not wrap, and beside any masked-out cell.
"""
function interior(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1))
    return _interior(IndexTopology(grid), _stencil_val(stencil))
end

function _interior(t::IndexTopology{N}, sten::Stencils.AbstractStencil) where {N}
    sz = t.size
    per = t.periodic
    offs = _stencil_offsets(Val{N}(), sten)
    out = fill(false, sz)
    @inbounds for ci in CartesianIndices(sz)
        I = Tuple(ci)
        _active(t, I...) || continue
        ok = true
        for δ in offs
            J = ntuple(d -> _wrap_or_clip(I[d], δ[d], sz[d], per[d]), Val(N))
            if any(==(0), J) || !_active(t, J...)
                ok = false
                break
            end
        end
        out[ci] = ok
    end
    return out
end

"""
    boundary_cells(grid; stencil = Stencils.Axial(1)) -> Array{Bool}

Which active cells are NOT [`interior`](@ref): the active cells that touch an edge or a masked-out
neighbour.
"""
function boundary_cells(grid::Grids.AbstractGrid; stencil = Stencils.Axial(1))
    t = IndexTopology(grid)
    int = _interior(t, _stencil_val(stencil))
    out = similar(int)
    @inbounds for ci in CartesianIndices(t.size)
        out[ci] = _active(t, Tuple(ci)...) && !int[ci]
    end
    return out
end

"""
    connected_components(grid; stencil = Stencils.Axial(1), active = true) -> (labels, ncomponents)

Label the connected components of the active region (or of the inactive region with
`active = false`), by flood fill honouring the grid's own wrapping. `labels` is `0` off the region and
`1:ncomponents` on it.
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

A region that reaches a non-wrapping edge is outside rather than enclosed. Along a wrapping direction
there is no edge to reach, so enclosure there is decided by the fill alone.
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
