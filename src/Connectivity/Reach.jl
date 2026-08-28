# ---------------------------------------------------------------------------
# Reach: the ball, or the part of it reachable from the seed
# ---------------------------------------------------------------------------
#
# `Unrestricted` and `Connected` are declared with the image conventions above, since the entry points'
# signatures name them; what follows is how `Connected` is computed.

# Adjacency as a callable, so the walk below is written once: `adj(lin) do nb … end` visits the
# neighbours of `lin` without materializing a list per cell.
struct _IndexSpaceAdjacency{N,O}
    size::NTuple{N,Int}
    periodic::NTuple{N,Bool}
    offsets::O
end

@inline function (a::_IndexSpaceAdjacency{N})(cb::F, lin::Int) where {N,F}
    I = Tuple(@inbounds CartesianIndices(a.size)[lin])
    for δ in a.offsets
        J = ntuple(d -> _wrap_or_clip(I[d], δ[d], a.size[d], a.periodic[d]), Val(N))
        any(==(0), J) && continue
        cb(_linidx(a.size, J...))
    end
    return nothing
end

struct _StoredAdjacency{VN,VP}
    nbrs::VN
    ptr::VP
end

@inline function (a::_StoredAdjacency)(cb::F, k::Int) where {F}
    @inbounds for t in a.ptr[k]:(a.ptr[k + 1] - 1)
        cb(Int(a.nbrs[t]))
    end
    return nothing
end

# Sort the ball by linear index so membership is a binary search rather than a `Set`: the walk tests
# membership once per (cell, neighbour) pair, and `searchsortedfirst` over a sorted `Vector{Int}` beats
# hashing at these sizes without allocating a dictionary per query.
function _sort_ball!(idxs::Vector{Int}, ds::Vector)
    p = sortperm(idxs)
    permute!(idxs, p)
    permute!(ds, p)
    return nothing
end

# The seed's component, as positions into `idxs` in breadth-first order.
function _component(idxs::Vector{Int}, seed::Int, adj::A) where {A}
    m = length(idxs)
    pos = searchsortedfirst(idxs, seed)
    (pos ≤ m && @inbounds(idxs[pos]) == seed) || return Int[]   # inactive seed: an empty ball
    visited = falses(m)
    order = Vector{Int}()
    sizehint!(order, m)
    queue = Int[pos]
    @inbounds visited[pos] = true
    while !isempty(queue)
        p = popfirst!(queue)
        push!(order, p)
        adj(@inbounds idxs[p]) do nb
            q = searchsortedfirst(idxs, nb)
            (q ≤ m && @inbounds(idxs[q]) == nb && !@inbounds(visited[q])) || return nothing
            @inbounds visited[q] = true
            push!(queue, q)
            return nothing
        end
    end
    return order
end

# The ball as (linear indices, distances), seed included — the walk needs random access to it, which a
# fold cannot give. `self = true` here regardless of the caller's `self`, since the seed is the walk's
# root; it is dropped afterwards if the caller did not ask for it.
function _ball_lists(
    grid::Grids.AbstractGrid{G,T}, I, ball, images::AbstractImageConvention, active_only::Bool,
    mt::MetricTopology, scratch,
) where {G,T}
    idxs, ds = Int[], T[]
    _fold_within(nothing, grid, I, ball, images, active_only, true, mt, scratch) do _, J, d
        push!(idxs, _sweep_linear(grid, J))
        push!(ds, d)
        return nothing
    end
    _sort_ball!(idxs, ds)
    return idxs, ds
end

# Fold over the component in breadth-first order. Distances come from the ball pass, so a `Connected`
# fold sees exactly the distances an `Unrestricted` one does.
@inline function _emit_component(
    f::F, init, idxs::Vector{Int}, ds::Vector, order::Vector{Int}, seed::Int, self::Bool, to_index::C,
) where {F,C}
    acc = init
    @inbounds for p in order
        lin = idxs[p]
        (lin == seed && !self) && continue
        acc = f(acc, to_index(lin), ds[p])
    end
    return acc
end

function _connected_fold(
    f::F, init, grid::Grids.AbstractGrid, I, ball, images::AbstractImageConvention,
    active_only::Bool, self::Bool, mt::MetricTopology, scratch, reach::Connected,
) where {F}
    images isa NearestImage || throw(ArgumentError(
        "`Connected` is a graph query, so a cell must be one node: use `NearestImage`, not `$(images)`",
    ))
    idxs, ds = _ball_lists(grid, I, ball, images, active_only, mt, scratch)
    seed = _sweep_linear(grid, I)
    order = _component(idxs, seed, _walk_adjacency(grid, reach))
    return _emit_component(f, init, idxs, ds, order, seed, self,
                           lin -> Grids._cell_from_linear(grid, lin))
end

# The graph the component walk expands along, from `Grids.adjacency_source` — the mesh's own neighbour
# relation, which is a different question from how the ball's candidates were enumerated.
@inline _walk_adjacency(grid::Grids.AbstractGrid, reach::Connected) =
    _walk_adjacency(grid, reach, Grids.adjacency_source(grid))

@inline function _walk_adjacency(grid, reach::Connected, ::Grids.IndexStencilNeighbors)
    sz = Grids.size_tuple(grid)
    return _IndexSpaceAdjacency(sz, _periodic_flags(grid),
                                _stencil_offsets(Val(length(sz)), _reach_stencil(reach)))
end

@inline function _walk_adjacency(grid, reach::Connected, ::Grids.StoredMeshNeighbors)
    reach isa Connected{Nothing} || throw(ArgumentError(
        "$(nameof(typeof(grid))) has no index space for a stencil to mean anything in; its adjacency " *
        "is the neighbour lists it stores, so use `Connected()`",
    ))
    return _StoredAdjacency(Grids.neighbor_nbrs(grid), Grids.neighbor_ptr(grid))
end

# The one place reach is selected. Dispatching on the singleton keeps the choice at compile time, so
# `Unrestricted` pays nothing for `Connected` existing.
@inline _route_fold(f::F, init, ::Unrestricted, args...) where {F} = _fold_within(f, init, args...)
@inline _route_fold(f::F, init, r::Connected, args...) where {F} = _connected_fold(f, init, args..., r)
