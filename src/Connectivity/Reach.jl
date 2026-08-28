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

# The third adjacency: arithmetic. The tuple is a stack value, so the walk expands a cell's neighbours
# without a buffer, exactly as the per-cell queries do.
struct _FormulaAdjacency{GR}
    grid::GR
end

@inline function (a::_FormulaAdjacency)(cb::F, k::Int) where {F}
    ids, n = Grids.formula_neighbors(a.grid, k)
    @inbounds for t in 1:n
        cb(ids[t])
    end
    return nothing
end

"""
    ConnectedScratch{T}

The buffers a [`Connected`](@ref) query needs: the ball's cells and their distances, the visited flags,
the breadth-first order, the queue, and the candidate buffer the ball pass itself takes.

**One per task**, exactly as [`ball_scratch`](@ref) is — every buffer is written per query, so two tasks
cannot share a set. Build one with [`connected_scratch`](@ref) and pass it as `scratch`; without one the
query allocates its buffers each time, which is correct but is the cost its `Unrestricted` sibling does
not have.
"""
struct ConnectedScratch{
    T, VI<:AbstractVector{Int}, VT<:AbstractVector{T}, VB<:AbstractVector{Bool},
}
    idxs::VI      # the ball's cells, sorted
    ds::VT        # their distances, moved with them
    visited::VB   # per ball position
    order::VI     # the component, as positions into `idxs`
    queue::VI     # the breadth-first frontier
    cand::VI      # what the ball pass uses, i.e. `ball_scratch`'s role
end

"""
    connected_scratch([T = Float64]) -> ConnectedScratch{T}

Buffers for a [`Connected`](@ref) query, so a caller making many of them allocates none. `T` is the
grid's coordinate type, which is what its distances are.

The buffers grow to the largest ball seen and are reused, so the first query on a new size is the only
one that allocates.
"""
connected_scratch(::Type{T} = Float64) where {T} =
    ConnectedScratch{T,Vector{Int},Vector{T},Vector{Bool}}(Int[], T[], Bool[], Int[], Int[], Int[])

# The buffers for one query, from whatever the caller passed as `scratch`: a `ConnectedScratch` supplies
# all of them, a bare vector supplies the ball pass's candidate buffer alone, and `nothing` supplies
# none. Each branch returns concretely typed buffers, and the caller is specialized on the scratch type,
# so no union crosses into the walk.
@inline _conn_bufs(::Nothing, ::Type{T}) where {T} =
    (Int[], T[], Bool[], Int[], Int[], nothing)
@inline _conn_bufs(v::AbstractVector{<:Integer}, ::Type{T}) where {T} =
    (Int[], T[], Bool[], Int[], Int[], v)
@inline _conn_bufs(s::ConnectedScratch{T}, ::Type{T}) where {T} =
    (s.idxs, s.ds, s.visited, s.order, s.queue, s.cand)

@inline _swap2!(a, b, i::Int, j::Int) = @inbounds begin
    a[i], a[j] = a[j], a[i]
    b[i], b[j] = b[j], b[i]
    nothing
end

"""
    _sort_ball!(idxs, ds, lo, hi) -> nothing

Sort the ball by linear index, carrying each cell's distance with it.

Membership is then a binary search rather than a `Set`: the walk tests it once per (cell, neighbour)
pair, and `searchsortedfirst` over a sorted `Vector{Int}` beats hashing at these sizes without allocating
a dictionary per query.

Both arrays move together in place, so there is no permutation vector — `sortperm` plus two `permute!`
is three allocations on a path whose sibling is free. Quicksort with a median-of-three pivot, recursing
on the smaller side so the depth stays `O(log m)`, and insertion sort for a short span.
"""
function _sort_ball!(
    idxs::AbstractVector{Int}, ds::AbstractVector, lo::Int = 1, hi::Int = length(idxs),
)
    while lo < hi
        if hi - lo < 16
            @inbounds for a in (lo + 1):hi
                x = idxs[a]
                y = ds[a]
                b = a - 1
                while b ≥ lo && idxs[b] > x
                    idxs[b + 1] = idxs[b]
                    ds[b + 1] = ds[b]
                    b -= 1
                end
                idxs[b + 1] = x
                ds[b + 1] = y
            end
            return nothing
        end
        mid = (lo + hi) >>> 1
        @inbounds begin
            idxs[mid] < idxs[lo] && _swap2!(idxs, ds, mid, lo)
            idxs[hi] < idxs[lo] && _swap2!(idxs, ds, hi, lo)
            idxs[hi] < idxs[mid] && _swap2!(idxs, ds, hi, mid)
            pivot = idxs[mid]
        end
        i = lo
        j = hi
        @inbounds while i ≤ j
            while idxs[i] < pivot
                i += 1
            end
            while idxs[j] > pivot
                j -= 1
            end
            if i ≤ j
                _swap2!(idxs, ds, i, j)
                i += 1
                j -= 1
            end
        end
        if (j - lo) < (hi - i)
            _sort_ball!(idxs, ds, lo, j)
            lo = i
        else
            _sort_ball!(idxs, ds, i, hi)
            hi = j
        end
    end
    return nothing
end

# The seed's component, as positions into `idxs` in breadth-first order, written into the caller's
# buffers. The frontier is walked by a head index rather than `popfirst!`, so `queue` is only appended to.
function _component!(
    order::AbstractVector{Int}, queue::AbstractVector{Int}, visited::AbstractVector{Bool},
    idxs::AbstractVector{Int}, seed::Int, adj::A,
) where {A}
    m = length(idxs)
    empty!(order)
    pos = searchsortedfirst(idxs, seed)
    (pos ≤ m && @inbounds(idxs[pos]) == seed) || return order   # inactive seed: an empty ball
    resize!(visited, m)
    fill!(visited, false)
    empty!(queue)
    push!(queue, pos)
    @inbounds visited[pos] = true
    head = 1
    while head ≤ length(queue)
        p = @inbounds queue[head]
        head += 1
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
function _ball_lists!(
    idxs::AbstractVector{Int}, ds::AbstractVector,
    grid::Grids.AbstractGrid{G,T}, I, ball, images::AbstractImageConvention, active_only::Bool,
    mt::MetricTopology, cand,
) where {G,T}
    empty!(idxs)
    empty!(ds)
    _fold_within(nothing, grid, I, ball, images, active_only, true, mt, cand) do _, J, d
        push!(idxs, _sweep_linear(grid, J))
        push!(ds, d)
        return nothing
    end
    _sort_ball!(idxs, ds)
    return nothing
end

# Fold over the component in breadth-first order. Distances come from the ball pass, so a `Connected`
# fold sees exactly the distances an `Unrestricted` one does.
@inline function _emit_component(
    f::F, init, idxs::AbstractVector{Int}, ds::AbstractVector, order::AbstractVector{Int},
    seed::Int, self::Bool, to_index::C,
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
    f::F, init, grid::Grids.AbstractGrid{G,T}, I, ball, images::AbstractImageConvention,
    active_only::Bool, self::Bool, mt::MetricTopology, scratch, reach::Connected,
) where {F,G,T}
    images isa NearestImage || throw(ArgumentError(
        "`Connected` is a graph query, so a cell must be one node: use `NearestImage`, not `$(images)`",
    ))
    idxs, ds, visited, order, queue, cand = _conn_bufs(scratch, T)
    _ball_lists!(idxs, ds, grid, I, ball, images, active_only, mt, cand)
    seed = _sweep_linear(grid, I)
    _component!(order, queue, visited, idxs, seed, _walk_adjacency(grid, reach))
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

@inline function _walk_adjacency(grid, reach::Connected, ::Grids.FormulaNeighbors)
    reach isa Connected{Nothing} || throw(ArgumentError(
        "$(nameof(typeof(grid))) has no index space for a stencil to mean anything in; its adjacency " *
        "is arithmetic on the cell id, so use `Connected()`",
    ))
    return _FormulaAdjacency(grid)
end

# The one place reach is selected. Dispatching on the singleton keeps the choice at compile time, so
# `Unrestricted` pays nothing for `Connected` existing.
@inline _route_fold(f::F, init, ::Unrestricted, args...) where {F} = _fold_within(f, init, args...)
@inline _route_fold(f::F, init, r::Connected, args...) where {F} = _connected_fold(f, init, args..., r)
