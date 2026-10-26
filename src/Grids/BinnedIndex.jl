# ---------------------------------------------------------------------------
# Binned index of a Cartesian point set
# ---------------------------------------------------------------------------

"""
    AbstractBinLayout

How a [`BinnedIndex`](@ref) stores its boxes: [`DenseBins`](@ref), [`HashedBins`](@ref), or
[`OccupancyBins`](@ref), which builds one of the two by how full the lattice is.
"""
abstract type AbstractBinLayout end

"""
    DenseBins()

`ptr` runs over every box of the lattice, so box `b`'s cells are the positions `ptr[b]:ptr[b+1]-1` and the
boxes of a line along axis 1 are one range of positions.
"""
struct DenseBins <: AbstractBinLayout end

"""
    HashedBins()

`ptr` runs over `length(ptr) - 1` buckets of a hash of the box, and `bins[s]` is the box of position `s`,
compared before a cell is a candidate: the layout for a set clustered in parts of its bounding box, whose
lattice is mostly empty.
"""
struct HashedBins <: AbstractBinLayout end

"""
    OccupancyBins()

[`DenseBins`](@ref) where the lattice holds at most eight boxes per cell, as for cells that fill their
bounding box, and [`HashedBins`](@ref) otherwise.
"""
struct OccupancyBins <: AbstractBinLayout end

"""
    BinnedIndex{D,T,L}

The cells of a grid of Cartesian geometry binned on a lattice of boxes of widths `h` from `lo`, about one
mean cell spacing wide, `nb` boxes per direction, axis 1 fastest, stored in layout `L`
([`DenseBins`](@ref) or [`HashedBins`](@ref)). `order[s]` is the linear index of the cell at sorted position
`s`, `x[d][s]` its coordinate along `d`, reduced into `[lo, lo + period)` along a wrapping direction, and
`active[s]` whether it is active (`active === nothing` on a grid with no inactive cell). `bins` is empty in
the dense layout.

A query of radius `r` visits the boxes its ball reaches, along each line of the dense layout the run of
boxes its chord spans, and gates each candidate on the stored coordinates before its cell is read. `slack`
bounds the difference between an offset formed from the stored coordinates and the grid's own.
"""
struct BinnedIndex{D,T<:AbstractFloat,L<:Union{DenseBins,HashedBins},VI<:AbstractVector{Int},
                   VT<:AbstractVector{T},VB<:AbstractVector{NTuple{D,Int}},MK}
    lo::NTuple{D,T}
    h::NTuple{D,T}
    nb::NTuple{D,Int}
    wrap::NTuple{D,Bool}
    period::NTuple{D,T}
    layout::L
    order::VI
    ptr::VI
    bins::VB
    x::NTuple{D,VT}
    active::MK
    slack::T
    active_only::Bool
end

Base.show(io::IO, ix::BinnedIndex{D}) where {D} =
    print(io, "BinnedIndex{", D, "}(", length(ix.order), " cells, ", join(ix.nb, "×"), " bins, ",
          nameof(typeof(ix.layout)), ")")

"""
    _binned_lattice(grid, width, Val(D)) -> (lo, h, nb, wrap, period)

The lattice a [`BinnedIndex`](@ref) bins on: side `width`, by default the mean cell size
`(Σ measure / ncells)^{1/D}`, rounded along a wrapping direction to divide its period; the origin is the
grid's along a wrapping direction and its lowest coordinate along a bounded one.
"""
function _binned_lattice(grid::AbstractGrid{G,T}, width, ::Val{D}) where {G,T,D}
    ncell = length(mask(grid))
    h0 = width !== nothing ? T(width) :
         ncell == 0 ? one(T) : (T(_total_measure(grid)) / ncell)^(one(T) / D)
    (isfinite(h0) && h0 > 0) || throw(ArgumentError("the bin side must be positive and finite, got $h0"))
    wrap = ntuple(d -> isperiodic(grid, d), Val(D))
    prd = ntuple(d -> wrap[d] ? T(period(grid, d)) : zero(T), Val(D))
    lo = ntuple(d -> wrap[d] ? T(origin(grid, d)) : T(bounds(grid, d)[1]), Val(D))
    nb = ntuple(Val(D)) do d
        wrap[d] && return max(1, floor(Int, prd[d] / h0))
        return floor(Int, (T(bounds(grid, d)[2]) - lo[d]) / h0) + 1
    end
    h = ntuple(d -> wrap[d] ? prd[d] / nb[d] : h0, Val(D))
    return lo, h, nb, wrap, prd
end

"""
    binned_index(grid; active_only = false, width = nothing, layout = OccupancyBins()) -> BinnedIndex

A [`BinnedIndex`](@ref) over the cells of `grid`, a grid of Cartesian geometry. `width` is the bin side, by
default the mean cell size; `active_only` as for [`cell_list`](@ref); `layout` an
[`AbstractBinLayout`](@ref).
"""
function binned_index(grid::AbstractGrid{G,T}; active_only::Bool = false, width = nothing,
                      layout::AbstractBinLayout = OccupancyBins()) where {G,T}
    embedding_of(grid) isa CartesianEmbedding || throw(ArgumentError(
        "a BinnedIndex bins Cartesian coordinates; $(nameof(typeof(grid_geometry(grid)))) embeds in " *
        "another space. Build a `cell_list` for it.",
    ))
    return _binned_index(grid, active_only, width, layout, Val(ncoordinates(grid)))
end

function _binned_index(grid::AbstractGrid{G,T}, active_only::Bool, width, layout::AbstractBinLayout,
                       ::Val{D}) where {G,T,D}
    everything = !active_only || mask(grid) isa AllActive
    n = everything ? length(cells(grid)) : count(mask(grid))
    ids = Vector{Int}(undef, n)
    xs = ntuple(_ -> Vector{T}(undef, n), Val(D))
    act = Vector{Bool}(undef, n)
    k = 0
    @inbounds for c in cells(grid)
        cell = cell_at(grid, c)
        a = _cell_active(grid, cell)
        (everything || a) || continue
        k += 1
        ids[k] = _linear_of(grid, cell)
        _store_point!(xs, k, map(T, embedded_at(grid, cell)))
        act[k] = a
    end
    lo, h, nb, wrap, prd = _binned_lattice(grid, width, Val(D))
    @inbounds for d in 1:D
        wrap[d] || continue
        v = xs[d]
        for j in 1:n
            r = lo[d] + mod(v[j] - lo[d], prd[d])
            v[j] = r < lo[d] + prd[d] ? r : lo[d]
        end
    end
    cellbin = Vector{NTuple{D,Int}}(undef, n)
    @inbounds for j in 1:n
        cellbin[j] = _binned_box(xs, j, lo, h, nb)
    end
    return _binned_store(_bin_layout(layout, nb, n), grid, ids, xs, act, cellbin, lo, h, nb, wrap, prd,
                         active_only)
end

# The layout an index is stored in. Under `OccupancyBins` this reads the data, so the index's type does
# too, decided once here at the build.
@inline _bin_layout(l::Union{DenseBins,HashedBins}, ::NTuple, ::Int) = l
@inline _bin_layout(::OccupancyBins, nb::NTuple, n::Int) =
    prod(big, nb) <= 8 * max(n, 1) ? DenseBins() : HashedBins()

function _nkeys(::DenseBins, nb::NTuple, ::Int)
    p = prod(big, nb)
    p < typemax(Int) || throw(ArgumentError(
        "a dense layout over $(join(nb, "×")) boxes has more boxes than it can index; take " *
        "`HashedBins()` or a wider `width`",
    ))
    return Int(p)
end
@inline _nkeys(::HashedBins, ::NTuple, n::Int) = max(1, nextpow(2, max(n, 1)))

@inline _bin_key(::DenseBins, box::NTuple{D,Int}, strides::NTuple{D,Int}, ::Int) where {D} =
    1 + sum(ntuple(d -> box[d] * strides[d], Val(D)))
@inline _bin_key(::HashedBins, box::NTuple, ::NTuple, nkeys::Int) = _bin_hash(box, nkeys)

@inline _stored_bins(::DenseBins, cellbin::AbstractVector, ::AbstractVector{Int}) = similar(cellbin, 0)
@inline _stored_bins(::HashedBins, cellbin::AbstractVector, perm::AbstractVector{Int}) = cellbin[perm]

function _binned_store(layout::Union{DenseBins,HashedBins}, grid::AbstractGrid{G,T}, ids, xs::NTuple{D},
                       act, cellbin, lo, h, nb, wrap, prd, active_only::Bool) where {G,T,D}
    n = length(ids)
    strides = accumulate(*, (1, Base.front(nb)...))
    nkeys = _nkeys(layout, nb, n)
    key = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        key[j] = _bin_key(layout, cellbin[j], strides, nkeys)
    end
    ptr = zeros(Int, nkeys + 1)
    @inbounds for b in key
        ptr[b + 1] += 1
    end
    ptr[1] = 1
    @inbounds for b in 1:nkeys
        ptr[b + 1] += ptr[b]
    end
    perm = Vector{Int}(undef, n)
    next = ptr[1:nkeys]
    @inbounds for j in 1:n
        perm[next[key[j]]] = j
        next[key[j]] += 1
    end
    x = ntuple(d -> xs[d][perm], Val(D))
    xscale = maximum(d -> (n == 0 ? zero(T) : maximum(abs, x[d])) + prd[d], 1:D)
    return BinnedIndex(lo, h, nb, wrap, prd, layout, ids[perm], ptr, _stored_bins(layout, cellbin, perm), x,
                       _index_activity(mask(grid), act, perm), 16 * eps(T) * xscale, active_only)
end

@inline _index_activity(::AllActive, ::AbstractVector{Bool}, ::AbstractVector{Int}) = nothing
@inline _index_activity(::AbstractArray{Bool}, act::AbstractVector{Bool}, perm::AbstractVector{Int}) = act[perm]

@inline function _store_point!(xs::NTuple{D}, k::Int, p::NTuple{D}) where {D}
    @inbounds for d in 1:D
        xs[d][k] = p[d]
    end
    return xs
end

# The 0-based box of point `j`.
@inline _binned_box(xs::NTuple{D}, j::Int, lo, h, nb) where {D} =
    ntuple(d -> clamp(_bin_floor(@inbounds(xs[d][j]), lo[d], h[d]), 0, nb[d] - 1), Val(D))

# The raw bin a coordinate falls in, any integer: a point off the lattice, or a radius wider than it,
# gives an index past either end, which a caller clips or wraps.
@inline function _bin_floor(x::T, lo::T, h::T) where {T}
    f = floor((x - lo) / h)
    return Base.unsafe_trunc(Int, clamp(f, -T(2)^52, T(2)^52))
end

# Raw bin `i` of a direction of `n` bins as the bin on the lattice and the offset of the image it is,
# `(-1, 0)` past a bounded edge.
@inline function _binned_wrap(i::Int, n::Int, wrap::Bool, P::T) where {T}
    0 <= i < n && return i, zero(T)
    wrap || return -1, zero(T)
    return mod(i, n), T(fld(i, n)) * P
end

# The raw bins `a…b` a walk visits along a direction: clipped to the lattice where it is bounded, and where
# it wraps at most one period of them with `once`.
@inline function _bin_span(a::Int, b::Int, n::Int, wrap::Bool, once::Bool)
    wrap || return max(a, 0):min(b, n - 1)
    return (once && b - a + 1 > n) ? (a:(a + n - 1)) : (a:b)
end

# The raw bins along direction `d` a ball of radius `re` about `q` reaches.
@inline _bin_reach(ix::BinnedIndex, q, re, d::Int, once::Bool) =
    _bin_span(_bin_floor(q[d] - re, ix.lo[d], ix.h[d]), _bin_floor(q[d] + re, ix.lo[d], ix.h[d]),
              ix.nb[d], ix.wrap[d], once)

# Distance from `x` to the interval `[a, a + h]`.
@inline _gap(x::T, a::T, h::T) where {T} = max(zero(T), a - x, x - (a + h))

"""
    _binned_walk(g, acc, ix, q, re, once) -> acc

`acc = g(acc, s0, s1, shift, box)` over the runs of positions holding the cells in the boxes a ball of
radius `re` about `q` (in the index's coordinates, reduced into its period) reaches, `shift` the offset of
the image a run is. `box` is `nothing` on the dense layout, where every position of a run is in a box the
ball reaches, and on the hashed one the box whose cells, among the bucket's positions `s0:s1`, are the
candidates. With `once = true` a wrapping direction is walked over at most one period of boxes, so each cell
is a candidate once; with `once = false` each image the ball reaches is its own.
"""
@inline _binned_walk(g::F, acc, ix::BinnedIndex{D,T,DenseBins}, q, re, once::Bool) where {F,D,T} =
    _binned_walk_lines(g, acc, ix, q, re, once)
@inline _binned_walk(g::F, acc, ix::BinnedIndex{D,T,HashedBins}, q, re, once::Bool) where {F,D,T} =
    _binned_walk_boxes(g, acc, ix, q, re, once)

# The dense layout, line by line: per line of boxes along axis 1 within reach, the run its chord spans.
@inline function _binned_walk_lines(g::F, acc, ix::BinnedIndex{D,T}, q::NTuple{D,T}, re::T,
                                    once::Bool) where {F,D,T}
    strides = Base.tail(accumulate(*, (1, Base.front(ix.nb)...)))
    spans = ntuple(t -> _bin_reach(ix, q, re, t + 1, once), Val(D - 1))
    re2 = re * re
    @inbounds for Jc in CartesianIndices(spans)
        lin, tshift, gap2 = _binned_line(ix, q, Tuple(Jc), strides)
        (lin < 0 || gap2 > re2) && continue
        e = sqrt(re2 - gap2)
        span = _bin_span(_bin_floor(q[1] - e, ix.lo[1], ix.h[1]), _bin_floor(q[1] + e, ix.lo[1], ix.h[1]),
                         ix.nb[1], ix.wrap[1], once)
        acc = _binned_axis1(g, acc, ix, first(span), last(span), lin, tshift)
    end
    return acc
end

# The line of boxes at raw tail index `Jr` (axes `2…D`): the linear offset of its first box, the offsets of
# the image it is, and the squared distance from `q` to its boxes across those axes; offset `-1` past a
# bounded edge.
@inline function _binned_line(ix::BinnedIndex{D,T}, q::NTuple{D,T}, Jr::NTuple{M,Int}, strides) where {D,T,M}
    w = ntuple(t -> _binned_wrap(Jr[t], ix.nb[t + 1], ix.wrap[t + 1], ix.period[t + 1]), Val(M))
    tshift = map(last, w)
    any(wt -> first(wt) < 0, w) && return -1, tshift, zero(T)
    lin = sum(ntuple(t -> first(w[t]) * strides[t], Val(M)); init = 0)
    gap2 = sum(ntuple(t -> _gap(q[t + 1], ix.lo[t + 1] + Jr[t] * ix.h[t + 1], ix.h[t + 1])^2, Val(M));
               init = zero(T))
    return lin, tshift, gap2
end

# The raw boxes `a…b` along axis 1 of the line at linear offset `lin`: one run per image of the line, the
# line's boxes being consecutive positions.
@inline function _binned_axis1(g::F, acc, ix::BinnedIndex{D,T}, a::Int, b::Int, lin::Int,
                               tshift::NTuple{M,T}) where {F,D,T,M}
    a <= b || return acc
    nb1 = ix.nb[1]
    @inbounds if !ix.wrap[1]
        return g(acc, ix.ptr[1 + lin + a], ix.ptr[2 + lin + b] - 1, (zero(T), tshift...), nothing)
    end
    i = a
    @inbounds while i <= b
        iw, sx = _binned_wrap(i, nb1, true, ix.period[1])
        e = min(b, i + nb1 - 1 - iw)
        acc = g(acc, ix.ptr[1 + lin + iw], ix.ptr[2 + lin + iw + e - i] - 1, (sx, tshift...), nothing)
        i = e + 1
    end
    return acc
end

# The hashed layout, box by box: every box within reach whose nearest point lies within `re`.
@inline function _binned_walk_boxes(g::F, acc, ix::BinnedIndex{D,T}, q::NTuple{D,T}, re::T,
                                    once::Bool) where {F,D,T}
    spans = ntuple(d -> _bin_reach(ix, q, re, d, once), Val(D))
    nkeys = length(ix.ptr) - 1
    re2 = re * re
    @inbounds for Jc in CartesianIndices(spans)
        Jr = Tuple(Jc)
        w = ntuple(d -> _binned_wrap(Jr[d], ix.nb[d], ix.wrap[d], ix.period[d]), Val(D))
        any(wd -> first(wd) < 0, w) && continue
        gap2 = sum(ntuple(d -> _gap(q[d], ix.lo[d] + Jr[d] * ix.h[d], ix.h[d])^2, Val(D)); init = zero(T))
        gap2 > re2 && continue
        box = map(first, w)
        bucket = _bin_hash(box, nkeys)
        acc = g(acc, ix.ptr[bucket], ix.ptr[bucket + 1] - 1, map(last, w), box)
    end
    return acc
end

# Whether position `s` of a run lies in its box: always on the dense layout.
@inline _in_box(::BinnedIndex, ::Nothing, ::Int) = true
@inline _in_box(ix::BinnedIndex, box::Tuple, s::Int) = @inbounds ix.bins[s] == box

# A point in the index's coordinates: reduced into the period along each wrapping direction.
@inline _binned_reduce(ix::BinnedIndex{D,T}, q::NTuple{D,T}) where {D,T} =
    ntuple(d -> ix.wrap[d] ? ix.lo[d] + mod(q[d] - ix.lo[d], ix.period[d]) : q[d], Val(D))

function fold_candidates(f::F, acc, ix::BinnedIndex{D,T}, grid, I, r) where {F,D,T}
    return fold_candidates_at(f, acc, ix, map(T, embedded_at(grid, I)), r, nothing)
end

# Each cell once, at its image nearest `q`: the walk covers at most one period, and the gate measures to
# the nearest image. Its radius is widened by the slack, so the set is a superset of the exact ball.
function fold_candidates_at(f::F, acc, ix::BinnedIndex{D,T}, q::NTuple{D,T}, r, _scratch) where {F,D,T}
    re = T(r) + ix.slack + 16 * eps(T) * T(r)
    qq = _binned_reduce(ix, q)
    return _binned_walk(acc, ix, qq, re, true) do a, s0, s1, shift, box
        _binned_nearest(f, a, ix, qq, shift, box, re * re, s0, s1)
    end
end

@inline function _binned_nearest(f::F, acc, ix::BinnedIndex{D,T}, q::NTuple{D,T}, shift::NTuple{D,T}, box,
                                 re2::T, s0::Int, s1::Int) where {F,D,T}
    @inbounds for s in s0:s1
        _in_box(ix, box, s) || continue
        d2 = zero(T)
        for d in 1:D
            δ = ix.x[d][s] + shift[d] - q[d]
            ix.wrap[d] && (δ -= round(δ / ix.period[d]) * ix.period[d])
            d2 += δ * δ
        end
        d2 <= re2 && (acc = f(acc, ix.order[s]))
    end
    return acc
end

"""
    fold_images(f, acc, ix::BinnedIndex, q, r) -> acc

`acc = f(acc, s, d², shift)` for every image of every indexed cell within `r` of the point `q` (in the
index's coordinates): `s` the cell's sorted position, `ix.order[s]` its linear index, `d²` the squared
distance from the stored coordinates and `shift` the image's offset. Across a wrapping direction each image
inside the ball is its own entry.
"""
function fold_images(f::F, acc, ix::BinnedIndex{D,T}, q::NTuple{D,T}, r) where {F,D,T}
    rr = T(r)
    qq = _binned_reduce(ix, q)
    # The walk is widened by the slack, a stored coordinate lying in its box only to that much; the gate
    # is the exact squared radius.
    return _binned_walk(acc, ix, qq, rr + ix.slack + 16 * eps(T) * rr, false) do a, s0, s1, shift, box
        _binned_images(f, a, ix, qq, shift, box, rr * rr, s0, s1)
    end
end

@inline function _binned_images(f::F, acc, ix::BinnedIndex{D,T}, q::NTuple{D,T}, shift::NTuple{D,T}, box,
                                r2::T, s0::Int, s1::Int) where {F,D,T}
    @inbounds for s in s0:s1
        _in_box(ix, box, s) || continue
        d2 = zero(T)
        for d in 1:D
            δ = ix.x[d][s] + shift[d] - q[d]
            d2 += δ * δ
        end
        d2 <= r2 && (acc = f(acc, s, d2, shift))
    end
    return acc
end

function index_within!(buf::AbstractVector{<:Integer}, ix::BinnedIndex, grid, I, r)
    empty!(buf)
    fold_candidates(nothing, ix, grid, I, r) do _, k
        push!(buf, k)
        return nothing
    end
    return buf
end
