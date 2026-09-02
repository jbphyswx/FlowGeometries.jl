# ---------------------------------------------------------------------------
# Minimum image, and the distance between two cells
# ---------------------------------------------------------------------------

"""
    _wrap_lengths(grid, Val(N)) -> NTuple{N,T}

Wrap length per direction, zero where the direction is bounded, so [`_min_image`](@ref) leaves those
components alone.
"""
@inline _wrap_lengths(grid::AbstractGrid, ::Val{N}) where {N} =
    ntuple(d -> isperiodic(grid, d) ? period(grid, d) : zero(period(grid, d)), Val(N))

"""
    _min_image(p0, pt, prd) -> NTuple

`pt` brought to the image nearest `p0`, per component, for each direction with a nonzero wrap length.

For an angular coordinate the geometry's own distance is already `2π`-periodic and this changes nothing
(it also keeps Vincenty inside its `|Δλ| ≤ π` regime); on a periodic Cartesian coordinate it carries a
pair across the seam. Per-component minimum image is the global minimum for a separable metric, which
the Euclidean one is.
"""
@inline function _min_image(p0::NTuple{N,Any}, pt::NTuple{N,Any}, prd::NTuple{N,Any}) where {N}
    return ntuple(Val(N)) do d
        p = prd[d]
        p > zero(p) ? p0[d] + (pt[d] - p0[d] - p * round((pt[d] - p0[d]) / p)) : pt[d]
    end
end

"""
    displacement(grid, I, J) -> NTuple{N,T}
    displacement(grid::UnstructuredGrid, i, j) -> NTuple{N,T}

The signed per-direction coordinate offset from cell `I` to cell `J`, reduced to the nearest image in
every periodic direction — the offset [`Geometry.distance`](@ref) is taken from.

A coordinate quantity, so it lives here, while the distance extends `Geometry.distance`. Across a
periodic seam the two cells' *stored* coordinates differ by nearly a full period, and this reports the
short way round.
"""
function displacement end

"""
    _cell_indices(grid, cell) -> Tuple{Vararg{Int}}

A cell as the index tuple every mask and coordinate accessor here takes: the tuple itself where cells
are [`CartesianCells`](@ref), and `(i,)` where they are [`FlatCells`](@ref).

Both the trait and the cell's own type are dispatched on, because they answer different questions: the
trait says how this layout names a cell, the cell's type how the caller named it. A pair that does not
agree — an integer for a tuple-addressed grid — reaches the message below, at the entry point the
caller wrote.
"""
@inline _cell_indices(grid::AbstractGrid, c) = _cell_indices(grid, c, cell_address(grid))
@inline _cell_indices(_grid, I::Tuple{Vararg{Integer}}, ::CartesianCells) = map(Int, I)
@inline _cell_indices(_grid, i::Integer, ::FlatCells) = (Int(i),)

_cell_indices(grid, c, ::CartesianCells) = throw(ArgumentError(
    "a cell of $(nameof(typeof(grid))) is named by $(ncoordinates(grid)) indices; " *
    "got a $(typeof(c))",
))
_cell_indices(grid, c, ::FlatCells) = throw(ArgumentError(
    "a cell of $(nameof(typeof(grid))) is named by one integer; got a $(typeof(c))",
))

# The three things a traversal asks about a cell, each one expression for every layout.
@inline _cell_coords(grid::AbstractGrid, c) = _raw_coords(grid, _cell_indices(grid, c)...)
@inline _cell_active(grid::AbstractGrid, c) = isactive(grid, _cell_indices(grid, c)...)
@inline _cell_checkbounds(grid::AbstractGrid, c) =
    checkbounds(Bool, mask(grid), _cell_indices(grid, c)...) ||
        throw(BoundsError(mask(grid), c))

"""
    _cell_named_by(grid, I::Tuple) -> cell

The cell a caller named with the indices `I`: the tuple itself where cells are [`CartesianCells`](@ref),
and its single element where they are [`FlatCells`](@ref). An entry point taking `I::Vararg{Integer}`
hands the traversals this.
"""
@inline _cell_named_by(grid::AbstractGrid, I::Tuple{Vararg{Integer}}) =
    _cell_named_by(grid, I, cell_address(grid))

# Both lengths are known from their types, so the check folds away where the call is right. It sits
# here, at the entry point; a traversal reaches `_cell_indices` per cell with a cell it built itself.
@inline function _cell_named_by(grid, I::Tuple{Vararg{Integer}}, ::CartesianCells)
    length(I) == ndims(grid) || throw(ArgumentError(
        "a cell of $(nameof(typeof(grid))) is named by $(ndims(grid)) indices; got $(length(I))",
    ))
    return map(Int, I)
end

@inline _cell_named_by(_grid, I::Tuple{Integer}, ::FlatCells) = Int(@inbounds I[1])

_cell_named_by(grid, I, ::FlatCells) = throw(ArgumentError(
    "a cell of $(nameof(typeof(grid))) is named by one integer; got $(length(I))",
))

"""
    _cell_from_linear(grid, lin) -> cell

The cell a linear index names: the inverse of the linear index a traversal reports, and what turns a
spatial index's answer back into a cell.
"""
@inline _cell_from_linear(grid::AbstractGrid, lin::Integer) =
    _cell_from_linear(grid, Int(lin), cell_address(grid))
@inline _cell_from_linear(grid, lin::Int, ::CartesianCells) =
    Tuple(@inbounds CartesianIndices(size_tuple(grid))[lin])
@inline _cell_from_linear(_grid, lin::Int, ::FlatCells) = lin

# `Val(length(p0))`: `_raw_coords` returns an `NTuple` whose length is in its type, and that length is
# the coordinate count on every architecture — the grid rank where the two coincide, and the node set's
# coordinate count where they do not.
@inline function _min_image_pair(grid::AbstractGrid, I, J)
    p0 = _cell_coords(grid, I)
    q = _min_image(p0, _cell_coords(grid, J), _wrap_lengths(grid, Val(length(p0))))
    return p0, q
end

@inline function displacement(grid::AbstractGrid, I, J)
    p0, q = _min_image_pair(grid, I, J)
    return ntuple(d -> q[d] - p0[d], Val(length(p0)))
end

"""
    Geometry.distance(grid, I, J) -> T
    Geometry.distance(grid::UnstructuredGrid, i, j) -> T

Distance between the centres of two cells under the grid's own geometry and topology: the coordinates
are resolved from the indices, reduced to the nearest image in every periodic direction, and handed to
the point form of [`Geometry.distance`](@ref).

Across a periodic seam this is the short way round: one spacing between the first and last cell of a
periodic direction, where the point form on the raw coordinates gives the full extent. A bounded
direction contributes its plain coordinate difference. See [`displacement`](@ref) for the offset it was
taken from.
"""
@inline function Geometry.distance(grid::AbstractGrid, I, J)
    p0, q = _min_image_pair(grid, I, J)
    return Geometry.distance(grid_geometry(grid), p0, q)
end

# `CartesianIndex` is what `CartesianIndices` hands a traversal, so accept it directly.
@inline Geometry.distance(grid::AbstractGrid, I::CartesianIndex, J::CartesianIndex) =
    Geometry.distance(grid, Tuple(I), Tuple(J))
@inline displacement(grid::AbstractGrid, I::CartesianIndex, J::CartesianIndex) =
    displacement(grid, Tuple(I), Tuple(J))
