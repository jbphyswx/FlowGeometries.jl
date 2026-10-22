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
Per-component minimum image is the global minimum for a separable metric, which the Euclidean one is.
"""
@inline _min_image(p0::NTuple{N,Any}, pt::NTuple{N,Any}, prd::NTuple{N,Any}) where {N} =
    map((a, b, p) -> p > zero(p) ? a + _image_offset(a, b, p) : b, p0, pt, prd)

"""
    _image_offset(a, b, P) -> T

`b − a` reduced to the image nearest `a` for wrap length `P > 0`. The raw difference is held exactly as
its rounded value and the rounding error (Knuth's TwoSum), and a whole number of wraps is removed from
the rounded value alone, so a pair either side of the seam keeps the precision of its gap.
"""
@inline function _image_offset(a::T, b::T, P::T) where {T<:AbstractFloat}
    s = b - a
    z = s - b
    e = (b - (s - z)) + (-a - z)
    return (s - round(s / P) * P) + e
end

@inline _image_offset(a::Real, b::Real, P::Real) = _image_offset(promote(float(a), b, P)...)

# Each component's offset `b − a`, reduced where its direction wraps.
@inline _offsets(p0::Tuple, pt::Tuple, prd::Tuple) =
    map((a, b, p) -> p > zero(p) ? _image_offset(a, b, p) : b - a, p0, pt, prd)

"""
    _image_pair(geo, p0, pt, prd) -> (a, b)

Two points whose distance under `geo` is the distance from `p0` to the image of `pt` nearest it, `prd`
being each direction's wrap length (zero where it does not wrap).

On the plane the distance depends on the offset alone, which is returned against the origin, each
component formed by [`_image_offset`](@ref). On a sphere or a spheroid the geometry reduces a longitude
difference by an exact `2π` itself, so the longitude is left as stored.
"""
@inline function _image_pair(::Geometry.AbstractCartesianGeometry, p0::Tuple, pt::Tuple, prd::Tuple)
    Δ = _offsets(p0, pt, prd)
    return (map(zero, Δ), Δ)
end

@inline _image_pair(::Geometry.AbstractLonLatGeometry, p0::Tuple, pt::Tuple, prd::Tuple) =
    (p0, (first(pt), _min_image(Base.tail(p0), Base.tail(pt), Base.tail(prd))...))

@inline _image_pair(::Geometry.AbstractGeometry, p0::Tuple, pt::Tuple, prd::Tuple) =
    (p0, _min_image(p0, pt, prd))

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
@inline function _cell_image_pair(grid::AbstractGrid, I, J)
    p0 = _cell_coords(grid, I)
    return _image_pair(grid_geometry(grid), p0, _cell_coords(grid, J), _wrap_lengths(grid, Val(length(p0))))
end

@inline function displacement(grid::AbstractGrid, I, J)
    p0 = _cell_coords(grid, I)
    return _displacement(grid_geometry(grid), p0, _cell_coords(grid, J), _wrap_lengths(grid, Val(length(p0))))
end

@inline _displacement(::Geometry.AbstractGeometry, p0::Tuple, pt::Tuple, prd::Tuple) = _offsets(p0, pt, prd)

# A wrapping longitude is reduced by an exact `2π`, as the geometry's own distance reduces it.
@inline function _displacement(::Geometry.AbstractLonLatGeometry, p0::Tuple, pt::Tuple, prd::Tuple)
    λ = first(prd) > zero(first(prd)) ? Geometry._lon_diff(first(p0), first(pt)) : first(pt) - first(p0)
    return (λ, _offsets(Base.tail(p0), Base.tail(pt), Base.tail(prd))...)
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
@inline Geometry.distance(grid::AbstractGrid, I, J) =
    Geometry.distance(grid_geometry(grid), _cell_image_pair(grid, I, J)...)

"""
    Geometry.normal_coordinates(grid, I, J) -> NamedTuple

[`Geometry.normal_coordinates`](@ref) of cell `J` about cell `I`, the pair reduced to the nearest image
in every periodic direction as for [`Geometry.distance`](@ref)`(grid, I, J)`, whose value is their
length.
"""
@inline Geometry.normal_coordinates(grid::AbstractGrid, I, J) =
    Geometry.normal_coordinates(grid_geometry(grid), _cell_image_pair(grid, I, J)...)

# `CartesianIndex` is what `CartesianIndices` hands a traversal, so accept it directly.
@inline Geometry.distance(grid::AbstractGrid, I::CartesianIndex, J::CartesianIndex) =
    Geometry.distance(grid, Tuple(I), Tuple(J))
@inline displacement(grid::AbstractGrid, I::CartesianIndex, J::CartesianIndex) =
    displacement(grid, Tuple(I), Tuple(J))
@inline Geometry.normal_coordinates(grid::AbstractGrid, I::CartesianIndex, J::CartesianIndex) =
    Geometry.normal_coordinates(grid, Tuple(I), Tuple(J))
