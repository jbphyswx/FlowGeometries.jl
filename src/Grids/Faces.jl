# ---------------------------------------------------------------------------
# Faces and corners of a grid whose cells meet across directional faces
# ---------------------------------------------------------------------------
#
# Face `F` along direction `d` separates cell `F − e_d` from cell `F`. A direction of `n` cells carries `n`
# faces where it wraps, the face below cell 1 being the face above cell `n`, and `n + 1` where it ends, so
# each direction has its own face array. A corner of the pair `(a, b)` sits where an `a`-face meets a
# `b`-face; a bounded direction carries its `n − 1` interior corners, the two outermost having no closed
# loop of cells around them, and a wrapping one all `n`.

"""
    FaceIndexedGrid{T,G,N}

The grids whose cells meet across directional faces, face `F` along direction `d` lying between cells
`F − e_d` and `F`: [`StructuredGrid`](@ref) and [`CurvilinearGrid`](@ref).
"""
const FaceIndexedGrid{T,G,N} = Union{StructuredGrid{T,G,N},CurvilinearGrid{T,G,N}}

"""
    nfaces(grid, d) -> Int

The faces normal to direction `d`: `n` where the direction wraps, `n + 1` where it ends.
"""
@inline nfaces(grid::FaceIndexedGrid, d::Integer) =
    isperiodic(grid, d) ? @inbounds(size_tuple(grid)[d]) : @inbounds(size_tuple(grid)[d]) + 1

"""
    face_dims(grid, d) -> NTuple{N,Int}

The shape of the face array normal to direction `d`: [`nfaces`](@ref) along `d`, the cell count along
every other direction.
"""
@inline face_dims(grid::FaceIndexedGrid{T,G,N}, d::Integer) where {T,G,N} =
    ntuple(e -> e == d ? nfaces(grid, e) : @inbounds(size_tuple(grid)[e]), Val(N))

"""
    ncorners(grid, d) -> Int

The corners along direction `d` of a pair staggered in it: `n` where the direction wraps, `n − 1`, the
interior ones, where it ends.
"""
@inline ncorners(grid::FaceIndexedGrid, d::Integer) = isperiodic(grid, d) ? nfaces(grid, d) : nfaces(grid, d) - 2

"""
    corner_offset(grid, d) -> Int

What a corner index along `d` adds to become the index of the face it sits on: `1` where the direction
ends and its outermost corner was dropped, `0` where it wraps.
"""
@inline corner_offset(grid::FaceIndexedGrid, d::Integer) = isperiodic(grid, d) ? 0 : 1

"""
    corner_dims(grid, a, b) -> NTuple{N,Int}

The shape of an array of `(a, b)` corners: [`ncorners`](@ref) along `a` and `b`, the cell count along
every other direction.
"""
@inline corner_dims(grid::FaceIndexedGrid{T,G,N}, a::Integer, b::Integer) where {T,G,N} =
    ntuple(e -> (e == a || e == b) ? ncorners(grid, e) : @inbounds(size_tuple(grid)[e]), Val(N))

"""
    corner_to_face(grid, Q, a, b) -> CartesianIndex
    face_to_corner(grid, F, a, b) -> CartesianIndex

Between a corner index of the pair `(a, b)` and the index of the face it sits on, along `a` and `b`. A
face index can name a dropped corner, outside [`corner_dims`](@ref).
"""
@inline corner_to_face(grid::FaceIndexedGrid, Q::CartesianIndex{N}, a::Integer, b::Integer) where {N} =
    CartesianIndex(ntuple(e -> @inbounds(Q[e]) + (e == a ? corner_offset(grid, a) :
                                                  e == b ? corner_offset(grid, b) : 0), Val(N)))

@inline face_to_corner(grid::FaceIndexedGrid, F::CartesianIndex{N}, a::Integer, b::Integer) where {N} =
    CartesianIndex(ntuple(e -> @inbounds(F[e]) - (e == a ? corner_offset(grid, a) :
                                                  e == b ? corner_offset(grid, b) : 0), Val(N)))

@inline _unit_index(::Val{N}, d::Integer) where {N} = CartesianIndex(ntuple(i -> i == d ? 1 : 0, Val(N)))

"""
    cell_below(grid, F, d) -> CartesianIndex or nothing
    cell_above(grid, F, d) -> CartesianIndex or nothing

The cell on the low and on the high side of face `F` along `d`; `nothing` past the end of a bounded
direction. Every face of a wrapping direction has both.
"""
@inline function cell_below(grid::FaceIndexedGrid, F::CartesianIndex{N}, d::Integer) where {N}
    @inbounds F[d] > 1 && return F - _unit_index(Val(N), d)
    isperiodic(grid, d) || return nothing
    return CartesianIndex(ntuple(j -> j == d ? @inbounds(size_tuple(grid)[d]) : @inbounds(F[j]), Val(N)))
end

@inline cell_above(grid::FaceIndexedGrid, F::CartesianIndex, d::Integer) =
    @inbounds(F[d]) ≤ @inbounds(size_tuple(grid)[d]) ? F : nothing

"""
    face_below(I, d) -> CartesianIndex
    face_above(grid, I, d) -> CartesianIndex

The two faces of cell `I` normal to `d`, as indices into face array `d`. Along a wrapping direction the
face above the last cell is face 1.
"""
@inline face_below(I::CartesianIndex, ::Integer) = I

@inline function face_above(grid::FaceIndexedGrid, I::CartesianIndex{N}, d::Integer) where {N}
    isperiodic(grid, d) && @inbounds(I[d]) == @inbounds(size_tuple(grid)[d]) &&
        return CartesianIndex(ntuple(j -> j == d ? 1 : @inbounds(I[j]), Val(N)))
    return I + _unit_index(Val(N), d)
end

"""
    face_coordinates(grid, d) -> AbstractVector

The coordinates of the faces normal to direction `d` of a rectilinear grid, [`nfaces`](@ref) of them:
midway between neighbouring centres, the first of a wrapping direction midway across its seam, the two
outermost of a bounded one half a cell beyond its end centres. The face axis of an
[`Axes.PeriodicAxis`](@ref) is another and that of a uniform axis is uniform; they are the face axes of
[`StaggeredGrid`](@ref). With `d` a `Val`, the axis's type is known where it is called.
"""
@inline face_coordinates(grid::StructuredGrid, d::Integer) =
    _face_axis(coordinates(grid, d), topology(grid)[d], period(grid, d))
@inline face_coordinates(grid::StructuredGrid, ::Val{d}) where {d} =
    _face_axis(coordinates(grid)[d], topology(grid)[d], period(grid, d))

"""
    corner_coordinates(grid, d) -> AbstractVector

The coordinates along direction `d` of the corners of a pair staggered in `d`: the
[`face_coordinates`](@ref) of a wrapping direction, and the interior ones of a bounded direction.
"""
@inline corner_coordinates(grid::StructuredGrid, d::Integer) =
    _corner_axis(face_coordinates(grid, d), topology(grid)[d])
@inline corner_coordinates(grid::StructuredGrid, ::Val{d}) where {d} =
    _corner_axis(face_coordinates(grid, Val(d)), topology(grid)[d])

@inline _corner_axis(f::AbstractVector, ::Periodic) = f
@inline _corner_axis(f::AbstractVector, ::Bounded) = f[(firstindex(f) + 1):(lastindex(f) - 1)]
