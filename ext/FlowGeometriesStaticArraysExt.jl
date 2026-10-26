module FlowGeometriesStaticArraysExt

using StaticArrays: StaticArrays as SA
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# Static vectors in and out: a point enters every geometry function as its tuple, and a leading
# `S <: StaticVector` builds the result as one.

@inline Geometry.as_ntuple(p::SA.StaticVector{N,T}) where {N,T} = Tuple(p)::NTuple{N,T}

@inline Geometry.build_point(::Type{S}, ::NTuple{N,Symbol}, vals::NTuple{N,Any}) where {N,S<:SA.StaticVector} =
    S(vals)

# An unparameterized `SVector`/`MVector` request still infers a concrete `SVector{N,T}`.
@inline Grids.coords(::Type{SA.SVector}, grid::Grids.AbstractGrid, I::Vararg{Integer,K}) where {K} =
    SA.SVector(Grids._raw_coords(grid, I...))

@inline Grids.coords(::Type{SA.MVector}, grid::Grids.AbstractGrid, I::Vararg{Integer,K}) where {K} =
    SA.MVector(Grids._raw_coords(grid, I...))

end # module
