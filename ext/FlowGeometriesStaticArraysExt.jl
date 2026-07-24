module FlowGeometriesStaticArraysExt

using StaticArrays: StaticArrays as SA
using FlowGeometries.Geometry: Geometry

# Accept `SVector` points by converting to the core `NTuple` API; results stay `NTuple`
# (call `SA.SVector(result)` if you want a static vector back).

@inline Geometry.distance(geo::Geometry.AbstractCartesianGeometry, p1::SA.SVector{N,T}, p2::SA.SVector{N,T}) where {N,T} =
    Geometry.distance(geo, Tuple(p1)::NTuple{N,T}, Tuple(p2)::NTuple{N,T})

@inline Geometry.distance(geo::Geometry.AbstractSphericalGeometry, p1::SA.SVector{2,T}, p2::SA.SVector{2,T}) where {T} =
    Geometry.distance(geo, Tuple(p1)::NTuple{2,T}, Tuple(p2)::NTuple{2,T})

@inline Geometry.distance(geo::Geometry.AbstractSphericalGeometry, p1::SA.SVector{3,T}, p2::SA.SVector{3,T}) where {T} =
    Geometry.distance(geo, Tuple(p1)::NTuple{3,T}, Tuple(p2)::NTuple{3,T})

@inline Geometry.local_tangent_basis(geo::Geometry.AbstractCartesianGeometry, coords::SA.SVector{2,T}) where {T} =
    Geometry.local_tangent_basis(geo, Tuple(coords)::NTuple{2,T})

@inline Geometry.local_tangent_basis(geo::Geometry.AbstractSphericalGeometry, coords::SA.SVector{2,T}) where {T} =
    Geometry.local_tangent_basis(geo, Tuple(coords)::NTuple{2,T})

@inline Geometry.project_to_tangent_plane(
    geo::Geometry.AbstractCartesianGeometry, center::SA.SVector{2,T}, neighbor::SA.SVector{2,T},
) where {T} =
    Geometry.project_to_tangent_plane(geo, Tuple(center)::NTuple{2,T}, Tuple(neighbor)::NTuple{2,T})

@inline Geometry.project_to_tangent_plane(
    geo::Geometry.AbstractSphericalGeometry, center::SA.SVector{2,T}, neighbor::SA.SVector{2,T},
) where {T} =
    Geometry.project_to_tangent_plane(geo, Tuple(center)::NTuple{2,T}, Tuple(neighbor)::NTuple{2,T})

end # module
