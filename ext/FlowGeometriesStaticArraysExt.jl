module FlowGeometriesStaticArraysExt

using StaticArrays: StaticArrays as SA
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# Static-vector points and returns, in vector form end to end. Measured against the generic path
# over 10⁶ points: `distance` is a wash (0.96–1.02×), `project_to_tangent_plane` is 1.36×.

@inline _dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

@inline Geometry.as_ntuple(p::SA.StaticVector{N,T}) where {N,T} = Tuple(p)::NTuple{N,T}

@inline Geometry.build_point(::Type{S}, ::NTuple{N,Symbol}, vals::NTuple{N,Any}) where {N,S<:SA.StaticVector} =
    S(vals)

# An unparameterized `SVector`/`MVector` request still infers a concrete `SVector{N,T}`.
@inline Grids.coords(::Type{SA.SVector}, grid::Grids.AbstractGrid, I::Vararg{Integer}) =
    SA.SVector(Grids._raw_coords(grid, I...))

@inline Grids.coords(::Type{SA.MVector}, grid::Grids.AbstractGrid, I::Vararg{Integer}) =
    SA.MVector(Grids._raw_coords(grid, I...))

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

@inline function Geometry.distance(
    ::Geometry.AbstractCartesianGeometry{T}, p1::SA.StaticVector{N}, p2::SA.StaticVector{N},
) where {T,N}
    d = SA.SVector{N,T}(p1) - SA.SVector{N,T}(p2)
    return sqrt(sum(abs2, d))
end

@inline function Geometry.distance(
    geo::Geometry.AbstractSphericalGeometry{T}, p1::SA.StaticVector{2}, p2::SA.StaticVector{2},
) where {T}
    λ1, φ1 = T(p1[1]), T(p1[2])
    λ2, φ2 = T(p2[1]), T(p2[2])
    a = sin((φ2 - φ1) / T(2))^2 + cos(φ1) * cos(φ2) * sin((λ2 - λ1) / T(2))^2
    return geo.R * T(2) * atan(sqrt(a), sqrt(max(zero(T), one(T) - a)))
end

@inline function Geometry.distance(
    geo::Geometry.AbstractSphericalGeometry{T}, p1::SA.StaticVector{3}, p2::SA.StaticVector{3},
) where {T}
    d = Geometry.spherical_to_cartesian(SA.SVector{3,T}, geo, p1) -
        Geometry.spherical_to_cartesian(SA.SVector{3,T}, geo, p2)
    return sqrt(sum(abs2, d))
end

# ---------------------------------------------------------------------------
# Spherical ↔ Cartesian, kept in vector form
# ---------------------------------------------------------------------------

@inline function Geometry.spherical_to_cartesian(
    ::Type{S}, geo::Geometry.AbstractSphericalGeometry{T}, p::SA.StaticVector{2},
) where {T<:AbstractFloat, S<:SA.StaticVector}
    sinλ, cosλ = sincos(T(p[1]))
    sinφ, cosφ = sincos(T(p[2]))
    R = geo.R
    return S((R * cosφ * cosλ, R * cosφ * sinλ, R * sinφ))
end

@inline function Geometry.spherical_to_cartesian(
    ::Type{S}, ::Geometry.AbstractSphericalGeometry{T}, p::SA.StaticVector{3},
) where {T<:AbstractFloat, S<:SA.StaticVector}
    sinλ, cosλ = sincos(T(p[1]))
    sinφ, cosφ = sincos(T(p[2]))
    r = T(p[3])
    return S((r * cosφ * cosλ, r * cosφ * sinλ, r * sinφ))
end

@inline function Geometry.cartesian_to_spherical(
    ::Type{S}, ::Geometry.AbstractSphericalGeometry{T}, v::SA.StaticVector{3},
) where {T<:AbstractFloat, S<:SA.StaticVector}
    x, y, z = T(v[1]), T(v[2]), T(v[3])
    r = sqrt(x * x + y * y + z * z)
    iszero(r) && return S((zero(T), zero(T), r))
    return S((atan(y, x), asin(clamp(z / r, -one(T), one(T))), r))
end

# ---------------------------------------------------------------------------
# Tangent-plane geometry
# ---------------------------------------------------------------------------

@inline function Geometry.local_tangent_basis(
    ::Type{S}, ::Geometry.AbstractCartesianGeometry{T}, ::SA.StaticVector{2},
) where {T<:AbstractFloat, S<:SA.StaticVector}
    return (; x = S((one(T), zero(T))), y = S((zero(T), one(T))))
end

@inline function Geometry.local_tangent_basis(
    ::Type{S}, ::Geometry.AbstractSphericalGeometry{T}, p::SA.StaticVector{2},
) where {T<:AbstractFloat, S<:SA.StaticVector}
    sinλ, cosλ = sincos(T(p[1]))
    sinφ, cosφ = sincos(T(p[2]))
    return (; λ = S((-sinλ, cosλ, zero(T))), φ = S((-sinφ * cosλ, -sinφ * sinλ, cosφ)))
end

@inline function Geometry.project_to_tangent_plane(
    ::Type{S}, ::Geometry.AbstractCartesianGeometry{T},
    center::SA.StaticVector{2}, neighbor::SA.StaticVector{2},
) where {T<:AbstractFloat, S<:SA.StaticVector}
    return S(SA.SVector{2,T}(neighbor) - SA.SVector{2,T}(center))
end

@inline function Geometry.project_to_tangent_plane(
    ::Type{S}, geo::Geometry.AbstractSphericalGeometry{T},
    center::SA.StaticVector{2}, neighbor::SA.StaticVector{2},
) where {T<:AbstractFloat, S<:SA.StaticVector}
    Pc = Geometry.spherical_to_cartesian(SA.SVector{3,T}, geo, center)
    Pn = Geometry.spherical_to_cartesian(SA.SVector{3,T}, geo, neighbor)
    chord = Pn - Pc
    ê = Geometry.local_tangent_basis(SA.SVector{3,T}, geo, center)
    return S((_dot3(chord, ê.λ), _dot3(chord, ê.φ)))
end

end # module
