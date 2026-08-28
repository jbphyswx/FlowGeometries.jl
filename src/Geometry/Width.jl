# The element type a geometry computes in, and the rebuild that changes it. Generic over every
# geometry, so this comes after all of them are defined.

"""
    float_type(geo) -> Type{<:AbstractFloat}

The element type `geo` computes in. Anything that pairs a geometry with an optional element type
should default to this rather than to `Float64`, or supplying a geometry of one width and no element
type silently rebuilds it at the other.
"""
float_type(::AbstractGeometry{T}) where {T<:AbstractFloat} = T

"""
    similar_geometry(T, geo) -> AbstractGeometry{T}

`geo` with its element type changed to `T`, keeping its shape parameters.

A geometry fixes the width of everything computed against it — coordinates, distances, metric
factors — so building a grid at one element type around a geometry at another silently promotes the
whole grid back. Anything that takes an element type and a geometry together should pass the
geometry through here first.

A geometry already at `T` is returned unchanged, whatever its type, so a geometry defined outside
this package inherits the whole stack without supplying anything. Only an actual change of width
needs a method, and only a geometry that wants to support one has to define it.
"""
function similar_geometry end

similar_geometry(::Type{T}, g::AbstractGeometry{T}) where {T<:AbstractFloat} = g

similar_geometry(::Type{T}, ::CartesianGeometry) where {T<:AbstractFloat} = CartesianGeometry{T}()
similar_geometry(::Type{T}, g::SphericalGeometry) where {T<:AbstractFloat} = SphericalGeometry{T}(g.R)
similar_geometry(::Type{T}, g::SpheroidGeometry) where {T<:AbstractFloat} =
    SpheroidGeometry{T}(g.a, g.f)

# The identity above and the rebuilds above it both match a built-in geometry already at `T`, and
# neither is more specific; these settle it in favour of the identity.
similar_geometry(::Type{T}, g::CartesianGeometry{T}) where {T<:AbstractFloat} = g
similar_geometry(::Type{T}, g::SphericalGeometry{T}) where {T<:AbstractFloat} = g
similar_geometry(::Type{T}, g::SpheroidGeometry{T}) where {T<:AbstractFloat} = g
