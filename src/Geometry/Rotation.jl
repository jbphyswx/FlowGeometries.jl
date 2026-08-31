# ---------------------------------------------------------------------------
# Pole rotation
# ---------------------------------------------------------------------------

"""
    PoleRotation(λp, φp)

The frame whose north pole sits at `(λp, φp)` of the original one — a rotated-pole grid's coordinate
change. Apply it with [`rotate`](@ref) and undo it with [`unrotate`](@ref).

The tilt's sine and cosine are part of the rotation: `θ = φp − π/2` is a property of the frame, so it
is resolved once here and read by every point carried through it.

`rotate` and `unrotate` work at the width of the points given them, carrying the rotation there;
`T` is the width the frame itself is stored and carried from. [`similar_rotation`](@ref) moves a
rotation between widths explicitly.
"""
struct PoleRotation{T<:AbstractFloat}
    λp::T
    φp::T
    sinθ::T      # θ = φp − π/2, the tilt about y
    cosθ::T
end

function PoleRotation{T}(λp::Real, φp::Real) where {T<:AbstractFloat}
    lp = convert(T, λp)
    fp = convert(T, φp)
    sinθ, cosθ = sincos(fp - T(π) / 2)
    return PoleRotation{T}(lp, fp, sinθ, cosθ)
end

PoleRotation(λp::Real, φp::Real) =
    PoleRotation{float(promote_type(typeof(λp), typeof(φp)))}(λp, φp)

"""
    similar_rotation(T, rot) -> PoleRotation{T}

`rot` at element width `T`, the counterpart of [`similar_geometry`](@ref) for a frame. The tilt is
re-resolved at the new width from `(λp, φp)`.
"""
@inline similar_rotation(::Type{T}, rot::PoleRotation{T}) where {T<:AbstractFloat} = rot
@inline similar_rotation(::Type{T}, rot::PoleRotation) where {T<:AbstractFloat} =
    PoleRotation{T}(rot.λp, rot.φp)

# The width a pair of coordinates is worked at, and the frame carried to it. `similar_rotation` is the
# identity when the widths already agree, which is every in-package call: a `RotatedGrid` carries its
# rotation to the grid's `T` at construction.
@inline _point_width(λ, φ) = float(promote_type(typeof(λ), typeof(φ)))
@inline _at_point_width(rot::PoleRotation, λ, φ) =
    (W = _point_width(λ, φ); (similar_rotation(W, rot), W(λ), W(φ)))

"""
    rotate(rot, λ, φ) -> (λ′, φ′)

`(λ, φ)` expressed in the rotated frame. The rotation's own pole maps to `φ′ = π/2`.
"""
@inline rotate(rot::PoleRotation, λ::Real, φ::Real) = _rotate(_at_point_width(rot, λ, φ)...)

function _rotate(rot::PoleRotation{T}, λ::T, φ::T) where {T}
    sinλ, cosλ = sincos(λ - rot.λp)
    sinφ, cosφ = sincos(φ)
    # about z by -λp, then about y by φp - π/2
    x = cosφ * cosλ
    y = cosφ * sinλ
    z = sinφ
    xr = rot.cosθ * x + rot.sinθ * z
    zr = -rot.sinθ * x + rot.cosθ * z
    return (mod(atan(y, xr), T(2π)), asin(clamp(zr, -one(T), one(T))))
end

"""
    unrotate(rot, λ′, φ′) -> (λ, φ)

Inverse of [`rotate`](@ref).
"""
@inline unrotate(rot::PoleRotation, λ::Real, φ::Real) = _unrotate(_at_point_width(rot, λ, φ)...)

function _unrotate(rot::PoleRotation{T}, λ::T, φ::T) where {T}
    sinλ, cosλ = sincos(λ)
    sinφ, cosφ = sincos(φ)
    x = cosφ * cosλ
    y = cosφ * sinλ
    z = sinφ
    xr = rot.cosθ * x - rot.sinθ * z
    zr = rot.sinθ * x + rot.cosθ * z
    return (mod(atan(y, xr) + rot.λp, T(2π)), asin(clamp(zr, -one(T), one(T))))
end

"""
    rotate!(λ, φ, rot) -> (λ, φ)
    unrotate!(λ, φ, rot) -> (λ, φ)

Rotate a whole point set in place — the form a sampling's `spherical_points` output takes. `λ` and `φ`
are any arrays of matching shape, so this covers a scattered node set and a grid's 2-D coordinate
fields alike. Allocates nothing.
"""
function rotate! end
function unrotate! end

# The arrays' own width, reached by carrying the rotation to it once outside the loop: a Float32 point
# set is worked and returned at Float32. `similar_rotation` is the explicit way to ask for another.
for (f!, f, k) in ((:rotate!, :rotate, :_rotate), (:unrotate!, :unrotate, :_unrotate))
    @eval function $f!(λ::AbstractArray, φ::AbstractArray, rot::PoleRotation)
        axes(λ) == axes(φ) || throw(DimensionMismatch(
            "λ has axes $(axes(λ)) but φ has $(axes(φ))",
        ))
        W = float(promote_type(eltype(λ), eltype(φ)))
        r = similar_rotation(W, rot)
        @inbounds for i in eachindex(λ, φ)
            λ[i], φ[i] = $k(r, W(λ[i]), W(φ[i]))
        end
        return (λ, φ)
    end

    @eval function $f(rot::PoleRotation, λ::AbstractArray, φ::AbstractArray)
        axes(λ) == axes(φ) || throw(DimensionMismatch(
            "λ has axes $(axes(λ)) but φ has $(axes(φ))",
        ))
        W = float(promote_type(eltype(λ), eltype(φ)))
        r = similar_rotation(W, rot)
        out_λ = similar(λ, W)
        out_φ = similar(φ, W)
        @inbounds for i in eachindex(λ, φ)
            out_λ[i], out_φ[i] = $k(r, W(λ[i]), W(φ[i]))
        end
        return (out_λ, out_φ)
    end
end
