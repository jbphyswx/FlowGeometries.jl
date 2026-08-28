# ---------------------------------------------------------------------------
# Pole rotation
# ---------------------------------------------------------------------------

"""
    PoleRotation(λp, φp)

The frame whose north pole sits at `(λp, φp)` of the original one — a rotated-pole grid's coordinate
change. Apply it with [`rotate`](@ref) and undo it with [`unrotate`](@ref).

The tilt's sine and cosine are part of the rotation: `θ = φp − π/2` is a property of the frame, not of
the point being carried through it, so it is resolved once here and read by every point.

`T` fixes the width of a scalar `rotate`/`unrotate`. The array forms work at their arrays' own width;
[`similar_rotation`](@ref) is how to move a rotation between widths explicitly.
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
re-resolved at the new width rather than rounded from the old one.
"""
@inline similar_rotation(::Type{T}, rot::PoleRotation{T}) where {T<:AbstractFloat} = rot
@inline similar_rotation(::Type{T}, rot::PoleRotation) where {T<:AbstractFloat} =
    PoleRotation{T}(rot.λp, rot.φp)

"""
    rotate(rot, λ, φ) -> (λ′, φ′)

`(λ, φ)` expressed in the rotated frame. The rotation's own pole maps to `φ′ = π/2`.
"""
function rotate(rot::PoleRotation{T}, λ::Real, φ::Real) where {T}
    sinλ, cosλ = sincos(convert(T, λ) - rot.λp)
    sinφ, cosφ = sincos(convert(T, φ))
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
function unrotate(rot::PoleRotation{T}, λ::Real, φ::Real) where {T}
    sinλ, cosλ = sincos(convert(T, λ))
    sinφ, cosφ = sincos(convert(T, φ))
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

for (f!, f) in ((:rotate!, :rotate), (:unrotate!, :unrotate))
    @eval function $f!(λ::AbstractArray, φ::AbstractArray, rot::PoleRotation)
        axes(λ) == axes(φ) || throw(DimensionMismatch(
            "λ has axes $(axes(λ)) but φ has $(axes(φ))",
        ))
        @inbounds for i in eachindex(λ, φ)
            λ[i], φ[i] = $f(rot, λ[i], φ[i])
        end
        return (λ, φ)
    end

    # The arrays' own width, reached by carrying the rotation to it once: a Float32 point set comes
    # back Float32. `similar_rotation` is the explicit way to ask for another.
    @eval function $f(rot::PoleRotation, λ::AbstractArray, φ::AbstractArray)
        axes(λ) == axes(φ) || throw(DimensionMismatch(
            "λ has axes $(axes(λ)) but φ has $(axes(φ))",
        ))
        W = float(promote_type(eltype(λ), eltype(φ)))
        r = similar_rotation(W, rot)
        out_λ = similar(λ, W)
        out_φ = similar(φ, W)
        @inbounds for i in eachindex(λ, φ)
            out_λ[i], out_φ[i] = $f(r, λ[i], φ[i])
        end
        return (out_λ, out_φ)
    end
end
