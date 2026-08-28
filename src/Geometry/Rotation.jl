# ---------------------------------------------------------------------------
# Pole rotation
# ---------------------------------------------------------------------------

"""
    PoleRotation(λp, φp)

The frame whose north pole sits at `(λp, φp)` of the original one — a rotated-pole grid's coordinate
change. Apply it with [`rotate`](@ref) and undo it with [`unrotate`](@ref).
"""
struct PoleRotation{T<:AbstractFloat}
    λp::T
    φp::T
end

PoleRotation(λp::Real, φp::Real) =
    PoleRotation{float(promote_type(typeof(λp), typeof(φp)))}(λp, φp)

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
    sinθ, cosθ = sincos(rot.φp - T(π) / 2)
    xr = cosθ * x + sinθ * z
    zr = -sinθ * x + cosθ * z
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
    sinθ, cosθ = sincos(rot.φp - T(π) / 2)
    xr = cosθ * x - sinθ * z
    zr = sinθ * x + cosθ * z
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

    @eval function $f(rot::PoleRotation{T}, λ::AbstractArray, φ::AbstractArray) where {T}
        axes(λ) == axes(φ) || throw(DimensionMismatch(
            "λ has axes $(axes(λ)) but φ has $(axes(φ))",
        ))
        out_λ = similar(λ, T)
        out_φ = similar(φ, T)
        @inbounds for i in eachindex(λ, φ)
            out_λ[i], out_φ[i] = $f(rot, λ[i], φ[i])
        end
        return (out_λ, out_φ)
    end
end
