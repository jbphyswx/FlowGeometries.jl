# ---------------------------------------------------------------------------
# The physical derivative
# ---------------------------------------------------------------------------

"""
    derivative!(out, field, grid, dim; order=1, nodes=order+1, policy=BlankMasked(),
                masked=zero, active_only=true, backend=nothing) -> out

The derivative with respect to **distance** along direction `dim`, rather than with respect to the
coordinate: [`apply_stencil!`](@ref) divided by the metric factor,

    ∂f/∂sᵈ = (1/hᵈ) · ∂f/∂ξᵈ,    hᵈ = Geometry.scale_factors(geo, p)[d]

On a Cartesian metric every `hᵈ` is `1` and this is `apply_stencil!` exactly, at no cost. Anywhere else
it is the derivative a physical law is written in, and assembling it from the parts was the one thing
every geometry-aware caller had to add — including the coordinate singularity below, which is not
theirs to get right.

**Where the metric degenerates the derivative does not exist**, and `masked` is written rather than a
number invented. Longitude at a pole is the case: `h_λ = R cos φ → 0`, so `1/h_λ` diverges. The test is
relative to the geometry's own size and to the precision, `|h| ≤ L·√eps(T)` — an absolute threshold
cannot be right for both, since `1e-12` is below `eps(Float32)` and in `Float32`
`cos(Float32(π/2)) ≈ -4.4e-8`, so a pole row would quietly receive a large finite number instead.

No scale factor in this package depends on **longitude**, so `hᵈ` is constant along the first axis
whichever direction is differenced. The scaling is applied once per remaining index and swept along
that contiguous axis. (It is *not* generally constant along the differenced direction — on a spheroid
`h_φ = M(φ)` varies with `φ` — so it is not hoisted that way.)

A divergence or a curl is still the caller's to assemble, needing a result location and a boundary
policy this does not choose. Note the flux form when doing so: on a sphere

    ∇·u = (1/(R cos φ)) [ ∂u_λ/∂λ + ∂(u_φ cos φ)/∂φ ]

so the second term differentiates `u_φ cos φ`, not `u_φ`; taking two physical derivatives and adding
them is a different, wrong expression.
"""
function derivative! end

"""
    metric_floor(geometry) -> T

The magnitude below which a scale factor is treated as degenerate: `L·√eps(T)` for a curved geometry of
size `L`, and `0` for a Cartesian one, whose metric never degenerates.

Relative to both the geometry's size and the element type on purpose — see [`derivative!`](@ref) for
what an absolute constant does in `Float32`.
"""
function metric_floor end

"""
    GradientPlan{T}

The geometry of a least-squares gradient, separated from any field: for each cell, the coefficient on
each neighbour's *difference* from it. Built by `Connectivity.gradient_plan`, applied by
[`gradient!`](@ref).

`apply_stencil!` covers the separable case. A `CurvilinearGrid` has no separable axis to build a scalar
stencil along, and a node set's neighbours come from connectivity rather than an index offset, so
neither has a gradient without this.

The construction, with tangent-plane displacements `Δrₖ` from
[`Geometry.project_to_tangent_plane`](@ref), differences `Δfₖ = fₖ - f₀` and weights `wₖ = 1/|Δrₖ|²`:
minimising `Σₖ wₖ (∇f·Δrₖ - Δfₖ)²` gives `A ∇f = b` with `A = Σₖ wₖ Δrₖ⊗Δrₖ` and
`b = Σₖ wₖ Δrₖ Δfₖ`. `A` depends only on the geometry, so it is inverted once here and the per-neighbour
coefficients `A⁻¹ wₖ Δrₖ` are what the plan stores; each apply is then one dot product per cell and
allocates nothing.

What this buys over inverting an index-space Jacobian:

- **exact for a linear field on any stencil**, however skewed — the least-squares combination cancels
  the leading truncation term, which a 2×2 Jacobian inverse does not;
- second order on locally symmetric stencils, degrading toward first on strongly skewed cells;
- it reduces to the centred difference where the stencil is separable and orthogonal (`A` diagonal), so
  it agrees with `apply_stencil!` where both apply.

Where `A` is rank deficient — a tangent direction with no data, at a boundary or beside a mask — that
component is **zeroed rather than invented**, by the pseudo-inverse. Same rule `apply_stencil!` states
for a mask: not determined by the active data, so not produced.
"""
struct GradientPlan{T,VI<:AbstractVector{Int},VT<:AbstractVector{T}}
    ptr::VI          # CSR row pointers into `nbr`/`c1`/`c2`, length n+1
    nbr::VI          # neighbour cell, as a linear index
    c1::VT           # coefficient on this neighbour's difference, first tangent component
    c2::VT           # …and second
    names::NTuple{2,Symbol}   # what those two components are, from the geometry
end

Base.length(p::GradientPlan) = length(p.ptr) - 1

function Base.show(io::IO, p::GradientPlan{T}) where {T}
    return print(io, "GradientPlan{", T, "}(", length(p), " cells, ", length(p.nbr),
                 " neighbour coefficients, ", p.names, ")")
end

"""
    gradient!(g1, g2, field, plan) -> (g1, g2)

Apply a [`GradientPlan`](@ref): the two tangent components of `∇field` at every cell, written into
`g1` and `g2`. One dot product per cell over its neighbours, allocating nothing.

`field`, `g1` and `g2` are indexed linearly, so an `N`-D array of the grid's shape works as is. The
components are named by `plan.names` — `(:λ, :φ)` on a sphere, `(:x, :y)` on a plane — and are per unit
**distance**, the tangent plane being metric already.
"""
function gradient!(
    g1::AbstractArray{S}, g2::AbstractArray{S}, field::AbstractArray, plan::GradientPlan,
) where {S}
    n = length(plan)
    length(g1) == length(g2) == length(field) || throw(DimensionMismatch(
        "field $(length(field)) and outputs $(length(g1))/$(length(g2)) must have the same length",
    ))
    # A field may carry trailing BATCH axes beyond the grid's: the plan is over cells, so batch element
    # `b` is the contiguous linear span `(b-1)*n .+ (1:n)` and one call covers all of them. The plan's
    # coefficients are geometry, not data, so they are reused across the batch rather than rebuilt.
    (length(field) % n == 0) || throw(DimensionMismatch(
        "plan is for $n cells; a field of $(length(field)) is not a whole number of them",
    ))
    nb = length(field) ÷ n
    # Indexed linearly rather than through `vec`, which would build three array wrappers per call —
    # 192 bytes on an operation that otherwise allocates nothing. An `N`-D array of the grid's shape
    # indexes linearly as it stands, and a batched one is that shape repeated.
    @inbounds for b in 0:(nb - 1)
        off = b * n
        for i in 1:n
            f0 = field[off + i]
            s1 = zero(S)
            s2 = zero(S)
            for t in plan.ptr[i]:(plan.ptr[i + 1] - 1)
                df = S(field[off + plan.nbr[t]] - f0)
                s1 += S(plan.c1[t]) * df
                s2 += S(plan.c2[t]) * df
            end
            g1[off + i] = s1
            g2[off + i] = s2
        end
    end
    return (g1, g2)
end

"""
    _sympinv2(a, b, c, tol) -> (p11, p12, p22)

Pseudo-inverse of the symmetric 2×2 `[a b; b c]`, dropping any eigendirection whose eigenvalue is below
`tol`. Closed form: a 2×2 symmetric eigenproblem has one.

Dropping rather than regularising is the point. A stencil that carries no information along some
tangent direction — every neighbour on one line, which happens at a boundary and beside a mask — leaves
`A` singular in that direction, and the gradient there is not determined by the data. Inverting a
nudged matrix would answer anyway, with a number governed by the nudge.
"""
@inline function _sympinv2(a::T, b::T, c::T, tol::T) where {T}
    τ = a + c
    disc = sqrt(max((a - c)^2 + 4b^2, zero(T)))
    λ1 = (τ + disc) / 2                      # ≥ λ2, both ≥ 0 for a Gram matrix
    λ2 = (τ - disc) / 2
    λ1 ≤ tol && return (zero(T), zero(T), zero(T))          # no direction resolved at all
    if λ2 > tol
        δ = a * c - b * b
        return (c / δ, -b / δ, a / δ)                        # full rank: the ordinary inverse
    end
    # Rank one: keep the resolved direction only, `A⁺ = v vᵀ / λ1`.
    v1, v2 = abs(b) > eps(T) * max(abs(a), abs(c), one(T)) ? (λ1 - c, b) :
             (a ≥ c ? (one(T), zero(T)) : (zero(T), one(T)))
    nrm = sqrt(v1 * v1 + v2 * v2)
    nrm ≤ zero(T) && return (zero(T), zero(T), zero(T))
    u1, u2 = v1 / nrm, v2 / nrm
    return (u1 * u1 / λ1, u1 * u2 / λ1, u2 * u2 / λ1)
end

"""
    interpolate(field, grid, p; policy=BlankMasked(), masked=NaN, …) -> value

The value of `field` at the coordinate `p`, which is the question observational data asks: a station, a
float or a ship track has a coordinate, not a cell index.

`interpolation_weights` gives this along **one** axis, and nothing composed them, so a caller had to
build the tensor product themselves on a rectilinear grid and had nothing at all on the others.

- `StructuredGrid` — multilinear, the tensor product of the per-axis weights. A periodic direction
  interpolates *across* its seam rather than clamping at the last sample.
- `CurvilinearGrid`, `UnstructuredGrid` — a weighted least-squares plane fitted to the `k` nearest
  cells in the tangent plane at `p`, which is **exact for a linear field** and reproduces a cell's own
  value at its centre. Falls back to the weighted mean where the fit is rank deficient, that being the
  part of it the data still determines.

`p` may be written any way a point is accepted elsewhere.

The mask policies say what an inactive contributor means, as they do for a stencil:
[`BlankMasked`](@ref) — the default — returns `masked` if any contributor is inactive, and
[`ReduceInRun`](@ref) renormalizes over the active ones. [`ShiftWithinRun`](@ref) has no meaning here,
there being no window to shift, and says so.

A field carrying trailing BATCH axes beyond the grid's own — many tracers, or an ensemble, sharing one
geometry — is evaluated for every element in one call: see [`interpolate!`](@ref) for the form that
writes into a caller's buffer, which this one wraps.
"""
function interpolate end

"""
    interpolate!(out, field, grid, p; …) -> out

[`interpolate`](@ref) for a batched field, writing one value per batch element into `out`.

The bracketing cell — or, off a rectilinear grid, the neighbour set and the least-squares fit — depends
on the point and the geometry and not on the data, so it is solved once and applied to every element.
That is why this is not the same as calling `interpolate` per slice.
"""
function interpolate! end

@inline function _interp_mask_error(policy)
    return throw(ArgumentError(
        "$(policy) has no meaning when interpolating — there is no window to shift. Use " *
        "`ReduceInRun()` to renormalize over the active contributors, or `BlankMasked()`.",
    ))
end

@inline metric_floor(::Geometry.AbstractCartesianGeometry{T}) where {T} = zero(T)
@inline metric_floor(g::Geometry.AbstractSphericalGeometry{T}) where {T} =
    Geometry.radius(g) * sqrt(eps(T))
@inline metric_floor(g::Geometry.AbstractEllipsoidalGeometry{T}) where {T} =
    Geometry.semimajor_axis(g) * sqrt(eps(T))
