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
