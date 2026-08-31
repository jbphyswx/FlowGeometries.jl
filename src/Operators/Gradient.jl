# A `#` comment between a docstring and its definition drops the docstring, so this sits above.
"""
    GradientPlan

The two index buffers are typed independently and hold any `Integer`, so a plan built off a CSR keeps
that CSR's own width — see `Connectivity._index_type`.

The geometry of a least-squares gradient, separated from any field: for each cell, the coefficient on
each neighbour's *difference* from it, once per coordinate direction. Built by
`Connectivity.gradient_plan`, applied by [`gradient!`](@ref).

`D` is the number of directions the fit resolves, which is the grid's coordinate count: two on a
surface — a `(λ, φ)` or `(x, y)` mesh — and three in a volume, where the neighbourhood spans a solid
angle.

`apply_stencil!` covers the separable case. A `CurvilinearGrid` has no separable axis to build a scalar
stencil along, and a node set's neighbours come from connectivity, so this is the gradient for both.

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

Where `A` is rank deficient — a direction with no data, at a boundary or beside a mask — the
pseudo-inverse **zeroes** that component, under the rule `apply_stencil!` states for a mask: a value
the active data does not determine is not produced.
"""
struct GradientPlan{D,T,VP<:AbstractVector{<:Integer},VI<:AbstractVector{<:Integer},
                    VT<:AbstractVector{T}}
    ptr::VP                   # CSR row pointers into `nbr` and each of `coeffs`, length n+1
    nbr::VI                   # neighbour cell, as a linear index
    coeffs::NTuple{D,VT}      # coefficient on this neighbour's difference, one per direction
    names::NTuple{D,Symbol}   # what those directions are, from the geometry
end

Base.length(p::GradientPlan) = length(p.ptr) - 1

"""
    ncomponents(plan::GradientPlan{D}) -> D

How many directions the plan resolves, and so how many outputs [`gradient!`](@ref) writes.
"""
@inline ncomponents(::GradientPlan{D}) where {D} = D

function Base.show(io::IO, p::GradientPlan{D,T}) where {D,T}
    return print(io, "GradientPlan{", D, ",", T, "}(", length(p), " cells, ", length(p.nbr),
                 " neighbour coefficients, ", p.names, ")")
end

"""
    gradient!(outs::Tuple, field, plan) -> outs
    gradient!(g1, g2, field, plan) -> (g1, g2)
    gradient!(g1, g2, g3, field, plan) -> (g1, g2, g3)

Apply a [`GradientPlan`](@ref): the components of `∇field` at every cell, written one per output array.
One dot product per cell over its neighbours, allocating nothing.

There must be exactly [`ncomponents`](@ref) of them. `field` and the outputs are indexed linearly, so
an `N`-D array of the grid's shape works as is. The components are named by `plan.names` — `(:λ, :φ)`
on a sphere, `(:x, :y)` on a plane, `(:x, :y, :z)` in a volume — and are per unit **distance**, the
displacements being metric already.
"""
function gradient! end

function gradient!(
    outs::Tuple{AbstractArray{S},Vararg{AbstractArray{S}}}, field::AbstractArray,
    plan::GradientPlan{D},
) where {D,S}
    n = length(plan)
    length(outs) == D || throw(DimensionMismatch(
        "the plan resolves $D components; got $(length(outs)) output arrays",
    ))
    all(length(o) == length(field) for o in outs) || throw(DimensionMismatch(
        "field $(length(field)) and outputs $(map(length, outs)) must have the same length",
    ))
    # A field may carry trailing batch axes beyond the grid's: the plan is over cells, so batch element
    # `b` is the contiguous linear span `(b-1)*n .+ (1:n)` and one call covers all of them. The plan's
    # coefficients depend on the geometry alone and are reused across the batch.
    (length(field) % n == 0) || throw(DimensionMismatch(
        "plan is for $n cells; a field of $(length(field)) is not a whole number of them",
    ))
    nb = length(field) ÷ n
    # Indexed linearly: an `N`-D array of the grid's shape indexes linearly as it stands, and a batched
    # one is that shape repeated, so `vec` adds an array wrapper per output per call for nothing.
    #
    # The per-direction accumulator is a tuple rebuilt each step, so one body serves any `D`: `ntuple`
    # over a `Val` unrolls and stays in registers.
    @inbounds for b in 0:(nb - 1)
        off = b * n
        for i in 1:n
            f0 = field[off + i]
            s = ntuple(_ -> zero(S), Val(D))
            for t in plan.ptr[i]:(plan.ptr[i + 1] - 1)
                s = _grad_accum(s, _grad_coefs(plan.coeffs, t, S),
                                S(field[off + plan.nbr[t]] - f0))
            end
            _grad_store!(outs, off + i, s)
        end
    end
    return outs
end

# The three tuple steps of the loop above, each taking everything it touches as an argument. Written
# inline in `gradient!`, the accumulator is a local that is both reassigned each step and captured by
# the `ntuple` closure, which Julia boxes.
@inline _grad_coefs(coeffs::NTuple{D,AbstractVector}, t::Int, ::Type{S}) where {D,S} =
    ntuple(d -> @inbounds(S(coeffs[d][t])), Val(D))

@inline _grad_accum(s::NTuple{D,S}, c::NTuple{D,S}, df::S) where {D,S} =
    ntuple(d -> s[d] + c[d] * df, Val(D))

@inline function _grad_store!(outs::NTuple{D,AbstractArray}, k::Int, s::NTuple{D}) where {D}
    ntuple(d -> @inbounds(outs[d][k] = s[d]), Val(D))
    return nothing
end

@inline gradient!(g1::AbstractArray, g2::AbstractArray, field::AbstractArray, plan::GradientPlan{2}) =
    gradient!((g1, g2), field, plan)

@inline gradient!(
    g1::AbstractArray, g2::AbstractArray, g3::AbstractArray, field::AbstractArray,
    plan::GradientPlan{3},
) = gradient!((g1, g2, g3), field, plan)

"""
    _sympinv2(a, b, c, tol) -> (p11, p12, p22)

Pseudo-inverse of the symmetric 2×2 `[a b; b c]`, dropping any eigendirection whose eigenvalue is below
`tol`. Closed form: a 2×2 symmetric eigenproblem has one.

A direction is dropped, never regularised. A stencil that carries no information along some tangent
direction — every neighbour on one line, which happens at a boundary and beside a mask — leaves `A`
singular there, and the data does not determine that component of the gradient. Inverting a nudged
matrix answers with a number governed by the nudge.
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
    _sympinv(A::NTuple{D,NTuple{D,T}}, tol) -> NTuple{D,NTuple{D,T}}

Pseudo-inverse of a symmetric positive-semidefinite `A` given as its rows, dropping any eigendirection
whose eigenvalue is at or below `tol`. `D = 2` and `D = 3`, which are the tangent plane and the volume.

A direction is dropped, never regularised — see [`_sympinv2`](@ref), which this calls at `D = 2`.
"""
@inline function _sympinv(A::NTuple{2,NTuple{2,T}}, tol::T) where {T}
    p11, p12, p22 = _sympinv2(A[1][1], A[1][2], A[2][2], tol)
    return ((p11, p12), (p12, p22))
end

"""
    _sym_eigvals3(A, tol) -> (λ₁, λ₂, λ₃)

Eigenvalues of a symmetric 3×3, largest first, by the closed form for the characteristic cubic: shift
by the mean eigenvalue, scale, and read the three roots off `cos` at thirds of a turn.

Eigenvalues only. A closed-form eigenvector of a 3×3 loses accuracy when two eigenvalues are close, and
[`_sympinv3`](@ref) needs none: a pseudo-inverse of a symmetric matrix is a polynomial in that matrix,
and its coefficients are functions of the eigenvalues alone.
"""
@inline function _sym_eigvals3(A::NTuple{3,NTuple{3,T}}, ::T) where {T}
    a11, a22, a33 = A[1][1], A[2][2], A[3][3]
    a12, a13, a23 = A[1][2], A[1][3], A[2][3]
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    q = (a11 + a22 + a33) / 3
    if iszero(p1)                                   # already diagonal
        λa, λb, λc = a11, a22, a33
        λa < λb && ((λa, λb) = (λb, λa))
        λb < λc && ((λb, λc) = (λc, λb))
        λa < λb && ((λa, λb) = (λb, λa))
        return (λa, λb, λc)
    end
    p2 = (a11 - q)^2 + (a22 - q)^2 + (a33 - q)^2 + 2 * p1
    p = sqrt(p2 / 6)
    # `det((A - qI)/p) / 2`, whose value in [-1, 1] is the cosine of three times the root angle.
    b11, b22, b33 = (a11 - q) / p, (a22 - q) / p, (a33 - q) / p
    b12, b13, b23 = a12 / p, a13 / p, a23 / p
    r = (b11 * (b22 * b33 - b23 * b23) -
         b12 * (b12 * b33 - b23 * b13) +
         b13 * (b12 * b23 - b22 * b13)) / 2
    φ = r ≤ -one(T) ? T(π) / 3 : (r ≥ one(T) ? zero(T) : acos(r) / 3)
    λ1 = q + 2 * p * cos(φ)
    λ3 = q + 2 * p * cos(φ + 2 * T(π) / 3)
    return (λ1, 3 * q - λ1 - λ3, λ3)                # the trace fixes the middle one
end

"""
    _sympinv3(A, tol) -> NTuple{3,NTuple{3,T}}

Pseudo-inverse of a symmetric positive-semidefinite 3×3.

At every rank `A⁺` is a polynomial in `A` with no constant term, so this is closed form and needs no
eigenvector. Such a polynomial annihilates the null space and acts as `1/λ` on the range, which is the
Moore–Penrose conditions:

- rank 3: `adj(A)/det(A)`, the ordinary inverse;
- rank 2: `αA² + βA` with `α = -(λ₁+λ₂)/(λ₁λ₂)²` and `β = (λ₁² + λ₁λ₂ + λ₂²)/(λ₁λ₂)²`;
- rank 1: `A/λ₁²`, since `A = λ₁vvᵀ` there;
- rank 0: zero.
"""
@inline function _sympinv3(A::NTuple{3,NTuple{3,T}}, tol::T) where {T}
    Z = ntuple(_ -> ntuple(_ -> zero(T), Val(3)), Val(3))
    λ1, λ2, λ3 = _sym_eigvals3(A, tol)
    λ1 ≤ tol && return Z                                    # no direction resolved at all
    if λ3 > tol
        a11, a12, a13 = A[1]
        a21, a22, a23 = A[2]
        a31, a32, a33 = A[3]
        c11 = a22 * a33 - a23 * a32
        c12 = a13 * a32 - a12 * a33
        c13 = a12 * a23 - a13 * a22
        δ = a11 * c11 + a12 * (a23 * a31 - a21 * a33) + a13 * (a21 * a32 - a22 * a31)
        iszero(δ) && return Z
        c22 = a11 * a33 - a13 * a31
        c23 = a13 * a21 - a11 * a23
        c33 = a11 * a22 - a12 * a21
        return ((c11 / δ, c12 / δ, c13 / δ),
                (c12 / δ, c22 / δ, c23 / δ),
                (c13 / δ, c23 / δ, c33 / δ))
    end
    A² = ntuple(i -> ntuple(j -> A[i][1] * A[1][j] + A[i][2] * A[2][j] + A[i][3] * A[3][j], Val(3)),
                Val(3))
    if λ2 > tol
        d = (λ1 * λ2)^2
        α = -(λ1 + λ2) / d
        β = (λ1 * λ1 + λ1 * λ2 + λ2 * λ2) / d
        return ntuple(i -> ntuple(j -> α * A²[i][j] + β * A[i][j], Val(3)), Val(3))
    end
    invλ² = inv(λ1 * λ1)
    return ntuple(i -> ntuple(j -> A[i][j] * invλ², Val(3)), Val(3))
end

@inline _sympinv(A::NTuple{3,NTuple{3,T}}, tol::T) where {T} = _sympinv3(A, tol)
