# ---------------------------------------------------------------------------
# The physical derivative
# ---------------------------------------------------------------------------

"""
    derivative!(out, field, grid, dim; order=1, nodes=order+1, policy=BlankMasked(),
                masked=zero, active_only=true, backend=nothing) -> out

The derivative with respect to **distance** along direction `dim`: [`apply_stencil!`](@ref) divided by
the metric factor,

    ∂f/∂sᵈ = (1/hᵈ) · ∂f/∂ξᵈ,    hᵈ = Geometry.scale_factors(geo, p)[d]

On a Cartesian metric every `hᵈ` is `1` and this is `apply_stencil!` exactly, at no cost. Anywhere else
it is the derivative a physical law is written in, singular points included.

**Where the metric degenerates the derivative does not exist**, and `masked` is written. Longitude at a
pole is the case: `h_λ = R cos φ → 0`, so `1/h_λ` diverges. The test scales with the geometry's own size
and with the precision, `|h| ≤ L·√eps(T)`. A fixed threshold serves neither: `1e-12` is below
`eps(Float32)`, and at `Float32` on Earth's radius `cos(Float32(π/2)) ≈ -4.4e-8` puts `h_λ` at the pole
around 0.28 m.

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
