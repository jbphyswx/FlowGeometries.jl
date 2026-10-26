# ---------------------------------------------------------------------------
# The physical derivative
# ---------------------------------------------------------------------------

"""
    derivative!(out, field, grid, dim; order=1, nodes=order+1, policy=BlankMasked(),
                masked=zero, active_only=true, execution=SerialBackend(), scratch=nothing) -> out

The derivative with respect to **distance** along direction `dim`: [`apply_stencil!`](@ref) divided by
the metric factor,

    ∂f/∂sᵈ = (1/hᵈ) · ∂f/∂ξᵈ,    hᵈ = Geometry.scale_factors(geo, p)[d]

On a Cartesian metric every `hᵈ` is `1` and this is `apply_stencil!` exactly, at no cost. Anywhere else
it is the derivative a physical law is written in, singular points included.

At `order = p` it is the `p`-th derivative with respect to distance. Along a direction in
[`Geometry.scale_invariant_directions`](@ref), where `hᵈ` is constant along the direction, that is
`(1/hᵈ)ᵖ·∂ᵖf/∂ξᵈᵖ`. Elsewhere — a spheroid's latitude, where `h_φ = M(φ)` — the weights are solved over
the nodes' distances from the cell, [`Geometry.line_distance`](@ref).

**Where the metric degenerates the derivative does not exist**, and `masked` is written. Longitude at a
pole is the case: `h_λ = R cos φ` is zero at a latitude of `±T(π)/2`, the pole as stored, so `1/h_λ`
does not exist there.

Each value is multiplied by `(1/hᵈ)ᵖ` as it is written, so a cell written `masked` holds exactly `masked`.
No scale factor in this package depends on **longitude**, so `hᵈ` is constant along the first axis
whichever direction is differenced, and it is solved once per direction-1 run. (It is *not* generally
constant along the differenced direction — on a spheroid `h_φ = M(φ)` varies with `φ` — so it is not
hoisted that way.)

A divergence or a curl is still the caller's to assemble, needing a result location and a boundary
policy this does not choose. Note the flux form when doing so: on a sphere

    ∇·u = (1/(R cos φ)) [ ∂u_λ/∂λ + ∂(u_φ cos φ)/∂φ ]

so the second term differentiates `u_φ cos φ`, not `u_φ`; taking two physical derivatives and adding
them is a different, wrong expression.
"""
function derivative! end
