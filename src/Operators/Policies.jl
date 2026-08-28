# ---------------------------------------------------------------------------
# Applying a weight set along one direction
# ---------------------------------------------------------------------------

"""
    AbstractMaskPolicy

What [`apply_stencil!`](@ref) does at the edge of the active region: [`BlankMasked`](@ref),
[`ShiftWithinRun`](@ref) or [`ReduceInRun`](@ref).

A **type**, like the image and reach conventions in `Connectivity`: which cells carry a number and which
carry `masked` is a property of the result, so it belongs in the call rather than in a runtime tag.
"""
abstract type AbstractMaskPolicy end

"""
    BlankMasked()

Write `masked` at a cell that is inactive **or** whose stencil reads an inactive cell. The default, and
the only policy that never invents a value: where the stencil cannot be formed from active data, there
is no derivative.

Its cost is a dead band. Every active cell within `nodes - 1` of a masked cell is blanked, so a
five-point derivative loses two cells either side of every coastline.
"""
struct BlankMasked <: AbstractMaskPolicy end

"""
    ShiftWithinRun()

Shift the stencil to fit inside the run of active samples containing the cell, keeping the full node
count — the same thing the stencil already does at the end of a bounded axis, with the end of the active
run as the boundary. `masked` only where the run is shorter than `nodes`.

The accuracy order is therefore the same everywhere a value is written, which is the property
[`fd_weights`](@ref) exists to preserve. On a run of at least `nodes` active samples the weights are
**identical** to the unmasked ones, so the interior of an active region is bit-for-bit unchanged.
"""
struct ShiftWithinRun <: AbstractMaskPolicy end

"""
    ReduceInRun()

[`ShiftWithinRun`](@ref), and where the run cannot hold `nodes`, use the largest window it can, down to
`order + 1` samples. `masked` below that, where no derivative of that order exists.

This trades accuracy order for coverage — a five-point scheme becomes three-point in a strait three
cells wide — so it is named rather than reached by fallback. Ask for it when a value everywhere matters
more than a uniform order.

Under this policy `nodes` is a **ceiling**, not a demand, and that applies to the end of the axis as
well as the end of a run: an axis with fewer than `nodes` samples uses as many as it has instead of
raising, and one with fewer than `order + 1` is `masked` throughout. A single-latitude strip, a
two-level column and a one-cell-wide channel are ordinary grids, and asking for "second order where the
axis allows it" should not require the caller to clamp `nodes` themselves. The other two policies keep
the error, since neither claims to degrade.
"""
struct ReduceInRun <: AbstractMaskPolicy end
