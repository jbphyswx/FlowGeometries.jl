"""
    interpolate(field, grid, p; policy=BlankMasked(), masked=NaN, …) -> value

The value of `field` at the coordinate `p`, which is how observational data arrives: a station, a float
or a ship track carries a coordinate.

`Discretization.interpolation_weights` gives the weights along **one** axis; this composes them, on
every layout.

- `StructuredGrid` — multilinear, the tensor product of the per-axis weights. A periodic direction
  interpolates *across* its seam, so a coordinate past the last sample wraps to the first.
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
on the point and the geometry alone, so it is solved once and applied to every element. One call is
therefore less work than `interpolate` per slice.
"""
function interpolate! end

@inline function _interp_mask_error(policy)
    return throw(ArgumentError(
        "$(policy) has no meaning when interpolating — there is no window to shift. Use " *
        "`ReduceInRun()` to renormalize over the active contributors, or `BlankMasked()`.",
    ))
end
