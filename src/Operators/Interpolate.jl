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

Off a rectilinear grid the `k` nearest cells come from a ball search, and three keywords hold its
state across calls. `topology` and `scratch` are as for [`Connectivity.neighbors_within!`](@ref):
with no index the search visits every cell, and [`Connectivity.indexed`](@ref) brings it to
`O(log n + k)`. `nearest = (idx, dist)`, two vectors of at least `k` entries, holds the cells found.
With all three held, a query allocates nothing.

`p` may be written any way a point is accepted elsewhere.

The mask policies say what an inactive contributor means, as they do for a stencil:
[`BlankMasked`](@ref) — the default — returns `masked` if any contributor is inactive, and
[`ReduceInRun`](@ref) renormalizes over the active ones. [`ShiftWithinRun`](@ref) has no meaning here,
there being no window to shift, and says so.

A field carrying trailing batch axes beyond the grid's own — many tracers, or an ensemble, sharing one
geometry — is evaluated for every element in one call: see [`interpolate!`](@ref) for the form that
writes into a caller's buffer, which this one wraps.

    interpolate(field, grid, points::AbstractVector; execution = SerialBackend(), …) -> values

The value at every point of `points`, a vector of coordinates: a vector, or for a field carrying trailing
batch axes a `(batch elements, points)` matrix. Off a rectilinear grid the neighbour search for all of
them runs on one cell list over the grid, built once, unless `topology` is passed.
"""
function interpolate end

"""
    interpolate!(out, field, grid, p; …) -> out
    interpolate!(out, field, grid, points::AbstractVector; execution = SerialBackend(), …) -> out

[`interpolate`](@ref) for a batched field, writing one value per batch element into `out`.

The bracketing cell — or, off a rectilinear grid, the neighbour set and the least-squares fit — depends
on the point and the geometry alone, so it is solved once and applied to every element. One call is
therefore less work than `interpolate` per slice.

At many points, `out[i]` (or `out[:, i]` for a batched field) receives `points[i]`'s values. The points
run under `execution` on any host or process backend.
"""
function interpolate! end

# A point as the many-point forms take one: written any way a single point is, never a bare number, so a
# vector of points and a single point written as a vector of numbers are told apart by element type.
const _PointElement = Union{Tuple,NamedTuple,AbstractVector{<:Real}}

# The batch length of `field` on `grid`, after checking `out` is shaped for `np` points: a vector for an
# unbatched field, `(batch elements, points)` for a batched one.
function _check_points_out(out::AbstractArray, field::AbstractArray, grid, np::Int)
    msk = Grids.mask(grid)
    n = length(msk)
    (n > 0 && length(field) % n == 0) || throw(DimensionMismatch(
        "grid has $n cells; a field of $(length(field)) is not a whole number of them",
    ))
    nb = length(field) ÷ n
    want = ndims(field) > ndims(msk) ? (nb, np) : (np,)
    size(out) == want || throw(DimensionMismatch(
        "out is $(size(out)); $np points of a field carrying $nb batch elements take $want",
    ))
    return nb
end

# The output the allocating many-point form returns.
_points_out(::Type{S}, field::AbstractArray, grid, np::Int) where {S} =
    ndims(field) > ndims(Grids.mask(grid)) ?
        Matrix{S}(undef, length(field) ÷ length(Grids.mask(grid)), np) : Vector{S}(undef, np)

@inline function _interp_mask_error(policy)
    return throw(ArgumentError(
        "$(policy) has no meaning when interpolating — there is no window to shift. Use " *
        "`ReduceInRun()` to renormalize over the active contributors, or `BlankMasked()`.",
    ))
end
