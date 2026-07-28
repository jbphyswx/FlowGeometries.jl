module Stencils

# Neighbourhood SHAPES, in index space, for any dimension and any radius.
#
# A stencil describes a shape but not a dimensionality: the same `Face(2)` applies to a 2-D and a 5-D
# grid, and `offsets(s, Val(N))` materializes it for the `N` the grid has. Radius and shape live in the
# type, so the offset tuple is built at compile time and a loop over it unrolls.
#
# Two different things are called a "radius" and they are kept apart deliberately:
# `CellRadius` counts CELLS along the index directions, and `MetricBall` is a physical distance
# measured through the geometry. `Axial(1)` is the four face neighbours of a 2-D cell;
# `MetricBall(500e3)` is everything within 500 km.

"""
    AbstractStencil

A neighbourhood shape in index space. Materialize it with [`offsets`](@ref).

Concrete shapes: [`Axial`](@ref), [`VonNeumann`](@ref), [`Moore`](@ref) (alias [`Vertex`](@ref)),
[`Diagonal`](@ref), [`Anisotropic`](@ref), [`Custom`](@ref).
"""
abstract type AbstractStencil end

"""
    CellRadius(r)

A stencil extent of `r` cells, as distinct from a physical [`MetricBall`](@ref). Stencil constructors
accept a bare `Integer` too; this exists to name the distinction where it matters.
"""
struct CellRadius
    r::Int
    function CellRadius(r::Integer)
        r ≥ 1 || throw(ArgumentError("a CellRadius must be ≥ 1, got $r"))
        return new(Int(r))
    end
end

# `Base.@constprop :aggressive` is what makes `Moore(2)` infer to `Moore{2}` rather than to the abstract
# `Moore`: the radius is a type parameter, so it has to be constant-folded from the call site or the
# stencil — and every iterator built from it — is not concretely typed.
Base.@constprop :aggressive @inline function _radius(r::Integer)
    r ≥ 1 || throw(ArgumentError("a stencil radius must be ≥ 1, got $r"))
    return Int(r)
end
Base.@constprop :aggressive @inline _radius(r::CellRadius) = r.r

"""
    Axial(r = 1)

Axis-aligned neighbours out to `r` cells: `±k·ê_d` for every direction `d` and `k = 1…r`. `2·N·r`
offsets. `Axial(1)` is the classic 4-in-2-D / 6-in-3-D face stencil; beyond radius 1 "face" would be a
misnomer, since these are the axis lines rather than the touching faces.
"""
struct Axial{R} <: AbstractStencil end
Base.@constprop :aggressive @inline Axial(r = 1) = Axial{_radius(r)}()

"""
    VonNeumann(r = 1)

Every cell within `r` steps measured in the ``L^1`` (taxicab) metric: `0 < |δ|₁ ≤ r`. Equal to
[`Axial`](@ref) at `r = 1` and strictly larger beyond it, since it admits diagonal combinations whose
step count still fits.
"""
struct VonNeumann{R} <: AbstractStencil end
Base.@constprop :aggressive @inline VonNeumann(r = 1) = VonNeumann{_radius(r)}()

"""
    Moore(r = 1)
    Vertex(r = 1)

Every cell in the surrounding box: `0 < |δ|_∞ ≤ r`, i.e. `(2r+1)^N - 1` offsets. `Moore(1)` is the
8-in-2-D / 26-in-3-D vertex stencil.
"""
struct Moore{R} <: AbstractStencil end
Base.@constprop :aggressive @inline Moore(r = 1) = Moore{_radius(r)}()

"""Alias for [`Moore`](@ref), under the name the face/vertex pairing uses."""
const Vertex = Moore

"""
    Diagonal(r = 1)

Pure diagonals only: every offset whose components are all `±k` for a single `k = 1…r`. `2^N·r`
offsets, and no axis-aligned neighbour among them.
"""
struct Diagonal{R} <: AbstractStencil end
Base.@constprop :aggressive @inline Diagonal(r = 1) = Diagonal{_radius(r)}()

"""
    Anisotropic(radii)

A box with its own radius per direction: `|δ_d| ≤ radii[d]`, origin excluded. A direction with radius
`0` contributes no offset of its own, which is how a stencil is confined to a subset of the
directions.
"""
struct Anisotropic{Rs} <: AbstractStencil end
Base.@constprop :aggressive @inline function Anisotropic(radii::Tuple{Vararg{Integer}})
    all(≥(0), radii) || throw(ArgumentError("Anisotropic radii must be ≥ 0, got $radii"))
    any(>(0), radii) || throw(ArgumentError("Anisotropic needs a positive radius somewhere"))
    return Anisotropic{map(Int, radii)}()
end
Anisotropic(radii::Integer...) = Anisotropic(radii)

"""
    Custom(offsets)

An explicit offset set, e.g. `Custom(((1, 0), (0, 1)))` for a forward-only two-point stencil. The
origin is rejected: a cell is not its own neighbour.
"""
struct Custom{O} <: AbstractStencil end
Base.@constprop :aggressive @inline function Custom(offs::Tuple{Vararg{Tuple{Vararg{Integer}}}})
    isempty(offs) && throw(ArgumentError("Custom needs at least one offset"))
    n = length(first(offs))
    all(o -> length(o) == n, offs) ||
        throw(ArgumentError("every Custom offset must have the same length"))
    any(o -> all(iszero, o), offs) &&
        throw(ArgumentError("the zero offset is not a neighbour; drop it from the Custom set"))
    return Custom{map(o -> map(Int, o), offs)}()
end
Custom(offs::AbstractVector) = Custom(Tuple(map(Tuple, offs)))

# ---------------------------------------------------------------------------
# Offsets
# ---------------------------------------------------------------------------

"""
    offsets(stencil, Val(N)) -> NTuple{K,NTuple{N,Int}}

The stencil's offsets in `N` dimensions, as a tuple built at compile time so a loop over it unrolls.

Offsets come out in column-major order of the enclosing box (direction 1 varying fastest), which puts
`Axial(1)` and `Moore(1)` in the conventional order.
"""
@generated function offsets(s::AbstractStencil, ::Val{N}) where {N}
    return :($(Tuple(_offset_list(s, N))))
end

@inline _axis_unit(N::Int, d::Int, k::Int) = ntuple(i -> i == d ? k : 0, N)

# Built at generation time, never at run time. `offsets` returns the whole tuple, which is convenient
# but heap-allocates once it outgrows a register — measured at 412 bytes per call for `Moore(2)` in 2-D.
# `foreach_offset` exists for that reason: it unrolls the body over the offsets, so each one reaches the
# body as a literal register-sized tuple and nothing is materialized.
function _offset_list(::Type{Axial{R}}, N::Int) where {R}
    offs = NTuple{N,Int}[]
    for d in 1:N, k in 1:R
        push!(offs, _axis_unit(N, d, -k))
        push!(offs, _axis_unit(N, d, k))
    end
    return offs
end

function _offset_list(::Type{VonNeumann{R}}, N::Int) where {R}
    offs = NTuple{N,Int}[]
    for δ in Iterators.product(ntuple(_ -> (-R):R, N)...)
        t = sum(abs, δ)
        (t == 0 || t > R) && continue
        push!(offs, δ)
    end
    return offs
end

function _offset_list(::Type{Moore{R}}, N::Int) where {R}
    offs = NTuple{N,Int}[]
    for δ in Iterators.product(ntuple(_ -> (-R):R, N)...)
        all(iszero, δ) && continue
        push!(offs, δ)
    end
    return offs
end

function _offset_list(::Type{Diagonal{R}}, N::Int) where {R}
    offs = NTuple{N,Int}[]
    for k in 1:R, signs in Iterators.product(ntuple(_ -> (-1, 1), N)...)
        push!(offs, map(t -> t * k, signs))
    end
    return offs
end

function _offset_list(::Type{Anisotropic{Rs}}, N::Int) where {Rs}
    length(Rs) == N || throw(DimensionMismatch(
        "Anisotropic radii have length $(length(Rs)) but the grid has $N directions"))
    offs = NTuple{N,Int}[]
    for δ in Iterators.product(ntuple(d -> (-Rs[d]):Rs[d], N)...)
        all(iszero, δ) && continue
        push!(offs, δ)
    end
    return offs
end

function _offset_list(::Type{Custom{O}}, N::Int) where {O}
    length(first(O)) == N || throw(DimensionMismatch(
        "Custom offsets are $(length(first(O)))-dimensional but the grid has $N directions"))
    return collect(O)
end

"""
    foreach_offset(f, stencil, Val(N))

Apply `f` to each of the stencil's offsets, with the loop unrolled at compile time.

Each offset reaches `f` as a literal `NTuple{N,Int}`, so nothing is materialized — unlike
[`offsets`](@ref), whose returned tuple is heap-allocated once it outgrows a register. Every bulk
neighbour kernel goes through this.
"""
@generated function foreach_offset(f, s::AbstractStencil, ::Val{N}) where {N}
    offs = _offset_list(s, N)
    return Expr(:block, (:(f($(o))) for o in offs)..., :nothing)
end

"""
    fold_offsets(f, init, stencil, Val(N))

Fold `f(acc, δ)` over the stencil's offsets, unrolled at compile time.

The accumulator is threaded through as a value rather than mutated, so nothing is captured and nothing
is boxed — which is what a counting or filling kernel needs, and what a closure over a mutated local
would cost.
"""
@generated function fold_offsets(f, init, s::AbstractStencil, ::Val{N}) where {N}
    offs = _offset_list(s, N)
    body = Expr(:block, :(acc = init))
    for o in offs
        push!(body.args, :(acc = f(acc, $(o))))
    end
    push!(body.args, :acc)
    return body
end







"""
    nstencil(stencil, Val(N)) -> Int

How many offsets the stencil has in `N` dimensions — the buffer length a `neighbors!` call needs.
"""
@inline nstencil(s::AbstractStencil, ::Val{N}) where {N} = length(offsets(s, Val(N)))

"""
    reach(stencil, Val(N)) -> NTuple{N,Int}

The stencil's extent in cells along each direction: `maximum(|δ_d|)` over its offsets. This is the
halo width a traversal must leave, and the window a distance query has to scan.
"""
@inline function reach(s::AbstractStencil, ::Val{N}) where {N}
    offs = offsets(s, Val(N))
    return ntuple(d -> maximum(o -> abs(o[d]), offs), Val(N))
end

# ---------------------------------------------------------------------------
# Metric neighbourhoods
# ---------------------------------------------------------------------------

"""
    MetricBall(r)

Every cell within physical distance `r`, measured through the geometry rather than in cells. Distinct
from a [`CellRadius`](@ref): on a stretched or spherical grid the number of cells within `r` varies
across the grid, so this cannot be reduced to a fixed offset set.
"""
struct MetricBall{T<:Real}
    radius::T
    function MetricBall(r::T) where {T<:Real}
        r ≥ 0 || throw(ArgumentError("a MetricBall radius must be ≥ 0, got $r"))
        return new{T}(r)
    end
end

@inline radius(b::MetricBall) = b.radius

end # module Stencils
