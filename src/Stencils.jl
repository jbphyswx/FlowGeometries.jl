module Stencils

# Neighbourhood shapes, in index space, for any dimension and any radius.
#
# A stencil describes a shape but not a dimensionality: the same `Face(2)` applies to a 2-D and a 5-D
# grid, and `offsets(s, Val(N))` materializes it for the `N` the grid has. Radius and shape live in the
# type, so the offset tuple is built at compile time and a loop over it unrolls.
#
# Two different things are called a radius. `CellRadius` counts cells along the index directions, and
# `MetricBall` is a physical distance measured through the geometry. `Axial(1)` is the four face
# neighbours of a 2-D cell; `MetricBall(500e3)` is everything within 500 km.

"""
    AbstractStencil

A neighbourhood shape in index space. Materialize it with [`offsets`](@ref).

Concrete shapes: [`Axial`](@ref), [`VonNeumann`](@ref), [`Moore`](@ref) (alias [`Vertex`](@ref)),
[`Diagonal`](@ref), [`Anisotropic`](@ref), [`Custom`](@ref).

A shape supplies one method, [`offsets`](@ref)`(s, ::Val{N})`, returning its offsets as a tuple;
`foreach_offset`, `fold_offsets`, `nstencil` and `reach` all follow from it, and their loops still
unroll at compile time. Put whatever the offsets depend on in the type, so the tuple is inferable:

    struct Upwind{R} <: AbstractStencil end
    Stencils.offsets(::Upwind{R}, ::Val{N}) where {R,N} =
        ntuple(i -> ntuple(d -> d == cld(i, R) ? mod1(i, R) : 0, Val(N)), Val(N * R))
"""
abstract type AbstractStencil end

"""
    CellRadius(r)

A stencil extent of `r` cells, as distinct from a physical [`MetricBall`](@ref). Stencil constructors
accept a bare `Integer` too; this names the distinction at a call site where it is worth naming.
"""
struct CellRadius
    r::Int
    function CellRadius(r::Integer)
        r ≥ 1 || throw(ArgumentError("a CellRadius must be ≥ 1, got $r"))
        return new(Int(r))
    end
end

# The radius is a type parameter, so `Base.@constprop :aggressive` folds it from the call site and
# `Moore(2)` infers to `Moore{2}`. Without that the stencil, and every iterator built from it, infers
# to the abstract `Moore`.
Base.@constprop :aggressive @inline function _radius(r::Integer)
    r ≥ 1 || throw(ArgumentError("a stencil radius must be ≥ 1, got $r"))
    return Int(r)
end
Base.@constprop :aggressive @inline _radius(r::CellRadius) = r.r

"""
    Axial(r = 1)

Axis-aligned neighbours out to `r` cells: `±k·ê_d` for every direction `d` and `k = 1…r`. `2·N·r`
offsets. `Axial(1)` is the classic 4-in-2-D / 6-in-3-D face stencil; beyond radius 1 the offsets run
along the axis lines, past the touching faces.
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

This is the one method a new shape defines; see [`AbstractStencil`](@ref).
"""
function offsets end

# The shapes defined here, whose offsets `_offset_list` can build at generation time from the type.
const _BuiltinStencil = Union{Axial,VonNeumann,Moore,Diagonal,Anisotropic,Custom}

@generated function offsets(s::_BuiltinStencil, ::Val{N}) where {N}
    return :($(Tuple(_offset_list(s, N))))
end

"""
    is_symmetric(stencil, Val(N)) -> Bool

Whether the offset set is closed under negation: `δ` is an offset exactly when `−δ` is.

A graph built from a symmetric stencil is a symmetric graph, and stays one under masking and under
clipping at a boundary — an edge and its reverse are dropped together, both being the same pair of
cells. A symmetric adjacency is then read as its own transpose.

`Axial`, `VonNeumann`, `Moore`, `Diagonal` and `Anisotropic` are symmetric by construction: each is
defined by a condition on `|δ|` and so contains `−δ` with `δ`. `Custom` is whatever it was given, and
is checked once at compile time. A caller's own shape answers `false` until it says otherwise, at the
cost of a transpose — see [`AbstractStencil`](@ref).
"""
function is_symmetric end

is_symmetric(::AbstractStencil, ::Val) = false
is_symmetric(::Union{Axial,VonNeumann,Moore,Diagonal,Anisotropic}, ::Val) = true

@generated function is_symmetric(::Custom{O}, ::Val{N}) where {O,N}
    all(o -> length(o) == N, O) || return :(false)
    set = Set(O)
    return :($(all(o -> map(-, o) in set, O)))
end

@inline _axis_unit(N::Int, d::Int, k::Int) = ntuple(i -> i == d ? k : 0, N)

# Built at generation time, never at run time. `offsets` returns the whole tuple, which heap-allocates
# once it outgrows a register: `Moore(3)` in 4-D is 2400 offsets. `foreach_offset` unrolls the body
# over them instead, so each offset reaches the body as a literal register-sized tuple and nothing is
# materialized. The same holds for a caller's own shape, whose offsets arrive as a tuple that never
# escapes.
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

Each offset reaches `f` as a register-sized `NTuple{N,Int}` and the offset set is never materialized on
the heap — unlike [`offsets`](@ref), whose returned tuple is allocated once it outgrows a register.
Every bulk neighbour kernel goes through this.
"""
@inline foreach_offset(f, s::AbstractStencil, ::Val{N}) where {N} =
    _foreach_tuple(f, offsets(s, Val(N)))

# For the shapes defined here the offset list exists at generation time, so each offset can be emitted
# as a literal and inference never carries the offset tuple as a value. That matters for a wide stencil:
# `Moore(3)` in 4-D is 2400 offsets, and compiling it through the tuple below costs ~30× as long.
@generated function foreach_offset(f, s::_BuiltinStencil, ::Val{N}) where {N}
    offs = _offset_list(s, N)
    return Expr(:block, (:(f($(o))) for o in offs)..., :nothing)
end

# Generated on the tuple's type: a shape defined anywhere answers `offsets` by ordinary dispatch, and
# the unroll over what it returns is still flat and compile-time.
@generated function _foreach_tuple(f, t::Tuple)
    K = length(t.parameters)
    return Expr(:block, (:(f(@inbounds t[$i])) for i in 1:K)..., :nothing)
end

"""
    fold_offsets(f, init, stencil, Val(N))

Fold `f(acc, δ)` over the stencil's offsets, unrolled at compile time.

The accumulator is threaded through as a value, so nothing is captured and nothing is boxed. This is
the form a counting or filling kernel takes.
"""
@inline fold_offsets(f, init, s::AbstractStencil, ::Val{N}) where {N} =
    _fold_tuple(f, init, offsets(s, Val(N)))

@generated function fold_offsets(f, init, s::_BuiltinStencil, ::Val{N}) where {N}
    body = Expr(:block, :(acc = init))
    for o in _offset_list(s, N)
        push!(body.args, :(acc = f(acc, $(o))))
    end
    push!(body.args, :acc)
    return body
end

@generated function _fold_tuple(f, init, t::Tuple)
    body = Expr(:block, :(acc = init))
    for i in 1:length(t.parameters)
        push!(body.args, :(acc = f(acc, @inbounds t[$i])))
    end
    push!(body.args, :acc)
    return body
end

"""
    nstencil(stencil, Val(N)) -> Int

How many offsets the stencil has in `N` dimensions — the buffer length a `neighbors!` call needs.
"""
@inline nstencil(s::AbstractStencil, ::Val{N}) where {N} = length(offsets(s, Val(N)))

# Both counts below are properties of the shape alone, so for the shapes defined here they are literals: the
# offset list exists at generation time, and neither answer needs it to reach inference as a value.
@generated nstencil(s::_BuiltinStencil, ::Val{N}) where {N} = length(_offset_list(s, N))

"""
    reach(stencil, Val(N)) -> NTuple{N,Int}

The stencil's extent in cells along each direction: `maximum(|δ_d|)` over its offsets. This is the
halo width a traversal must leave, and the window a distance query has to scan.
"""
@inline function reach(s::AbstractStencil, ::Val{N}) where {N}
    offs = offsets(s, Val(N))
    return ntuple(d -> maximum(o -> abs(o[d]), offs), Val(N))
end

@generated function reach(s::_BuiltinStencil, ::Val{N}) where {N}
    offs = _offset_list(s, N)
    r = ntuple(d -> maximum(abs(o[d]) for o in offs), N)
    return :($(r))
end

# ---------------------------------------------------------------------------
# Metric neighbourhoods
# ---------------------------------------------------------------------------

"""
    MetricBall(r)

Every cell within physical distance `r`, measured through the geometry. Distinct from a
[`CellRadius`](@ref): on a stretched or spherical grid the number of cells within `r` varies from cell
to cell, so this reduces to no fixed offset set.

Query it with `Connectivity.neighbors_within!` / `nneighbors_within` / `neighbors_within`, which take
it as their `ball` keyword.
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
