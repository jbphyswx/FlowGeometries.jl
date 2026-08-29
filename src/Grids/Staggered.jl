# ---------------------------------------------------------------------------
# Staggered (Arakawa C) grids
# ---------------------------------------------------------------------------

"""
    StaggeredGrid(geometry, axes...; mask = nothing, topology = nothing, period = nothing)
    StaggeredGrid(center::StructuredGrid)

One rectilinear mesh, read at any staggered LOCATION: a family of grids rather than a grid.

A location is one [`FlowGeometries.Discretization.AbstractLocation`](@ref) per direction — `Center()`
where the samples are the cell centres, `Face()` where they are the cell boundaries. On an Arakawa C
arrangement the tracer sits at all-centres, the velocity component along direction `d` at `Face()` in
`d` and `Center()` elsewhere, and the vorticity at `Face()` in the two directions it circulates about.
[`grid_at`](@ref) returns the ordinary [`StructuredGrid`](@ref) at any of them, so every operator that
already works on a rectilinear grid works at every location without knowing this type exists.

The face axes are built once, at construction. A face axis derived from a UNIFORM primal axis is
itself an [`Axes.UniformAxis`](@ref) — see [`FlowGeometries.Discretization.faces`](@ref) — so a uniform
mesh does not look stretched to the methods that dispatch on spacing.

A periodic direction of `N` cells has `N` faces, not `N+1`: its last face is its first, and storing
both would be one column of duplicated degrees of freedom. A bounded direction has `N+1`.

A mask is given over the CENTRE cells, and a staggered point is active where every centre it is built
from is. That is the finite-volume rule and not a new one: a velocity face between an active and an
inactive cell is a boundary, not a free value, on the same footing as the least-squares gradient's
unresolved direction and a stencil's masked cell.
"""
struct StaggeredGrid{
    T<:AbstractFloat,
    N,
    GC<:AbstractStructuredGrid,
    AF<:NTuple{N,AbstractVector{T}},
}
    center::GC
    faces::AF
end

function StaggeredGrid(center::StructuredGrid{T,G,N}) where {T,G,N}
    fs = ntuple(Val(N)) do d
        _face_axis(coordinates(center, d), isperiodic(center, d))
    end
    return StaggeredGrid{T,N,typeof(center),typeof(fs)}(center, fs)
end

StaggeredGrid(geometry::Geometry.AbstractGeometry, ax::Vararg{AbstractVector}; kwargs...) =
    StaggeredGrid(StructuredGrid(geometry, ax...; kwargs...))

# A wrapping direction's last face IS its first, so it carries `N` of them; a bounded one carries the
# `N+1` `faces` produces. Slicing a uniform axis keeps it uniform, so the guarantee survives either way.
@inline _face_axis(x::AbstractVector, periodic::Bool) =
    periodic ? (f = Discretization.faces(x); @inbounds f[firstindex(f):(firstindex(f) + length(x) - 1)]) :
               Discretization.faces(x)

"""
    center_grid(sg) -> StructuredGrid

The primal grid `sg` was built from: every direction at `Center()`, which is where a tracer sits.
"""
@inline center_grid(sg::StaggeredGrid) = sg.center

@inline grid_geometry(sg::StaggeredGrid) = grid_geometry(sg.center)
@inline Base.ndims(::StaggeredGrid{T,N}) where {T,N} = N
@inline ncoordinates(::StaggeredGrid{T,N}) where {T,N} = N
@inline coordinate_names(sg::StaggeredGrid) = coordinate_names(sg.center)
@inline Base.eltype(::StaggeredGrid{T}) where {T} = T

"""
    location_at(sg, d) -> NTuple{N,AbstractLocation}

Where the velocity component along direction `d` sits: `Face()` in `d`, `Center()` in every other
direction.
"""
@inline location_at(::StaggeredGrid{T,N}, d::Integer) where {T,N} =
    ntuple(e -> e == d ? Discretization.Face() : Discretization.Center(), Val(N))

"""
    locations(sg) -> NTuple{N,NTuple{N,AbstractLocation}}

All `N` velocity locations, one per direction — [`location_at`](@ref) for each, which together are the
Arakawa C staggering itself.
"""
@inline locations(sg::StaggeredGrid{T,N}) where {T,N} =
    ntuple(d -> location_at(sg, d), Val(N))

"""
    center_location(sg) -> NTuple{N,Center}

Every direction at `Center()`.
"""
@inline center_location(::StaggeredGrid{T,N}) where {T,N} =
    ntuple(_ -> Discretization.Center(), Val(N))

"""
    axis_at(sg, d, loc) -> AbstractVector

Direction `d`'s samples at location `loc`: the primal axis at [`Discretization.Center`](@ref), its faces at
[`Discretization.Face`](@ref).
"""
@inline axis_at(sg::StaggeredGrid, d::Integer, ::Discretization.Center) = coordinates(sg.center, d)
# The face axes of different directions may be different types, so a runtime `d` selects one by the
# same recursive tail-split `coordinates(grid, d)` uses rather than by indexing the tuple.
@inline axis_at(sg::StaggeredGrid, d::Integer, ::Discretization.Face) =
    _at_axis(identity, sg.faces, d)

"""
    grid_at(sg, loc) -> StructuredGrid

The grid of `sg` at location `loc`, one [`Discretization.Center`](@ref)/[`Discretization.Face`](@ref) per direction.

An ordinary [`StructuredGrid`](@ref), so every rectilinear operator applies to it unchanged. It is
BUILT here rather than stored — there are `2^N` locations and a mesh needs few of them — so a caller
differencing repeatedly holds the result, the way a stencil plan is held.
"""
function grid_at(
    sg::StaggeredGrid{T,N}, loc::NTuple{N,Discretization.AbstractLocation},
) where {T,N}
    ax = ntuple(d -> loc[d] isa Discretization.Face ? sg.faces[d] : coordinates(sg.center, d), Val(N))
    dims = map(length, ax)
    return StructuredGrid(
        grid_geometry(sg.center), ax...;
        mask = _staggered_mask(sg, loc, dims),
        topology = topology(sg.center),
        period = ntuple(d -> period(sg.center, d), Val(N)),
    )
end

@inline grid_at(sg::StaggeredGrid, d::Integer) = grid_at(sg, location_at(sg, d))

# The centres a staggered point is built from, in one direction. A pair always — duplicated where the
# point sits on a centre, or against a bounded end — so the activity test below is one shape.
@inline function _stagger_sources(i::Int, nc::Int, periodic::Bool)
    periodic && return (mod1(i - 1, nc), mod1(i, nc))
    i == 1 && return (1, 1)
    i ≥ nc + 1 && return (nc, nc)
    return (i - 1, i)
end

@inline _source_pair(::Discretization.Center, i::Int, ::Int, ::Bool) = (i, i)
@inline _source_pair(::Discretization.Face, i::Int, nc::Int, per::Bool) =
    _stagger_sources(i, nc, per)

# All-active stays all-active: a point built only from active centres is active, and every centre is.
function _staggered_mask(
    sg::StaggeredGrid{T,N}, loc::NTuple{N,Discretization.AbstractLocation}, dims::NTuple{N,Int},
) where {T,N}
    m = mask(sg.center)
    m isa AllActive && return AllActive(dims)
    nc = size_tuple(sg.center)
    per = ntuple(d -> isperiodic(sg.center, d), Val(N))
    out = BitArray(undef, dims)
    @inbounds for I in CartesianIndices(dims)
        src = ntuple(d -> _source_pair(loc[d], I[d], nc[d], per[d]), Val(N))
        ok = true
        # Every combination of the contributing centres, which is `2^(number of staggered directions)`
        # distinct cells and a repeat of one cell in each direction that is not staggered.
        for c in 0:(2^N - 1)
            J = ntuple(d -> src[d][1 + ((c >> (d - 1)) & 1)], Val(N))
            ok &= m[J...]
        end
        out[I] = ok
    end
    return out
end

function Base.show(io::IO, sg::StaggeredGrid{T,N}) where {T,N}
    print(io, "StaggeredGrid{", T, ",", N, "}(", join(size_tuple(sg.center), "×"), " cells, ",
          coordinate_names(sg.center), ")")
    return
end

function Base.show(io::IO, ::MIME"text/plain", sg::StaggeredGrid{T,N}) where {T,N}
    nm = coordinate_names(sg.center)
    println(io, "StaggeredGrid{", T, ",", N, "} over ", join(size_tuple(sg.center), "×"), " cells")
    print(io, "  centres  ", size_tuple(sg.center))
    for d in 1:N
        print(io, "\n  ", nm[d], "-faces ", size_tuple(grid_at(sg, d)))
    end
    return
end
