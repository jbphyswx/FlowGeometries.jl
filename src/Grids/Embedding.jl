# ---------------------------------------------------------------------------
# The Euclidean space a spatial index searches in
# ---------------------------------------------------------------------------

"""
    AbstractEmbedding

How a grid's cell centres sit in the Euclidean space an index searches, and therefore what a physical
radius means there. A **type**, for the reason stencils are: the conversion is applied once per query,
so a runtime tag would leave it unresolved and put a branch — and a boxed radius — in the hot path.

[`CartesianEmbedding`](@ref), [`ChordEmbedding`](@ref), [`ArcEmbedding`](@ref).
"""
abstract type AbstractEmbedding end

"""
    CartesianEmbedding()

The coordinates themselves, replicated at the periodic images. A radius passes through unchanged.
"""
struct CartesianEmbedding <: AbstractEmbedding end

"""
    ChordEmbedding()

The embedded distance already is the metric, or a lower bound on it: `(λ, φ, r)` on a sphere, where the
metric is the 3-D chord, and geodetic coordinates on a spheroid, where the ECEF chord is at most the
geodesic. A radius passes through unchanged; under-approximating means the query over-returns, which is
what an index is allowed to do.
"""
struct ChordEmbedding <: AbstractEmbedding end

"""
    ArcEmbedding(R)

Points on the reference sphere of radius `R`, where the metric is the great-circle arc. A radius has to
be converted to the chord it subtends.
"""
struct ArcEmbedding{T<:AbstractFloat} <: AbstractEmbedding
    radius::T
end

"""
    embed_point(grid, p) -> NTuple

One coordinate tuple through the same transform [`embedded_points`](@ref) applies to the cell centres,
so a query seeded by a point searches the space the index was built in.
"""
function embed_point end

@inline embed_point(grid::AbstractGrid{G,T}, p::NTuple{D,Real}) where {G,T,D} =
    Geometry.embed(grid_geometry(grid), p)

"""
    embedding_of(grid) -> AbstractEmbedding

The Euclidean space this grid's cell centres sit in, and therefore what a physical radius means there.

Named separately from [`embedded_points`](@ref) because an index that streams the centres still has to
know the space before it reads the first one.
"""
function embedding_of end

@inline embedding_of(grid::AbstractGrid) =
    _embedding_for(grid_geometry(grid), ncoordinates(grid))

"""
    _embedding_for(geo, ncoords) -> AbstractEmbedding

The space a geometry's `ncoords`-coordinate points sit in. [`embedding_of`](@ref) is this for a grid,
which knows its own coordinate count; a caller holding loose coordinate vectors states it.

One definition, so an index built from a grid and one built from bare coordinates convert a radius the
same way rather than by two implementations agreeing.
"""
@inline _embedding_for(::Geometry.AbstractCartesianGeometry, ::Integer) = CartesianEmbedding()

# `(λ, φ, r)` carries its own radius, so the metric there IS the Euclidean chord; on the surface it is
# the great-circle arc, and a radius has to be converted to the chord it subtends.
@inline _embedding_for(geo::Geometry.AbstractSphericalGeometry{T}, ncoords::Integer) where {T} =
    ncoords ≥ 3 ? ChordEmbedding() : ArcEmbedding(T(Geometry.radius(geo)))

# Geodetic coordinates always: the ECEF chord is at most the geodesic, so a chord query over-returns
# and the caller's own distance gate trims it. Height enters the position, never the radius.
@inline _embedding_for(::Geometry.AbstractEllipsoidalGeometry, ::Integer) = ChordEmbedding()

"""
    embedded_radius(embedding, r) -> T

A physical radius as a radius in the embedding.
"""
@inline embedded_radius(::CartesianEmbedding, r::T) where {T} = r
@inline embedded_radius(::ChordEmbedding, r::T) where {T} = r

# `2R·sin(σ/2)` is monotone only to `σ = π`, so an arc of an antipodal distance or more saturates at the
# diameter rather than turning back down.
@inline function embedded_radius(e::ArcEmbedding{T}, r::Real) where {T}
    σ = T(r) / e.radius
    return σ ≥ T(π) ? T(2) * e.radius : T(2) * e.radius * sin(σ / T(2))
end

"""
    _shift_set(periodic, period)

Offsets to replicate a point set by: `(0,)` in a non-wrapping direction, `(0, -L, +L)` in a wrapping
one. Zero first, so the originals occupy the first block of the replicated set.
"""
@inline _shift_set(p::Bool, L::T) where {T} = p ? (zero(T), -L, L) : (zero(T),)

"""
    _ghost_points(pts, periodic, period) -> (all_pts, nghost)

Replicate the `D × N` point matrix once per combination of periodic image offsets, originals in the
first `N` columns. A wrapping domain is then searched by an ordinary Euclidean query over the images.
"""
function _ghost_points(
    pts::AbstractMatrix{T}, periodic::NTuple{D,Bool}, period::NTuple{D,T},
) where {D,T}
    shifts = ntuple(d -> _shift_set(periodic[d], period[d]), Val(D))
    N = size(pts, 2)
    ng = prod(map(length, shifts))
    out = similar(pts, D, N * ng)
    g = 0
    for ci in CartesianIndices(map(eachindex, shifts))
        cols = (g * N + 1):((g + 1) * N)
        for d in 1:D
            δ = shifts[d][ci[d]]
            @views out[d, cols] .= pts[d, :] .+ δ
        end
        g += 1
    end
    return out, ng
end

"""
    _grid_points(grid) -> (raw, D)

Cell centres as a `D × n` matrix, in the grid's own coordinates.
"""
function _grid_points(grid::AbstractGrid{G,T}) where {G,T}
    msk = mask(grid)
    lin = LinearIndices(size(msk))
    D = length(_raw_coords(grid, Tuple(first(CartesianIndices(size(msk))))...))
    raw = Matrix{T}(undef, D, length(msk))
    @inbounds for ci in CartesianIndices(size(msk))
        p = _raw_coords(grid, Tuple(ci)...)
        for d in 1:D
            raw[d, lin[ci]] = p[d]
        end
    end
    return raw, D
end

function _grid_points(grid::UnstructuredGrid{T,G,N}) where {T,G,N}
    n = length(mask(grid))
    raw = Matrix{T}(undef, N, n)
    @inbounds for k in 1:n
        p = _raw_coords(grid, k)
        for d in 1:N
            raw[d, k] = p[d]
        end
    end
    return raw, N
end

"""
    embedded_points(grid) -> (pts, nghost, embedding)

The cell centres in the space an index searches, the number of periodic replications they carry, and
the [`AbstractEmbedding`](@ref) saying what a radius means there.

One definition, so every index searches the same space as every other and as the k-d-tree construction
path — the guarantee that an indexed query and a scan return the same cells rests on it.
"""
function embedded_points end

function embedded_points(
    grid::AbstractGrid{G,T}; ghosts::Bool = true,
) where {G<:Geometry.AbstractCartesianGeometry,T}
    raw, D = _grid_points(grid)
    per = ntuple(d -> isperiodic(grid, d), D)
    prd = ntuple(d -> T(period(grid, d)), D)
    # Images exist for an index that cannot wrap. One that wraps its own lattice asks for none, and so
    # never allocates the `3^d` replication only to discard it.
    pts, ng = (ghosts && any(per)) ? _ghost_points(raw, per, prd) : (raw, 1)
    return pts, ng, embedding_of(grid)
end

# `Geometry.embed` rather than either formula written again, so a cell centre reaches the index through
# the same map as a query point. Longitude needs no ghost images — `λ` and `λ+2π` embed together.
function embedded_points(
    grid::AbstractGrid{G,T}; ghosts::Bool = true,
) where {G<:Geometry.AbstractLonLatGeometry,T}
    geo = grid_geometry(grid)
    raw, D = _grid_points(grid)
    pts = Matrix{T}(undef, 3, size(raw, 2))
    @inbounds for k in axes(raw, 2)
        c = Geometry.embed(geo, ntuple(d -> raw[d, k], D))
        pts[1, k] = c[1]; pts[2, k] = c[2]; pts[3, k] = c[3]
    end
    return pts, 1, embedding_of(grid)
end
