# ---------------------------------------------------------------------------
# Iso-latitude rings
# ---------------------------------------------------------------------------

"""
    AbstractRingStructure

How a grid's cells lie on circles of constant latitude: [`TensorRings`](@ref),
[`IsoLatitudeRings`](@ref) or [`NoRings`](@ref), as [`ring_structure`](@ref) reports it.
"""
abstract type AbstractRingStructure end

"""
    TensorRings()

Every ring is a latitude row of a structured grid on a sphere or spheroid, and all of them share one
longitude axis. Longitude is direction 1, so ring `r` is `nlon` consecutive cells.
"""
struct TensorRings <: AbstractRingStructure end

"""
    IsoLatitudeRings()

Rings of constant latitude whose cell counts vary from ring to ring, each a contiguous run of cell ids:
HEALPix in RING order, and the reduced Gaussian grids.
"""
struct IsoLatitudeRings <: AbstractRingStructure end

"""
    NoRings()

No ring a contiguous run of cell ids: a node set, the cubed sphere, the icosahedral geodesic, Yin–Yang, a
rotated grid, HEALPix in NESTED order, and every Cartesian grid.
"""
struct NoRings <: AbstractRingStructure end

"""
    ring_structure(grid) -> AbstractRingStructure

Whether `grid`'s cells lie on iso-latitude rings its numbering keeps contiguous, and of which kind. The
ring accessors — [`nrings`](@ref), [`ring_range`](@ref), [`ring_latitude`](@ref), [`eachring`](@ref) —
answer for any grid that reports one, and raise for [`NoRings`](@ref).

A mask leaves the rings as they are: a ring covers its inactive cells too, and [`isactive`](@ref) says
which they are.
"""
@inline ring_structure(::AbstractGrid) = NoRings()
@inline ring_structure(::StructuredGrid{T,<:Geometry.AbstractLonLatGeometry}) where {T} = TensorRings()
@inline ring_structure(::RingGrid) = IsoLatitudeRings()
@inline ring_structure(::HEALPixGrid{T,G,<:SphericalSampling.Ring}) where {T,G} = IsoLatitudeRings()

"""
    nrings(grid) -> Int

How many iso-latitude rings `grid` has: one per latitude row (per latitude and level in three
directions) of a structured sphere, `4·nside − 1` on HEALPix, the ring count of a [`RingGrid`](@ref).
"""
function nrings end

"""
    ring_range(grid, r) -> UnitRange{Int}

The cells of ring `r`, counted from the first ring, as a contiguous range of linear cell indices — the
column-major order of the grid's arrays, which is the field's own. `O(1)`.
"""
function ring_range end

"""
    ring_latitude(grid, r) -> T

The latitude every cell of ring `r` shares, as the grid stores it: geographic on a sphere, geodetic on a
spheroid. A one-direction grid's single ring is the equator.
"""
function ring_latitude end

"""
    nlon_in_ring(grid, r) -> Int

How many cells ring `r` holds.
"""
@inline nlon_in_ring(grid::AbstractGrid, r::Integer) = length(ring_range(grid, r))

"""
    ring_of(grid, i) -> Int

The ring that cell `i` (a linear index) belongs to.
"""
function ring_of end

"""
    eachring(grid)

The rings of `grid` in order, each as its [`ring_range`](@ref).
"""
@inline eachring(grid::AbstractGrid) = (ring_range(grid, r) for r in Base.OneTo(nrings(grid)))

# A layout without rings states so where a ring is asked of it.
@noinline _no_rings(grid) = throw(ArgumentError(
    "$(nameof(typeof(grid))) has no iso-latitude rings its numbering keeps contiguous; " *
    "`ring_structure` reports NoRings()",
))

for f in (:nrings, :ring_range, :ring_latitude, :ring_of)
    @eval $f(grid::AbstractGrid, _...) = _ring_accessor(ring_structure(grid), grid)
end
@noinline _ring_accessor(::NoRings, grid) = _no_rings(grid)

@noinline _check_ring(r::Int, n::Int) =
    1 ≤ r ≤ n ? r : throw(ArgumentError("ring $r is outside 1:$n"))

# ---- a structured sphere or spheroid: rings are latitude rows ---------------

@inline _nlon(grid::StructuredGrid) = @inbounds size_tuple(grid)[1]

@inline nrings(grid::StructuredGrid{T,<:Geometry.AbstractLonLatGeometry}) where {T} =
    length(grid) ÷ _nlon(grid)

@inline function ring_range(grid::StructuredGrid{T,<:Geometry.AbstractLonLatGeometry}, r::Integer) where {T}
    n = _nlon(grid)
    k = _check_ring(Int(r), nrings(grid))
    return ((k - 1) * n + 1):(k * n)
end

@inline function ring_latitude(
    grid::StructuredGrid{T,<:Geometry.AbstractLonLatGeometry,N}, r::Integer,
) where {T,N}
    k = _check_ring(Int(r), nrings(grid))
    N == 1 && return zero(T)
    φ = coordinates(grid, 2)
    return @inbounds T(φ[mod1(k, length(φ))])
end

@inline function ring_of(grid::StructuredGrid{T,<:Geometry.AbstractLonLatGeometry}, i::Integer) where {T}
    checkbounds(mask(grid), Int(i))
    return (Int(i) - 1) ÷ _nlon(grid) + 1
end

# ---- HEALPix in RING order --------------------------------------------------

@inline nrings(grid::HEALPixGrid{T,G,<:SphericalSampling.Ring}) where {T,G} = 4 * nside(grid) - 1

@inline ring_range(grid::HEALPixGrid{T,G,<:SphericalSampling.Ring}, r::Integer) where {T,G} =
    SphericalSampling.ring_range(SphericalSampling.HEALPixSampling(nside(grid)), r)

@inline function ring_latitude(grid::HEALPixGrid{T,G,<:SphericalSampling.Ring}, r::Integer) where {T,G}
    k = _check_ring(Int(r), nrings(grid))
    return SphericalSampling._hp_ring_angles(T, nside(grid), k)[2]
end

@inline function ring_of(grid::HEALPixGrid{T,G,<:SphericalSampling.Ring}, i::Integer) where {T,G}
    checkbounds(mask(grid), Int(i))
    return SphericalSampling._healpix_ring_phi(nside(grid), Int(i) - 1, T)[1]
end

# ---- a ring grid: its own tables --------------------------------------------

@inline ring_latitude(grid::RingGrid, r::Integer) =
    @inbounds getfield(grid, :latitudes)[_check_ring(Int(r), nrings(grid))]
