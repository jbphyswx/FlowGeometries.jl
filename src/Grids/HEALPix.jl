# ---------------------------------------------------------------------------
# HEALPix
# ---------------------------------------------------------------------------

"""
    HEALPixGrid(geometry, nside; scheme = SphericalSampling.Ring(), mask = nothing)

The HEALPix pixelization as a grid: `12·nside²` equal-area pixels on `4·nside − 1` iso-latitude rings.

It stores `nside`, the scheme, the geometry and the mask. A pixel's coordinates, its neighbours and its
area are each closed-form in `(nside, pixel)`, so those four fields are the whole grid:
`Base.summarysize` is the same at `nside = 1024` — 12.6 million pixels — as at `nside = 1`.

`nlon` varies by ring, so this is a layout of its own, and it answers the three traits a neighbourhood
algorithm asks of one: cells are [`FlatCells`](@ref) (a pixel is one id), adjacency is
[`FormulaNeighbors`](@ref) (the face tables of Górski et al.), and a distance query enumerates through
[`IndexedCandidates`](@ref).

Coordinates are `(λ, φ)`: [`coords`](@ref)`(grid, i)` reads one pixel, and [`materialize`](@ref) gives
the whole cloud as dense vectors.
"""
struct HEALPixGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    S<:SphericalSampling.RingScheme,
    B<:AbstractVector{Bool},
} <: AbstractGrid{G,T}
    geometry::G
    nside::Int
    scheme::S
    mask::B
end

function HEALPixGrid(
    geometry::Geometry.AbstractSphericalGeometry{T}, nside::Integer;
    scheme::SphericalSampling.RingScheme = SphericalSampling.Ring(), mask = nothing,
) where {T<:AbstractFloat}
    ns = Int(nside)
    ns ≥ 1 || throw(ArgumentError("HEALPix nside must be ≥ 1, got $ns"))
    npix = SphericalSampling.healpix_npix(ns)
    m = mask === nothing ? AllActive((npix,)) : mask
    length(m) == npix || throw(DimensionMismatch(
        "mask holds $(length(m)) entries for a grid of $npix pixels",
    ))
    return HEALPixGrid{T,typeof(geometry),typeof(scheme),typeof(m)}(geometry, ns, scheme, m)
end

HEALPixGrid(nside::Integer; kwargs...) = HEALPixGrid(Geometry.SphericalGeometry(), nside; kwargs...)

@inline _from_fields(
    ::Type{<:HEALPixGrid},
    geometry::G, nside::Int, scheme::S, mask::B,
) where {T,G<:Geometry.AbstractSphericalGeometry{T},S,B} =
    HEALPixGrid{T,G,S,B}(geometry, nside, scheme, mask)

"""
    nside(grid::HEALPixGrid) -> Int

The resolution parameter: the grid has `12·nside²` pixels on `4·nside − 1` rings.
"""
@inline nside(grid::HEALPixGrid) = getfield(grid, :nside)

"""
    scheme(grid::HEALPixGrid) -> RingScheme

Which pixel ordering the ids are in — `Ring()` or `Nested()`.
"""
@inline scheme(grid::HEALPixGrid) = getfield(grid, :scheme)

"""
    npixels(grid::HEALPixGrid) -> Int

`12·nside²`, the pixelization's own count — which is `length(grid)` whether or not a mask leaves some
of them inactive.
"""
@inline npixels(grid::HEALPixGrid) = SphericalSampling.healpix_npix(nside(grid))

# ---- the three traits -------------------------------------------------------

@inline cell_address(::HEALPixGrid) = FlatCells()
@inline adjacency_source(::HEALPixGrid) = FormulaNeighbors()
@inline candidate_source(::HEALPixGrid) = IndexedCandidates()

# ---- coordinates ------------------------------------------------------------

@inline ncoordinates(::HEALPixGrid) = 2

# `pix2ang` gives colatitude; the package's spherical coordinates are `(λ, φ)` with `φ` the geographic
# latitude, which is the same angle measured from the equator.
@inline function _raw_coords(grid::HEALPixGrid{T}, i::Integer) where {T}
    θ, ϕ = SphericalSampling.pix2ang(T, nside(grid), Int(i) - 1; scheme = scheme(grid))
    return (ϕ, SphericalSampling.geographic_latitude(θ))
end

coordinates(grid::HEALPixGrid) = throw(ArgumentError(
    "a HEALPixGrid stores no coordinate arrays — a pixel's position is arithmetic in (nside, pixel). " *
    "Use `coords(grid, i)` for one pixel, or `Grids.materialize(grid)` for the whole cloud.",
))

# ---- topology and span ------------------------------------------------------

@inline topology(::HEALPixGrid) = (Periodic(), Bounded())
@inline period(grid::HEALPixGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? T(2π) : zero(T)

# The pixelization covers the whole sphere, so both spans are known without reading a pixel.
@inline bounds(grid::HEALPixGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? (zero(T), T(2π)) : (-T(π) / 2, T(π) / 2)
@inline origin(grid::HEALPixGrid, d::Integer) = bounds(grid, d)[1]

# ---- measure ----------------------------------------------------------------

# Equal-area is what the pixelization is for: every pixel is the sphere's area over the pixel count, so
# the measure is one number and a length, not `npix` copies of it.
@inline measure(grid::HEALPixGrid{T}) where {T} =
    Axes.ConstantVector(T(4π) * T(Geometry.radius(grid_geometry(grid)))^2 / T(npixels(grid)),
                        npixels(grid))

# ---- materialization --------------------------------------------------------

# The whole cloud as a ring walk, costing `4·nside − 1` `acos` calls: colatitude is constant along a
# ring. The RING ordering is what makes a ring's pixels contiguous, so it is the scheme this serves; the
# generic per-cell path covers `Nested`.
function materialize(grid::HEALPixGrid{T,G,<:SphericalSampling.Ring}) where {T,G}
    n = npixels(grid)
    p = SphericalSampling.spherical_points!(
        Vector{T}(undef, n), Vector{T}(undef, n), SphericalSampling.HEALPixSampling(nside(grid)),
    )
    return (p.λ, p.φ)
end

# ---- display ----------------------------------------------------------------

Base.show(io::IO, grid::HEALPixGrid{T}) where {T} =
    print(io, "HEALPixGrid{", T, "}(nside=", nside(grid), ", ", npixels(grid), " pixels)")

function Base.show(io::IO, ::MIME"text/plain", grid::HEALPixGrid{T}) where {T}
    println(io, "HEALPixGrid{", T, "} nside=", nside(grid), " (", count(mask(grid)), "/",
            npixels(grid), " active)")
    println(io, "  geometry:  ", grid_geometry(grid))
    println(io, "  scheme:    ", scheme(grid), ", ", 4 * nside(grid) - 1, " rings")
    print(io, "  measure:   ", sum(measure(grid)), " total")
end
