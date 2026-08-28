# ---------------------------------------------------------------------------
# Ring grids: reduced Gaussian, octahedral
# ---------------------------------------------------------------------------

"""
    RingGrid(geometry, sampling::AbstractReducedGaussianSampling; mask = nothing)
    RingGrid(geometry, latitudes, nlon_per_ring, ring_area; mask = nothing)

Iso-latitude rings whose longitude count varies by ring: the reduced Gaussian family, of which the
octahedral grid is the operational case.

Storage is `O(nrings) = O(√npoints)` — one latitude, one longitude count, one offset and one cell area
per ring. A point's longitude is `(j−1)·2π/nlon[r]`, and its latitude and area are its ring's, so all
three come from the ring's four numbers. At `N = 1280` that is four vectors of 2560 entries, covering
6.6 million points.

Cells are numbered ring by ring from the north, matching
[`SphericalSampling.ring_range`](@ref) and the order `spherical_points` writes.

Coordinates are `(λ, φ)`: [`coords`](@ref)`(grid, i)` reads one point, and [`materialize`](@ref) gives
the whole cloud as dense vectors.
"""
struct RingGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    VT<:AbstractVector{T},
    VI<:AbstractVector{Int},
    B<:AbstractVector{Bool},
} <: AbstractGrid{G,T}
    geometry::G
    latitudes::VT      # one per ring, north to south
    nlon::VI           # points in each ring
    offset::VI         # points BEFORE each ring; length nrings + 1, so offset[end] == npoints
    ring_area::VT      # cell area within each ring; every cell of a ring has the same one
    mask::B
end

function RingGrid(
    geometry::Geometry.AbstractSphericalGeometry{T}, latitudes::AbstractVector,
    nlon_per_ring::AbstractVector{<:Integer}, ring_area::AbstractVector; mask = nothing,
) where {T<:AbstractFloat}
    nring = length(latitudes)
    length(nlon_per_ring) == nring || throw(DimensionMismatch(
        "$(length(nlon_per_ring)) longitude counts for $nring latitudes",
    ))
    length(ring_area) == nring || throw(DimensionMismatch(
        "$(length(ring_area)) ring areas for $nring latitudes",
    ))
    all(>(0), nlon_per_ring) || throw(ArgumentError("every ring needs at least one point"))
    off = Vector{Int}(undef, nring + 1)
    off[1] = 0
    @inbounds for r in 1:nring
        off[r + 1] = off[r] + Int(nlon_per_ring[r])
    end
    n = off[end]
    m = mask === nothing ? AllActive((n,)) : mask
    length(m) == n || throw(DimensionMismatch("mask holds $(length(m)) entries for $n points"))
    lat = collect(T, latitudes)
    area = collect(T, ring_area)
    nlon = collect(Int, nlon_per_ring)
    return RingGrid{T,typeof(geometry),typeof(lat),typeof(nlon),typeof(m)}(
        geometry, lat, nlon, off, area, m,
    )
end

# A Gaussian ring's cell area is `R²·(2π/nlon_r)·w_r` with `w_r` the latitude quadrature weight, which
# sums to `4πR²` because the weights sum to 2.
function RingGrid(
    geometry::Geometry.AbstractSphericalGeometry{T},
    sampling::SphericalSampling.AbstractReducedGaussianSampling; mask = nothing,
) where {T<:AbstractFloat}
    lat = SphericalSampling.ring_latitudes(T, sampling)
    nlon = SphericalSampling.nlon_per_ring(sampling)
    w = SphericalSampling.latitude_weights(T, sampling)
    R² = T(Geometry.radius(geometry))^2
    area = [R² * (T(2π) / T(nlon[r])) * w[r] for r in eachindex(nlon)]
    return RingGrid(geometry, lat, nlon, area; mask = mask)
end

RingGrid(sampling::SphericalSampling.AbstractReducedGaussianSampling; kwargs...) =
    RingGrid(Geometry.SphericalGeometry(), sampling; kwargs...)

@inline _from_fields(
    geometry::G, latitudes::VT, nlon::VI, offset::VI, ring_area::VT, mask::B,
) where {T,G<:Geometry.AbstractSphericalGeometry{T},VT<:AbstractVector{T},VI,B} =
    RingGrid{T,G,VT,VI,B}(geometry, latitudes, nlon, offset, ring_area, mask)

"""
    nrings(grid::RingGrid) -> Int

How many iso-latitude rings the grid has.
"""
@inline nrings(grid::RingGrid) = length(getfield(grid, :nlon))

"""
    nlon_in_ring(grid::RingGrid, r::Integer) -> Int

Points in ring `r`, counted from the north.
"""
@inline nlon_in_ring(grid::RingGrid, r::Integer) = @inbounds getfield(grid, :nlon)[r]

"""
    ring_range(grid::RingGrid, r::Integer) -> UnitRange{Int}

The cells of ring `r`, as a contiguous slice of the flattened numbering. `O(1)`.
"""
@inline function ring_range(grid::RingGrid, r::Integer)
    off = getfield(grid, :offset)
    @inbounds return (off[r] + 1):off[r + 1]
end

"""
    ring_of(grid::RingGrid, i::Integer) -> Int

Which ring cell `i` belongs to.

`O(log nrings)` by bisection of the cumulative counts. A closed form exists for the octahedral rule
alone; a tabulated reduced grid's counts are arbitrary, so the search is what serves both.
"""
@inline ring_of(grid::RingGrid, i::Integer) =
    searchsortedlast(getfield(grid, :offset), Int(i) - 1)

# ---- the three traits -------------------------------------------------------

@inline cell_address(::RingGrid) = FlatCells()
@inline adjacency_source(::RingGrid) = FormulaNeighbors()
@inline candidate_source(::RingGrid) = IndexedCandidates()

# ---- coordinates ------------------------------------------------------------

@inline ncoordinates(::RingGrid) = 2

@inline function _raw_coords(grid::RingGrid{T}, i::Integer) where {T}
    r = ring_of(grid, i)
    @inbounds off = getfield(grid, :offset)[r]
    @inbounds m = getfield(grid, :nlon)[r]
    j = Int(i) - off                                  # 1-based position within the ring
    @inbounds return (T(j - 1) * (T(2π) / T(m)), getfield(grid, :latitudes)[r])
end

coordinates(grid::RingGrid) = throw(ArgumentError(
    "a RingGrid stores no coordinate arrays — longitude is `(j-1)·2π/nlon[r]` and latitude is the " *
    "ring's. Use `coords(grid, i)` for one point, or `Grids.materialize(grid)` for the whole cloud.",
))

# ---- topology and span ------------------------------------------------------

@inline topology(::RingGrid) = (Periodic(), Bounded())
@inline period(grid::RingGrid{T}, d::Integer) where {T} =
    _checked_direction((1, 2), d) == 1 ? T(2π) : zero(T)

@inline function bounds(grid::RingGrid{T}, d::Integer) where {T}
    _checked_direction((1, 2), d) == 1 && return (zero(T), T(2π))
    lat = getfield(grid, :latitudes)
    @inbounds return (min(lat[1], lat[end]), max(lat[1], lat[end]))   # monotone, north to south
end

@inline origin(grid::RingGrid, d::Integer) = bounds(grid, d)[1]

# ---- measure ----------------------------------------------------------------

@inline measure(grid::RingGrid) =
    RingwiseVector(getfield(grid, :ring_area), getfield(grid, :offset))

# ---- materialization --------------------------------------------------------

# A ring walk rather than the generic per-cell one: the ring a cell belongs to is known from the loop,
# so no cell pays the `O(log nrings)` bisection `ring_of` does.
function materialize(grid::RingGrid{T}) where {T}
    n = length(grid)
    λ = Vector{T}(undef, n)
    φ = Vector{T}(undef, n)
    lat = getfield(grid, :latitudes)
    @inbounds for r in 1:nrings(grid)
        m = nlon_in_ring(grid, r)
        Δλ = T(2π) / T(m)
        φr = lat[r]
        for (j, k) in enumerate(ring_range(grid, r))
            λ[k] = T(j - 1) * Δλ
            φ[k] = φr
        end
    end
    return (λ, φ)
end

# ---- display ----------------------------------------------------------------

Base.show(io::IO, grid::RingGrid{T}) where {T} =
    print(io, "RingGrid{", T, "}(", nrings(grid), " rings, ", length(grid), " points)")

function Base.show(io::IO, ::MIME"text/plain", grid::RingGrid{T}) where {T}
    nl = getfield(grid, :nlon)
    println(io, "RingGrid{", T, "} ", nrings(grid), " rings (", count(mask(grid)), "/",
            length(grid), " active)")
    println(io, "  geometry:  ", grid_geometry(grid))
    println(io, "  nlon:      ", minimum(nl), " … ", maximum(nl), " per ring")
    print(io, "  measure:   ", sum(measure(grid)), " total")
end
