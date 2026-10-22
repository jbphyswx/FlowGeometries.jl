# ---------------------------------------------------------------------------
# Ring grids: reduced Gaussian, octahedral
# ---------------------------------------------------------------------------

"""
    RingGrid(geometry, sampling::AbstractReducedGaussianSampling; mask = nothing)
    RingGrid(geometry, latitudes, nlon_per_ring, ring_area; mask = nothing)

Iso-latitude rings whose longitude count varies by ring: the reduced Gaussian family, of which the
octahedral grid is the operational case.

Storage is one latitude, one longitude count, one offset and one cell area per ring, and the ring of
every `w`-th point, `w` the shortest ring's length, which finds any point's ring in one comparison. A
point's longitude is `(j−1)·2π/nlon[r]`, and its latitude and area are its ring's.

Cells are numbered ring by ring from the north, matching
[`SphericalSampling.ring_range`](@ref) and the order `spherical_points` writes.

Coordinates are `(λ, φ)`: [`coords`](@ref)`(grid, i)` reads one point, and [`materialize`](@ref) gives
the whole cloud as dense vectors.
"""
struct RingGrid{
    T<:AbstractFloat,
    G<:Geometry.AbstractSphericalGeometry{T},
    VT<:AbstractVector{T},
    VA<:AbstractVector{T},
    VI<:AbstractVector{Int},
    VO<:AbstractVector{Int},
    B<:AbstractVector{Bool},
    Sm<:Union{Nothing,SphericalSampling.AbstractReducedGaussianSampling},
    VK<:AbstractVector{<:Integer},
} <: AbstractGrid{G,T}
    geometry::G
    latitudes::VT      # one per ring, north to south
    nlon::VI           # points in each ring
    offset::VO         # points before each ring; length nrings + 1, so offset[end] == npoints
    ring_area::VA      # cell area within each ring; every cell of a ring has the same one
    mask::B
    sampling::Sm       # the rule the rings came from, `nothing` where the tables were given
    bucket::VK         # the ring holding point `(b − 1)·width + 1`
    width::Int         # the shortest ring's length
end

function RingGrid(
    geometry::Geometry.AbstractSphericalGeometry{T}, latitudes::AbstractVector,
    nlon_per_ring::AbstractVector{<:Integer}, ring_area::AbstractVector; mask = nothing,
) where {T<:AbstractFloat}
    return _ring_grid(geometry, latitudes, nlon_per_ring, ring_area, mask, nothing)
end

function _ring_grid(
    geometry::Geometry.AbstractSphericalGeometry{T}, latitudes::AbstractVector,
    nlon_per_ring::AbstractVector{<:Integer}, ring_area::AbstractVector, mask, sampling,
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
    lat = _to_arr(T, latitudes)
    area = _to_arr(T, ring_area)
    nlon = _int_counts(nlon_per_ring)
    bucket, width = _ring_buckets(off)
    return RingGrid{T,typeof(geometry),typeof(lat),typeof(area),typeof(nlon),typeof(off),typeof(m),
                    typeof(sampling),typeof(bucket)}(geometry, lat, nlon, off, area, m, sampling, bucket, width)
end

# The caller's counts as given when they are `Int` already, converted otherwise.
_int_counts(v::AbstractVector{Int}) = v
_int_counts(v::AbstractVector{<:Integer}) = collect(Int, v)

# A Gaussian ring's cell area is `R²·(2π/nlon_r)·w_r` with `w_r` the latitude quadrature weight, which
# sums to `4πR²` because the weights sum to 2. One root solve gives the latitudes and the weights, both
# ascending, so ring `r` (from the north) is entry `nring + 1 − r`.
function RingGrid(
    geometry::Geometry.AbstractSphericalGeometry{T},
    sampling::SphericalSampling.AbstractReducedGaussianSampling; mask = nothing,
) where {T<:AbstractFloat}
    nring = SphericalSampling.nrings(sampling)
    φ = Vector{T}(undef, nring)
    w = Vector{T}(undef, nring)
    SphericalSampling._gauss_legendre!(T, nothing, w, nring; lat = φ)
    nlon = SphericalSampling.nlon_per_ring(sampling)
    R² = T(Geometry.radius(geometry))^2
    lat = reverse!(φ)
    area = [R² * (T(2π) / T(nlon[r])) * w[nring + 1 - r] for r in 1:nring]
    return _ring_grid(geometry, lat, nlon, area, mask, sampling)
end

RingGrid(sampling::SphericalSampling.AbstractReducedGaussianSampling; kwargs...) =
    RingGrid(Geometry.SphericalGeometry(), sampling; kwargs...)

@inline _from_fields(
    ::Type{<:RingGrid},
    geometry::G, latitudes::VT, nlon::VI, offset::VO, ring_area::VA, mask::B, sampling::Sm, bucket::VK,
    width::Int,
) where {T,G<:Geometry.AbstractSphericalGeometry{T},VT<:AbstractVector{T},VA<:AbstractVector{T},VI,VO,B,Sm,VK} =
    RingGrid{T,G,VT,VA,VI,VO,B,Sm,VK}(geometry, latitudes, nlon, offset, ring_area, mask, sampling, bucket,
                                       width)

@inline sampling(grid::RingGrid) = getfield(grid, :sampling)

# The ring accessors, read off the per-ring tables. Rings are counted from the north.
@inline nrings(grid::RingGrid) = length(getfield(grid, :nlon))

@inline nlon_in_ring(grid::RingGrid, r::Integer) =
    @inbounds getfield(grid, :nlon)[_check_ring(Int(r), nrings(grid))]

@inline function ring_range(grid::RingGrid, r::Integer)
    off = getfield(grid, :offset)
    k = _check_ring(Int(r), nrings(grid))
    @inbounds return (off[k] + 1):off[k + 1]
end

@inline function ring_of(grid::RingGrid, i::Integer)
    @boundscheck checkbounds(mask(grid), Int(i))
    return _ring_at(getfield(grid, :offset), getfield(grid, :bucket), getfield(grid, :width), Int(i))
end

# ---- the three traits -------------------------------------------------------

@inline cell_address(::RingGrid) = FlatCells()
@inline adjacency_source(::RingGrid) = FormulaNeighbors()
@inline candidate_source(::RingGrid) = IndexedCandidates()

# ---- coordinates ------------------------------------------------------------

@inline ncoordinates(::RingGrid) = 2

@inline function _raw_coords(grid::RingGrid{T}, i::Integer) where {T}
    r = @inbounds ring_of(grid, i)
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

@inline measure(grid::RingGrid) = RingwiseVector(getfield(grid, :ring_area), getfield(grid, :offset),
                                                  getfield(grid, :bucket), getfield(grid, :width))
@inline _total_measure(grid::RingGrid) = sum(measure(grid))

# ---- materialization --------------------------------------------------------

# A ring walk: the ring a cell belongs to is known from the loop.
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
