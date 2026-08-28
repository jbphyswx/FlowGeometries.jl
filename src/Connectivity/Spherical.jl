# Spherical *sampling* → mesh topology (CSR). Sampling places points; connectivity is
# the discrete neighbor graph for that layout. Included from Connectivity.jl.
#
# HEALPix RING neighbors follow the standard face-table algorithm (Górski et al. 2005; Reinecke 2003).

using ..SphericalSampling: SphericalSampling
using ..Geometry: Geometry

# ---------------------------------------------------------------------------
# CSR helpers
# ---------------------------------------------------------------------------

# CSR is built into contiguous storage directly — never through one heap-allocated neighbor vector
# per node, which costs `nnodes` allocations and turns every later traversal into pointer chasing.

"""
    _sort_unique_filter!(buf, lo, m, self, n) -> Int

Sort `buf[lo+1 : lo+m]` in place, drop duplicates, self-references, and out-of-range indices, and
return the surviving count (left packed at the front of the slice).
"""
@inline function _sort_unique_filter!(buf::AbstractVector{<:Integer}, lo::Int, m::Int, self::Int, n::Int)
    m == 0 && return 0
    v = view(buf, (lo + 1):(lo + m))
    # Insertion sort inline rather than `sort!`: these slices are stencil-short (≤ 8), so the generic
    # entry point's algorithm selection and scratch handling cost more than the ordering itself.
    @inbounds for a in 2:m
        x = v[a]
        b = a - 1
        while b ≥ 1 && v[b] > x
            v[b + 1] = v[b]
            b -= 1
        end
        v[b + 1] = x
    end
    w = 0
    prev = 0   # no valid neighbor is ever 0, so this can't suppress a real first entry
    @inbounds for k in 1:m
        j = v[k]
        (j == prev || j == self || j < 1 || j > n) && continue
        prev = j
        w += 1
        v[w] = j
    end
    return w
end

"""
    _csr_from_candidates(emit!, n, maxdeg) -> CSRConnectivity

Build CSR for `n` nodes whose degree is bounded by `maxdeg`. `emit!(buf, lo, i)` writes node `i`'s
candidate neighbors into `buf[lo+1 : lo+maxdeg]` and returns how many it wrote; duplicates, self and
out-of-range entries are removed here. `emit!` must touch only that slice and carry no state
between calls — nodes are emitted concurrently under a threaded `backend`.

Candidates land in one `n*maxdeg` block that is then compacted in place down to the exact CSR — one
allocation for the neighbor list, one for the offsets.
"""
function _csr_from_candidates(emit!::F, n::Integer, maxdeg::Integer; backend = nothing) where {F}
    n = Int(n); maxdeg = Int(maxdeg)
    buf = Vector{Int}(undef, n * maxdeg)
    deg = Vector{Int}(undef, n)
    # Emit and dedup in parallel: each node owns the block `[(i-1)·maxdeg, i·maxdeg)` and nothing
    # else, so `emit!` must likewise keep no state across calls.
    Execution.run_chunks(n, backend) do rng
        @inbounds for i in rng
            lo = (i - 1) * maxdeg
            m = emit!(buf, lo, i)
            deg[i] = _sort_unique_filter!(buf, lo, m, i, n)
        end
    end
    ptr = Vector{Int}(undef, n + 1)
    @inbounds ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    # Compact in place. The destination never overtakes the source: the first `i-1` degrees sum to at
    # most `(i-1)*maxdeg`, which is exactly where row `i`'s candidates start.
    #
    # This pass stays SERIAL. Row `j`'s destination can fall inside row `i`'s source block for some
    # `i < j`, so running rows concurrently would let one row's write land on another's unread
    # candidates. It is a compacting move over `nedges` entries, against an emit pass that does the
    # trigonometry and the dedup.
    @inbounds for i in 1:n
        src = (i - 1) * maxdeg
        dst = ptr[i] - 1
        d = ptr[i + 1] - ptr[i]
        if dst != src
            for k in 1:d
                buf[dst + k] = buf[src + k]
            end
        end
    end
    resize!(buf, ptr[end] - 1)
    return csr_connectivity(buf, ptr; validate = false)
end

"""
    _csr_from_undirected_edges(nnodes, edges) -> CSRConnectivity

Build CSR from an undirected edge list, by counting degrees first and then filling each node's slot
range directly — no intermediate per-node vectors, and no reallocation while filling.
"""
function _csr_from_undirected_edges(nnodes::Integer, edges)
    n = Int(nnodes)
    ptr = zeros(Int, n + 1)
    @inbounds for (a, b) in edges
        (1 ≤ a ≤ n && 1 ≤ b ≤ n && a != b) || continue
        ptr[a + 1] += 1
        ptr[b + 1] += 1
    end
    @inbounds ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] += ptr[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    cursor = copy(ptr)
    @inbounds for (a, b) in edges
        (1 ≤ a ≤ n && 1 ≤ b ≤ n && a != b) || continue
        nbrs[cursor[a]] = b; cursor[a] += 1
        nbrs[cursor[b]] = a; cursor[b] += 1
    end
    # An edge list may repeat an edge; dedup each row in place and re-tighten the offsets.
    write = 1
    @inbounds for i in 1:n
        lo = ptr[i] - 1
        m = ptr[i + 1] - ptr[i]
        deg = _sort_unique_filter!(nbrs, lo, m, i, n)
        for k in 1:deg
            nbrs[write + k - 1] = nbrs[lo + k]
        end
        ptr[i] = write
        write += deg
    end
    @inbounds ptr[n + 1] = write
    resize!(nbrs, write - 1)
    return csr_connectivity(nbrs, ptr; validate = false)
end

# ---------------------------------------------------------------------------
# Tensor-product samplings → StructuredGrid + CSR
# ---------------------------------------------------------------------------

"""
    structured_grid([T], sampling, nlat; geometry, nlon, mask, periodic) -> StructuredGrid

Build a spherical `StructuredGrid` from a tensor-product sampling (Clenshaw–Curtis,
Gauss–Legendre, Driscoll–Healy, McEwen–Wiaux, lat–lon, …). Longitude periodicity is
auto-detected unless `periodic` is set.

`T` is the element type to build in, and defaults to the `geometry`'s own — a geometry fixes the
width of every coordinate and metric factor computed against it. Naming `T` carries the geometry to
that width rather than letting it promote the grid back.
"""
structured_grid(
    s::SphericalSampling.AbstractTensorProductSphericalSampling, nlat::Integer;
    geometry::Geometry.AbstractSphericalGeometry = Geometry.SphericalGeometry(), kwargs...,
) = structured_grid(Geometry.float_type(geometry), s, nlat; geometry = geometry, kwargs...)

function structured_grid(
    ::Type{T},
    s::SphericalSampling.AbstractTensorProductSphericalSampling,
    nlat::Integer;
    geometry::Geometry.AbstractSphericalGeometry = Geometry.SphericalGeometry(),
    nlon::Union{Nothing,Integer} = nothing,
    mask = nothing,
    periodic = nothing,
) where {T<:AbstractFloat}
    ax = SphericalSampling.spherical_axes(T, s, nlat; nlon = nlon)
    λ = ax.λ
    φ = ax.φ
    m = mask === nothing ? Grids.AllActive((length(λ), length(φ))) : mask
    # The geometry fixes the width of every coordinate and metric factor, so a grid built at `T`
    # around a geometry of another width comes back promoted to that other width.
    return Grids.StructuredGrid(Geometry.similar_geometry(T, geometry), λ, φ, m;
                                periodic = periodic, sampling = s)
end

"""
    build_connectivity(sampling, nlat; nlon, mask, periodic, stencil, active_only)

Sampling topology straight to CSR. A tensor-product sampling's neighbor graph is fixed by its axis
LENGTHS and longitude wrapping alone, so the axes themselves are never evaluated — for
Gauss–Legendre that is an O(n²) root solve.
"""
function build_connectivity(
    s::SphericalSampling.AbstractTensorProductSphericalSampling,
    nlat::Integer;
    nlon::Union{Nothing,Integer} = nothing,
    mask = nothing,
    periodic::Union{Nothing,NTuple{2,Bool}} = nothing,
    stencil = Stencils.Axial(1),
    active_only::Bool = true,
)
    sz = SphericalSampling.axes_lengths(s, nlat; nlon = nlon)
    # These samplings tile the full circle in longitude and stop at the poles in latitude, which is
    # what `StructuredGrid`'s auto-detection concludes from the axes themselves.
    per = periodic === nothing ? (true, false) : periodic
    return build_connectivity(
        IndexTopology((sz.nlon, sz.nlat), per, mask);
        stencil = stencil, active_only = active_only,
    )
end

# ---------------------------------------------------------------------------
# Cubed sphere — six panels + gnomonic seam fold
# ---------------------------------------------------------------------------

"""
Exact face-neighbor under offset `(di,dj)` for the gnomonic cubed sphere matching
`SphericalSampling._cubed_face_to_xyz` / `cubed_sphere_points!`.

Panel interiors stay on-face. Crossing an edge uses cube face adjacency with index
maps derived by matching cube XYZ along shared edges. Diagonal (corner) exits
return `(0,0,0)` — no unique adjacent face.
"""
function _cubed_neighbor(f::Int, i::Int, j::Int, di::Int, dj::Int, n::Int)
    ii, jj = i + di, j + dj
    if 1 ≤ ii ≤ n && 1 ≤ jj ≤ n
        return f, ii, jj
    end
    (di != 0 && dj != 0) && return 0, 0, 0
    r(k) = n + 1 - k
    # Face local (i,j) → gnomonic (X,Y); cube XYZ as in `_cubed_face_to_xyz`.
    if f == 1  # +z (X,Y,1)
        jj < 1 && return 4, r(i), 1          # Y=-1 → -y, Y₄=-1
        jj > n && return 2, i, 1             # Y=+1 → +y, Y₂=-1
        ii < 1 && return 5, j, 1             # X=-1 → -x, Y₅=-1
        ii > n && return 3, r(j), 1          # X=+1 → +x, Y₃=-1
    elseif f == 2  # +y (X, 1, -Y)
        jj < 1 && return 1, i, n
        jj > n && return 6, n, r(i)          # (X,1,-1) → face6 X₆=1, Y₆=-X
        ii < 1 && return 5, n, j
        ii > n && return 3, 1, j
    elseif f == 3  # +x (1, -X, -Y)
        jj < 1 && return 1, n, r(i)
        jj > n && return 6, r(i), 1          # (1,-X,-1) → face6 Y₆=-1, X₆=-X
        ii < 1 && return 2, n, j
        ii > n && return 4, 1, j
    elseif f == 4  # -y (-X, -1, -Y)
        jj < 1 && return 1, r(i), 1
        jj > n && return 6, 1, i             # (-X,-1,-1) → face6 X₆=-1, Y₆=X
        ii < 1 && return 3, n, j
        ii > n && return 5, 1, j
    elseif f == 5  # -x (-1, X, -Y)
        jj < 1 && return 1, 1, i
        jj > n && return 6, i, n             # (-1,X,-1) → face6 Y₆=+1, X₆=X
        ii < 1 && return 4, n, j
        ii > n && return 2, 1, j
    else  # f == 6, -z (-Y, X, -1); i→X (cube y), j→Y (cube x = -Y)
        jj < 1 && return 3, r(i), n          # Y=-1 → cube x=+1 → +x top
        jj > n && return 5, i, n             # Y=+1 → cube x=-1 → -x top
        ii < 1 && return 4, j, n             # X=-1 → cube y=-1 → -y top
        ii > n && return 2, r(j), n          # X=+1 → cube y=+1 → +y top
    end
    return 0, 0, 0
end

"""
    build_connectivity(::CubedSphereSampling, n; stencil=Axial(1)) -> CSRConnectivity

Six-panel gnomonic cubed sphere with cross-face seams. Indexing matches
[`SphericalSampling.cubed_sphere_points!`](@ref).
"""
function build_connectivity(
    ::SphericalSampling.CubedSphereSampling, n::Integer;
    stencil = Stencils.Axial(1), backend = nothing,
)
    n = Int(n)
    n ≥ 1 || throw(ArgumentError("cubed-sphere n must be ≥ 1"))
    sv = _stencil_val(stencil)
    offs = _stencil_offsets(Val{2}(), sv)
    N = 6 * n * n
    nn = n
    return _csr_from_candidates(N, length(offs); backend = backend) do buf, lo, lin
        f, i, j = SphericalSampling._cubed_unlin(lin, nn)
        m = 0
        @inbounds for δ in offs
            f2, i2, j2 = _cubed_neighbor(f, i, j, δ[1], δ[2], nn)
            f2 == 0 && continue
            m += 1
            buf[lo + m] = SphericalSampling._cubed_lin(f2, i2, j2, nn)
        end
        return m
    end
end

# ---------------------------------------------------------------------------
# Yin–Yang — two non-periodic panels (no cross-panel edges)
# ---------------------------------------------------------------------------

"""
    build_connectivity(::YinYangSampling, nlon, nlat; stencil=Axial(1)) -> CSRConnectivity

Panel-local face/vertex stencils on yin then yang. Global ordering matches
[`SphericalSampling.spherical_points!`](@ref) for Yin–Yang: yin (`nlon×nlat`, lon
fastest), then yang with the same panel indexing. Overlap is *not* cross-linked —
that is the standard Yin–Yang discrete topology (panels couple through
interpolation, not shared mesh edges).
"""
function build_connectivity(
    ::SphericalSampling.YinYangSampling, nlon::Integer, nlat::Integer;
    stencil = Stencils.Axial(1), backend = nothing,
)
    nlon = Int(nlon); nlat = Int(nlat)
    nlon ≥ 1 && nlat ≥ 1 || throw(ArgumentError("Yin–Yang nlon/nlat must be ≥ 1"))
    sv = _stencil_val(stencil)
    offs = _stencil_offsets(Val{2}(), sv)
    npanel = nlon * nlat
    N = 2 * npanel
    sz = (nlon, nlat)
    return _csr_from_candidates(N, length(offs); backend = backend) do buf, lo, lin
        base = ((lin - 1) ÷ npanel) * npanel
        p = lin - base - 1
        j, i = divrem(p, nlon)
        i += 1; j += 1
        m = 0
        @inbounds for δ in offs
            ii = _wrap_or_clip(i, δ[1], nlon, false)
            jj = _wrap_or_clip(j, δ[2], nlat, false)
            (ii == 0 || jj == 0) && continue
            m += 1
            buf[lo + m] = base + _linidx(sz, ii, jj)
        end
        return m
    end
end

# ---------------------------------------------------------------------------
# HEALPix RING neighbors (face-table algorithm)
# ---------------------------------------------------------------------------

const _HP_NB_X = (-1, -1, 0, 1, 1, 1, 0, -1)
const _HP_NB_Y = (0, 1, 1, 1, 0, -1, -1, -1)

# Order in which to WALK the eight offsets above. RING indices increase north-to-south and, within a
# ring, with φ — so visiting the compass directions in this order emits ascending pixel ids. Measured
# at nside = 32: this single order already sorts 10,684 of the 10,800 interior pixels. The rest are
# the ring seam, where φ wraps inside a ring and the ids jump; `_sort_unique_filter!` still runs and
# still guarantees the result, it just has almost nothing left to move (9.01 inversions per node
# before, 0.04 after).
const _HP_NB_ORDER = (4, 3, 5, 2, 6, 1, 7, 8)
# nb_facearray[nbnum+1][face+1]; nbnum in 0:8 → rows S,SE,E,SW,center,NE,W,NW,N
const _HP_NB_FACE = (
    (8, 9, 10, 11, -1, -1, -1, -1, 10, 11, 8, 9),
    (5, 6, 7, 4, 8, 9, 10, 11, 9, 10, 11, 8),
    (-1, -1, -1, -1, 5, 6, 7, 4, -1, -1, -1, -1),
    (4, 5, 6, 7, 11, 8, 9, 10, 11, 8, 9, 10),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    (1, 2, 3, 0, 0, 1, 2, 3, 5, 6, 7, 4),
    (-1, -1, -1, -1, 7, 4, 5, 6, -1, -1, -1, -1),
    (3, 0, 1, 2, 3, 0, 1, 2, 4, 5, 6, 7),
    (2, 3, 0, 1, -1, -1, -1, -1, 0, 1, 2, 3),
)
# nb_swaparray[nbnum+1][face÷4 + 1]
const _HP_NB_SWAP = (
    (0, 0, 3),
    (0, 0, 6),
    (0, 0, 0),
    (0, 0, 5),
    (0, 0, 0),
    (5, 0, 0),
    (0, 0, 0),
    (6, 0, 0),
    (3, 0, 0),
)

"""
    _fold_healpix_neighbors(f, acc, nside, ipix0) -> acc

Thread `acc = f(acc, ipix)` over the RING-scheme topological neighbours of 0-based pixel `ipix0`, in the
order SW, W, NW, N, NE, E, SE, S, skipping the ones that do not exist at the eight singular pixels.

The walk itself, so the buffer form and the tuple form are the same arithmetic rather than two copies of
it. The accumulator is threaded as a value, which is what lets the tuple form build its result on the
stack.
"""
@inline function _fold_healpix_neighbors(f::F, acc, nside::Integer, ipix0::Integer) where {F}
    nside = Int(nside)
    pix = Int(ipix0)
    npix = 12 * nside * nside
    (0 ≤ pix < npix) || throw(ArgumentError("HEALPix pixel $pix out of range 0:$(npix - 1)"))
    ix, iy, face_num = SphericalSampling._hp_ring2xyf(nside, pix)
    nsm1 = nside - 1
    if (ix > 0) && (ix < nsm1) && (iy > 0) && (iy < nsm1)
        @inbounds for m in _HP_NB_ORDER
            acc = f(acc, SphericalSampling._hp_xyf2ring(nside, ix + _HP_NB_X[m], iy + _HP_NB_Y[m],
                                                        face_num))
        end
        return acc
    end
    @inbounds for i in _HP_NB_ORDER
        x = ix + _HP_NB_X[i]
        y = iy + _HP_NB_Y[i]
        nbnum = 4
        if x < 0
            x += nside; nbnum -= 1
        elseif x ≥ nside
            x -= nside; nbnum += 1
        end
        if y < 0
            y += nside; nbnum -= 3
        elseif y ≥ nside
            y -= nside; nbnum += 3
        end
        f_ = _HP_NB_FACE[nbnum + 1][face_num + 1]
        if f_ ≥ 0
            bits = _HP_NB_SWAP[nbnum + 1][(face_num >> 2) + 1]
            (bits & 1) != 0 && (x = nside - x - 1)
            (bits & 2) != 0 && (y = nside - y - 1)
            if (bits & 4) != 0
                x, y = y, x
            end
            acc = f(acc, SphericalSampling._hp_xyf2ring(nside, x, y, f_))
        end
    end
    return acc
end

"""
    healpix_neighbors!(out, nside, ipix0) -> n_written

RING-scheme topological neighbors of 0-based pixel `ipix0`. Writes up to 8 0-based neighbor indices
into `out`, skipping the neighbors that do not exist at the eight singular pixels.
Order: SW, W, NW, N, NE, E, SE, S.
"""
function healpix_neighbors!(out::AbstractVector{<:Integer}, nside::Integer, ipix0::Integer)
    length(out) ≥ 8 || throw(ArgumentError("out must have length ≥ 8"))
    return _fold_healpix_neighbors(0, nside, ipix0) do k, ipix
        m = k + 1
        @inbounds out[m] = ipix
        return m
    end
end

"""
    healpix_neighbor_ids(nside, ipix0) -> (NTuple{8,Int}, n)

[`healpix_neighbors!`](@ref) into a stack tuple: the `n` neighbours in the first `n` entries, still
0-based. Nothing is allocated, so a traversal over every pixel holds no buffer.
"""
@inline function healpix_neighbor_ids(nside::Integer, ipix0::Integer)
    return _fold_healpix_neighbors((ntuple(_ -> 0, Val(8)), 0), nside, ipix0) do (ids, k), ipix
        return (Base.setindex(ids, ipix, k + 1), k + 1)
    end
end

function healpix_neighbors(nside::Integer, ipix0::Integer)
    buf = Vector{Int}(undef, 8)
    n = healpix_neighbors!(buf, nside, ipix0)
    return resize!(buf, n)
end

"""
    build_connectivity(s::HEALPixSampling) -> CSRConnectivity

HEALPix RING topological adjacency (usually 8 neighbors; 7 or 6 at singular pixels).
Julia node indices are 1-based (`pixel 0` → node `1`).
"""
function build_connectivity(s::SphericalSampling.HEALPixSampling; backend = nothing, _...)
    nside = s.nside
    npix = SphericalSampling.healpix_npix(nside)
    return _csr_from_candidates(npix, 8; backend = backend) do buf, lo, node
        # Straight into this node's own slice — no scratch shared between calls, so the emit pass
        # parallelizes.
        slice = view(buf, (lo + 1):(lo + 8))
        nn = healpix_neighbors!(slice, nside, node - 1)
        @inbounds for k in 1:nn
            slice[k] += 1          # HEALPix pixel ids are 0-based; node ids here are 1-based
        end
        return nn
    end
end

# ---------------------------------------------------------------------------
# Icosahedral geodesic edges
# ---------------------------------------------------------------------------

"""
    build_connectivity(s::IcosahedralSampling) -> CSRConnectivity

Undirected edges of the frequency-`ν` geodesic triangulation (same vertex set as
[`SphericalSampling.icosahedral_vertices`](@ref)).
"""
function build_connectivity(s::SphericalSampling.IcosahedralSampling; _...)
    mesh = SphericalSampling.icosahedral_mesh(s.frequency)
    return _csr_from_undirected_edges(length(mesh.λ), mesh.edges)
end

# ---------------------------------------------------------------------------
# Unstructured grids from mesh samplings
# ---------------------------------------------------------------------------

"""
    unstructured_grid([T], sampling, args...; geometry, areas, mask) -> UnstructuredGrid

Points from `spherical_points`, exact sampling topology from `build_connectivity`, and exact cell areas
from the sampling's own tessellation. Pass `areas` to supply them instead.

`T` is the element type to build in, and defaults to the `geometry`'s own, as for
[`structured_grid`](@ref).

The spherical pixelizations are layouts rather than node sets — [`Grids.HEALPixGrid`](@ref),
[`Grids.CubedSphereGrid`](@ref), [`Grids.YinYangGrid`](@ref), [`Grids.RingGrid`](@ref) — with
coordinates, adjacency and measure arithmetic in their resolution parameters.
[`Grids.materialize`](@ref) turns one into the dense point cloud. What is left here is the genuinely
arbitrary mesh: the icosahedral geodesic, and a caller's own points.
"""
function unstructured_grid end

"""
    unstructured_grid(::AbstractScatteredSphericalSampling, λ, φ; geometry, k, radius, areas, mask)

Build an `UnstructuredGrid` on an arbitrary `(λ, φ)` point set. The points are the caller's, so there
is no resolution parameter. Adjacency comes from a k-d-tree query (`k` nearest, or everything within
a physical `radius`) and cell areas default to the spherical Voronoi dual; see
[`Grids.UnstructuredGrid`](@ref) for the extension each needs.
"""
function unstructured_grid(
    ::SphericalSampling.AbstractScatteredSphericalSampling,
    λ::AbstractVector, φ::AbstractVector;
    geometry::Geometry.AbstractGeometry = Geometry.SphericalGeometry(),
    areas = nothing,
    mask = nothing,
    k::Integer = 6,
    radius::Union{Nothing,Real} = nothing,
    periodic = nothing,
    period = nothing,
)
    n = length(λ)
    length(φ) == n || throw(DimensionMismatch("λ/φ length mismatch: $n vs $(length(φ))"))
    m = mask === nothing ? Grids.AllActive((n,)) : mask
    return Grids.UnstructuredGrid(
        geometry, λ, φ, m;
        k = k, radius = radius, areas = areas, periodic = periodic, period = period,
    )
end

unstructured_grid(
    s::SphericalSampling.IcosahedralSampling;
    geometry::Geometry.AbstractGeometry = Geometry.SphericalGeometry(), kwargs...,
) = unstructured_grid(Geometry.float_type(geometry), s; geometry = geometry, kwargs...)

function unstructured_grid(
    ::Type{T},
    s::SphericalSampling.IcosahedralSampling;
    geometry::Geometry.AbstractGeometry = Geometry.SphericalGeometry(),
    areas = nothing,
    mask = nothing,
) where {T<:AbstractFloat}
    mesh = SphericalSampling.icosahedral_mesh(T, s.frequency)
    conn = _csr_from_undirected_edges(length(mesh.λ), mesh.edges)
    geo = Geometry.similar_geometry(T, geometry)
    a = areas === nothing ?
        _icosahedral_dual_areas(geo, mesh.verts, mesh.triangles, length(mesh.λ)) : areas
    return _unstructured_from_points_conn(geo, mesh.λ, mesh.φ, conn, a; mask = mask)
end

"""
    _icosahedral_dual_areas(geometry, verts, triangles, nvert) -> Vector

Exact spherical-Voronoi dual-cell areas from the mesh's own triangulation — no convex hull, no
optional dependency.

Each triangle is divided among its three vertices by the three arcs from its circumcenter `O` to its
edge midpoints. Those arcs *are* the perpendicular bisectors of the edges — `O` is equidistant from
all three vertices and each midpoint is equidistant from its two — so vertex `a`'s share is the
spherical quadrilateral `(a, M_ab, O, M_ca)`, exactly its Voronoi cell restricted to that triangle.
The three shares tile the triangle, so accumulating over all `20ν²` triangles tiles the sphere and
the areas sum to `4πR²` identically. Ordering the cells' corners is never needed, so there is no
per-vertex `sortperm` and no incident-triangle list.
"""
function _icosahedral_dual_areas(
    geometry::Geometry.AbstractSphericalGeometry{TG},
    verts::AbstractVector{NTuple{3,TV}},
    triangles::AbstractVector{NTuple{3,Int}},
    nvert::Integer,
) where {TG<:AbstractFloat,TV<:AbstractFloat}
    T = promote_type(TG, TV)
    areas = zeros(T, Int(nvert))
    R2 = T(Geometry.radius(geometry))^2
    @inline nrm(p) = (r = sqrt(p[1]^2 + p[2]^2 + p[3]^2); (p[1] / r, p[2] / r, p[3] / r))
    @inbounds for (ia, ib, ic) in triangles
        A = verts[ia]; B = verts[ib]; C = verts[ic]
        # Circumcenter: the plane through A, B, C has a unit normal equidistant from all three.
        ux = B[1] - A[1]; uy = B[2] - A[2]; uz = B[3] - A[3]
        vx = C[1] - A[1]; vy = C[2] - A[2]; vz = C[3] - A[3]
        O = nrm((uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx))
        O[1] * A[1] + O[2] * A[2] + O[3] * A[3] < 0 && (O = (-O[1], -O[2], -O[3]))
        Mab = nrm((A[1] + B[1], A[2] + B[2], A[3] + B[3]))
        Mbc = nrm((B[1] + C[1], B[2] + C[2], B[3] + C[3]))
        Mca = nrm((C[1] + A[1], C[2] + A[2], C[3] + A[3]))
        areas[ia] += R2 * (Geometry.spherical_excess(A, Mab, O) + Geometry.spherical_excess(A, O, Mca))
        areas[ib] += R2 * (Geometry.spherical_excess(B, Mbc, O) + Geometry.spherical_excess(B, O, Mab))
        areas[ic] += R2 * (Geometry.spherical_excess(C, Mca, O) + Geometry.spherical_excess(C, O, Mbc))
    end
    return areas
end

_icosahedral_dual_areas(
    ::Geometry.AbstractCartesianGeometry{T}, ::AbstractVector, ::AbstractVector, nvert::Integer,
) where {T<:AbstractFloat} = ones(T, Int(nvert))

# `areas` is required: a node set's control volumes come from its own tessellation, and each sampling
# that reaches here computes them in closed form before calling.
function _unstructured_from_points_conn(
    geometry::Geometry.AbstractGeometry{T},
    λ::AbstractVector,
    φ::AbstractVector,
    conn::CSRConnectivity,
    areas::AbstractVector;
    mask = nothing,
) where {T<:AbstractFloat}
    n = length(λ)
    length(φ) == n || throw(DimensionMismatch("λ/φ length mismatch"))
    length(areas) == n || throw(DimensionMismatch("$(length(areas)) areas for $n points"))
    nnodes(conn) == n || throw(DimensionMismatch("connectivity nnodes=$(nnodes(conn)) ≠ npoints=$n"))
    m = mask === nothing ? Grids.AllActive((n,)) : mask
    return Grids.UnstructuredGrid(geometry, λ, φ, areas, m, conn.nbrs, conn.ptr)
end
