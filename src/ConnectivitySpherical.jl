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
    structured_grid(sampling, nlat; geometry, nlon, T, mask, periodic) -> StructuredGrid

Build a spherical `StructuredGrid` from a tensor-product sampling (Clenshaw–Curtis,
Gauss–Legendre, Driscoll–Healy, McEwen–Wiaux, lat–lon, …). Longitude periodicity is
auto-detected unless `periodic` is set.
"""
function structured_grid(
    s::SphericalSampling.AbstractTensorProductSphericalSampling,
    nlat::Integer;
    geometry::Geometry.AbstractSphericalGeometry = Geometry.SphericalGeometry(),
    nlon::Union{Nothing,Integer} = nothing,
    T::Type{<:AbstractFloat} = Float64,
    mask = nothing,
    periodic = nothing,
)
    ax = SphericalSampling.spherical_axes(s, nlat; nlon = nlon, T = T)
    λ = ax.λ
    φ = ax.φ
    m = mask === nothing ? Grids.AllActive((length(λ), length(φ))) : mask
    return Grids.StructuredGrid(geometry, λ, φ, m; periodic = periodic)
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
    stencil::Symbol = :face,
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

@inline function _cubed_lin(f::Int, i::Int, j::Int, n::Int)
    return (f - 1) * n * n + (j - 1) * n + i
end

# Inverse of `_cubed_lin`.
@inline function _cubed_unlin(lin::Int, n::Int)
    q, r = divrem(lin - 1, n * n)
    j, i = divrem(r, n)
    return (q + 1, i + 1, j + 1)
end

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
    build_connectivity(::CubedSphereSampling, n; stencil=:face) -> CSRConnectivity

Six-panel gnomonic cubed sphere with cross-face seams. Indexing matches
[`SphericalSampling.cubed_sphere_points!`](@ref).
"""
function build_connectivity(
    ::SphericalSampling.CubedSphereSampling, n::Integer;
    stencil::Symbol = :face, backend = nothing,
)
    n = Int(n)
    n ≥ 1 || throw(ArgumentError("cubed-sphere n must be ≥ 1"))
    sv = _stencil_val(stencil)
    offs = _stencil_offsets(Val{2}(), sv)
    N = 6 * n * n
    nn = n
    return _csr_from_candidates(N, length(offs); backend = backend) do buf, lo, lin
        f, i, j = _cubed_unlin(lin, nn)
        m = 0
        @inbounds for δ in offs
            f2, i2, j2 = _cubed_neighbor(f, i, j, δ[1], δ[2], nn)
            f2 == 0 && continue
            m += 1
            buf[lo + m] = _cubed_lin(f2, i2, j2, nn)
        end
        return m
    end
end

# ---------------------------------------------------------------------------
# Yin–Yang — two non-periodic panels (no cross-panel edges)
# ---------------------------------------------------------------------------

"""
    build_connectivity(::YinYangSampling, nlon, nlat; stencil=:face) -> CSRConnectivity

Panel-local face/vertex stencils on yin then yang. Global ordering matches
[`SphericalSampling.spherical_points!`](@ref) for Yin–Yang: yin (`nlon×nlat`, lon
fastest), then yang with the same panel indexing. Overlap is *not* cross-linked —
that is the standard Yin–Yang discrete topology (panels couple through
interpolation, not shared mesh edges).
"""
function build_connectivity(
    ::SphericalSampling.YinYangSampling, nlon::Integer, nlat::Integer;
    stencil::Symbol = :face, backend = nothing,
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

const _HP_JRLL = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
const _HP_JPLL = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)
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

@inline _hp_special_div(a::Int, b::Int) = (t = Int(a ≥ (b << 1)); a2 = a - t * (b << 1); (t << 1) + Int(a2 ≥ b))

@inline function _hp_ncap(nside::Int)
    # Pixels in the north polar cap ABOVE the ring at iring == nside, as the RING↔XYF conversion
    # needs it. Distinct from the classic 2 nside (nside+1) cap count used for pixel centers.
    return 2 * nside * (nside - 1)
end

@inline function _hp_get_ring_info_small(nside::Int, ring::Int)
    npix = 12 * nside * nside
    ncap = _hp_ncap(nside)
    if ring < nside
        return (startpix = 2 * ring * (ring - 1), ringpix = 4 * ring, shifted = true)
    elseif ring < 3 * nside
        ringpix = 4 * nside
        return (startpix = ncap + (ring - nside) * ringpix, ringpix = ringpix, shifted = ((ring - nside) & 1) == 0)
    else
        nr = 4 * nside - ring
        return (startpix = npix - 2 * nr * (nr + 1), ringpix = 4 * nr, shifted = true)
    end
end

function _hp_ring2xyf(nside::Int, pix::Int)
    # pix 0-based RING
    ncap = _hp_ncap(nside)
    npix = 12 * nside * nside
    nl2 = 2 * nside
    iring = 0
    iphi = 0
    kshift = 0
    nr = 0
    face_num = 0
    if pix < ncap
        iring = (1 + isqrt(1 + 2 * pix)) >> 1
        iphi = (pix + 1) - 2 * iring * (iring - 1)
        kshift = 0
        nr = iring
        face_num = _hp_special_div(iphi - 1, nr)
    elseif pix < (npix - ncap)
        ip = pix - ncap
        tmp = ip ÷ (4 * nside)
        iring = tmp + nside
        iphi = ip - tmp * 4 * nside + 1
        kshift = (iring + nside) & 1
        nr = nside
        ire = tmp + 1
        irm = nl2 + 1 - tmp
        ifm = (iphi - (ire >> 1) + nside - 1) ÷ nside
        ifp = (iphi - (irm >> 1) + nside - 1) ÷ nside
        face_num = (ifp == ifm) ? (ifp | 4) : ((ifp < ifm) ? ifp : (ifm + 8))
    else
        ip = npix - pix
        iring = (1 + isqrt(2 * ip - 1)) >> 1
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        kshift = 0
        nr = iring
        iring = 2 * nl2 - iring
        face_num = _hp_special_div(iphi - 1, nr) + 8
    end
    irt = iring - ((2 + (face_num >> 2)) * nside) + 1
    ipt = 2 * iphi - _HP_JPLL[face_num + 1] * nr - kshift - 1
    ipt ≥ nl2 && (ipt -= 8 * nside)
    ix = (ipt - irt) >> 1
    iy = (-ipt - irt) >> 1
    return ix, iy, face_num
end

function _hp_xyf2ring(nside::Int, ix::Int, iy::Int, face_num::Int)
    nl4 = 4 * nside
    jr = (_HP_JRLL[face_num + 1] * nside) - ix - iy - 1
    info = _hp_get_ring_info_small(nside, jr)
    nr = info.ringpix >> 2
    kshift = 1 - Int(info.shifted)
    jp = (_HP_JPLL[face_num + 1] * nr + ix - iy + 1 + kshift) ÷ 2
    jp < 1 && (jp += nl4)
    return info.startpix + jp - 1
end

"""
    healpix_neighbors!(out, nside, ipix0) -> n_written

RING-scheme topological neighbors of 0-based pixel `ipix0`. Writes up to 8 0-based neighbor indices
into `out`, skipping the neighbors that do not exist at the eight singular pixels.
Order: SW, W, NW, N, NE, E, SE, S.
"""
function healpix_neighbors!(out::AbstractVector{<:Integer}, nside::Integer, ipix0::Integer)
    nside = Int(nside)
    pix = Int(ipix0)
    npix = 12 * nside * nside
    (0 ≤ pix < npix) || throw(ArgumentError("HEALPix pixel $pix out of range 0:$(npix - 1)"))
    length(out) ≥ 8 || throw(ArgumentError("out must have length ≥ 8"))
    ix, iy, face_num = _hp_ring2xyf(nside, pix)
    nsm1 = nside - 1
    k = 0
    if (ix > 0) && (ix < nsm1) && (iy > 0) && (iy < nsm1)
        @inbounds for m in _HP_NB_ORDER
            k += 1
            out[k] = _hp_xyf2ring(nside, ix + _HP_NB_X[m], iy + _HP_NB_Y[m], face_num)
        end
        return k
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
        f = _HP_NB_FACE[nbnum + 1][face_num + 1]
        if f ≥ 0
            bits = _HP_NB_SWAP[nbnum + 1][(face_num >> 2) + 1]
            (bits & 1) != 0 && (x = nside - x - 1)
            (bits & 2) != 0 && (y = nside - y - 1)
            if (bits & 4) != 0
                x, y = y, x
            end
            k += 1
            out[k] = _hp_xyf2ring(nside, x, y, f)
        end
    end
    return k
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
    unstructured_grid(sampling, args...; geometry, T, areas, mask) -> UnstructuredGrid

Points from `spherical_points` plus exact sampling topology from `build_connectivity`.
Default cell areas are uniform (`4π R² / N` on a sphere, `1` on Cartesian).
"""
function unstructured_grid(
    s::SphericalSampling.HEALPixSampling;
    geometry::Geometry.AbstractGeometry = Geometry.SphericalGeometry(),
    T::Type{<:AbstractFloat} = Float64,
    areas = nothing,
    mask = nothing,
)
    pts = SphericalSampling.spherical_points(s; T = T)
    conn = build_connectivity(s)
    return _unstructured_from_points_conn(s, geometry, pts.λ, pts.φ, conn; areas = areas, mask = mask)
end

function unstructured_grid(
    s::SphericalSampling.CubedSphereSampling, n::Integer;
    geometry::Geometry.AbstractGeometry = Geometry.SphericalGeometry(),
    T::Type{<:AbstractFloat} = Float64,
    areas = nothing,
    mask = nothing,
)
    pts = SphericalSampling.spherical_points(s, n; T = T)
    conn = build_connectivity(s, n)
    return _unstructured_from_points_conn(s, geometry, pts.λ, pts.φ, conn; areas = areas, mask = mask)
end

function unstructured_grid(
    s::SphericalSampling.YinYangSampling, nlon::Integer, nlat::Integer;
    geometry::Geometry.AbstractGeometry = Geometry.SphericalGeometry(),
    T::Type{<:AbstractFloat} = Float64,
    areas = nothing,
    mask = nothing,
    stencil::Symbol = :face,
)
    pts = SphericalSampling.spherical_points(s, nlon, nlat; T = T)
    conn = build_connectivity(s, nlon, nlat; stencil = stencil)
    # Cell areas need the panel shape, which N = 2·nlon·nlat does not determine.
    a = areas === nothing ? _yin_yang_areas(geometry, nlon, nlat) : areas
    return _unstructured_from_points_conn(s, geometry, pts.λ, pts.φ, conn; areas = a, mask = mask)
end

function unstructured_grid(
    s::SphericalSampling.IcosahedralSampling;
    geometry::Geometry.AbstractGeometry = Geometry.SphericalGeometry(),
    T::Type{<:AbstractFloat} = Float64,
    areas = nothing,
    mask = nothing,
)
    mesh = SphericalSampling.icosahedral_mesh(s.frequency; T = T)
    conn = _csr_from_undirected_edges(length(mesh.λ), mesh.edges)
    a = areas === nothing ?
        _icosahedral_dual_areas(geometry, mesh.verts, mesh.triangles, length(mesh.λ)) : areas
    return _unstructured_from_points_conn(s, geometry, mesh.λ, mesh.φ, conn; areas = a, mask = mask)
end

"""
    _default_node_areas(sampling, geometry, λ, φ) -> Vector

Cell areas to use when the caller supplies none.

Dispatched on whether the sampling is **equal-area**, because a uniform `4πR²/N` is exact for one
family and simply wrong for the others: measured, an icosahedral geodesic's dual cells span a
min/max ratio of 0.69, so a uniform default would silently corrupt every area-weighted integral on
it. Non-equal-area samplings therefore get their true Voronoi dual areas, or — if the tessellation
extension is not loaded — the clear error `_voronoi_areas` already raises, rather than a plausible
wrong number.
"""
function _default_node_areas(
    ::SphericalSampling.AbstractEqualAreaSphericalSampling,
    geometry::Geometry.AbstractSphericalGeometry{T}, λ::AbstractVector, φ::AbstractVector,
) where {T<:AbstractFloat}
    # Equal-area by construction: every cell is exactly the sphere's area over the pixel count.
    return fill(T(4π) * geometry.R^2 / T(length(λ)), length(λ))
end

"""
    _default_node_areas(::CubedSphereSampling, geometry, λ, φ)

Exact cell areas in closed form. A cubed-sphere cell is the spherical quadrilateral cut by its own
panel coordinates, so its area is the spherical excess of the two triangles through its four corner
directions — no tessellation, no convex hull, no optional dependency. The `(n+1)²` corner directions
per panel are built once and shared by the four cells that meet at each, rather than re-derived per
cell.
"""
function _default_node_areas(
    ::SphericalSampling.CubedSphereSampling,
    geometry::Geometry.AbstractSphericalGeometry{T}, λ::AbstractVector, φ::AbstractVector,
) where {T<:AbstractFloat}
    N = length(λ)
    n2, r = divrem(N, 6)
    r == 0 || throw(DimensionMismatch("a cubed sphere has 6n² nodes; got $N"))
    n = isqrt(n2)
    n * n == n2 || throw(DimensionMismatch("a cubed sphere has 6n² nodes; got $N"))

    Δ = T(π) / 2 / T(n)
    edge = range(-T(π) / 4; step = Δ, length = n + 1)   # cell boundaries, not centres
    areas = Vector{T}(undef, N)
    corner = Matrix{NTuple{3,T}}(undef, n + 1, n + 1)
    R2 = geometry.R^2
    @inbounds for f in 1:6
        for jj in 1:(n + 1), ii in 1:(n + 1)
            p = SphericalSampling._cubed_face_to_xyz(f, tan(edge[ii]), tan(edge[jj]), T)
            s = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            corner[ii, jj] = (p.x / s, p.y / s, p.z / s)
        end
        for j in 1:n, i in 1:n
            c1 = corner[i, j]; c2 = corner[i + 1, j]
            c3 = corner[i + 1, j + 1]; c4 = corner[i, j + 1]
            areas[_cubed_lin(f, i, j, n)] =
                R2 * (Grids._tri_excess(c1, c2, c3) + Grids._tri_excess(c1, c3, c4))
        end
    end
    return areas
end

"""
    _yin_yang_areas(geometry, nlon, nlat) -> Vector

Exact cell areas in closed form. A Yin–Yang cell is a lat–lon patch in its own panel frame, so it
integrates to `R² Δλ (sin(φ+Δφ/2) - sin(φ-Δφ/2)) = R² Δλ 2sin(Δφ/2) cos φ` — independent of `λ`, and
identical on the two panels because yang is a rigid rotation of yin.

Each panel's cells then sum to exactly `√2 (3π/2) R²`, its full `[-3π/4, 3π/4] × [-π/4, π/4]` box.
The two panels **overlap by construction**, so the areas sum to `3√2 π R²` — 6.07% more than the
sphere, at every resolution. That excess is the grid's real geometry, not a discretisation error:
integrating over both panels needs a partition-of-unity weight for the shared region, which is a
modelling choice the consumer makes on top of these areas.
"""
function _yin_yang_areas(
    geometry::Geometry.AbstractSphericalGeometry{T}, nlon::Integer, nlat::Integer,
) where {T<:AbstractFloat}
    nlon = Int(nlon); nlat = Int(nlat)
    np = nlon * nlat
    Δλ = (T(3π) / 2) / T(nlon)
    Δφ = (T(π) / 2) / T(nlat)
    c = geometry.R^2 * Δλ * 2 * sin(Δφ / 2)
    areas = Vector{T}(undef, 2 * np)
    @inbounds for j in 1:nlat
        aj = c * cos(-T(π) / 4 + (T(j) - T(0.5)) * Δφ)
        base = (j - 1) * nlon
        for i in 1:nlon
            areas[base + i] = aj
            areas[np + base + i] = aj
        end
    end
    return areas
end

_yin_yang_areas(
    ::Geometry.AbstractCartesianGeometry{T}, nlon::Integer, nlat::Integer,
) where {T<:AbstractFloat} = ones(T, 2 * Int(nlon) * Int(nlat))

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
    R2 = T(geometry.R)^2
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
        areas[ia] += R2 * (Grids._tri_excess(A, Mab, O) + Grids._tri_excess(A, O, Mca))
        areas[ib] += R2 * (Grids._tri_excess(B, Mbc, O) + Grids._tri_excess(B, O, Mab))
        areas[ic] += R2 * (Grids._tri_excess(C, Mca, O) + Grids._tri_excess(C, O, Mbc))
    end
    return areas
end

_icosahedral_dual_areas(
    ::Geometry.AbstractCartesianGeometry{T}, ::AbstractVector, ::AbstractVector, nvert::Integer,
) where {T<:AbstractFloat} = ones(T, Int(nvert))

_default_node_areas(
    ::SphericalSampling.AbstractSphericalSampling,
    geometry::Geometry.AbstractSphericalGeometry{T}, λ::AbstractVector, φ::AbstractVector,
) where {T<:AbstractFloat} = Grids._voronoi_areas(geometry, λ, φ)

_default_node_areas(
    ::SphericalSampling.AbstractSphericalSampling,
    ::Geometry.AbstractCartesianGeometry{T}, λ::AbstractVector, ::AbstractVector,
) where {T<:AbstractFloat} = ones(T, length(λ))

function _unstructured_from_points_conn(
    sampling::SphericalSampling.AbstractSphericalSampling,
    geometry::Geometry.AbstractGeometry{T},
    λ::AbstractVector,
    φ::AbstractVector,
    conn::CSRConnectivity;
    areas = nothing,
    mask = nothing,
) where {T<:AbstractFloat}
    n = length(λ)
    length(φ) == n || throw(DimensionMismatch("λ/φ length mismatch"))
    nnodes(conn) == n || throw(DimensionMismatch("connectivity nnodes=$(nnodes(conn)) ≠ npoints=$n"))
    m = mask === nothing ? Grids.AllActive((n,)) : mask
    a = areas === nothing ? _default_node_areas(sampling, geometry, λ, φ) : areas
    return Grids.UnstructuredGrid(geometry, λ, φ, a, m, conn.nbrs, conn.ptr)
end
