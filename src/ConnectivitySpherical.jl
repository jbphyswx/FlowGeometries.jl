# Spherical *sampling* → mesh topology (CSR). Sampling places points; connectivity is
# the discrete neighbor graph for that layout. Included from Connectivity.jl.
#
# HEALPix RING neighbors follow the Healpix_cxx algorithm (Reinecke / Górski tables).

using ..SphericalSampling: SphericalSampling
using ..Geometry: Geometry

# ---------------------------------------------------------------------------
# CSR helpers
# ---------------------------------------------------------------------------

"""
    _csr_from_adjlists(adj) -> CSRConnectivity

`adj[i]` is the neighbor list of node `i` (1-based). Lists are sorted & uniqued.
"""
function _csr_from_adjlists(adj::Vector{Vector{Int}})
    n = length(adj)
    deg = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        sort!(unique!(adj[i]))
        filter!(j -> j != i && 1 ≤ j ≤ n, adj[i])
        deg[i] = length(adj[i])
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    @inbounds for i in 1:n
        copyto!(nbrs, ptr[i], adj[i], 1, deg[i])
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

function _csr_from_undirected_edges(nnodes::Integer, edges)
    n = Int(nnodes)
    adj = [Int[] for _ in 1:n]
    @inbounds for (a, b) in edges
        (1 ≤ a ≤ n && 1 ≤ b ≤ n && a != b) || continue
        push!(adj[a], b)
        push!(adj[b], a)
    end
    return _csr_from_adjlists(adj)
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
    m = mask === nothing ? trues(length(λ), length(φ)) : mask
    return Grids.StructuredGrid(geometry, λ, φ, m; periodic = periodic)
end

function build_connectivity(
    s::SphericalSampling.AbstractTensorProductSphericalSampling,
    nlat::Integer;
    stencil::Symbol = :face,
    active_only::Bool = true,
    kwargs...,
)
    return build_connectivity(structured_grid(s, nlat; kwargs...); stencil = stencil, active_only = active_only)
end

# ---------------------------------------------------------------------------
# Cubed sphere — six panels + gnomonic seam fold
# ---------------------------------------------------------------------------

@inline function _cubed_lin(f::Int, i::Int, j::Int, n::Int)
    return (f - 1) * n * n + (j - 1) * n + i
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
    stencil::Symbol = :face,
)
    n = Int(n)
    n ≥ 1 || throw(ArgumentError("cubed-sphere n must be ≥ 1"))
    sv = _stencil_val(stencil)
    offs = _stencil_offsets(Val{2}(), sv)
    N = 6 * n * n
    adj = [Int[] for _ in 1:N]
    @inbounds for f in 1:6, j in 1:n, i in 1:n
        lin = _cubed_lin(f, i, j, n)
        for δ in offs
            f2, i2, j2 = _cubed_neighbor(f, i, j, δ[1], δ[2], n)
            f2 == 0 && continue
            push!(adj[lin], _cubed_lin(f2, i2, j2, n))
        end
    end
    return _csr_from_adjlists(adj)
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
    stencil::Symbol = :face,
)
    nlon = Int(nlon); nlat = Int(nlat)
    nlon ≥ 1 && nlat ≥ 1 || throw(ArgumentError("Yin–Yang nlon/nlat must be ≥ 1"))
    sv = _stencil_val(stencil)
    offs = _stencil_offsets(Val{2}(), sv)
    npanel = nlon * nlat
    N = 2 * npanel
    adj = [Int[] for _ in 1:N]
    sz = (nlon, nlat)
    @inbounds for panel in 0:1
        base = panel * npanel
        for j in 1:nlat, i in 1:nlon
            lin = base + _linidx(sz, i, j)
            for δ in offs
                ii = _wrap_or_clip(i, δ[1], nlon, false)
                jj = _wrap_or_clip(j, δ[2], nlat, false)
                (ii == 0 || jj == 0) && continue
                push!(adj[lin], base + _linidx(sz, ii, jj))
            end
        end
    end
    return _csr_from_adjlists(adj)
end

# ---------------------------------------------------------------------------
# HEALPix RING neighbors (Healpix_cxx tables / algorithm)
# ---------------------------------------------------------------------------

const _HP_JRLL = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
const _HP_JPLL = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)
const _HP_NB_X = (-1, -1, 0, 1, 1, 1, 0, -1)
const _HP_NB_Y = (0, 1, 1, 1, 0, -1, -1, -1)
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
    # Healpix_cxx `ncap_` for RING↔XYF (excludes the ring at iring == nside).
    # Distinct from the classic 2 nside (nside+1) count used in pix2ang.
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

RING-scheme topological neighbors of 0-based pixel `ipix0`. Writes up to 8
0-based neighbor indices into `out` (skipping missing neighbors, coded `-1` in
Healpix_cxx). Order: SW, W, NW, N, NE, E, SE, S.
"""
function healpix_neighbors!(out::AbstractVector{Int}, nside::Integer, ipix0::Integer)
    nside = Int(nside)
    pix = Int(ipix0)
    npix = 12 * nside * nside
    (0 ≤ pix < npix) || throw(ArgumentError("HEALPix pixel $pix out of range 0:$(npix - 1)"))
    length(out) ≥ 8 || throw(ArgumentError("out must have length ≥ 8"))
    ix, iy, face_num = _hp_ring2xyf(nside, pix)
    nsm1 = nside - 1
    k = 0
    if (ix > 0) && (ix < nsm1) && (iy > 0) && (iy < nsm1)
        @inbounds for m in 1:8
            k += 1
            out[k] = _hp_xyf2ring(nside, ix + _HP_NB_X[m], iy + _HP_NB_Y[m], face_num)
        end
        return k
    end
    @inbounds for i in 1:8
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
function build_connectivity(s::SphericalSampling.HEALPixSampling; _...)
    nside = s.nside
    npix = SphericalSampling.healpix_npix(nside)
    adj = [Int[] for _ in 1:npix]
    buf = Vector{Int}(undef, 8)
    @inbounds for p0 in 0:(npix - 1)
        nn = healpix_neighbors!(buf, nside, p0)
        for k in 1:nn
            push!(adj[p0 + 1], buf[k] + 1)
        end
    end
    return _csr_from_adjlists(adj)
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
    return _unstructured_from_points_conn(geometry, pts.λ, pts.φ, conn; areas = areas, mask = mask)
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
    return _unstructured_from_points_conn(geometry, pts.λ, pts.φ, conn; areas = areas, mask = mask)
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
    return _unstructured_from_points_conn(geometry, pts.λ, pts.φ, conn; areas = areas, mask = mask)
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
    return _unstructured_from_points_conn(geometry, mesh.λ, mesh.φ, conn; areas = areas, mask = mask)
end

function _unstructured_from_points_conn(
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
    m = mask === nothing ? trues(n) : mask
    a = if areas === nothing
        if geometry isa Geometry.AbstractSphericalGeometry
            fill(T(4π) * geometry.R^2 / T(n), n)
        else
            ones(T, n)
        end
    else
        areas
    end
    return Grids.UnstructuredGrid(geometry, λ, φ, a, m, conn.nbrs, conn.ptr)
end
