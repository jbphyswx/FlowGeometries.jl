# ---- Cubed sphere -----------------------------------------------------------

"""
    cubed_sphere_points!(λ, φ, panel, n; backend=nothing) -> NamedTuple{(:λ,:φ,:panel)}

Gnomonic cubed-sphere CELL CENTRES into caller-owned buffers of length `6n²`, plus each point's panel
index. Pass `panel = nothing` when the panel id is not wanted, and it is not computed.

See [`cubed_sphere_points`](@ref) for the allocating form.
"""
function cubed_sphere_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T},
    panel::Union{Nothing,AbstractVector{<:Integer}}, n::Integer; backend = nothing,
) where {T<:AbstractFloat}
    n = Int(n)
    n ≥ 1 || throw(ArgumentError("cubed-sphere n must be ≥ 1, got $n"))
    N = 6 * n * n
    length(λ) == N && length(φ) == N || throw(DimensionMismatch("buffers must have length 6n²"))
    panel === nothing || length(panel) == N || throw(DimensionMismatch("buffers must have length 6n²"))
    # CELL CENTRES, not panel vertices: ξ_i = -π/4 + (i-½)·(π/2)/n.
    #
    # An endpoint-inclusive `range(-π/4, π/4; length=n)` puts nodes ON the panel edges, so adjacent
    # panels emit coincident points — measured, exactly 12(n-2)+16 duplicates — while
    # `_cubed_neighbor` simultaneously treats those edges as folding onto a *different* panel's
    # cells. Points and connectivity would then disagree, and any grid built from them carries
    # coincident nodes (degenerate tessellation, zero-area cells). Cell centres give 6n² genuinely
    # distinct points that match the connectivity, and make n=1 (one cell per face, at the face
    # centre) fall out of the formula instead of needing a special case.
    Δ = T(π) / 2 / T(n)
    a = range(-T(π) / 4 + Δ / 2; step = Δ, length = n)
    # Each output slot is a pure function of its own linear index, so chunks are independent.
    Execution.run_chunks(N, backend) do rng
        @inbounds for k in rng
            q, r0 = divrem(k - 1, n * n)
            j, i = divrem(r0, n)
            f = q + 1
            p = _cubed_face_to_xyz(f, tan(a[i + 1]), tan(a[j + 1]), T)
            r = sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            x = p.x / r; y = p.y / r; z = p.z / r
            θ = acos(clamp(z, -one(T), one(T)))
            ϕ = atan(y, x)
            ϕ < 0 && (ϕ += T(2π))
            λ[k] = ϕ
            φ[k] = geographic_latitude(θ)
            _put!(panel, k, f)
        end
    end
    return (; λ, φ, panel)
end

"""
    cubed_sphere_points([T = Float64], n; backend=nothing) -> NamedTuple{(:λ,:φ,:panel)}

Gnomonic cubed-sphere CELL CENTRES: `6n²` distinct points, plus each point's panel index.

Allocating wrapper around [`cubed_sphere_points!`](@ref). Use
[`spherical_points`](@ref)`(CubedSphereSampling(), n)` when the panel id is not needed.
"""
cubed_sphere_points(n::Integer; kwargs...) = cubed_sphere_points(Float64, n; kwargs...)

function cubed_sphere_points(::Type{T}, n::Integer; backend = nothing) where {T<:AbstractFloat}
    N = npoints(CubedSphereSampling(), n)
    return cubed_sphere_points!(
        Vector{T}(undef, N), Vector{T}(undef, N), Vector{Int}(undef, N), n; backend = backend,
    )
end

function spherical_points!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, ::CubedSphereSampling, n::Integer; backend = nothing,
) where {T<:AbstractFloat}
    cubed_sphere_points!(λ, φ, nothing, n; backend = backend)   # panel id is not part of this result
    return (; λ, φ)
end

spherical_points(s::CubedSphereSampling, n::Integer; kwargs...) =
    spherical_points(Float64, s, n; kwargs...)

function spherical_points(
    ::Type{T}, ::CubedSphereSampling, n::Integer; backend = nothing,
) where {T<:AbstractFloat}
    N = npoints(CubedSphereSampling(), n)
    return spherical_points!(
        Vector{T}(undef, N), Vector{T}(undef, N), CubedSphereSampling(), n; backend = backend,
    )
end

@inline function _cubed_face_to_xyz(face::Int, X::T, Y::T, ::Type{T}) where {T}
    if face == 1
        return (; x = X, y = Y, z = one(T))
    elseif face == 2
        return (; x = X, y = one(T), z = -Y)
    elseif face == 3
        return (; x = one(T), y = -X, z = -Y)
    elseif face == 4
        return (; x = -X, y = -one(T), z = -Y)
    elseif face == 5
        return (; x = -one(T), y = X, z = -Y)
    elseif face == 6
        return (; x = -Y, y = X, z = -one(T))
    else
        throw(ArgumentError("cubed-sphere face must be 1:6"))
    end
end

# ---- Yin–Yang ---------------------------------------------------------------

"""
    yin_yang_panels!(λyin, φyin, λyang, φyang, nlon, nlat) -> (; yin, yang)

The two Kageyama–Sato panels. `yin` is a pair of AXES (`nlon` and `nlat` long): in its own frame the
panel is a separable lat–lon patch. `yang` is that panel rotated onto the sphere, which is no longer
separable in global lon/lat, so it is a pair of `nlon × nlat` FIELDS — one `(λ, φ)` per cell.

The shapes differ because the geometry does, not by convention; the argument types say so.
"""
function yin_yang_panels!(
    λyin::AbstractVector{T}, φyin::AbstractVector{T},
    λyang::AbstractMatrix{T}, φyang::AbstractMatrix{T},
    nlon::Integer, nlat::Integer,
) where {T<:AbstractFloat}
    nlon = Int(nlon); nlat = Int(nlat)
    length(λyin) == nlon && length(φyin) == nlat ||
        throw(DimensionMismatch("yin axes must be nlon and nlat long"))
    size(λyang) == (nlon, nlat) && size(φyang) == (nlon, nlat) ||
        throw(DimensionMismatch("yang fields must be nlon × nlat"))
    # Cell centres, not panel edges: each node carries one cell, so the nlon×nlat cells tile
    # [-3π/4, 3π/4] × [-π/4, π/4] exactly. Sampling the endpoints instead would give the two
    # boundary columns/rows half-width cells while the connectivity still counts them whole.
    Δλ = (T(3π) / 2) / T(nlon)
    Δφ = (T(π) / 2) / T(nlat)
    @inbounds for i in 1:nlon
        λyin[i] = -T(3π) / 4 + (T(i) - T(0.5)) * Δλ
    end
    @inbounds for j in 1:nlat
        φyin[j] = -T(π) / 4 + (T(j) - T(0.5)) * Δφ
    end
    @inbounds for j in 1:nlat
        sinφ, cosφ = sincos(φyin[j])   # constant along the row; hoisted out of the longitude loop
        for i in 1:nlon
            sinλ, cosλ = sincos(λyin[i])
            X = -sinφ; Y = cosφ * cosλ; Z = -cosφ * sinλ
            θ = acos(clamp(Z, -one(T), one(T)))
            ϕ = atan(Y, X)
            ϕ < 0 && (ϕ += T(2π))
            λyang[i, j] = ϕ
            φyang[i, j] = geographic_latitude(θ)
        end
    end
    return (; yin = (; λ = λyin, φ = φyin), yang = (; λ = λyang, φ = φyang))
end

"""
    yin_yang_panels([T = Float64], nlon, nlat) -> (; yin, yang)

Allocating wrapper around [`yin_yang_panels!`](@ref).
"""
yin_yang_panels(nlon::Integer, nlat::Integer) = yin_yang_panels(Float64, nlon, nlat)

function yin_yang_panels(::Type{T}, nlon::Integer, nlat::Integer) where {T<:AbstractFloat}
    nlon = Int(nlon); nlat = Int(nlat)
    return yin_yang_panels!(
        Vector{T}(undef, nlon), Vector{T}(undef, nlat),
        Matrix{T}(undef, nlon, nlat), Matrix{T}(undef, nlon, nlat),
        nlon, nlat,
    )
end

function spherical_points!(Λ::AbstractVector{T}, Φ::AbstractVector{T}, ::YinYangSampling, nlon::Integer, nlat::Integer) where {T<:AbstractFloat}
    nlon = Int(nlon); nlat = Int(nlat)
    np = nlon * nlat
    n = 2 * np
    length(Λ) == n && length(Φ) == n || throw(DimensionMismatch("buffers must have length 2*nlon*nlat"))
    n == 0 && return (; λ = Λ, φ = Φ)
    # Yang is written straight into the second half of the outputs — exactly an nlon×nlat block in the
    # order the panel fields use — and yin's axes into the leading elements of the first half. Nothing
    # overlaps, so one call fills all four; yin is then expanded across its block in place.
    yang_λ = reshape(view(Λ, (np + 1):n), nlon, nlat)
    yang_φ = reshape(view(Φ, (np + 1):n), nlon, nlat)
    yin_yang_panels!(view(Λ, 1:nlon), view(Φ, 1:nlat), yang_λ, yang_φ, nlon, nlat)
    @inbounds for j in nlat:-1:1
        φj = Φ[j]
        base = (j - 1) * nlon
        for i in nlon:-1:1
            Λ[base + i] = Λ[i]
            Φ[base + i] = φj
        end
    end
    return (; λ = Λ, φ = Φ)
end

spherical_points(s::YinYangSampling, nlon::Integer, nlat::Integer) =
    spherical_points(Float64, s, nlon, nlat)

function spherical_points(::Type{T}, ::YinYangSampling, nlon::Integer,
                          nlat::Integer) where {T<:AbstractFloat}
    n = npoints(YinYangSampling(), nlon, nlat)
    return spherical_points!(Vector{T}(undef, n), Vector{T}(undef, n), YinYangSampling(), nlon, nlat)
end

# ---- Icosahedral ------------------------------------------------------------

function _xyz_to_lonlat!(λ::AbstractVector{T}, φ::AbstractVector{T}, verts) where {T}
    n = length(verts)
    length(λ) == n && length(φ) == n || throw(DimensionMismatch("buffers must match vertex count"))
    @inbounds for i in 1:n
        x, y, z = verts[i]
        θ = acos(clamp(T(z), -one(T), one(T)))
        ϕ = atan(T(y), T(x))
        ϕ < 0 && (ϕ += T(2π))
        λ[i] = ϕ
        φ[i] = geographic_latitude(θ)
    end
    return (; λ, φ)
end

"""
    icosahedral_mesh([T = Float64], frequency) -> (; λ, φ, edges, triangles, verts)

Geodesic vertices at frequency `ν` as both lon/lat (`λ`, `φ`) and unit vectors (`verts`), plus the
`10ν²+2` mesh's undirected edges `(i,j)` with `i < j` and its `20ν²` triangles `(i,j,k)` (1-based).
Vertex numbering is topological — corners, then macro-edge interiors, then face interiors — so it is
deterministic and every vertex is generated exactly once.
"""
icosahedral_mesh(frequency::Integer = 1; kwargs...) = icosahedral_mesh(Float64, frequency; kwargs...)

function icosahedral_mesh(
    ::Type{T}, frequency::Integer = 1; topology::Bool = true,
) where {T<:AbstractFloat}
    ν = Int(frequency)
    nexp = icosahedral_nvertices(ν)
    # Fixed combinatorial facts, so load-time constants rather than four allocations per call.
    base = _icosahedron_base(T)
    faces = _ICOSAHEDRON_FACES
    macro_edges = _ICOSAHEDRON_MACRO_EDGES      # the 30 canonical (lo, hi) corner pairs
    edge_index = _ICOSAHEDRON_EDGE_INDEX

    # Vertices are numbered by TOPOLOGY, not by hashing their coordinates: the 12 corners, then the
    # ν-1 interior points of each of the 30 macro-edges, then the (ν-1)(ν-2)/2 interior points of
    # each of the 20 faces — which sums to exactly 10ν²+2. Every vertex therefore has one owner and
    # is generated once, so no dedup dictionary, no quantized keys, and no per-vertex hashing.
    nint = ((ν - 1) * (ν - 2)) ÷ 2
    verts = Vector{NTuple{3,T}}(undef, nexp)
    @inline norm3(p) = (r = sqrt(p[1]^2 + p[2]^2 + p[3]^2); (p[1] / r, p[2] / r, p[3] / r))
    @inline lerp3(P, Q, a, b) = (a * P[1] + b * Q[1], a * P[2] + b * Q[2], a * P[3] + b * Q[3])

    @inbounds for v in 1:12
        verts[v] = base[v]
    end
    @inbounds for (e, (lo, hi)) in enumerate(macro_edges), t in 1:(ν - 1)
        verts[12 + (e - 1) * (ν - 1) + t] =
            norm3(lerp3(base[lo], base[hi], T(ν - t) / T(ν), T(t) / T(ν)))
    end
    face_base = 12 + 30 * (ν - 1)
    @inbounds for (f, (ia, ib, ic)) in enumerate(faces)
        A, B, C = base[ia], base[ib], base[ic]
        k = 0
        for i in 1:(ν - 1), j in 1:(ν - 1 - i)
            k += 1
            w = T(ν - i - j) / T(ν); u = T(i) / T(ν); v = T(j) / T(ν)
            verts[face_base + (f - 1) * nint + k] = norm3((
                w * A[1] + u * B[1] + v * C[1],
                w * A[2] + u * B[2] + v * C[2],
                w * A[3] + u * B[3] + v * C[3],
            ))
        end
    end

    # Triangles and edges from ONE walk of the three lattice directions on each face. A
    # face-boundary edge is generated by both adjacent faces, so the edge list is canonicalized and
    # deduped by sorting — a single sort of ~30ν² pairs, rather than hashing every edge into a Set.
    # Triangles need no dedup: each belongs to exactly one face. The two lattice orientations give
    # ν(ν+1)/2 upward plus ν(ν-1)/2 downward per face, i.e. 20ν² in total.
    per_face = 3 * ((ν * (ν + 1)) ÷ 2)
    edges = Vector{NTuple{2,Int}}(undef, topology ? 20 * per_face : 0)
    triangles = Vector{NTuple{3,Int}}(undef, topology ? 20 * ν * ν : 0)
    ne = 0
    nt = 0
    @inbounds topology && for (f, face) in enumerate(faces)
        for i in 0:ν, j in 0:(ν - i)
            v0 = _ico_node_id(f, face, i, j, ν, edge_index, nint, face_base)
            if i + j < ν
                vi = _ico_node_id(f, face, i + 1, j, ν, edge_index, nint, face_base)
                vj = _ico_node_id(f, face, i, j + 1, ν, edge_index, nint, face_base)
                ne += 1; edges[ne] = minmax(v0, vi)
                ne += 1; edges[ne] = minmax(v0, vj)
                nt += 1; triangles[nt] = (v0, vi, vj)
                if i + j < ν - 1
                    nt += 1
                    triangles[nt] =
                        (vi, vj, _ico_node_id(f, face, i + 1, j + 1, ν, edge_index, nint, face_base))
                end
            end
            if i ≥ 1
                ne += 1; edges[ne] = minmax(v0, _ico_node_id(f, face, i - 1, j + 1, ν, edge_index, nint, face_base))
            end
        end
    end
    if topology
        nt == 20 * ν * ν || throw(AssertionError("icosahedral triangle count $nt ≠ $(20 * ν * ν)"))
        resize!(edges, ne)
        sort!(edges)
        unique!(edges)
    end

    λ = Vector{T}(undef, nexp)
    φ = Vector{T}(undef, nexp)
    _xyz_to_lonlat!(λ, φ, verts)
    return (; λ, φ, edges, triangles, verts)
end

# The 30 canonical (lo, hi) corner pairs of the base icosahedron, in a deterministic order.
function _icosahedron_edges(faces)
    edges = NTuple{2,Int}[]
    for (a, b, c) in faces, (u, v) in ((a, b), (b, c), (c, a))
        push!(edges, minmax(u, v))
    end
    sort!(edges)
    unique!(edges)
    length(edges) == 30 || throw(ArgumentError("icosahedron edge recovery failed (got $(length(edges)))"))
    return edges
end

"""
    _ico_node_id(f, face, i, j, ν, edge_index, nint, face_base) -> Int

Global vertex id of barycentric lattice node `(i, j)` (with `i + j ≤ ν`, weights `ν-i-j`, `i`, `j`
on the face's corners `A`, `B`, `C`) of face `f`. Nodes on a corner or a macro-edge resolve to that
shared entity's id, which is what makes the two faces meeting at an edge agree without any lookup
table.
"""
@inline function _ico_node_id(
    f::Int, face::NTuple{3,Int}, i::Int, j::Int, ν::Int,
    edge_index::AbstractMatrix{Int}, nint::Int, face_base::Int,
)
    A, B, C = face
    # Corners.
    (i == 0 && j == 0) && return A
    (i == ν) && return B
    (j == ν) && return C
    # Macro-edge interiors: walk from `u` toward `v` with parameter `t`.
    if j == 0
        return _ico_edge_id(A, B, i, ν, edge_index)
    elseif i == 0
        return _ico_edge_id(A, C, j, ν, edge_index)
    elseif i + j == ν
        return _ico_edge_id(B, C, j, ν, edge_index)
    end
    # Face interior. Closed form for the position of (i, j) in the same `for ii, jj` order the
    # vertex pass used, so this stays O(1) instead of rescanning the lattice per lookup.
    k = (i - 1) * (ν - 1) - ((i - 1) * i) ÷ 2 + j
    return face_base + (f - 1) * nint + k
end

@inline function _ico_edge_id(u::Int, v::Int, t::Int, ν::Int, edge_index::AbstractMatrix{Int})
    lo, hi = minmax(u, v)
    e = @inbounds edge_index[lo, hi]
    # `t` counts from `u`; the stored interior points count from `lo`.
    s = (u == lo) ? t : (ν - t)
    return 12 + (e - 1) * (ν - 1) + s
end

"""
    _put_lonlat!(λ, φ, v, p, T)

Normalize `p` and write vertex `v`'s longitude/latitude. Top-level rather than a closure so nothing
is captured.
"""
@inline function _put_lonlat!(
    λ::AbstractVector{T}, φ::AbstractVector{T}, v::Int, p::NTuple{3,T}, ::Type{T},
) where {T<:AbstractFloat}
    r = sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3])
    x = p[1] / r; y = p[2] / r; z = p[3] / r
    θ = acos(clamp(z, -one(T), one(T)))
    ϕ = atan(y, x)
    ϕ < 0 && (ϕ += T(2π))
    @inbounds λ[v] = ϕ
    @inbounds φ[v] = geographic_latitude(θ)
    return nothing
end

"""
    icosahedral_vertices!(λ, φ, frequency = 1) -> NamedTuple{(:λ,:φ)}

Write the `10ν²+2` geodesic vertices' longitude/latitude into the caller's buffers, allocating
nothing. The numbering is [`icosahedral_mesh`](@ref)'s topological one — corners, then macro-edge
interiors, then face interiors — so each vertex is emitted at its own index with no intermediate
array and no lookup.
"""
function icosahedral_vertices!(λ::AbstractVector{T}, φ::AbstractVector{T}, frequency::Integer = 1) where {T<:AbstractFloat}
    ν = Int(frequency)
    ν ≥ 1 || throw(ArgumentError("icosahedral frequency must be ≥ 1, got $ν"))
    nexp = icosahedral_nvertices(ν)
    length(λ) == nexp && length(φ) == nexp ||
        throw(DimensionMismatch("buffers must have length 10ν²+2 = $nexp"))
    base = _icosahedron_base(T)
    @inbounds for v in 1:12
        _put_lonlat!(λ, φ, v, base[v], T)
    end
    @inbounds for (e, (lo, hi)) in enumerate(_ICOSAHEDRON_MACRO_EDGES)
        P = base[lo]; Q = base[hi]
        for t in 1:(ν - 1)
            a = T(ν - t) / T(ν); b = T(t) / T(ν)
            _put_lonlat!(λ, φ, 12 + (e - 1) * (ν - 1) + t,
                         (a * P[1] + b * Q[1], a * P[2] + b * Q[2], a * P[3] + b * Q[3]), T)
        end
    end
    nint = ((ν - 1) * (ν - 2)) ÷ 2
    face_base = 12 + 30 * (ν - 1)
    @inbounds for (f, (ia, ib, ic)) in enumerate(_ICOSAHEDRON_FACES)
        A = base[ia]; B = base[ib]; C = base[ic]
        k = 0
        for i in 1:(ν - 1), j in 1:(ν - 1 - i)
            k += 1
            w = T(ν - i - j) / T(ν); u = T(i) / T(ν); v = T(j) / T(ν)
            _put_lonlat!(λ, φ, face_base + (f - 1) * nint + k,
                         (w * A[1] + u * B[1] + v * C[1],
                          w * A[2] + u * B[2] + v * C[2],
                          w * A[3] + u * B[3] + v * C[3]), T)
        end
    end
    return (; λ, φ)
end

"""
    icosahedral_vertices([T = Float64], frequency=1) -> NamedTuple{(:λ,:φ)}

The `10ν²+2` geodesic vertices as longitude/latitude, without building the mesh topology — several
times faster than [`icosahedral_mesh`](@ref) at large `ν`, which also returns edges and triangles.
"""
icosahedral_vertices(frequency::Integer = 1) = icosahedral_vertices(Float64, frequency)

function icosahedral_vertices(::Type{T}, frequency::Integer = 1) where {T<:AbstractFloat}
    n = icosahedral_nvertices(frequency)
    return icosahedral_vertices!(Vector{T}(undef, n), Vector{T}(undef, n), frequency)
end

# The 20 faces, as index triples into the 12 base vertices in the order `icosahedral_mesh` lists them.
const _ICOSAHEDRON_FACES = (
    (1, 2, 9), (1, 2, 11), (1, 5, 6), (1, 5, 9), (1, 6, 11),
    (2, 7, 8), (2, 7, 9), (2, 8, 11), (3, 4, 10), (3, 4, 12),
    (3, 5, 6), (3, 5, 10), (3, 6, 12), (4, 7, 8), (4, 7, 10),
    (4, 8, 12), (5, 9, 10), (6, 11, 12), (7, 9, 10), (8, 11, 12),
)

"""
    _icosahedron_base(T) -> NTuple{12,NTuple{3,T}}

The 12 unit vertices of the base icosahedron, in the order `_ICOSAHEDRON_FACES` indexes them.
A tuple, so it costs no allocation. Every raw vertex is a permutation of `(0, ±1, ±φ)` and so shares
the norm `√(1+φ²)`, formed once.
"""
@inline function _icosahedron_base(::Type{T}) where {T<:AbstractFloat}
    φg = (one(T) + sqrt(T(5))) / T(2)
    r = sqrt(one(T) + φg * φg)
    z = zero(T)
    o = one(T) / r
    g = φg / r
    return (
        (z, o, g), (z, -o, g), (z, o, -g), (z, -o, -g),
        (o, g, z), (-o, g, z), (o, -g, z), (-o, -g, z),
        (g, z, o), (g, z, -o), (-g, z, o), (-g, z, -o),
    )
end

# Derived once at load rather than rebuilt, with a sort and a dedup, on every call.
const _ICOSAHEDRON_MACRO_EDGES = Tuple(_icosahedron_edges(_ICOSAHEDRON_FACES))

const _ICOSAHEDRON_EDGE_INDEX = let m = zeros(Int, 12, 12)
    for (e, (lo, hi)) in enumerate(_ICOSAHEDRON_MACRO_EDGES)
        m[lo, hi] = e
    end
    m
end

function spherical_points!(λ::AbstractVector{T}, φ::AbstractVector{T}, s::IcosahedralSampling) where {T<:AbstractFloat}
    return icosahedral_vertices!(λ, φ, s.frequency)
end

spherical_points(s::IcosahedralSampling) = spherical_points(Float64, s)

spherical_points(::Type{T}, s::IcosahedralSampling) where {T<:AbstractFloat} =
    icosahedral_vertices(T, s.frequency)

"""
    spherical_points(::AbstractScatteredSphericalSampling, λ, φ) -> NamedTuple{(:λ,:φ)}

A scattered sampling's points are the caller's arrays, so this hands them back. It exists so a
scattered set can be driven through the same entry point as a generated one.
"""
spherical_points(::AbstractScatteredSphericalSampling, λ::AbstractVector, φ::AbstractVector) =
    (; λ, φ)

"""
    spherical_points!(λ_out, φ_out, ::AbstractScatteredSphericalSampling, λ, φ) -> NamedTuple

Copy a scattered point set into caller-owned buffers.

The source arrays are required: a scattered sampling carries no rule from which points could be
generated, so a form taking only the destinations would have nothing to write.
"""
function spherical_points!(
    λ_out::AbstractVector, φ_out::AbstractVector, ::AbstractScatteredSphericalSampling,
    λ::AbstractVector, φ::AbstractVector,
)
    n = length(λ)
    length(φ) == n || throw(DimensionMismatch("λ/φ length mismatch: $n vs $(length(φ))"))
    length(λ_out) == n && length(φ_out) == n ||
        throw(DimensionMismatch("buffers must have length npoints = $n"))
    copyto!(λ_out, λ)
    copyto!(φ_out, φ)
    return (; λ = λ_out, φ = φ_out)
end

"""
    npoints(::AbstractScatteredSphericalSampling, λ, φ) -> Int

The number of points in a scattered set, i.e. `length(λ)`.
"""
function npoints(::AbstractScatteredSphericalSampling, λ::AbstractVector, φ::AbstractVector)
    length(φ) == length(λ) ||
        throw(DimensionMismatch("λ/φ length mismatch: $(length(λ)) vs $(length(φ))"))
    return length(λ)
end
