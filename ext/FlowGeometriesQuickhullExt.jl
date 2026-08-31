module FlowGeometriesQuickhullExt

using Quickhull: Quickhull as QH
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# Exact per-node spherical Voronoi-cell areas: the spherical Voronoi diagram is the dual of the
# convex hull of the unit-sphere embedding, so a node's cell is the spherical polygon through the
# circumcenters of its incident hull facets.
#
# Everything stays in UNIT VECTORS from hull to area. The circumcenters come out of the hull as
# directions already, so converting them to (λ, φ) only to have the area formula convert them back
# costs four transcendentals per (node, facet) pair and buys nothing.

@inline function _circumcenter_direction(a::NTuple{3,T}, b::NTuple{3,T}, c::NTuple{3,T}) where {T}
    ab = (b[1] - a[1], b[2] - a[2], b[3] - a[3])
    ac = (c[1] - a[1], c[2] - a[2], c[3] - a[3])
    n = (
        ab[2] * ac[3] - ab[3] * ac[2],
        ab[3] * ac[1] - ab[1] * ac[3],
        ab[1] * ac[2] - ab[2] * ac[1],
    )
    nn = sqrt(n[1]^2 + n[2]^2 + n[3]^2)
    n = (n[1] / nn, n[2] / nn, n[3] / nn)
    s = n[1] * (a[1] + b[1] + c[1]) + n[2] * (a[2] + b[2] + c[2]) + n[3] * (a[3] + b[3] + c[3])
    return s < 0 ? (-n[1], -n[2], -n[3]) : n
end

@inline _dot3(a::NTuple{3,T}, b::NTuple{3,T}) where {T} = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

function Grids._voronoi_tessellation(
    geo::Geometry.AbstractSphericalGeometry{T}, x::AbstractVector{T}, y::AbstractVector{T},
) where {T<:AbstractFloat}
    N = length(x)
    length(y) == N || throw(DimensionMismatch("λ/φ length mismatch"))
    N ≥ 4 || throw(ArgumentError(
        "a spherical Voronoi tessellation needs at least 4 non-coplanar points (got $N)",
    ))

    # Hull input as a contiguous 3×N matrix: one allocation for the whole node set.
    pts = Matrix{T}(undef, 3, N)
    @inbounds for i in 1:N
        sinλ, cosλ = sincos(x[i])
        sinφ, cosφ = sincos(y[i])
        pts[1, i] = cosφ * cosλ
        pts[2, i] = cosφ * sinλ
        pts[3, i] = sinφ
    end
    @inline vert(i) = (@inbounds(pts[1, i]), @inbounds(pts[2, i]), @inbounds(pts[3, i]))

    hull = QH.quickhull(pts)
    fs = QH.facets(hull)
    nf = length(fs)

    # `facets` is a lazy mapped view that rebuilds a face object on EVERY element access, so the
    # three passes below would each pay for it. Materialize the vertex indices once into a plain
    # contiguous 3×nf block and read that instead.
    fv = Matrix{Int}(undef, 3, nf)
    @inbounds for (fi, f) in enumerate(fs)
        fv[1, fi] = f[1]
        fv[2, fi] = f[2]
        fv[3, fi] = f[3]
    end

    centers = Vector{NTuple{3,T}}(undef, nf)
    @inbounds for fi in 1:nf
        centers[fi] = _circumcenter_direction(vert(fv[1, fi]), vert(fv[2, fi]), vert(fv[3, fi]))
    end

    # Node → incident facets as CSR: count, scan, fill. No per-node vectors.
    ptr = zeros(Int, N + 1)
    @inbounds for fi in 1:nf, k in 1:3
        ptr[fv[k, fi] + 1] += 1
    end
    @inbounds ptr[1] = 1
    @inbounds for i in 1:N
        ptr[i + 1] += ptr[i]
    end
    adj = Vector{Int}(undef, ptr[end] - 1)
    cursor = copy(ptr)
    @inbounds for fi in 1:nf, k in 1:3
        v = fv[k, fi]
        adj[cursor[v]] = fi
        cursor[v] += 1
    end

    # Sized to the largest valence once, so the per-node sort allocates nothing.
    maxdeg = 0
    @inbounds for i in 1:N
        maxdeg = max(maxdeg, ptr[i + 1] - ptr[i])
    end
    angs = Vector{T}(undef, maxdeg)
    order = Vector{Int}(undef, maxdeg)

    areas = Vector{T}(undef, N)
    @inbounds for i in 1:N
        lo, hi = ptr[i], ptr[i + 1] - 1
        m = hi - lo + 1
        if m < 3
            areas[i] = zero(T)   # degenerate: no polygon to enclose
            continue
        end
        # Local tangent frame at the node, computed once; the incident circumcenters are ordered by
        # their azimuth in it so the polygon is traversed consistently.
        sinλ, cosλ = sincos(x[i])
        sinφ, cosφ = sincos(y[i])
        ci = (cosφ * cosλ, cosφ * sinλ, sinφ)
        êλ = (-sinλ, cosλ, zero(T))
        êφ = (-sinφ * cosλ, -sinφ * sinλ, cosφ)
        va = view(angs, 1:m)
        vo = view(order, 1:m)
        for k in 1:m
            c = centers[adj[lo + k - 1]]
            chord = (c[1] - ci[1], c[2] - ci[2], c[3] - ci[3])
            va[k] = atan(_dot3(chord, êφ), _dot3(chord, êλ))
        end
        sortperm!(vo, va)
        A = zero(T)
        for k in 1:m
            c1 = centers[adj[lo + vo[k] - 1]]
            c2 = centers[adj[lo + vo[mod1(k + 1, m)] - 1]]
            A += Geometry.spherical_excess(ci, c1, c2)
        end
        areas[i] = Geometry.radius(geo)^2 * A
    end
    # The hull's facets ARE the mesh's cells and `ptr`/`adj` is already the node→cell transpose, both
    # built above for the areas.
    #
    # The narrowest integer that indexes this node set types the IDS — node numbers in `cell_nodes`,
    # facet numbers in `adj`, both bounded by the counts they name. The two OFFSET arrays run to the
    # entry count, `3·nf ≈ 6N`, so they stay `Int`: there are `O(n)` of them against the ids' `O(n)`
    # either way, and every traversal of the mesh reads the ids.
    I = N ≤ typemax(Int32) ? Int32 : Int
    cell_ptr = Vector{Int}(undef, nf + 1)
    cell_nodes = Vector{I}(undef, 3 * nf)
    @inbounds for fi in 1:nf
        cell_ptr[fi] = 3 * (fi - 1) + 1
        cell_nodes[3 * (fi - 1) + 1] = I(fv[1, fi])
        cell_nodes[3 * (fi - 1) + 2] = I(fv[2, fi])
        cell_nodes[3 * (fi - 1) + 3] = I(fv[3, fi])
    end
    @inbounds cell_ptr[nf + 1] = 3 * nf + 1
    return areas, Grids.CellMesh(cell_ptr, cell_nodes, ptr, Vector{I}(adj))
end

end # module
