module FlowGeometriesQuickhullExt

using Quickhull: Quickhull as QH
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# Exact per-node spherical Voronoi-cell areas: the spherical Voronoi diagram is the dual of the
# convex hull of the unit-sphere embedding, so a node's cell is the spherical polygon through the
# circumcenters of its incident hull facets.
#
# Each facet's circumcenter is held as its offset `O − v` from each of the facet's three nodes, formed
# from the chords between the nodes' coordinates, so the fan of excesses around a node keeps the
# relative precision of `T` however small its cell.

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

    # `facets` is a lazy mapped view that rebuilds a face object on each element access, and three
    # passes below read it. Materialize the vertex indices once into a contiguous 3×nf block.
    fv = Matrix{Int}(undef, 3, nf)
    @inbounds for (fi, f) in enumerate(fs)
        fv[1, fi] = f[1]
        fv[2, fi] = f[2]
        fv[3, fi] = f[3]
    end

    offsets = Vector{NTuple{3,NTuple{3,T}}}(undef, nf)
    @inbounds for fi in 1:nf
        ia, ib, ic = fv[1, fi], fv[2, fi], fv[3, fi]
        pa = (x[ia], y[ia])
        u = Geometry._lonlat_chord(pa, (x[ib], y[ib]))
        v = Geometry._lonlat_chord(pa, (x[ic], y[ic]))
        oa = Geometry._circumcenter_offset(vert(ia), u, v)
        offsets[fi] = (oa, map(-, oa, u), map(-, oa, v))
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
    offs = Vector{NTuple{3,T}}(undef, maxdeg)

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
        for k in 1:m
            fi = adj[lo + k - 1]
            slot = fv[1, fi] == i ? 1 : (fv[2, fi] == i ? 2 : 3)
            o = offsets[fi][slot]
            offs[k] = o
            angs[k] = atan(Geometry._dot3(o, êφ), Geometry._dot3(o, êλ))
        end
        vo = view(order, 1:m)
        sortperm!(vo, view(angs, 1:m))
        A = zero(T)
        for k in 1:m
            o1 = offs[vo[k]]
            o2 = offs[vo[mod1(k + 1, m)]]
            A += Geometry._excess(ci, map(+, ci, o1), map(+, ci, o2), o1, o2)
        end
        areas[i] = Geometry.radius(geo)^2 * A
    end
    # The hull's facets are the mesh's cells and `ptr`/`adj` is already the node→cell transpose, both
    # built above for the areas.
    #
    # The narrowest integer that indexes this node set types the ids — node numbers in `cell_nodes`,
    # facet numbers in `adj`, both bounded by the counts they name. The two offset arrays run to the
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
