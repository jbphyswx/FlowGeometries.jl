module FlowGeometriesQuickhullExt

using Quickhull: Quickhull as QH
using LinearAlgebra: LinearAlgebra as LA
using FlowGeometries.Geometry: Geometry
using FlowGeometries.Grids: Grids

# Exact per-node spherical Voronoi-cell areas via 3D convex hull of the unit-sphere embedding.

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

@inline function _dir_to_lonlat(v::NTuple{3,T}) where {T}
    return (; λ = atan(v[2], v[1]), φ = asin(clamp(v[3], -one(T), one(T))))
end

function Grids._voronoi_areas(
    geo::Geometry.AbstractSphericalGeometry{T}, x::AbstractVector{T}, y::AbstractVector{T},
) where {T<:AbstractFloat}
    N = length(x)
    pts = [
        NTuple{3,T}(cos(y[i]) * cos(x[i]), cos(y[i]) * sin(x[i]), sin(y[i]))
        for i in eachindex(x)
    ]
    hull = QH.quickhull(pts)
    fs = QH.facets(hull)
    nf = length(fs)

    centers = Vector{NTuple{3,T}}(undef, nf)
    @inbounds for (fi, f) in enumerate(fs)
        centers[fi] = _circumcenter_direction(pts[f[1]], pts[f[2]], pts[f[3]])
    end

    deg = zeros(Int, N)
    @inbounds for f in fs, v in f
        deg[v] += 1
    end
    ptr = Vector{Int}(undef, N + 1)
    ptr[1] = 1
    @inbounds for i in 1:N
        ptr[i + 1] = ptr[i] + deg[i]
    end
    adj = Vector{Int}(undef, ptr[end] - 1)
    cursor = copy(ptr[1:(end - 1)])
    @inbounds for (fi, f) in enumerate(fs), v in f
        adj[cursor[v]] = fi
        cursor[v] += 1
    end

    areas = Vector{T}(undef, N)
    angs = Vector{T}(undef, 0)
    cidxs = Vector{Int}(undef, 0)
    @inbounds for i in 1:N
        lo, hi = ptr[i], ptr[i + 1] - 1
        m = hi - lo + 1
        if m < 3
            areas[i] = zero(T)
            continue
        end
        center_i = (x[i], y[i])
        resize!(angs, m)
        resize!(cidxs, m)
        for (k, idx) in enumerate(lo:hi)
            fi = adj[idx]
            nb = _dir_to_lonlat(centers[fi])
            d = Geometry.project_to_tangent_plane(geo, center_i, nb)
            angs[k] = atan(d.φ, d.λ)
            cidxs[k] = fi
        end
        order = sortperm(angs)
        A = zero(T)
        for k in 1:m
            v1 = _dir_to_lonlat(centers[cidxs[order[k]]])
            v2 = _dir_to_lonlat(centers[cidxs[order[mod1(k + 1, m)]]])
            A += Grids._sph_triangle_area(geo, center_i, v1, v2)
        end
        areas[i] = A
    end
    return areas
end

end # module
