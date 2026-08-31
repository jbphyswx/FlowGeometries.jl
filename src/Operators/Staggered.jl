# ---------------------------------------------------------------------------
# Staggered differences on an Arakawa C arrangement
# ---------------------------------------------------------------------------
#
# All three operators below are the orthogonal-curvilinear forms, built from the geometry's own
# `scale_factors`, so one body serves every geometry. With `h_d` the scale factor of direction `d` and
# `J = ∏ h_d`,
#
#     (∇f)_d = (1/h_d)·∂f/∂x_d
#     ∇·u    = (1/J)·Σ_d ∂/∂x_d[ (J/h_d)·u_d ]
#     (∇×u)  = (1/J)·[ ∂(h_2 u_2)/∂x_1 − ∂(h_1 u_1)/∂x_2 ]      (two directions)
#
# On a sphere those are `(1/(R cosφ))∂/∂λ`, `(1/(R cosφ))[∂u_λ/∂λ + ∂(cosφ·u_φ)/∂φ]` and
# `(1/(R cosφ))[∂u_φ/∂λ − ∂(cosφ·u_λ)/∂φ]`; on a plane they are the plain differences. Nothing here is
# specific to either — a geometry that answers `scale_factors` gets all three.
#
# The differences are the ones the C arrangement makes exact: each is a single difference across ONE
# cell, evaluated where the result lives, with no averaging and so no dispersive error from it.

"""
    _face_span(nc, periodic) -> Int

How many samples a direction of `nc` cells carries at [`Discretization.Face`](@ref): `nc` where it wraps, its last
face being its first, and `nc + 1` where it does not.
"""
@inline _face_span(nc::Int, periodic::Bool) = periodic ? nc : nc + 1

# The centres that face `k` of a direction lies between, and whether that pair is complete. Incomplete
# only at the outer two faces of a BOUNDED direction, which have a cell on one side only.
@inline function _face_pair(k::Int, nc::Int, periodic::Bool)
    periodic && return (true, mod1(k - 1, nc), k)
    (k == 1 || k == nc + 1) && return (false, 1, 1)
    return (true, k - 1, k)
end

# The faces that cell `i` of a direction lies between. Always complete: a cell has two boundaries.
@inline _cell_faces(i::Int, nc::Int, periodic::Bool) =
    periodic ? (i, mod1(i + 1, nc)) : (i, i + 1)

@inline _period_or_nothing(g, d::Integer) = Grids.isperiodic(g, d) ? Grids.period(g, d) : nothing

# One index tuple with direction `d` replaced.
@inline _set(I::NTuple{N,Int}, d::Int, v::Int) where {N} =
    ntuple(e -> e == d ? v : I[e], Val(N))

# The coordinates of a point given one location per direction and one index per direction.
@inline _point_at(sg::Grids.StaggeredGrid{T,N}, loc::NTuple{N,Any}, I::NTuple{N,Int}) where {T,N} =
    ntuple(e -> @inbounds(Grids.axis_at(sg, e, loc[e])[I[e]]), Val(N))

# `∏_{e≠d} h_e`, which is `J/h_d` without the division — the weight a flux through a `d`-face carries.
@inline _omit_prod(h::Tuple{T,Vararg{T,M}}, d::Int) where {T,M} =
    prod(ntuple(e -> e == d ? one(T) : h[e], Val(M + 1)))

"""
    gradient!(outs, f, sg; masked = NaN) -> outs

The gradient of a centre field `f` on a [`Grids.StaggeredGrid`](@ref), component `d` written to
`outs[d]` at the `d`-face location — the Arakawa C gradient.

Each component is one difference across one cell, evaluated where it lives:
`(f[i] − f[i−1]) / (h_d · (x[i] − x[i−1]))`, with `h_d` the scale factor at the face. No averaging
enters, so the C arrangement's gradient is second order on a uniform mesh and free of the two-grid null
space a collocated difference has.

The outer two faces of a bounded direction have a cell on one side only, so no difference exists there
and they are written `masked`. That is where a boundary condition goes: pass `masked = 0` for a
zero-gradient edge, or write the edge yourself. A wrapping direction has no such face, its faces all
lying between two cells.

A point whose cells are not all active is `masked` too, on the same rule.
"""
function gradient!(
    outs::Tuple{AbstractArray{S,N},Vararg{AbstractArray{S,N}}}, f::AbstractArray,
    sg::Grids.StaggeredGrid{T,N}; masked = T(NaN),
) where {N,S,T}
    center = Grids.center_grid(sg)
    nc = Grids.size_tuple(center)
    length(outs) == N || throw(DimensionMismatch(
        "a $N-direction grid has $N gradient components; got $(length(outs)) output arrays",
    ))
    length(f) == prod(nc) || throw(DimensionMismatch(
        "field has $(length(f)) values for $(prod(nc)) centres",
    ))
    per = ntuple(d -> Grids.isperiodic(center, d), Val(N))
    geo = Grids.grid_geometry(center)
    msk = Grids.mask(center)
    mv = convert(S, masked)
    _grad_dirs(sg, geo, center, msk, outs, _shaped(f, nc), nc, per, mv, Val(N), 1)
    return outs
end

# A field already carrying the shape it is about to be read at. `reshape` builds an array header even
# where the shape matches, and on a per-direction path that header is one per call; a field of the
# right rank is handed back untouched, so the common call reaches the loop having allocated nothing.
@inline function _shaped(f::AbstractArray{<:Any,N}, dims::NTuple{N,Int}) where {N}
    size(f) == dims || throw(DimensionMismatch("field is $(size(f)) for a $(dims) location"))
    return f
end
@inline _shaped(f::AbstractArray, dims::NTuple{N,Int}) where {N} = reshape(f, dims)

# One component per direction, walked by a recursive tail-split.
@inline _grad_dirs(sg, geo, center, msk, outs, f, nc, per, mv, ::Val{N}, d::Int) where {N} =
    d > N ? nothing :
    (_grad_dir!(sg, geo, center, msk, outs[d], f, nc, per, mv, Val(N), d);
     _grad_dirs(sg, geo, center, msk, outs, f, nc, per, mv, Val(N), d + 1))

function _grad_dir!(
    sg, geo, center, msk, out::AbstractArray{S,N}, fc, nc, per, mv, ::Val{N}, d::Int,
) where {S,N}
    loc = Grids.location_at(sg, d)
    dims = ntuple(e -> length(Grids.axis_at(sg, e, loc[e])), Val(N))
    size(out) == dims || throw(DimensionMismatch(
        "component $d holds $(size(out)) values for a $(dims) face location",
    ))
    cax = Grids.coordinates(center, d)
    pd = _period_or_nothing(center, d)
    @inbounds for I in CartesianIndices(dims)
        It = Tuple(I)
        ok, lo, hi = _face_pair(It[d], nc[d], per[d])
        if !ok
            out[I] = mv
            continue
        end
        Ilo = _set(It, d, lo)
        Ihi = _set(It, d, hi)
        if !(msk[Ilo...] && msk[Ihi...])
            out[I] = mv
            continue
        end
        # `local_spacing` at the upper centre is `x[hi] − x[lo]`, wrapped where the direction is.
        Δξ, _ = Discretization.local_spacing(cax, It[d], pd)
        h = Geometry.scale_factors(geo, _point_at(sg, loc, It))
        out[I] = S((fc[Ihi...] - fc[Ilo...]) / (h[d] * Δξ))
    end
    return nothing
end

"""
    divergence!(out, us, sg; masked = NaN) -> out

The divergence of a C-staggered vector field — `us[d]` at the `d`-face location — written to `out` at
the cell centres.

`(1/J)·Σ_d [ (J/h_d)·u_d ]` differenced across the cell: the finite-volume divergence, the net flux
through the cell's own faces over its own volume. It is discretely conservative — summing it against
the cell measures telescopes, two neighbours' shared face cancelling to the bit, so a closed or
wrapping domain integrates to zero to round-off.

A centre whose bounding faces are not all active reads `masked`.
"""
function divergence!(
    out::AbstractArray{S,N}, us::NTuple{N,AbstractArray}, sg::Grids.StaggeredGrid{T,N};
    masked = T(NaN),
) where {N,S,T}
    center = Grids.center_grid(sg)
    nc = Grids.size_tuple(center)
    size(out) == nc || throw(DimensionMismatch(
        "out holds $(size(out)) values for $(nc) centres",
    ))
    per = ntuple(d -> Grids.isperiodic(center, d), Val(N))
    geo = Grids.grid_geometry(center)
    msk = Grids.mask(center)
    locs = Grids.locations(sg)
    mv = convert(S, masked)
    # Each component carries its own face location's shape, so the loop below indexes it
    # Cartesian-wise. A component already of that shape is taken as it is — see [`_shaped`](@ref).
    uf = ntuple(Val(N)) do d
        dims = ntuple(e -> length(Grids.axis_at(sg, e, locs[d][e])), Val(N))
        length(us[d]) == prod(dims) || throw(DimensionMismatch(
            "component $d holds $(length(us[d])) values for a $(dims) face location",
        ))
        return _shaped(us[d], dims)
    end
    ctr = Grids.center_location(sg)
    faxes = ntuple(d -> Grids.axis_at(sg, d, Discretization.Face()), Val(N))
    pers = ntuple(d -> _period_or_nothing(center, d), Val(N))
    @inbounds for I in CartesianIndices(nc)
        It = Tuple(I)
        if !msk[I]
            out[I] = mv
            continue
        end
        # One `(contribution, determined)` per direction, returned as a tuple, so nothing is mutated
        # from inside the unrolling closure and nothing is captured.
        terms = _div_terms(sg, geo, msk, uf, locs, faxes, pers, It, nc, per, 1)
        if all(t -> t[2], terms)
            hc = Geometry.scale_factors(geo, _point_at(sg, ctr, It))
            out[I] = S(sum(t -> t[1], terms) / prod(hc))
        else
            out[I] = mv
        end
    end
    return out
end

# The per-direction terms for one cell, walked by a recursive tail-split, the same shape `_at_axis`
# uses. An `ntuple` over a closure captures five of the loop's locals, and a capture the compiler does
# not elide is a heap object per cell.
@inline _div_terms(sg, geo, msk, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, It::Tuple, ::Tuple{},
                   ::Tuple{}, d::Int) = ()

@inline _div_terms(
    sg, geo, msk, uf::Tuple, locs::Tuple, faxes::Tuple, pers::Tuple, It::Tuple,
    nc::Tuple, per::Tuple, d::Int,
) = (
    _div_term(sg, geo, msk, uf[1], locs[1], faxes[1], pers[1], It, d, nc[1], per[1]),
    _div_terms(sg, geo, msk, Base.tail(uf), Base.tail(locs), Base.tail(faxes), Base.tail(pers),
               It, Base.tail(nc), Base.tail(per), d + 1)...,
)

# Direction `d`'s flux difference across cell `I`, and whether both its faces are determined.
@inline function _div_term(
    sg::Grids.StaggeredGrid{T,N}, geo, msk, u_d, loc_d, fax, pd, It::NTuple{N,Int},
    d::Int, nc_d::Int, per_d::Bool,
) where {T,N}
    flo, fhi = _cell_faces(It[d], nc_d, per_d)
    Ilo = _set(It, d, flo)
    Ihi = _set(It, d, fhi)
    # A face is active where every centre it is built from is — the same rule the grid's own face masks
    # use, so an operator and a mask cannot disagree about which points exist. At a bounded boundary
    # that is the one cell inside it: the flux THROUGH that face is data the caller supplies.
    alo, blo = Grids._stagger_sources(flo, nc_d, per_d)
    ahi, bhi = Grids._stagger_sources(fhi, nc_d, per_d)
    ok = @inbounds(msk[_set(It, d, alo)...] && msk[_set(It, d, blo)...] &&
                   msk[_set(It, d, ahi)...] && msk[_set(It, d, bhi)...])
    hlo = Geometry.scale_factors(geo, _point_at(sg, loc_d, Ilo))
    hhi = Geometry.scale_factors(geo, _point_at(sg, loc_d, Ihi))
    # The SIGNED face-to-face gap: on a descending axis the flux difference and the coordinate
    # difference change sign together, so the derivative does not.
    _, gap = Discretization.local_spacing(fax, flo, pd)
    val = @inbounds (_omit_prod(hhi, d) * u_d[Ihi...] - _omit_prod(hlo, d) * u_d[Ilo...]) / gap
    return (val, ok)
end

"""
    _curl_location(sg, i) -> NTuple{N,AbstractLocation}

Where `(∇×u)_i` lives: [`Discretization.Center`](@ref) in direction `i` and
[`Discretization.Face`](@ref) in every other — the vorticity point of an Arakawa C cell for that
component. In two directions the scalar curl is the `i = 3` case, giving `(Face, Face)`, the corner.
"""
@inline _curl_location(::Grids.StaggeredGrid{T,N}, i::Int) where {T,N} =
    ntuple(e -> e == i ? Discretization.Center() : Discretization.Face(), Val(N))

"""
    curl!(out, u1, u2, sg; masked = NaN) -> out
    curl!(out, us, sg; masked = NaN) -> out
    curl!(outs, us, sg; masked = NaN) -> outs

The curl of a C-staggered vector field, `us[d]` at the `d`-face location.

In **two** directions it is one scalar, written to `out` at the corner where both directions are at
[`Discretization.Face`](@ref):

    (1/J)·[ ∂(h_2 u_2)/∂x_1 − ∂(h_1 u_1)/∂x_2 ]

In **three** it is a vector, component `i` written to `outs[i]` at that component's own vorticity point
— see [`_curl_location`](@ref) — with `(i, j, k)` running cyclically:

    (∇×u)_i = (1/(h_j·h_k))·[ ∂(h_k u_k)/∂x_j − ∂(h_j u_j)/∂x_k ]

Each term is one difference across one cell. Like the divergence this is the circulation around the
cell over its area, so the circulation telescopes: the curl summed against the corner measures over a
closed or wrapping domain vanishes to round-off. Being the same differences the gradient makes,
`curl(gradient(f))` is zero to round-off as well.

A point missing either difference — the outer faces of a bounded direction — is `masked`, as is one
whose four bounding centres are not all active.
"""
function curl!(
    out::AbstractArray{S,2}, u1::AbstractArray, u2::AbstractArray,
    sg::Grids.StaggeredGrid{T,2}; masked = T(NaN),
) where {S,T}
    center = Grids.center_grid(sg)
    nc = Grids.size_tuple(center)
    per = ntuple(d -> Grids.isperiodic(center, d), Val(2))
    geo = Grids.grid_geometry(center)
    msk = Grids.mask(center)
    corner = (Discretization.Face(), Discretization.Face())
    dims = ntuple(e -> length(Grids.axis_at(sg, e, Discretization.Face())), Val(2))
    size(out) == dims || throw(DimensionMismatch(
        "out holds $(size(out)) values for a $(dims) corner location",
    ))
    l1 = Grids.location_at(sg, 1)
    l2 = Grids.location_at(sg, 2)
    d1 = ntuple(e -> length(Grids.axis_at(sg, e, l1[e])), Val(2))
    d2 = ntuple(e -> length(Grids.axis_at(sg, e, l2[e])), Val(2))
    length(u1) == prod(d1) || throw(DimensionMismatch("u1 holds $(length(u1)) values for $(d1)"))
    length(u2) == prod(d2) || throw(DimensionMismatch("u2 holds $(length(u2)) values for $(d2)"))
    f1 = _shaped(u1, d1)
    f2 = _shaped(u2, d2)
    c1 = Grids.coordinates(center, 1)
    c2 = Grids.coordinates(center, 2)
    p1 = _period_or_nothing(center, 1)
    p2 = _period_or_nothing(center, 2)
    mv = convert(S, masked)
    @inbounds for j in 1:dims[2], i in 1:dims[1]
        ok1, ilo, ihi = _face_pair(i, nc[1], per[1])
        ok2, jlo, jhi = _face_pair(j, nc[2], per[2])
        if !(ok1 && ok2)
            out[i, j] = mv
            continue
        end
        if !(msk[ilo, jlo] && msk[ihi, jlo] && msk[ilo, jhi] && msk[ihi, jhi])
            out[i, j] = mv
            continue
        end
        # `u2` sits at (centre, face): differenced along direction 1 between the two centres the
        # corner lies between. `u1` sits at (face, centre), differenced along direction 2.
        h2hi = Geometry.scale_factors(geo, _point_at(sg, l2, (ihi, j)))
        h2lo = Geometry.scale_factors(geo, _point_at(sg, l2, (ilo, j)))
        h1hi = Geometry.scale_factors(geo, _point_at(sg, l1, (i, jhi)))
        h1lo = Geometry.scale_factors(geo, _point_at(sg, l1, (i, jlo)))
        Δξ1, _ = Discretization.local_spacing(c1, i, p1)
        Δξ2, _ = Discretization.local_spacing(c2, j, p2)
        hc = Geometry.scale_factors(geo, _point_at(sg, corner, (i, j)))
        out[i, j] = S(((h2hi[2] * f2[ihi, j] - h2lo[2] * f2[ilo, j]) / Δξ1 -
                       (h1hi[1] * f1[i, jhi] - h1lo[1] * f1[i, jlo]) / Δξ2) / prod(hc))
    end
    return out
end

@inline curl!(
    out::AbstractArray{S,2}, us::NTuple{2,AbstractArray}, sg::Grids.StaggeredGrid{T,2}; kwargs...,
) where {S,T} = curl!(out, us[1], us[2], sg; kwargs...)

function curl!(
    outs::NTuple{3,AbstractArray}, us::NTuple{3,AbstractArray}, sg::Grids.StaggeredGrid{T,3};
    masked = T(NaN),
) where {T}
    center = Grids.center_grid(sg)
    nc = Grids.size_tuple(center)
    per = ntuple(d -> Grids.isperiodic(center, d), Val(3))
    geo = Grids.grid_geometry(center)
    msk = Grids.mask(center)
    locs = Grids.locations(sg)
    caxes = ntuple(d -> Grids.coordinates(center, d), Val(3))
    pers = ntuple(d -> _period_or_nothing(center, d), Val(3))
    # Each component carries its own face location's shape — see [`_shaped`](@ref).
    uf = ntuple(Val(3)) do d
        dims = ntuple(e -> length(Grids.axis_at(sg, e, locs[d][e])), Val(3))
        length(us[d]) == prod(dims) || throw(DimensionMismatch(
            "component $d holds $(length(us[d])) values for a $(dims) face location",
        ))
        return _shaped(us[d], dims)
    end
    _curl_dirs(sg, geo, msk, outs, uf, locs, caxes, pers, nc, per, masked, 1)
    return outs
end

# One component per direction, walked by a recursive tail-split, the same shape `_grad_dirs` uses.
@inline _curl_dirs(sg, geo, msk, outs, uf, locs, caxes, pers, nc, per, masked, i::Int) =
    i > 3 ? nothing :
    (_curl_dir!(sg, geo, msk, outs[i], uf, locs, caxes, pers, nc, per, masked,
                i, mod1(i + 1, 3), mod1(i + 2, 3));
     _curl_dirs(sg, geo, msk, outs, uf, locs, caxes, pers, nc, per, masked, i + 1))

function _curl_dir!(
    sg::Grids.StaggeredGrid{T,3}, geo, msk, out::AbstractArray{S,3}, uf, locs, caxes, pers, nc, per,
    masked, i::Int, j::Int, k::Int,
) where {S,T}
    loc = _curl_location(sg, i)
    dims = ntuple(e -> length(Grids.axis_at(sg, e, loc[e])), Val(3))
    size(out) == dims || throw(DimensionMismatch(
        "component $i holds $(size(out)) values for a $(dims) vorticity location",
    ))
    mv = convert(S, masked)
    @inbounds for I in CartesianIndices(dims)
        It = Tuple(I)
        okj, jlo, jhi = _face_pair(It[j], nc[j], per[j])
        okk, klo, khi = _face_pair(It[k], nc[k], per[k])
        if !(okj && okk)
            out[I] = mv
            continue
        end
        # The four centres the two differences are built from: the `(j, k)` plaquette this component's
        # point sits in the middle of, at its own direction-`i` centre.
        Jlo = _set(It, j, jlo)
        Jhi = _set(It, j, jhi)
        if !(msk[_set(Jlo, k, klo)...] && msk[_set(Jhi, k, klo)...] &&
             msk[_set(Jlo, k, khi)...] && msk[_set(Jhi, k, khi)...])
            out[I] = mv
            continue
        end
        Klo = _set(It, k, klo)
        Khi = _set(It, k, khi)
        # `u_k` sits at Face in `k` and Center elsewhere, so it is read at the two direction-`j`
        # centres the point lies between, and differenced along `j`. `u_j` mirrors that.
        hkhi = Geometry.scale_factors(geo, _point_at(sg, locs[k], Jhi))
        hklo = Geometry.scale_factors(geo, _point_at(sg, locs[k], Jlo))
        hjhi = Geometry.scale_factors(geo, _point_at(sg, locs[j], Khi))
        hjlo = Geometry.scale_factors(geo, _point_at(sg, locs[j], Klo))
        Δξj, _ = Discretization.local_spacing(caxes[j], It[j], pers[j])
        Δξk, _ = Discretization.local_spacing(caxes[k], It[k], pers[k])
        hc = Geometry.scale_factors(geo, _point_at(sg, loc, It))
        out[I] = S(((hkhi[k] * uf[k][Jhi...] - hklo[k] * uf[k][Jlo...]) / Δξj -
                    (hjhi[j] * uf[j][Khi...] - hjlo[j] * uf[j][Klo...]) / Δξk) / _omit_prod(hc, i))
    end
    return nothing
end

"""
    gradient(f, sg; kwargs...) -> NTuple{N,Array}

[`gradient!`](@ref) into fresh arrays, one per direction at that direction's face location.
"""
function gradient(f::AbstractArray, sg::Grids.StaggeredGrid{T,N}; kwargs...) where {T,N}
    locs = Grids.locations(sg)
    outs = ntuple(Val(N)) do d
        Array{T}(undef, ntuple(e -> length(Grids.axis_at(sg, e, locs[d][e])), Val(N)))
    end
    return gradient!(outs, f, sg; kwargs...)
end

"""
    divergence(us, sg; kwargs...) -> Array

[`divergence!`](@ref) into a fresh array at the cell centres.
"""
divergence(us::NTuple{N,AbstractArray}, sg::Grids.StaggeredGrid{T,N}; kwargs...) where {T,N} =
    divergence!(Array{T}(undef, Grids.size_tuple(Grids.center_grid(sg))), us, sg; kwargs...)

"""
    curl(u1, u2, sg; kwargs...) -> Array
    curl(us, sg; kwargs...) -> Array | NTuple{3,Array}

[`curl!`](@ref) into fresh arrays: one at the corner in two directions, and one per component at its own
vorticity point in three.
"""
function curl(u1::AbstractArray, u2::AbstractArray, sg::Grids.StaggeredGrid{T,2}; kwargs...) where {T}
    dims = ntuple(e -> length(Grids.axis_at(sg, e, Discretization.Face())), Val(2))
    return curl!(Array{T}(undef, dims), u1, u2, sg; kwargs...)
end

@inline curl(us::NTuple{2,AbstractArray}, sg::Grids.StaggeredGrid{T,2}; kwargs...) where {T} =
    curl(us[1], us[2], sg; kwargs...)

function curl(us::NTuple{3,AbstractArray}, sg::Grids.StaggeredGrid{T,3}; kwargs...) where {T}
    outs = ntuple(Val(3)) do i
        loc = _curl_location(sg, i)
        Array{T}(undef, ntuple(e -> length(Grids.axis_at(sg, e, loc[e])), Val(3)))
    end
    return curl!(outs, us, sg; kwargs...)
end
