"""
    gradient_plan(grid; stencil=Stencils.Axial(1), active_only=true, conn=nothing) -> GradientPlan

Build the least-squares gradient of `grid` — the geometry of it, with no field involved. See
[`GradientPlan`](@ref) for what it is and why it is that; apply it with
[`gradient!`](@ref).

This is the counterpart of `apply_stencil!` for the two architectures that have no separable axis to
difference along: a `CurvilinearGrid`, whose neighbours come from its index topology, and an
`UnstructuredGrid`, whose come from its stored adjacency. `conn` overrides the neighbour set; otherwise
one is built, from `stencil` where the architecture takes one.

Surface fields only — the tangent plane is two-dimensional, so the grid's coordinates must be a
`(λ, φ)` or `(x, y)` pair.

A masked cell gets no coefficients at all and reads zero gradient, and an inactive neighbour is not
offered to the fit, on the same rule as everywhere else: not determined by the active data, so not
invented.
"""
function gradient_plan end

function gradient_plan(
    grid::Grids.AbstractGrid{G,T}; stencil = Stencils.Axial(1), active_only::Bool = true,
    conn = nothing,
) where {G,T}
    Grids.ncoordinates(grid) == 2 || throw(ArgumentError(
        "a least-squares gradient is built in the tangent plane, so it needs a 2-coordinate grid; " *
        "got $(Grids.ncoordinates(grid))",
    ))
    # `build_connectivity` already resolves this per architecture: an index-space stencil on a
    # curvilinear grid, the stored adjacency on a node set, which ignores the stencil.
    c = conn === nothing ?
        Connectivity.build_connectivity(grid; stencil = stencil, active_only = active_only) : conn
    geo = Grids.grid_geometry(grid)
    msk = Grids.mask(grid)
    sz = grid isa Grids.UnstructuredGrid ? nothing : Grids.size_tuple(grid)
    n = length(msk)
    ptr = Vector{Int}(undef, n + 1)
    nbr = Int[]
    c1 = T[]
    c2 = T[]
    sizehint!(nbr, length(c.nbrs)); sizehint!(c1, length(c.nbrs)); sizehint!(c2, length(c.nbrs))
    # Scratch for one cell's neighbours: the displacements are needed twice, once to accumulate `A`
    # and once to weight it by `A⁺`, and re-projecting them would double the trigonometry.
    d1 = T[]; d2 = T[]; wk = T[]
    @inbounds ptr[1] = 1
    @inbounds for i in 1:n
        empty!(d1); empty!(d2); empty!(wk)
        if !(active_only && !msk[i])
            p0 = _grad_coords(grid, sz, i)
            for t in c.ptr[i]:(c.ptr[i + 1] - 1)
                j = Int(c.nbrs[t])
                active_only && !msk[j] && continue
                Δ = Geometry.project_to_tangent_plane(geo, p0, _grad_coords(grid, sz, j))
                δ1, δ2 = T(Δ[1]), T(Δ[2])
                q = δ1 * δ1 + δ2 * δ2
                q > 0 || continue                     # a coincident neighbour carries no direction
                push!(d1, δ1); push!(d2, δ2); push!(wk, inv(q)); push!(nbr, j)
            end
            a = zero(T); b = zero(T); cc = zero(T)
            for t in eachindex(wk)
                a += wk[t] * d1[t] * d1[t]
                b += wk[t] * d1[t] * d2[t]
                cc += wk[t] * d2[t] * d2[t]
            end
            # Relative tolerance: `A` scales with the weights, and `wₖ = 1/|Δrₖ|²` makes it O(number
            # of neighbours), so the cut has to be against its own size rather than an absolute number.
            tol = max(a + cc, one(T)) * sqrt(eps(T))
            p11, p12, p22 = _sympinv2(a, b, cc, tol)
            for t in eachindex(wk)
                push!(c1, wk[t] * (p11 * d1[t] + p12 * d2[t]))
                push!(c2, wk[t] * (p12 * d1[t] + p22 * d2[t]))
            end
        end
        ptr[i + 1] = ptr[i] + length(wk)
    end
    names = Geometry.point_names(geo, Val(2))
    return GradientPlan(ptr, nbr, c1, c2, names)
end

# The scalar fit: one value per cell in, one value out. Reached through the rank-matched methods below,
# which is what keeps `interpolate` type-stable — a scalar for an unbatched field, a vector for a
# batched one, decided by the rank rather than by a length at run time.
function _interp_scattered(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}},
    p::NTuple{D,Real}; k::Integer = 8, active_only::Bool = true, masked = T(NaN),
    topology = Connectivity.MetricTopology(grid), scratch = nothing,
    policy::AbstractMaskPolicy = BlankMasked(),
) where {T,D}
    policy isa ShiftWithinRun && _interp_mask_error(policy)
    Grids.ncoordinates(grid) == 2 || throw(ArgumentError(
        "interpolation off a rectilinear grid is fitted in the tangent plane, so it needs a " *
        "2-coordinate grid; got $(Grids.ncoordinates(grid))",
    ))
    n = length(Grids.mask(grid))
    length(field) == n || throw(DimensionMismatch(
        "field has $(length(field)) values for a grid of $n cells",
    ))
    geo = Grids.grid_geometry(grid)
    p0 = ntuple(d -> T(p[d]), Val(D))
    idx, dist = Connectivity.k_nearest(grid, p0; k = k, active_only = active_only, topology = topology,
                          scratch = scratch)
    isempty(idx) && return masked
    # `BlankMasked` refuses where the neighbourhood is not wholly active, on the same rule a stencil
    # uses; `k_nearest` with `active_only` has already dropped those, so the test is whether doing so
    # left a hole — a cell nearer than the farthest one kept, that was skipped.
    if policy isa BlankMasked && active_only
        msk = Grids.mask(grid)
        rmax = dist[end]
        n_in = Connectivity.nneighbors_within(grid, p0; ball = rmax, active_only = false, topology = topology,
                                 scratch = scratch)
        n_in > length(idx) && return masked
    end
    return _scattered_fit(field, grid, geo, p0, idx, 0, masked)
end

# The fit at one batch element, `off` into the field. The neighbour set and the mask verdict are
# properties of the POINT and the geometry, so a batched call solves those once — they are the k-d tree
# query, the expensive part — and calls this per element. The per-neighbour projections are eight
# arithmetic ops and are recomputed rather than buffered, which keeps this allocation-free.
function _scattered_fit(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, geo,
    p0::NTuple{D,T}, idx, off::Int, masked,
) where {T,D}
    # Weighted least squares for `f ≈ a + g·Δr` in the tangent plane at `p`. `a` is the value there,
    # and including `g` is what makes it exact for a linear field rather than a smoothed average.
    m11 = zero(T); m12 = zero(T); m13 = zero(T)
    m22 = zero(T); m23 = zero(T); m33 = zero(T)
    b1 = zero(T); b2 = zero(T); b3 = zero(T)
    sz = grid isa Grids.UnstructuredGrid ? nothing : Grids.size_tuple(grid)
    fsum = zero(T); wsum = zero(T)
    @inbounds for t in eachindex(idx)
        j = Int(idx[t])
        Δ = Geometry.project_to_tangent_plane(geo, p0, _grad_coords(grid, sz, j))
        δ1, δ2 = T(Δ[1]), T(Δ[2])
        # Inverse-square distance, floored so a query exactly on a cell centre stays finite.
        q = δ1 * δ1 + δ2 * δ2
        w = inv(max(q, eps(T)))
        fv = T(field[off + j])
        m11 += w;            m12 += w * δ1;      m13 += w * δ2
        m22 += w * δ1 * δ1;  m23 += w * δ1 * δ2; m33 += w * δ2 * δ2
        b1 += w * fv;        b2 += w * δ1 * fv;  b3 += w * δ2 * fv
        fsum += w * fv;      wsum += w
    end
    # A symmetric 3×3 by its adjugate. Where it is singular — collinear neighbours, or a single one —
    # the plane is not determined and only its constant is, which is the weighted mean.
    a11 = m22 * m33 - m23 * m23
    a12 = m13 * m23 - m12 * m33
    a13 = m12 * m23 - m13 * m22
    det = m11 * a11 + m12 * a12 + m13 * a13
    scale = max(m11 * m22 * m33, one(T))
    abs(det) ≤ scale * sqrt(eps(T)) && return wsum > 0 ? fsum / wsum : masked
    return (a11 * b1 + a12 * b2 + a13 * b3) / det        # the constant term: the value at `p`
end

# One value per cell — a curvilinear grid's cells are an `N`-D array, a node grid's a vector — so the
# field's rank matching the cells' says this is a single field and the answer is a scalar.
@inline interpolate(
    field::AbstractArray{<:Any,N}, grid::Grids.CurvilinearGrid{T,G,N}, p::NTuple{D,Real}; kwargs...,
) where {T,G,N,D} = _interp_scattered(field, grid, p; kwargs...)

@inline interpolate(
    field::AbstractVector, grid::Grids.UnstructuredGrid{T}, p::NTuple{D,Real}; kwargs...,
) where {T,D} = _interp_scattered(field, grid, p; kwargs...)

# A higher rank than the cells means trailing batch axes, and the answer is one value per element. The
# allocating form of [`interpolate!`](@ref), as everywhere else in the package.
function interpolate(
    field::AbstractArray{<:Any,NA}, grid::Grids.CurvilinearGrid{T,G,N}, p::NTuple{D,Real}; kwargs...,
) where {T,G,N,NA,D}
    n = length(Grids.mask(grid))
    return interpolate!(Vector{T}(undef, length(field) ÷ n), field, grid, p; kwargs...)
end

function interpolate(
    field::AbstractArray{<:Any,NA}, grid::Grids.UnstructuredGrid{T}, p::NTuple{D,Real}; kwargs...,
) where {T,NA,D}
    n = length(Grids.mask(grid))
    return interpolate!(Vector{T}(undef, length(field) ÷ n), field, grid, p; kwargs...)
end

@inline interpolate(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid,Grids.UnstructuredGrid},
    p::Geometry.PointLike; kwargs...,
) = interpolate(field, grid, Geometry.as_ntuple(p); kwargs...)

"""
    interpolate!(out, field, grid, p; k=8, …) -> out

[`interpolate`](@ref) off a rectilinear grid for a field carrying trailing BATCH axes,
writing one value per batch element.

The `k` nearest cells and the mask verdict are a property of the point and the geometry — and the k-d
tree query is the expensive part — so they are solved once here and the tangent-plane fit is then
applied to every element.
"""
function interpolate!(
    out::AbstractVector, field::AbstractArray,
    grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, p::NTuple{D,Real};
    k::Integer = 8, active_only::Bool = true, masked = T(NaN),
    topology = Connectivity.MetricTopology(grid), scratch = nothing,
    policy::AbstractMaskPolicy = BlankMasked(),
) where {T,D}
    policy isa ShiftWithinRun && _interp_mask_error(policy)
    Grids.ncoordinates(grid) == 2 || throw(ArgumentError(
        "interpolation off a rectilinear grid is fitted in the tangent plane, so it needs a " *
        "2-coordinate grid; got $(Grids.ncoordinates(grid))",
    ))
    n = length(Grids.mask(grid))
    (length(field) % n == 0) || throw(DimensionMismatch(
        "grid has $n cells; a field of $(length(field)) is not a whole number of them",
    ))
    nb = length(field) ÷ n
    length(out) == nb || throw(DimensionMismatch(
        "out holds $(length(out)) values but the field carries $nb batch elements",
    ))
    geo = Grids.grid_geometry(grid)
    p0 = ntuple(d -> T(p[d]), Val(D))
    idx, dist = Connectivity.k_nearest(grid, p0; k = k, active_only = active_only, topology = topology,
                          scratch = scratch)
    if isempty(idx)
        fill!(out, masked)
        return out
    end
    if policy isa BlankMasked && active_only
        rmax = dist[end]
        n_in = Connectivity.nneighbors_within(grid, p0; ball = rmax, active_only = false, topology = topology,
                                 scratch = scratch)
        if n_in > length(idx)
            fill!(out, masked)
            return out
        end
    end
    @inbounds for b in 1:nb
        out[b] = _scattered_fit(field, grid, geo, p0, idx, (b - 1) * n, masked)
    end
    return out
end

@inline interpolate!(
    out::AbstractVector, field::AbstractArray,
    grid::Union{Grids.CurvilinearGrid,Grids.UnstructuredGrid}, p::Geometry.PointLike; kwargs...,
) = interpolate!(out, field, grid, Geometry.as_ntuple(p); kwargs...)

@inline _grad_coords(grid, ::Nothing, i::Int) = Grids._raw_coords(grid, i)
@inline _grad_coords(grid, sz::Tuple, i::Int) =
    Grids._raw_coords(grid, Tuple(@inbounds CartesianIndices(sz)[i])...)
