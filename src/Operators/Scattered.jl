"""
    gradient_plan(grid; stencil=Stencils.Axial(1), active_only=true, conn=nothing) -> GradientPlan

Build the least-squares gradient of `grid` — the geometry of it, with no field involved. See
[`GradientPlan`](@ref) for what it is and why it is that; apply it with
[`gradient!`](@ref).

This is the counterpart of `apply_stencil!` for the two architectures that have no separable axis to
difference along: a `CurvilinearGrid`, whose neighbours come from its index topology, and an
`UnstructuredGrid`, whose come from its stored adjacency. `conn` overrides the neighbour set; otherwise
one is built, from `stencil` where the architecture takes one.

One direction per coordinate: a `(λ, φ)` or `(x, y)` surface resolves two, in its tangent plane, and a
three-coordinate volume resolves three, on the local frame — see
[`Geometry.local_displacement`](@ref). The fixed-size symmetric solve behind it covers those two cases.

A masked cell gets no coefficients at all and reads zero gradient, and an inactive neighbour is not
offered to the fit, on the same rule as everywhere else: not determined by the active data, so not
invented.
"""
function gradient_plan end

function gradient_plan(
    grid::Grids.AbstractGrid{G,T}; stencil = Stencils.Axial(1), active_only::Bool = true,
    conn = nothing,
) where {G,T}
    D = Grids.ncoordinates(grid)
    (D == 2 || D == 3) || throw(ArgumentError(
        "a least-squares gradient resolves one direction per coordinate, and the fixed-size solve " *
        "behind it covers two or three; got $D",
    ))
    # `build_connectivity` already resolves this per architecture: an index-space stencil on a
    # curvilinear grid, the stored adjacency on a node set, which ignores the stencil.
    c = conn === nothing ?
        Connectivity.build_connectivity(grid; stencil = stencil, active_only = active_only) : conn
    # `Val(D)` from a runtime count: the body below is written once for any `D`, and the tuple
    # arithmetic in it only stays in registers when the width is a compile-time fact.
    return D == 2 ? _gradient_plan(grid, c, active_only, Val(2)) :
                    _gradient_plan(grid, c, active_only, Val(3))
end

# The `Δrₖ` contraction, one method per width, so each is a flat sum with no accumulator.
@inline _contract(p::NTuple{2,T}, δ::NTuple{2,T}) where {T} = p[1] * δ[1] + p[2] * δ[2]
@inline _contract(p::NTuple{3,T}, δ::NTuple{3,T}) where {T} = p[1] * δ[1] + p[2] * δ[2] + p[3] * δ[3]

function _gradient_plan(
    grid::Grids.AbstractGrid{G,T}, c, active_only::Bool, ::Val{D},
) where {G,T,D}
    geo = Grids.grid_geometry(grid)
    msk = Grids.mask(grid)
    sz = grid isa Grids.UnstructuredGrid ? nothing : Grids.size_tuple(grid)
    n = length(msk)
    # The connectivity already gives every degree, so the result is sized up front. Dropping an
    # inactive or coincident neighbour only lowers the count, so `length(c.nbrs)` is an upper bound
    # and one `resize!` at the end reclaims the difference.
    #
    # The plan's index buffers keep the connectivity's own width: this graph is a subset of that one,
    # so whatever held its ids and offsets holds these.
    cap = length(c.nbrs)
    ptr = similar(c.ptr, n + 1)
    nbr = similar(c.nbrs, cap)
    coef = ntuple(_ -> Vector{T}(undef, cap), Val(D))
    # Scratch for one cell's neighbours: the displacements are read twice, once to accumulate `A` and
    # once to weight it by `A⁺`, and re-projecting them doubles the trigonometry. Sized to the largest
    # degree once, so it is allocated once.
    maxdeg = 0
    @inbounds for i in 1:n
        maxdeg = max(maxdeg, c.ptr[i + 1] - c.ptr[i])
    end
    dcol = ntuple(_ -> Vector{T}(undef, maxdeg), Val(D))
    wk = Vector{T}(undef, maxdeg)
    w = 0                                              # write cursor into `nbr`/`coef`
    @inbounds ptr[1] = 1
    @inbounds for i in 1:n
        m = 0                                          # this cell's kept neighbours
        if !(active_only && !msk[i])
            p0 = _grad_coords(grid, sz, i)
            for t in c.ptr[i]:(c.ptr[i + 1] - 1)
                j = Int(c.nbrs[t])
                active_only && !msk[j] && continue
                Δ = Geometry.local_displacement(geo, p0, _grad_coords(grid, sz, j))
                δ = ntuple(d -> T(Δ[d]), Val(D))
                q = _contract(δ, δ)
                q > 0 || continue                     # a coincident neighbour carries no direction
                m += 1
                for d in 1:D
                    dcol[d][m] = δ[d]
                end
                wk[m] = inv(q)
                nbr[w + m] = j
            end
            A = ntuple(r -> ntuple(s -> begin
                    acc = zero(T)
                    for t in 1:m
                        acc += wk[t] * dcol[r][t] * dcol[s][t]
                    end
                    acc
                end, Val(D)), Val(D))
            # Relative tolerance: `A` scales with the weights, and `wₖ = 1/|Δrₖ|²` makes it O(number
            # of neighbours), so the cut is against its own trace.
            trA = zero(T)
            for d in 1:D
                trA += A[d][d]
            end
            P = _sympinv(A, max(trA, one(T)) * sqrt(eps(T)))
            for t in 1:m
                δ = ntuple(d -> dcol[d][t], Val(D))
                for d in 1:D
                    coef[d][w + t] = wk[t] * _contract(P[d], δ)
                end
            end
        end
        w += m
        ptr[i + 1] = ptr[i] + m
    end
    resize!(nbr, w)
    for d in 1:D
        resize!(coef[d], w)
    end
    return GradientPlan(ptr, nbr, coef, Geometry.point_names(geo, Val(D)))
end

"""
    _bary_weights(grid, geo, p0, nearest, active_only) -> (a, b, c, (w₁, w₂, w₃))

The cell of the grid's [`Grids.CellMesh`](@ref) containing `p0`, and the barycentric weights of `p0`
in it. `a == 0` where there is none — outside the mesh, or against an inactive node.

Trying the cells incident on `nearest` is exhaustive: `p0` lies in the Voronoi cell of its nearest node,
and in a Delaunay triangulation that region is covered by the triangles incident on that node.

The weights are computed in the tangent plane at `p0`, where the three vertices sit at their
[`Geometry.local_displacement`](@ref)s and `p0` itself is the origin.
"""
@inline function _bary_weights(
    grid, geo, p0::NTuple{2,T}, nearest::Int, active_only::Bool,
) where {T}
    mesh = Grids.cell_mesh(grid)
    msk = Grids.mask(grid)
    z = (zero(T), zero(T), zero(T))
    mesh === nothing && return (0, 0, 0, z)
    (1 ≤ nearest ≤ length(mesh.node_ptr) - 1) || return (0, 0, 0, z)
    # A vertex a hair outside is still in: the test decides a boundary shared by two cells, and a
    # query exactly on an edge lands in one of them.
    tol = -sqrt(eps(T))
    @inbounds for cc in Grids.node_cells(mesh, nearest)
        nds = Grids.cell_nodes(mesh, Int(cc))
        length(nds) == 3 || continue
        a, b, c = Int(nds[1]), Int(nds[2]), Int(nds[3])
        active_only && !(msk[a] && msk[b] && msk[c]) && continue
        Δa = Geometry.local_displacement(geo, p0, Grids._raw_coords(grid, a))
        Δb = Geometry.local_displacement(geo, p0, Grids._raw_coords(grid, b))
        Δc = Geometry.local_displacement(geo, p0, Grids._raw_coords(grid, c))
        a1, a2 = T(Δa[1]), T(Δa[2])
        b1, b2 = T(Δb[1]), T(Δb[2])
        c1, c2 = T(Δc[1]), T(Δc[2])
        det = (b2 - c2) * (a1 - c1) + (c1 - b1) * (a2 - c2)
        iszero(det) && continue                       # a degenerate cell spans no area
        w1 = ((b2 - c2) * (-c1) + (c1 - b1) * (-c2)) / det
        w2 = ((c2 - a2) * (-c1) + (a1 - c1) * (-c2)) / det
        w3 = one(T) - w1 - w2
        (w1 ≥ tol && w2 ≥ tol && w3 ≥ tol) && return (a, b, c, (w1, w2, w3))
    end
    return (0, 0, 0, z)
end

# `f` at the query point from the cell containing it: three reads, exact for a field linear in the
# tangent plane.
@inline _bary_value(field::AbstractArray, off::Int, a::Int, b::Int, c::Int, w::NTuple{3,T}) where {T} =
    @inbounds T(field[off + a]) * w[1] + T(field[off + b]) * w[2] + T(field[off + c]) * w[3]

"""
    _mask_may_hide(grid, policy, active_only) -> Bool

Whether the neighbourhood test has anything to find: under [`BlankMasked`](@ref) an active-only
`k_nearest` may have skipped an inactive cell nearer than the farthest it kept, and the test is a
second range query over the same index. On an [`Grids.AllActive`](@ref) grid there is no inactive cell
to skip, so the answer is known without the query.
"""
@inline _mask_may_hide(grid, policy::AbstractMaskPolicy, active_only::Bool) =
    policy isa BlankMasked && active_only && !(Grids.mask(grid) isa Grids.AllActive)

# The scalar fit: one value per cell in, one value out. The rank-matched methods below dispatch to it,
# so `interpolate` returns a scalar for an unbatched field and a vector for a batched one, decided by
# the field's rank.
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
    # Where the grid carries its cells, the containing one gives the answer exactly, from three
    # values. `k_nearest` returns nearest first, which is the node whose incident cells to search.
    #
    # Tried before the mask test below, and exempt from it: that test asks whether the whole
    # neighbourhood is active, which is the question for a fit over a neighbourhood. A value taken
    # from one triangle needs only that triangle, and `_bary_weights` already refuses a cell with an
    # inactive vertex.
    a, b, c, w = _bary_weights(grid, geo, p0, Int(idx[1]), active_only)
    a == 0 || return _bary_value(field, 0, a, b, c, w)
    # `BlankMasked` refuses where the neighbourhood is not wholly active, on the same rule a stencil
    # uses; `k_nearest` with `active_only` has already dropped those, so the test is whether doing so
    # left a hole — a cell nearer than the farthest one kept, that was skipped.
    if _mask_may_hide(grid, policy, active_only)
        rmax = dist[end]
        n_in = Connectivity.nneighbors_within(grid, p0; ball = rmax, active_only = false, topology = topology,
                                 scratch = scratch)
        n_in > length(idx) && return masked
    end
    return _scattered_fit(field, grid, geo, p0, idx, 0, masked)
end

# The fit at one batch element, `off` into the field. The neighbour set and the mask verdict depend on
# the point and the geometry alone, so a batched call solves them once — that is the k-d tree query,
# the expensive part — and calls this per element. The per-neighbour projections are eight arithmetic
# ops, recomputed here, so this allocates nothing.
function _scattered_fit(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, geo,
    p0::NTuple{D,T}, idx, off::Int, masked,
) where {T,D}
    # Weighted least squares for `f ≈ a + g·Δr` in the tangent plane at `p`. `a` is the value there;
    # fitting the gradient `g` alongside it makes the result exact for a linear field.
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

[`interpolate`](@ref) off a rectilinear grid for a field carrying trailing batch axes,
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
    # The cell and its weights are a property of the point alone, so they are solved once and applied
    # to every batch element — the same structure the k-d tree query above already has, and tried
    # before the neighbourhood mask test as the scalar form does.
    ba, bb, bc, bw = _bary_weights(grid, geo, p0, Int(idx[1]), active_only)
    if ba != 0
        @inbounds for b in 1:nb
            out[b] = _bary_value(field, (b - 1) * n, ba, bb, bc, bw)
        end
        return out
    end
    if _mask_may_hide(grid, policy, active_only)
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
