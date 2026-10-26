"""
    gradient_plan(grid; stencil=Stencils.Axial(1), active_only=true, conn=nothing) -> GradientPlan

Build the least-squares gradient of `grid` — the geometry of it, with no field involved. See
[`GradientPlan`](@ref) for what it is and why it is that; apply it with
[`gradient!`](@ref).

This is the counterpart of `apply_stencil!` for the two architectures that have no separable axis to
difference along: a `CurvilinearGrid`, whose neighbours come from its index topology, and an
`UnstructuredGrid`, whose come from its stored adjacency. `conn` overrides the neighbour set; otherwise
one is built, from `stencil` where the architecture takes one. A neighbour set with no edges at all
raises an `ArgumentError`, the fit having nothing to read.

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
    msk = Grids.mask(grid)
    (length(c.nbrs) == 0 && length(msk) > 1 && (!active_only || any(msk))) && throw(ArgumentError(
        "gradient_plan fits each gradient from a cell's neighbours, and no cell of this grid has " *
        "one; give the grid an adjacency (`k` or `radius` on a node set) or pass `conn`",
    ))
    # `D` is the length of the grid's coordinate tuple, fixed by its type, so this branch folds and the
    # body below runs at a literal width, its tuple arithmetic in registers.
    return D == 2 ? _gradient_plan(grid, c, active_only, Val(2)) :
                    _gradient_plan(grid, c, active_only, Val(3))
end

# The `Δrₖ` contraction, one method per width, so each is a flat sum with no accumulator.
@inline _contract(p::NTuple{2,T}, δ::NTuple{2,T}) where {T} = p[1] * δ[1] + p[2] * δ[2]
@inline _contract(p::NTuple{3,T}, δ::NTuple{3,T}) where {T} = p[1] * δ[1] + p[2] * δ[2] + p[3] * δ[3]

# `nb`'s displacement on the frame `fr` at `c`, `nb` taken at its image nearest `c` along each direction
# with a wrap length in `prd` (see `Grids._image_pair`). On the plane the offset is formed exactly and
# resolved against the origin.
@inline _image_displacement(geo::Geometry.AbstractCartesianGeometry, fr, c, nb, prd) =
    Geometry._displacement_on(geo, map(zero, fr), Grids._offsets(c, nb, prd))
@inline _image_displacement(geo, fr, c, nb, prd) =
    Geometry._displacement_on(geo, fr, last(Grids._image_pair(geo, c, nb, prd)))

# The largest row of a CSR offset array over its first `n` rows.
@inline function _max_degree(ptr::AbstractVector{<:Integer}, n::Int)
    d = 0
    @inbounds for i in 1:n
        d = max(d, Int(ptr[i + 1] - ptr[i]))
    end
    return d
end

# The weighted Gram matrix `Σₜ wₜ δₜ δₜᵀ` over the first `m` displacements.
@inline function _gram(dcol::NTuple{D,AbstractVector{T}}, wk::AbstractVector{T}, m::Int,
                       ::Val{D}) where {T,D}
    return ntuple(r -> ntuple(s -> begin
            acc = zero(T)
            @inbounds for t in 1:m
                acc += wk[t] * dcol[r][t] * dcol[s][t]
            end
            acc
        end, Val(D)), Val(D))
end

function _gradient_plan(
    grid::Grids.AbstractGrid{G,T}, c, active_only::Bool, ::Val{D},
) where {G,T,D}
    geo = Grids.grid_geometry(grid)
    msk = Grids.mask(grid)
    sz = Connectivity._linear_shape(grid)
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
    # degree once, so it is allocated once. `maxdeg` is assigned exactly once: the closure below
    # captures it, and a captured variable that is reassigned is boxed.
    maxdeg = _max_degree(c.ptr, n)
    dcol = ntuple(_ -> Vector{T}(undef, maxdeg), Val(D))
    wk = Vector{T}(undef, maxdeg)
    w = 0                                              # write cursor into `nbr`/`coef`
    prd = Grids._wrap_lengths(grid, Val(D))
    @inbounds ptr[1] = 1
    @inbounds for i in 1:n
        m = 0                                          # this cell's kept neighbours
        if !(active_only && !msk[i])
            ci = _grad_coords(grid, sz, i)
            fr = Geometry._local_frame(geo, ci)
            for t in c.ptr[i]:(c.ptr[i + 1] - 1)
                j = Int(c.nbrs[t])
                active_only && !msk[j] && continue
                Δ = _image_displacement(geo, fr, ci, _grad_coords(grid, sz, j), prd)
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
            A = _gram(dcol, wk, m, Val(D))
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

The cells incident on `nearest` are tried first, and usually hold `p0`. They need not: a point near
an edge can lie nearer the vertex across it than any of its own. From the best of them the search
walks across the edge opposite the most negative weight, which ends at the containing cell on a
Delaunay mesh, or at the mesh's boundary.

The weights are computed in the tangent plane at `p0`, where the three vertices sit at their
[`Geometry.local_displacement`](@ref)s and `p0` itself is the origin.
"""
@inline function _bary_weights(
    grid, geo, p0::NTuple{2,T}, nearest::Int, active_only::Bool,
) where {T}
    mesh = Grids.cell_mesh(grid)
    msk = Grids.mask(grid)
    none = (0, 0, 0, (zero(T), zero(T), zero(T)))
    mesh === nothing && return none
    (1 ≤ nearest ≤ length(mesh.node_ptr) - 1) || return none
    # A vertex a hair outside is still in: the test decides a boundary shared by two cells, and a
    # query exactly on an edge lands in one of them.
    tol = -sqrt(eps(T))
    fr = Geometry._local_frame(geo, p0)
    prd = Grids._wrap_lengths(grid, Val(2))
    start, least, held = 0, T(-Inf), false
    @inbounds for cc in Grids.node_cells(mesh, nearest)
        tri = _cell_weights(grid, geo, fr, p0, prd, mesh, Int(cc), T)
        a, b, c, w = tri
        a == 0 && continue
        m = min(w[1], w[2], w[3])
        if m ≥ tol
            (!active_only || (msk[a] && msk[b] && msk[c])) && return tri
            held = true                                # contained, against an inactive node
        elseif m > least
            start, least = Int(cc), m
        end
    end
    (held || start == 0) && return none
    t = start
    tri = _cell_weights(grid, geo, fr, p0, prd, mesh, t, T)
    # Each step leaves through the edge `p0` lies beyond, so on a Delaunay mesh no cell repeats; the cap
    # ends a cycle on a mesh that is not one.
    @inbounds for _ in 1:Grids.ncells(mesh)
        a, b, c, w = tri
        u, v = w[1] ≤ min(w[2], w[3]) ? (b, c) : w[2] ≤ w[3] ? (a, c) : (a, b)
        t = _cell_across(mesh, t, u, v)
        t == 0 && return none                          # off the mesh's edge
        tri = _cell_weights(grid, geo, fr, p0, prd, mesh, t, T)
        a, b, c, w = tri
        a == 0 && return none
        if min(w[1], w[2], w[3]) ≥ tol
            return !active_only || (msk[a] && msk[b] && msk[c]) ? tri : none
        end
    end
    return none
end

# Cell `cc`'s vertices and the barycentric weights of the frame's origin `p0` in it, or `a == 0` where
# the cell is no triangle or spans no area. Each vertex is taken at its image nearest `p0`, so a cell
# joining nodes across a periodic face is whole.
@inline function _cell_weights(grid, geo, fr, p0, prd, mesh, cc::Int, ::Type{T}) where {T}
    nds = Grids.cell_nodes(mesh, cc)
    length(nds) == 3 || return (0, 0, 0, (zero(T), zero(T), zero(T)))
    @inbounds a, b, c = Int(nds[1]), Int(nds[2]), Int(nds[3])
    Δa = _image_displacement(geo, fr, p0, Grids._raw_coords(grid, a), prd)
    Δb = _image_displacement(geo, fr, p0, Grids._raw_coords(grid, b), prd)
    Δc = _image_displacement(geo, fr, p0, Grids._raw_coords(grid, c), prd)
    a1, a2 = T(Δa[1]), T(Δa[2])
    b1, b2 = T(Δb[1]), T(Δb[2])
    c1, c2 = T(Δc[1]), T(Δc[2])
    det = (b2 - c2) * (a1 - c1) + (c1 - b1) * (a2 - c2)
    iszero(det) && return (0, 0, 0, (zero(T), zero(T), zero(T)))
    w1 = ((b2 - c2) * (-c1) + (c1 - b1) * (-c2)) / det
    w2 = ((c2 - a2) * (-c1) + (a1 - c1) * (-c2)) / det
    return (a, b, c, (w1, w2, one(T) - w1 - w2))
end

# The cell other than `t` that has both `u` and `v` among its nodes, or 0 where the edge is a boundary.
@inline function _cell_across(mesh, t::Int, u::Int, v::Int)
    @inbounds for c1 in Grids.node_cells(mesh, u)
        Int(c1) == t && continue
        for c2 in Grids.node_cells(mesh, v)
            c1 == c2 && return Int(c1)
        end
    end
    return 0
end

# `f` at the query point from the cell containing it: three reads, exact for a field linear in the
# tangent plane.
@inline _bary_value(field::AbstractArray, off::Int, a::Int, b::Int, c::Int, w::NTuple{3,T}) where {T} =
    @inbounds T(field[off + a]) * w[1] + T(field[off + b]) * w[2] + T(field[off + c]) * w[3]

"""
    _mask_may_hide(grid, policy, active_only) -> Bool

Whether the neighbourhood test has anything to find: under [`BlankMasked`](@ref) an active-only search
may pass over an inactive cell nearer than the farthest active one it keeps. On an
[`Grids.AllActive`](@ref) grid there is no inactive cell to pass over.
"""
@inline _mask_may_hide(grid, policy::AbstractMaskPolicy, active_only::Bool) =
    policy isa BlankMasked && active_only && !(Grids.mask(grid) isa Grids.AllActive)

# The `k` nearest cells' ids and distances: the caller's pair where one is given, so that a stream of
# queries allocates nothing, else a fresh pair.
@inline _nearest_buffers(::Nothing, k::Int, ::Type{T}) where {T} =
    (Vector{Int}(undef, k), Vector{T}(undef, k))

@inline function _nearest_buffers(b::Tuple{AbstractVector{<:Integer},AbstractVector}, k::Int, ::Type)
    (length(b[1]) ≥ k && length(b[2]) ≥ k) || throw(ArgumentError(
        "nearest must hold at least k = $k ids and distances; got $(length(b[1])) and $(length(b[2]))",
    ))
    return b
end

# The neighbour search both forms share: the `k` nearest active cells to `p0`, nearest first, how many
# were found, and whether an inactive cell lies at least as near as the farthest of them. `BlankMasked`
# refuses a fit over such a neighbourhood, on the rule a stencil uses; the search answers it in the same
# pass, visiting the inactive cells without keeping them.
#
# Without a cell mesh that verdict is the whole answer, so the search stops once it is settled. With
# one, the containing cell is tried first and needs the nearest active node, so the search runs on.
@inline function _scattered_neighbours(
    grid::Grids.AbstractGrid{G,T}, p0::Tuple, k::Integer, active_only::Bool, policy, topology, scratch,
    nearest,
) where {G,T}
    kk = Int(k)
    kk ≥ 0 || throw(ArgumentError("k must be non-negative, got $k"))
    idx, dist = _nearest_buffers(nearest, kk, T)
    n, hidden = Connectivity._k_nearest_at!(idx, dist, grid, p0, kk, active_only,
                                            _mask_may_hide(grid, policy, active_only),
                                            !Grids.has_cell_mesh(grid), topology, scratch)
    return idx, dist, n, hidden
end

# The scalar fit: one value per cell in, one value out. The rank-matched methods below dispatch to it,
# so `interpolate` returns a scalar for an unbatched field and a vector for a batched one, decided by
# the field's rank.
function _interp_scattered(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}},
    p::NTuple{D,Real}; k::Integer = 8, active_only::Bool = true, masked = T(NaN),
    topology = Connectivity.MetricTopology(grid), scratch = nothing, nearest = nothing,
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
    idx, _, m, hidden = _scattered_neighbours(grid, p0, k, active_only, policy, topology, scratch,
                                              nearest)
    m == 0 && return masked
    # Where the grid carries its cells, the containing one gives the answer exactly, from three
    # values. The search returns nearest first, which is the node whose incident cells to search.
    #
    # Tried before the mask test, and exempt from it: that test asks whether the whole neighbourhood
    # is active, which is the question for a fit over a neighbourhood. A value taken from one triangle
    # needs only that triangle, and `_bary_weights` already refuses a cell with an inactive vertex.
    a, b, c, w = _bary_weights(grid, geo, p0, Int(idx[1]), active_only)
    a == 0 || return _bary_value(field, 0, a, b, c, w)
    hidden && return masked
    return _scattered_fit(field, grid, geo, p0, idx, m, 0, masked)
end

# The fit at one batch element, `off` into the field, over the first `m` cells of `idx`. The neighbour
# set and the mask verdict depend on the point and the geometry alone, so a batched call solves them
# once and calls this per element. The per-neighbour projections are recomputed here, so this
# allocates nothing.
function _scattered_fit(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, geo,
    p0::NTuple{D,T}, idx, m::Int, off::Int, masked,
) where {T,D}
    # Weighted least squares for `f ≈ a + g·Δr` in the tangent plane at `p`. `a` is the value there;
    # fitting the gradient `g` alongside it makes the result exact for a linear field.
    m11 = zero(T); m12 = zero(T); m13 = zero(T)
    m22 = zero(T); m23 = zero(T); m33 = zero(T)
    b1 = zero(T); b2 = zero(T); b3 = zero(T)
    sz = Connectivity._linear_shape(grid)
    fsum = zero(T); wsum = zero(T)
    fr = Geometry._local_frame(geo, p0)
    prd = Grids._wrap_lengths(grid, Val(D))
    @inbounds for t in 1:m
        j = Int(idx[t])
        Δ = _image_displacement(geo, fr, p0, _grad_coords(grid, sz, j), prd)
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
    # the plane is not determined and only its constant is, which is the weighted mean. Its entries
    # carry different powers of length, so the test is against the product of the diagonal, which
    # carries the same power as `det`: their ratio is the determinant of the unit-diagonal matrix.
    a11 = m22 * m33 - m23 * m23
    a12 = m13 * m23 - m12 * m33
    a13 = m12 * m23 - m13 * m22
    det = m11 * a11 + m12 * a12 + m13 * a13
    abs(det) ≤ m11 * m22 * m33 * sqrt(eps(T)) && return wsum > 0 ? fsum / wsum : masked
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

The `k` nearest cells and the mask verdict are a property of the point and the geometry, so they are
solved once here and the tangent-plane fit is then applied to every element.
"""
function interpolate!(
    out::AbstractVector, field::AbstractArray,
    grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, p::NTuple{D,Real};
    k::Integer = 8, active_only::Bool = true, masked = T(NaN),
    topology = Connectivity.MetricTopology(grid), scratch = nothing, nearest = nothing,
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
    idx, _, m, hidden = _scattered_neighbours(grid, p0, k, active_only, policy, topology, scratch,
                                              nearest)
    if m == 0
        fill!(out, masked)
        return out
    end
    # The cell and its weights are a property of the point alone, so they are solved once and applied
    # to every batch element, tried before the neighbourhood mask test as the scalar form does.
    ba, bb, bc, bw = _bary_weights(grid, geo, p0, Int(idx[1]), active_only)
    if ba != 0
        @inbounds for b in 1:nb
            out[b] = _bary_value(field, (b - 1) * n, ba, bb, bc, bw)
        end
        return out
    end
    if hidden
        fill!(out, masked)
        return out
    end
    @inbounds for b in 1:nb
        out[b] = _scattered_fit(field, grid, geo, p0, idx, m, (b - 1) * n, masked)
    end
    return out
end

@inline interpolate!(
    out::AbstractVector, field::AbstractArray,
    grid::Union{Grids.CurvilinearGrid,Grids.UnstructuredGrid}, p::Geometry.PointLike; kwargs...,
) = interpolate!(out, field, grid, Geometry.as_ntuple(p); kwargs...)

# Every point searches one cell list over the grid, binned at the radius a `k`-nearest search opens at,
# and each chunk of points holds its own `nearest` buffers, so a point allocates nothing.
function interpolate!(
    out::AbstractArray, field::AbstractArray,
    grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}}, points::AbstractVector{<:_PointElement};
    k::Integer = 8, active_only::Bool = true, masked = T(NaN), policy::AbstractMaskPolicy = BlankMasked(),
    topology = nothing, execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {T}
    nb = _check_points_out(out, field, grid, length(points))
    kk = Int(k)
    tp = topology === nothing ? Connectivity.default_sweep_topology(
        grid, Connectivity._knn_seed_radius_at(grid, nothing, max(kk, 1)), false) : topology
    Execution.run_chunks(length(points), execution, Execution.Written(out, Execution.ByBlock(nb))) do rng
        nearest = (Vector{Int}(undef, kk), Vector{T}(undef, kk))
        @inbounds for i in rng
            interpolate!(view(out, ((i - 1) * nb + 1):(i * nb)), field, grid, points[i];
                         k = kk, active_only = active_only, masked = masked, policy = policy,
                         topology = tp, nearest = nearest)
        end
    end
    return out
end

interpolate(
    field::AbstractArray, grid::Union{Grids.CurvilinearGrid{T},Grids.UnstructuredGrid{T}},
    points::AbstractVector{<:_PointElement}; kwargs...,
) where {T} = interpolate!(_points_out(T, field, grid, length(points)), field, grid, points; kwargs...)

@inline _grad_coords(grid, ::Nothing, i::Int) = Grids._raw_coords(grid, i)
@inline _grad_coords(grid, sz::Tuple, i::Int) =
    Grids._raw_coords(grid, Tuple(@inbounds CartesianIndices(sz)[i])...)
