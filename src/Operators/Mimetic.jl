# ---------------------------------------------------------------------------
# Mimetic operators on staggered faces and corners
# ---------------------------------------------------------------------------
#
# A scalar sits at cell centres, a normal velocity on faces — one array per direction, shaped by
# `Grids.face_dims` — and each component `R_ab` of an antisymmetric rotation potential on the `(a, b)`
# corners, shaped by `Grids.corner_dims`. One gradient `G`, the metric-free difference across a face over
# the distance between its centres, defines the rest:
#
#     D = −G*        the adjoint under the cell measure `V` and the face weight `A·g`
#     L = D G        symmetric and negative semidefinite under `V`
#
# The metric-free differences compose to zero, so `curl(G χ) = 0` and `D(rot R) = 0` hold on every grid.
# The metric enters through a face's area `A` and gap `g` and, in three directions, the length `ℓ` of the
# primal edge through a corner, all held by a `FaceMetrics`; a boundary condition and a mask enter as faces
# of zero area.

"""
    AbstractBoundaryCondition

How the faces past the end of a bounded direction couple: [`Dirichlet`](@ref) or [`Neumann`](@ref). A
wrapping direction has no such face.
"""
abstract type AbstractBoundaryCondition end

"""
    Dirichlet()

A zero value one cell beyond the end of every bounded direction: an end face couples through its
geometric area against that zero.
"""
struct Dirichlet <: AbstractBoundaryCondition end

"""
    Neumann()

No flux through the end of a bounded direction: an end face has no area in the operators.
"""
struct Neumann <: AbstractBoundaryCondition end

# ---- rotation bookkeeping ------------------------------------------------------

"""
    n_rotation_components(N) -> Int

`N(N−1)/2`, the independent components of an antisymmetric tensor in `N` directions.
"""
@inline n_rotation_components(N::Integer) = (N * (N - 1)) ÷ 2

"""
    rotation_pairs(Val(N)) -> NTuple{P,NTuple{2,Int}}

The pairs `(a, b)`, `a < b`, in lexicographic order: the components of an antisymmetric tensor, and the
directions each is staggered in.
"""
@generated function rotation_pairs(::Val{N}) where {N}
    pairs = Expr[]
    for a in 1:(N - 1), b in (a + 1):N
        push!(pairs, Expr(:tuple, a, b))
    end
    return Expr(:tuple, pairs...)
end

# For each direction `c`, the `(e, p, s)` that assemble `(rot R)_c`: component `p` differenced along `e`,
# with sign `s`. Each face then gathers its own terms.
@generated function _rotation_terms(::Val{N}) where {N}
    pairs = Tuple{Int,Int}[]
    for a in 1:(N - 1), b in (a + 1):N
        push!(pairs, (a, b))
    end
    per_dir = map(1:N) do c
        terms = Expr[]
        for (p, (a, b)) in enumerate(pairs)
            a == c && push!(terms, Expr(:tuple, b, p, -1))
            b == c && push!(terms, Expr(:tuple, a, p, 1))
        end
        return Expr(:tuple, terms...)
    end
    return Expr(:tuple, per_dir...)
end

# ---- metric storage -------------------------------------------------------------

"""
    SeparableMetrics()
    DenseMetrics()

How a face metric is held: as one factor per axis where it is a product of them, and as a full array
elsewhere. [`metric_layout`](@ref) reads which from a grid's geometry and mask types.
"""
struct SeparableMetrics end
struct DenseMetrics end

"""
    metric_layout(grid) -> SeparableMetrics | DenseMetrics

`SeparableMetrics()` on an unmasked Cartesian or spherical rectilinear grid, where every face area, gap
and edge length is a product of one factor per axis; `DenseMetrics()` on every other grid.
"""
@inline metric_layout(grid::Grids.StructuredGrid) =
    _metric_layout(Grids.grid_geometry(grid), Grids.mask(grid))
@inline metric_layout(::Grids.CurvilinearGrid) = DenseMetrics()

@inline _metric_layout(::Geometry.AbstractCartesianGeometry, ::Grids.AllActive) = SeparableMetrics()
@inline _metric_layout(::Geometry.SphericalGeometry, ::Grids.AllActive) = SeparableMetrics()
@inline _metric_layout(_, _) = DenseMetrics()

"""
    metric_array(f, dims, layout, T) -> AbstractArray{T,N}

The array `a[I] = f(I)` over `dims`: a [`Grids.SeparableMeasure`](@ref) of its axis factors under
`SeparableMetrics()`, read off `f` along each axis through one index where `f` is nonzero, and a dense
array under `DenseMetrics()`.

With `G_e(i) = f(ref with e → i)` and `P = f(ref)`, `∏_e G_e(I[e]) = f(I)·P^{N−1}` for a separable `f`, so
the trailing factors are divided by `P`. A function zero everywhere factors as zeros.
"""
function metric_array(f::F, dims::NTuple{N,Int}, ::SeparableMetrics, ::Type{T}) where {F,N,T}
    ref = _separable_reference(f, dims)
    ref === nothing && return Grids.SeparableMeasure(ntuple(e -> zeros(T, dims[e]), Val(N)))
    P = T(f(ref))
    return Grids.SeparableMeasure(ntuple(Val(N)) do e
        g = Vector{T}(undef, dims[e])
        @inbounds for i in 1:dims[e]
            g[i] = T(f(CartesianIndex(ntuple(k -> k == e ? i : ref[k], Val(N)))))
        end
        e == 1 || (g ./= P)
        return g
    end)
end

function metric_array(f::F, dims::NTuple{N,Int}, ::DenseMetrics, ::Type{T}) where {F,N,T}
    a = Array{T,N}(undef, dims)
    @inbounds for I in CartesianIndices(dims)
        a[I] = T(f(I))
    end
    return a
end

# An index where `f` is nonzero, the middle tried first: a vanishing entry sits at the end of an axis.
function _separable_reference(f::F, dims::NTuple{N,Int}) where {F,N}
    mid = CartesianIndex(ntuple(e -> cld(dims[e], 2), Val(N)))
    iszero(f(mid)) || return mid
    for I in CartesianIndices(dims)
        iszero(f(I)) || return I
    end
    return nothing
end

# ---- face metrics ---------------------------------------------------------------

"""
    FaceMetrics

The staggered metric of a grid under a boundary condition, built once by [`face_metrics`](@ref) and read
by [`gradient!`](@ref), [`divergence!`](@ref), [`curl!`](@ref), [`rotational_velocity!`](@ref),
[`to_faces!`](@ref), [`to_centres!`](@ref) and [`laplacian_coefficients`](@ref). Fields, each face array
one per direction shaped by [`Grids.face_dims`](@ref):

- `grid`, `boundary`: what it describes;
- `area_op`: the area through which the operators couple across a face: its geometric area between two
  active cells, zero against an inactive one, and at the end of a bounded direction its geometric area
  under `Dirichlet()` and zero under `Neumann()`;
- `area_flux`: the area through which data crosses a face, `area_op` with every bounded end open; the
  same array under `Dirichlet()`;
- `gap`: the physical distance between the two centres a face separates, and from its one centre to the
  face at the end of a bounded direction;
- `edge`: per rotation pair `(a, b)`, over its corners, the length of the primal edge through the corner,
  `∏` of the physical cell widths along the other directions; `(nothing,)` in two directions, where it is
  one;
- `corner`: per rotation pair `(a, b)`, over its corners, the area `μ` of the dual cell around the corner
  in the `(a, b)` plane, spanned by the four centres around it: on a rectilinear grid
  `|h_a h_b Δξ_a Δξ_b|` with the scale factors read at the corner and `Δξ` the coordinate distances between
  those centres;
- `orientation`: per direction `σ = ±1`, the sign of the coordinate's step along the index. A face value
  and a derivative along `d` are taken along increasing coordinate, whichever way an axis is stored; on a
  curvilinear grid, whose directions have no coordinate of their own, along increasing index.

Every area, gap, edge length and corner area is positive. Under `Neumann()` the two areas differ at the
ends of bounded directions only, where data crosses and the operators do not couple.

An operator reads it in `execution`'s memory: one held on the host is moved there for the call, and one
moved once with `Execution.on_backend` is read as it is.
"""
struct FaceMetrics{N,T,Gr<:Grids.AbstractGrid,B<:AbstractBoundaryCondition,A<:NTuple{N,AbstractArray{T,N}},
                   F<:NTuple{N,AbstractArray{T,N}},P<:NTuple{N,AbstractArray{T,N}},E<:Tuple,M<:Tuple}
    grid::Gr
    boundary::B
    area_op::A
    area_flux::F
    gap::P
    edge::E
    corner::M
    orientation::NTuple{N,T}
end

Base.show(io::IO, fm::FaceMetrics{N,T}) where {N,T} =
    print(io, "FaceMetrics{", N, ",", T, "}(", join(Grids.size_tuple(fm.grid), "×"), " cells, ",
          nameof(typeof(fm.boundary)), ")")

"""
    face_metrics(grid, boundary = Neumann()) -> FaceMetrics

The [`FaceMetrics`](@ref) of a [`Grids.StructuredGrid`](@ref), of the centres of a
[`Grids.StaggeredGrid`](@ref), or of a two-dimensional [`Grids.CurvilinearGrid`](@ref) holding its cell
vertices, under `boundary`.

On a rectilinear grid a face normal to `d` has area `∏_{e≠d} |h_e| w_e` and gap `|h_d| Δξ_d`, the scale
factors read at the face, `w_e` the cell widths and `Δξ_d` the coordinate distance between the centres it
separates, so [`gradient!`](@ref), [`divergence!`](@ref) and [`curl!`](@ref) are exact for a linear field
on a Cartesian one. On a curvilinear grid a face is the edge between two cell vertices, its area their
distance and its gap the distance between the two centres it separates.
"""
function face_metrics(grid::Grids.StructuredGrid{T,G,N}, bc::AbstractBoundaryCondition = Neumann()) where {T,G,N}
    layout = metric_layout(grid)
    faces = ntuple(d -> Grids.face_coordinates(grid, d), Val(N))
    area = ntuple(d -> _face_area_array(grid, faces[d], d, bc, layout), Val(N))
    flux = bc isa Dirichlet ? area : ntuple(d -> _face_area_array(grid, faces[d], d, Dirichlet(), layout), Val(N))
    gap = ntuple(d -> _face_gap_array(grid, faces[d], d, layout), Val(N))
    corner = map(ab -> _corner_array(grid, faces[ab[1]], faces[ab[2]], ab[1], ab[2], layout),
                 rotation_pairs(Val(N)))
    σ = ntuple(d -> T(Axes.wrap_sign(Grids.coordinates(grid, d))), Val(N))
    return FaceMetrics(grid, bc, area, flux, gap, _edge_lengths(grid, faces, layout, Val(N)), corner, σ)
end

face_metrics(sg::Grids.StaggeredGrid, bc::AbstractBoundaryCondition = Neumann()) =
    face_metrics(Grids.center_grid(sg), bc)

function _face_area_array(grid::Grids.StructuredGrid{T}, fd, d::Int, bc, layout) where {T}
    return metric_array(F -> _face_area(grid, fd, F, d, bc, T), Grids.face_dims(grid, d), layout, T)
end

# The coupling area of face `F` along `d`: zero against an inactive cell, and past a bounded end zero unless
# the condition is `Dirichlet`.
@inline function _face_area(grid, fd, F::CartesianIndex, d::Int, bc, ::Type{T}) where {T}
    Cl = Grids.cell_below(grid, F, d)
    Cu = Grids.cell_above(grid, F, d)
    if Cl === nothing || Cu === nothing
        C = Cu === nothing ? Cl : Cu
        C === nothing && return zero(T)
        (bc isa Dirichlet && Grids.isactive(grid, Tuple(C)...)) || return zero(T)
        return _geometric_area(grid, fd, F, d, C, T)
    end
    (Grids.isactive(grid, Tuple(Cl)...) && Grids.isactive(grid, Tuple(Cu)...)) || return zero(T)
    return _geometric_area(grid, fd, F, d, Cu, T)
end

# `∏_{e≠d} |h_e| w_e` at the face: its coordinate along `d` from the face axis `fd`, and the centre
# coordinates of the cell `ref` it bounds along every other direction.
@inline function _geometric_area(grid, fd, F::CartesianIndex{N}, d::Int, ref::CartesianIndex{N},
                                 ::Type{T}) where {N,T}
    c = Grids.coords(Tuple, grid, Tuple(ref)...)
    x = T(@inbounds fd[F[d]])
    h = Geometry.scale_factors(Grids.grid_geometry(grid), ntuple(e -> e == d ? x : T(c[e]), Val(N)))
    return prod(ntuple(e -> e == d ? one(T) : abs(T(h[e])) * T(Grids.cell_width(grid, e, ref[e])), Val(N)))
end

function _face_gap_array(grid::Grids.StructuredGrid{T}, fd, d::Int, layout) where {T}
    return metric_array(F -> _face_gap(grid, fd, F, d, T), Grids.face_dims(grid, d), layout, T)
end

# `|h_d|` at the face times the coordinate distance between the two centres it separates, or from its one
# centre to the face.
@inline function _face_gap(grid, fd, F::CartesianIndex{N}, d::Int, ::Type{T}) where {N,T}
    Cl = Grids.cell_below(grid, F, d)
    Cu = Grids.cell_above(grid, F, d)
    Cl === nothing && Cu === nothing && return one(T)
    ref = Cu === nothing ? Cl : Cu
    c = Grids.coords(Tuple, grid, Tuple(ref)...)
    f = T(@inbounds fd[F[d]])
    Δ = (Cl === nothing || Cu === nothing) ? T(c[d]) - f : T(first(Grids.local_spacing(grid, d, F[d])))
    h = Geometry.scale_factors(Grids.grid_geometry(grid), ntuple(e -> e == d ? f : T(c[e]), Val(N)))
    return abs(T(h[d]) * Δ)
end

_edge_lengths(_, _, _, ::Val{1}) = ()
_edge_lengths(_, _, _, ::Val{2}) = (nothing,)
_edge_lengths(grid, faces, layout, ::Val{N}) where {N} =
    map(ab -> _edge_array(grid, faces[ab[1]], faces[ab[2]], ab[1], ab[2], layout), rotation_pairs(Val(N)))

function _edge_array(grid::Grids.StructuredGrid{T}, fa, fb, a::Int, b::Int, layout) where {T}
    return metric_array(Qc -> _edge_length(grid, fa, fb, Qc, a, b, T), Grids.corner_dims(grid, a, b), layout, T)
end

# The corner sits at face coordinates along `a` and `b` and at cell centres along every other direction,
# where the scale factors are read.
@inline function _edge_length(grid, fa, fb, Qc::CartesianIndex{N}, a::Int, b::Int, ::Type{T}) where {N,T}
    Q = Grids.corner_to_face(grid, Qc, a, b)
    ref = ntuple(e -> e == a || e == b ? 1 : Q[e], Val(N))
    c = Grids.coords(Tuple, grid, ref...)
    x = ntuple(e -> e == a ? T(@inbounds fa[Q[a]]) : e == b ? T(@inbounds fb[Q[b]]) : T(c[e]), Val(N))
    h = Geometry.scale_factors(Grids.grid_geometry(grid), x)
    return prod(ntuple(e -> e == a || e == b ? one(T) : abs(T(h[e])) * T(Grids.cell_width(grid, e, Q[e])), Val(N)))
end

@inline _edge(::Nothing, _, ::Type{T}) where {T} = one(T)
@inline _edge(ℓ::AbstractArray, Q, ::Type{T}) where {T} = @inbounds T(ℓ[Q])

function _corner_array(grid::Grids.StructuredGrid{T}, fa, fb, a::Int, b::Int, layout) where {T}
    return metric_array(Qc -> _corner_area(grid, fa, fb, Qc, a, b, T), Grids.corner_dims(grid, a, b), layout, T)
end

# `|h_a h_b| Δξ_a Δξ_b` at the corner, `Δξ` the coordinate distance between the centres either side of it.
@inline function _corner_area(grid, fa, fb, Qc::CartesianIndex{N}, a::Int, b::Int, ::Type{T}) where {N,T}
    Q = Grids.corner_to_face(grid, Qc, a, b)
    ref = ntuple(e -> e == a || e == b ? 1 : Q[e], Val(N))
    c = Grids.coords(Tuple, grid, ref...)
    x = ntuple(e -> e == a ? T(@inbounds fa[Q[a]]) : e == b ? T(@inbounds fb[Q[b]]) : T(c[e]), Val(N))
    h = Geometry.scale_factors(Grids.grid_geometry(grid), x)
    Δa = T(first(Grids.local_spacing(grid, a, Q[a])))
    Δb = T(first(Grids.local_spacing(grid, b, Q[b])))
    return abs(T(h[a]) * Δa) * abs(T(h[b]) * Δb)
end

function face_metrics(grid::Grids.CurvilinearGrid{T,G,2}, bc::AbstractBoundaryCondition = Neumann()) where {T,G}
    Grids.has_corners(grid) || throw(ArgumentError(
        "the face metric of a CurvilinearGrid is built from its cell vertices, which this grid does not " *
        "hold; build it with `corners` given"))
    k = Grids.corners(grid)
    area = ntuple(d -> _curvilinear_area(grid, k, d, bc, T), Val(2))
    flux = bc isa Dirichlet ? area : ntuple(d -> _curvilinear_area(grid, k, d, Dirichlet(), T), Val(2))
    gap = ntuple(d -> _curvilinear_gap(grid, k, d, T), Val(2))
    return FaceMetrics(grid, bc, area, flux, gap, (nothing,), (_curvilinear_corner(grid, T),), (one(T), one(T)))
end

# The area of the quadrilateral of the four centres around each corner, half the cross product of its
# diagonals; a centre across a seam is taken one period back.
function _curvilinear_corner(grid, ::Type{T}) where {T}
    c = Grids.coordinates(grid)
    dims = Grids.corner_dims(grid, 1, 2)
    n = Grids.size_tuple(grid)
    P = ntuple(d -> Grids.isperiodic(grid, d) ? T(Grids.period(grid, d)) : zero(T), Val(2))
    centre(i, j) = (T(c[1][i < 1 ? n[1] : i, j < 1 ? n[2] : j]) - (i < 1 ? P[1] : zero(T)),
                    T(c[2][i < 1 ? n[1] : i, j < 1 ? n[2] : j]) - (j < 1 ? P[2] : zero(T)))
    μ = zeros(T, dims)
    @inbounds for Qc in CartesianIndices(dims)
        Q = Grids.corner_to_face(grid, Qc, 1, 2)
        p00 = centre(Q[1] - 1, Q[2] - 1)
        p10 = centre(Q[1], Q[2] - 1)
        p11 = centre(Q[1], Q[2])
        p01 = centre(Q[1] - 1, Q[2])
        d1 = (p11[1] - p00[1], p11[2] - p00[2])
        d2 = (p01[1] - p10[1], p01[2] - p10[2])
        μ[Qc] = abs(d1[1] * d2[2] - d1[2] * d2[1]) / 2
    end
    return μ
end

# Whether a face couples at all: the mask and the condition, with no geometry.
@inline function _face_open(grid, F::CartesianIndex, d::Int, bc)
    Cl = Grids.cell_below(grid, F, d)
    Cu = Grids.cell_above(grid, F, d)
    Cl === nothing && Cu === nothing && return false
    if Cl === nothing || Cu === nothing
        C = Cu === nothing ? Cl : Cu
        return bc isa Dirichlet && Grids.isactive(grid, Tuple(C)...)
    end
    return Grids.isactive(grid, Tuple(Cl)...) && Grids.isactive(grid, Tuple(Cu)...)
end

# Face `F` normal to `d` spans the vertices at `F` and at `F` stepped along the other direction.
function _curvilinear_area(grid, k, d::Int, bc, ::Type{T}) where {T}
    e = d == 1 ? 2 : 1
    dims = Grids.face_dims(grid, d)
    a = zeros(T, dims)
    @inbounds for F in CartesianIndices(dims)
        _face_open(grid, F, d, bc) || continue
        Q = F + Grids._unit_index(Val(2), e)
        a[F] = hypot(T(k[1][Q]) - T(k[1][F]), T(k[2][Q]) - T(k[2][F]))
    end
    return a
end

# The distance between the centres a face separates, across a seam one period back; half the cell's own
# extent between its two bounding vertices where the face has one cell.
function _curvilinear_gap(grid, k, d::Int, ::Type{T}) where {T}
    c = Grids.coordinates(grid)
    dims = Grids.face_dims(grid, d)
    shift = Grids.isperiodic(grid, d) ? T(Grids.period(grid, d)) : zero(T)
    g = ones(T, dims)
    @inbounds for F in CartesianIndices(dims)
        Cl = Grids.cell_below(grid, F, d)
        Cu = Grids.cell_above(grid, F, d)
        if Cl !== nothing && Cu !== nothing
            s = F[d] == 1 ? shift : zero(T)
            g[F] = hypot(T(c[1][Cu]) - (T(c[1][Cl]) - (d == 1 ? s : zero(T))),
                         T(c[2][Cu]) - (T(c[2][Cl]) - (d == 2 ? s : zero(T))))
        else
            C = Cu === nothing ? Cl : Cu
            C === nothing && continue
            Q = C + Grids._unit_index(Val(2), d)
            g[F] = hypot(T(k[1][Q]) - T(k[1][C]), T(k[2][Q]) - T(k[2][C])) / 2
        end
    end
    return g
end

# ---- sweeps ---------------------------------------------------------------------

# `f(I)` at every index of `dims`. On the host a chunk walks its run of linear indices carrying the
# Cartesian index from one cell to the next; elsewhere each index is a work item.
@inline function _each_cell(f::F, dims::NTuple{N,Int}, b::Execution._HostBackend,
                            outs::Execution.Written...) where {F,N}
    cart = CartesianIndices(dims)
    Execution.run_chunks(length(cart), b, outs...) do rng
        _walk_cells(f, cart, rng)
    end
    return nothing
end

@inline function _each_cell(f::F, dims::NTuple{N,Int}, b::CB.AbstractExecutionBackend,
                            outs::Execution.Written...) where {F,N}
    cart = CartesianIndices(dims)
    Execution.run_indices(length(cart), b, outs...) do lin
        f(@inbounds cart[lin])
        return nothing
    end
    return nothing
end

@inline function _walk_cells(f::F, cart::CartesianIndices, rng::UnitRange{Int}) where {F}
    isempty(rng) && return nothing
    I = @inbounds cart[first(rng)]
    for _ in rng
        f(I)
        nxt = iterate(cart, I)
        nxt === nothing && return nothing
        I = nxt[1]
    end
    return nothing
end

# `f(Val(d))` for `d = 1:N`, written out, so a tuple indexed by `d` inside `f` reads one fixed slot.
@generated function _each_dir(f::F, ::Val{N}) where {F,N}
    ex = Expr(:block)
    for d in 1:N
        push!(ex.args, :(f(Val($d))))
    end
    push!(ex.args, :(return nothing))
    return ex
end

@inline _val(::Val{d}) where {d} = d

# The value on one side of a face, or the zero ghost past the end of a bounded direction.
@inline function _cell_value(χ, grid, F::CartesianIndex, d::Int, upper::Bool, ::Type{T}) where {T}
    C = upper ? Grids.cell_above(grid, F, d) : Grids.cell_below(grid, F, d)
    C === nothing && return zero(T)
    return @inbounds T(χ[C])
end

@inline _side_count(grid, F::CartesianIndex, d::Int, upper::Bool) =
    (upper ? Grids.cell_above(grid, F, d) : Grids.cell_below(grid, F, d)) === nothing ? 0 : 1

function _check_cells(a::AbstractArray, grid, what)
    size(a) == Grids.size_tuple(grid) || throw(DimensionMismatch(
        "$what holds $(size(a)) values for $(Grids.size_tuple(grid)) cells"))
    return nothing
end

function _check_faces(v::Tuple, grid::Grids.FaceIndexedGrid{T,G,N}, what) where {T,G,N}
    length(v) == N || throw(DimensionMismatch(
        "a $N-direction grid has $N face arrays; $what has $(length(v))"))
    for d in 1:N
        size(v[d]) == Grids.face_dims(grid, d) || throw(DimensionMismatch(
            "$what[$d] holds $(size(v[d])) values for $(Grids.face_dims(grid, d)) faces"))
    end
    return nothing
end

function _check_corners(W::Tuple, grid::Grids.FaceIndexedGrid{T,G,N}, what) where {T,G,N}
    pairs = rotation_pairs(Val(N))
    length(W) == length(pairs) || throw(DimensionMismatch(
        "a $N-direction grid has $(length(pairs)) corner arrays; $what has $(length(W))"))
    for (p, (a, b)) in enumerate(pairs)
        size(W[p]) == Grids.corner_dims(grid, a, b) || throw(DimensionMismatch(
            "$what[$p] holds $(size(W[p])) values for $(Grids.corner_dims(grid, a, b)) corners"))
    end
    return nothing
end

# ---- the operators --------------------------------------------------------------

"""
    allocate_faces(T, grid; execution = SerialBackend()) -> NTuple{N,AbstractArray{T,N}}
    allocate_corners(T, grid; execution = SerialBackend()) -> NTuple{P,AbstractArray{T,N}}

Zeroed face arrays, one per direction shaped by [`Grids.face_dims`](@ref), and corner arrays, one per
[`rotation_pairs`](@ref) pair shaped by [`Grids.corner_dims`](@ref), in the memory `execution`'s loops
write.
"""
allocate_faces(::Type{T}, grid::Grids.FaceIndexedGrid{TT,G,N};
               execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {T,TT,G,N} =
    ntuple(d -> fill!(Execution.allocate(execution, T, Grids.face_dims(grid, d)...), zero(T)), Val(N))

allocate_corners(::Type{T}, grid::Grids.FaceIndexedGrid{TT,G,N};
                 execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {T,TT,G,N} =
    map(ab -> fill!(Execution.allocate(execution, T, Grids.corner_dims(grid, ab[1], ab[2])...), zero(T)),
        rotation_pairs(Val(N)))

"""
    gradient!(g, χ, fm::FaceMetrics; execution = SerialBackend()) -> g

`g[d][F] = σ_d (χ_above − χ_below)/gap` on every face of nonzero `area_op` normal to `d`, a zero ghost
past a `Dirichlet` end, and zero on every other face; `σ_d` is `fm.orientation[d]`.
"""
function gradient!(g::NTuple{N,AbstractArray}, χ::AbstractArray, fm::FaceMetrics{N,T};
                   execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_cells(χ, fm.grid, "χ")
    _check_faces(g, fm.grid, "g")
    b = Execution.resolve(execution)
    m = Execution.on_backend(b, fm)
    grid = m.grid
    _each_dir(Val(N)) do vd
        d = _val(vd)
        _gradient_dir!(g[d], χ, grid, m.area_op[d], m.gap[d], m.orientation[d], d, b)
    end
    return g
end

function _gradient_dir!(gd::AbstractArray{S}, χ, grid, ad, gpd, σ, d::Int, b) where {S}
    _each_cell(Grids.face_dims(grid, d), b, Execution.Written(gd)) do F
        @inbounds gd[F] = iszero(ad[F]) ? zero(S) :
                          S(σ) * (_cell_value(χ, grid, F, d, true, S) - _cell_value(χ, grid, F, d, false, S)) /
                          gpd[F]
        return nothing
    end
    return nothing
end

"""
    divergence!(δ, v, fm::FaceMetrics; execution = SerialBackend()) -> δ

`δ[I] = (1/V_I) Σ_d σ_d ((A v)[face above] − (A v)[face below])` over the `area_op` of each face of cell
`I`, `V_I` its measure, and zero on an inactive cell: the negative adjoint of [`gradient!`](@ref) under
the cell measure and the face weight `area_op · gap`, so `L = D G` is symmetric.
"""
function divergence!(δ::AbstractArray, v::NTuple{N,AbstractArray}, fm::FaceMetrics{N,T};
                     execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_cells(δ, fm.grid, "δ")
    _check_faces(v, fm.grid, "v")
    b = Execution.resolve(execution)
    m = Execution.on_backend(b, fm)
    grid = m.grid
    msk = Grids.mask(grid)
    meas = Grids.measure(grid)
    area = m.area_op
    σ = m.orientation
    S = eltype(δ)
    _each_cell(Grids.size_tuple(grid), b, Execution.Written(δ)) do I
        @inbounds δ[I] = msk[I] ? S(_flux_out(v, area, σ, grid, I, Val(N), S) / meas[I]) : zero(S)
        return nothing
    end
    return δ
end

@inline _flux_out(v, area, σ, grid, I::CartesianIndex, ::Val{0}, ::Type{S}) where {S} = zero(S)
@inline function _flux_out(v, area, σ, grid, I::CartesianIndex, ::Val{d}, ::Type{S}) where {d,S}
    @inbounds begin
        ad, vd = area[d], v[d]
        Fhi = Grids.face_above(grid, I, d)
        here = S(σ[d]) * (ad[Fhi] * vd[Fhi] - ad[I] * vd[I])
    end
    return _flux_out(v, area, σ, grid, I, Val(d - 1), S) + here
end

"""
    laplacian!(out, χ, fm::FaceMetrics, g; execution = SerialBackend()) -> out

`L χ = D G χ` from [`gradient!`](@ref) into the face arrays `g`, then [`divergence!`](@ref). For repeated
application [`laplacian_coefficients`](@ref) holds the same operator in one pass.
"""
function laplacian!(out::AbstractArray, χ::AbstractArray, fm::FaceMetrics{N}, g::NTuple{N,AbstractArray};
                    execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N}
    gradient!(g, χ, fm; execution = execution)
    return divergence!(out, g, fm; execution = execution)
end

"""
    corner_mask(fm::FaceMetrics, a, b) -> AbstractArray{Bool,N}

The `(a, b)` corners in the complex: those whose four surrounding faces all couple, so a loop of active
cells closes around them. [`Grids.AllActive`](@ref) where every one does.
"""
function corner_mask(fm::FaceMetrics{N}, a::Integer, b::Integer) where {N}
    grid = fm.grid
    dims = Grids.corner_dims(grid, a, b)
    aa, ab = fm.area_op[a], fm.area_op[b]
    m = falses(dims)
    @inbounds for Qc in CartesianIndices(dims)
        m[Qc] = _corner_closes(grid, aa, ab, Grids.corner_to_face(grid, Qc, a, b), Int(a), Int(b))
    end
    return all(m) ? Grids.AllActive(dims) : m
end

@inline function _corner_closes(grid, aa, ab, Q::CartesianIndex, a::Int, b::Int)
    Cla = Grids.cell_below(grid, Q, a)
    Cua = Grids.cell_above(grid, Q, a)
    Clb = Grids.cell_below(grid, Q, b)
    Cub = Grids.cell_above(grid, Q, b)
    (Cla === nothing || Cua === nothing || Clb === nothing || Cub === nothing) && return false
    return @inbounds !(iszero(ab[Cla]) || iszero(ab[Cua]) || iszero(aa[Clb]) || iszero(aa[Cub]))
end

"""
    curl!(W, v, fm::FaceMetrics; execution = SerialBackend()) -> W

`W_ab = ∂_a v_b − ∂_b v_a` on the `(a, b)` corners from face-normal velocity `v`, one array per
[`rotation_pairs`](@ref) pair: the circulation `σ_a Δ_a(v_b g_b) − σ_b Δ_b(v_a g_a)` around the corner
over the area `μ` of the dual cell it bounds, `fm.corner`. A corner is written where it is in the
complex, [`corner_mask`](@ref), and zero elsewhere.
"""
function curl!(W::Tuple{Vararg{AbstractArray}}, v::NTuple{N,AbstractArray}, fm::FaceMetrics{N,T};
               execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_faces(v, fm.grid, "v")
    _check_corners(W, fm.grid, "W")
    b = Execution.resolve(execution)
    m = Execution.on_backend(b, fm)
    grid = m.grid
    pairs = rotation_pairs(Val(N))
    σ = m.orientation
    _each_dir(Val(n_rotation_components(N))) do vp
        p = _val(vp)
        a, c = pairs[p]
        _curl_pair!(W[p], v[a], v[c], m.area_op[a], m.area_op[c], m.gap[a], m.gap[c], m.corner[p],
                    σ[a], σ[c], grid, a, c, b)
    end
    return W
end

function _curl_pair!(Wp::AbstractArray{S}, va, vb, aa, ab, ga, gb, μ, σa, σb, grid, a::Int, b::Int,
                     bk) where {S}
    _each_cell(size(Wp), bk, Execution.Written(Wp)) do Qc
        @inbounds begin
            Q = Grids.corner_to_face(grid, Qc, a, b)
            if _corner_closes(grid, aa, ab, Q, a, b)
                # The `b`-faces either side of the corner along `a` are indexed by the cells there, `Q`
                # above and `Cla` below, and the `a`-faces likewise along `b`.
                Cla = something(Grids.cell_below(grid, Q, a))
                Clb = something(Grids.cell_below(grid, Q, b))
                circ_b = vb[Q] * gb[Q] - vb[Cla] * gb[Cla]
                circ_a = va[Q] * ga[Q] - va[Clb] * ga[Clb]
                Wp[Qc] = (S(σa) * circ_b - S(σb) * circ_a) / μ[Qc]
            else
                Wp[Qc] = zero(S)
            end
        end
        return nothing
    end
    return nothing
end

"""
    rotational_velocity!(u, R, fm::FaceMetrics; execution = SerialBackend()) -> u

`u_a = −Σ_b ∂_b R_ab` on the faces, from the rotation potential `R` on the corners, one array per
[`rotation_pairs`](@ref) pair: the flux `area_op · u` through a face is `σ_b` times the metric-free
difference of `R ℓ` across its bounding edges, `ℓ` the edge length, so `D u = 0` on every grid. A corner
outside `R`'s arrays, past the end of a bounded direction, holds zero.

For `R` zero off [`corner_mask`](@ref), `⟨u, v⟩ = −⟨R, curl v⟩` under the face weight `area_op · gap` and
the corner weight `ℓ μ`, with [`curl!`](@ref).
"""
function rotational_velocity!(u::NTuple{N,AbstractArray}, R::Tuple{Vararg{AbstractArray}},
                              fm::FaceMetrics{N,T};
                              execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_faces(u, fm.grid, "u")
    _check_corners(R, fm.grid, "R")
    b = Execution.resolve(execution)
    m = Execution.on_backend(b, fm)
    grid = m.grid
    terms = _rotation_terms(Val(N))
    _each_dir(Val(N)) do vc
        c = _val(vc)
        _rotational_dir!(u[c], R, m.edge, m.area_op[c], m.orientation, terms[c], grid, c, b)
    end
    return u
end

function _rotational_dir!(uc::AbstractArray{S}, R, edge, ac, σ, terms, grid, c::Int, b) where {S}
    _each_cell(Grids.face_dims(grid, c), b, Execution.Written(uc)) do F
        @inbounds if iszero(ac[F])
            uc[F] = zero(S)
        else
            acc = zero(S)
            # Along `e` the `c`-face holds a cell index; its two corners are the `e`-faces either side.
            for (e, p, s) in terms
                Rp, ℓp = R[p], edge[p]
                Qlo = Grids.face_to_corner(grid, Grids.face_below(F, e), c, e)
                Qhi = Grids.face_to_corner(grid, Grids.face_above(grid, F, e), c, e)
                lo = checkbounds(Bool, Rp, Qlo) ? S(Rp[Qlo]) * _edge(ℓp, Qlo, S) : zero(S)
                hi = checkbounds(Bool, Rp, Qhi) ? S(Rp[Qhi]) * _edge(ℓp, Qhi, S) : zero(S)
                acc += S(s) * S(σ[e]) * (hi - lo)
            end
            uc[F] = acc / ac[F]
        end
        return nothing
    end
    return nothing
end

"""
    to_faces!(v, u, fm::FaceMetrics; execution = SerialBackend()) -> v

The face-normal velocity `v` of a cell-centred field `u`, one array per direction: the mean of the two
cells a face separates, the one cell's value at the end of a bounded direction, and zero on a face of zero
`area_flux`. `u` is a tuple of one array of the grid's shape per direction.
"""
function to_faces!(v::NTuple{N,AbstractArray}, u::NTuple{N,AbstractArray}, fm::FaceMetrics{N,T};
                   execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_faces(v, fm.grid, "v")
    foreach(ud -> _check_cells(ud, fm.grid, "a component of u"), u)
    b = Execution.resolve(execution)
    m = Execution.on_backend(b, fm)
    grid = m.grid
    _each_dir(Val(N)) do vd
        d = _val(vd)
        _to_faces_dir!(v[d], u[d], m.area_flux[d], grid, d, b)
    end
    return v
end

function _to_faces_dir!(vd::AbstractArray{S}, ud, ad, grid, d::Int, b) where {S}
    _each_cell(Grids.face_dims(grid, d), b, Execution.Written(vd)) do F
        @inbounds if iszero(ad[F])
            vd[F] = zero(S)
        else
            w = _side_count(grid, F, d, false) + _side_count(grid, F, d, true)
            vd[F] = (_cell_value(ud, grid, F, d, false, S) + _cell_value(ud, grid, F, d, true, S)) / S(w)
        end
        return nothing
    end
    return nothing
end

"""
    to_centres!(u, v, fm::FaceMetrics; execution = SerialBackend()) -> u

The cell-centred field of the face-normal velocity `v`: along each direction the mean over a cell's two
faces of nonzero `area_flux`, zero where it has none and on an inactive cell. On a uniform axis
`to_centres! ∘ to_faces!` has the symbol `cos²(kh/2)`.
"""
function to_centres!(u::NTuple{N,AbstractArray}, v::NTuple{N,AbstractArray}, fm::FaceMetrics{N,T};
                     execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_faces(v, fm.grid, "v")
    foreach(ud -> _check_cells(ud, fm.grid, "a component of u"), u)
    b = Execution.resolve(execution)
    m = Execution.on_backend(b, fm)
    grid = m.grid
    msk = Grids.mask(grid)
    _each_dir(Val(N)) do vd
        d = _val(vd)
        _to_centres_dir!(u[d], v[d], m.area_flux[d], msk, grid, d, b)
    end
    return u
end

function _to_centres_dir!(ud::AbstractArray{S}, vd, ad, msk, grid, d::Int, b) where {S}
    _each_cell(Grids.size_tuple(grid), b, Execution.Written(ud)) do I
        @inbounds if msk[I]
            Fhi = Grids.face_above(grid, I, d)
            olo, ohi = !iszero(ad[I]), !iszero(ad[Fhi])
            w = olo + ohi
            ud[I] = iszero(w) ? zero(S) : ((olo ? S(vd[I]) : zero(S)) + (ohi ? S(vd[Fhi]) : zero(S))) / S(w)
        else
            ud[I] = zero(S)
        end
        return nothing
    end
    return nothing
end
