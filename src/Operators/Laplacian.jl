# ---------------------------------------------------------------------------
# The assembled Laplacian `L = D G`, its null space, and its Galerkin coarsening
# ---------------------------------------------------------------------------
#
# Written as a flux balance, `L`'s row at cell `I` is
#
#     (L Φ)_I = (1/V_I) Σ_F c_F (Φ_nbr(F) − Φ_I),      c_F = A_F / g_F
#
# over the faces of `I`: a face of zero area does not appear, and a face past a `Dirichlet` end reads a
# zero ghost. A coarse level's operator is `R L P` for injection `P` and the measure-weighted average `R`,
# which in this flux form is a sum of fine conductances per coarse face and of fine measures per coarse
# cell.

"""
    ComponentNullspace

The constants `L` annihilates, one per closed component: a connected component of the graph of faces of
nonzero coefficient on which `L·1 = 0`. Component `k`'s cells are the linear indices
`perm[offsets[k]:offsets[k+1]-1]` and `totals[k]` its measure; `full` marks one closed component holding
every active cell, which needs no `perm`. Kernels read `perm`, in the execution backend's memory;
`offsets` and `totals` stay in host memory, read by the loop over components.
"""
struct ComponentNullspace{P<:AbstractVector{<:Integer},O<:AbstractVector{Int},V<:AbstractVector{<:AbstractFloat}}
    full::Bool
    perm::P
    offsets::O
    totals::V
end

"""
    LaplacianCoefficients(grid, coef, measure)

`L` on a [`Grids.FaceIndexedGrid`](@ref) as data: the face coefficients `coef[d][F] = A_F/g_F`, one
array per direction shaped by [`Grids.face_dims`](@ref), and the cell measures, zero on an inactive cell.
Its fields:

- `grid`, `coef`, `measure`;
- `diag`: `(L e_I)_I`, minus the sum of a cell's own face coefficients over its measure;
- `total`: the measure of the active cells;
- `singular`, `nullspace`: whether `L` has a null space, and the [`ComponentNullspace`](@ref) spanning it.

[`laplacian_coefficients`](@ref) builds it from a [`FaceMetrics`](@ref), [`galerkin_coefficients`](@ref)
from a finer level. An operator reads it in `execution`'s memory, as it reads a `FaceMetrics`.
"""
struct LaplacianCoefficients{N,T,Gr<:Grids.AbstractGrid,C<:NTuple{N,AbstractArray{T,N}},D<:AbstractArray{T,N},
                             M<:AbstractArray{T,N},NS}
    grid::Gr
    coef::C
    diag::D
    measure::M
    total::T
    singular::Bool
    nullspace::NS
end

function LaplacianCoefficients(grid::Grids.FaceIndexedGrid{TG,G,N}, coef::NTuple{N,AbstractArray{T,N}},
                               measure::AbstractArray{T,N}) where {TG,G,N,T}
    diag = _dense_diagonal(grid, coef, measure, T)
    ns = _nullspace(grid, coef, diag, measure, Val(N), T)
    return LaplacianCoefficients{N,T,typeof(grid),typeof(coef),typeof(diag),typeof(measure),typeof(ns)}(
        grid, coef, diag, measure, T(sum(measure)), !isempty(ns.totals), ns)
end

Base.show(io::IO, c::LaplacianCoefficients{N,T}) where {N,T} =
    print(io, "LaplacianCoefficients{", N, ",", T, "}(", join(Grids.size_tuple(c.grid), "×"), " cells",
          c.singular ? ", singular" : "", ")")

"""
    laplacian_coefficients(fm::FaceMetrics) -> LaplacianCoefficients

`L = D G` of [`divergence!`](@ref) and [`gradient!`](@ref) under `fm`, with the coefficients `A/g` of
`fm.area_op` and `fm.gap`: separable where the metrics are, and the grid's own measure on an unmasked
grid.
"""
function laplacian_coefficients(fm::FaceMetrics{N,T}) where {N,T}
    coef = ntuple(d -> _quotient(fm.area_op[d], fm.gap[d]), Val(N))
    return LaplacianCoefficients(fm.grid, coef, _measure_array(fm.grid, Grids.mask(fm.grid), T))
end

# `A/g` per face, and zero on a face of no area, whose gap can be zero too: a cell centred on a pole
# meets its face at the pole.
@inline _conductance(a, g) = iszero(a) ? zero(a / one(g)) : a / g
@inline _quotient(a::AbstractArray, g::AbstractArray) = _conductance.(a, g)
@inline _quotient(a::Grids.SeparableMeasure, g::Grids.SeparableMeasure) =
    Grids.SeparableMeasure(map((x, y) -> _conductance.(x, y), a.factors, g.factors))

@inline _measure_array(grid, ::Grids.AllActive, ::Type{T}) where {T} = _measure_of(Grids.measure(grid), T)
function _measure_array(grid, msk, ::Type{T}) where {T}
    m = zeros(T, Grids.size_tuple(grid))
    @inbounds for I in CartesianIndices(m)
        msk[I] && (m[I] = T(Grids.measure(grid, Tuple(I)...)))
    end
    return m
end

@inline _measure_of(m::AbstractArray{T}, ::Type{T}) where {T} = m
@inline _measure_of(m::AbstractArray, ::Type{T}) where {T} = T.(m)

function _dense_diagonal(grid, coef::NTuple{N,Any}, meas, ::Type{T}) where {N,T}
    diag = zeros(T, Grids.size_tuple(grid))
    @inbounds for I in CartesianIndices(diag)
        m = meas[I]
        iszero(m) && continue
        diag[I] = -_coef_sum(coef, grid, I, Val(N), T) / m
    end
    return diag
end

@inline _coef_sum(coef, grid, I, ::Val{0}, ::Type{T}) where {T} = zero(T)
@inline _coef_sum(coef, grid, I, ::Val{d}, ::Type{T}) where {d,T} =
    @inbounds(coef[d][I] + coef[d][Grids.face_above(grid, I, d)]) + _coef_sum(coef, grid, I, Val(d - 1), T)

# The constants in `L`'s null space. With `1` on the active cells, `(L·1)_I` is nonzero only through a
# face whose far side reads zero — a `Dirichlet` ghost, or an inactive cell a coefficient reaches — and a
# component counts as closed where `L·1` stays within `√eps` of the diagonal's scale, so a face whose area
# is round-off, as at a pole, closes it.
function _nullspace(grid, coef, diag, meas, v::Val{N}, ::Type{T}) where {N,T}
    dims = Grids.size_tuple(grid)
    msk = Grids.mask(grid)
    labels = zeros(Int32, dims)
    ncomp = _label_components!(labels, grid, coef, v)
    ones_ = zeros(T, dims)
    @inbounds for I in CartesianIndices(dims)
        msk[I] && (ones_[I] = one(T))
    end
    Lone = similar(ones_)
    _apply_laplacian!(_lap_value, Lone, ones_, grid, msk, coef, meas, v, T, CB.SerialBackend())
    scale = zero(T)
    worst = zeros(T, ncomp)
    counts = zeros(Int, ncomp)
    @inbounds for I in CartesianIndices(dims)
        k = labels[I]
        k == 0 && continue
        scale = max(scale, abs(diag[I]))
        worst[k] = max(worst[k], abs(Lone[I]))
        counts[k] += 1
    end
    closed = falses(ncomp)
    @inbounds for k in 1:ncomp
        closed[k] = iszero(scale) || worst[k] ≤ sqrt(eps(T)) * scale
    end
    K = count(closed)
    K == 0 && return ComponentNullspace(false, Int[], [1], T[])
    ncomp == 1 && return ComponentNullspace(true, Int[], [1], [T(sum(meas))])
    slot = zeros(Int, ncomp)
    offsets = ones(Int, K + 1)
    s = 0
    for k in 1:ncomp
        closed[k] || continue
        s += 1
        slot[k] = s
        offsets[s + 1] = offsets[s] + counts[k]
    end
    perm = Vector{Int}(undef, offsets[end] - 1)
    next = offsets[1:K]
    totals = zeros(T, K)
    @inbounds for (lin, I) in enumerate(CartesianIndices(dims))
        k = labels[I]
        (k == 0 || slot[k] == 0) && continue
        s = slot[k]
        perm[next[s]] = lin
        next[s] += 1
        totals[s] += T(meas[I])
    end
    return ComponentNullspace(false, perm, offsets, totals)
end

# Each active cell's component of the coupling graph, joined across faces of nonzero coefficient.
function _label_components!(labels, grid, coef, v::Val{N}) where {N}
    msk = Grids.mask(grid)
    lin = LinearIndices(labels)
    cart = CartesianIndices(labels)
    stack = Int[]
    ncomp = Int32(0)
    @inbounds for I0 in cart
        (labels[I0] == 0 && msk[I0]) || continue
        ncomp += Int32(1)
        labels[I0] = ncomp
        push!(stack, lin[I0])
        while !isempty(stack)
            _join_neighbours!(labels, stack, lin, msk, grid, coef, cart[pop!(stack)], ncomp, v)
        end
    end
    return Int(ncomp)
end

@inline _join_neighbours!(labels, stack, lin, msk, grid, coef, I::CartesianIndex, k, ::Val{0}) = nothing
@inline function _join_neighbours!(labels, stack, lin, msk, grid, coef, I::CartesianIndex, k, ::Val{d}) where {d}
    @inbounds begin
        cd = coef[d]
        iszero(cd[I]) || _join!(labels, stack, lin, msk, Grids.cell_below(grid, I, d), k)
        Fhi = Grids.face_above(grid, I, d)
        iszero(cd[Fhi]) || _join!(labels, stack, lin, msk, Grids.cell_above(grid, Fhi, d), k)
    end
    return _join_neighbours!(labels, stack, lin, msk, grid, coef, I, k, Val(d - 1))
end

@inline _join!(labels, stack, lin, msk, ::Nothing, k) = nothing
@inline function _join!(labels, stack, lin, msk, J::CartesianIndex, k)
    @inbounds (msk[J] && labels[J] == 0) || return nothing
    @inbounds labels[J] = k
    push!(stack, @inbounds lin[J])
    return nothing
end

"""
    apply_laplacian!(out, Φ, c::LaplacianCoefficients; execution = SerialBackend()) -> out
    apply_laplacian!(f, out, Φ, c::LaplacianCoefficients; execution = SerialBackend()) -> out

`out = L Φ` from the coefficients, zero on every inactive cell. A face past the end of a bounded direction
contributes `c·(0 − Φ_I)`, the zero ghost of `Dirichlet`; under `Neumann` its coefficient is zero. A
nonzero coefficient into an inactive cell reads `Φ` there, which holds zero for an operator that pins it.

With `f`, `out[I] = f(I, (L Φ)_I)` on each active cell, `I` its `CartesianIndex`: the application and the
cellwise step that reads it in one pass, as a relaxation sweep or a residual takes them. `out` is a
different array from `Φ`, and `f` writes nothing a neighbouring cell's `(L Φ)` reads.
"""
apply_laplacian!(out::AbstractArray, Φ::AbstractArray, c::LaplacianCoefficients; kwargs...) =
    apply_laplacian!(_lap_value, out, Φ, c; kwargs...)

function apply_laplacian!(f::F, out::AbstractArray, Φ::AbstractArray, c::LaplacianCoefficients{N,T};
                          execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {F,N,T}
    _check_cells(out, c.grid, "out")
    _check_cells(Φ, c.grid, "Φ")
    b = Execution.resolve(execution)
    grid = Execution.on_backend(b, c.grid)
    _apply_laplacian!(f, out, Φ, grid, Grids.mask(grid), Execution.on_backend(b, c.coef),
                      Execution.on_backend(b, c.measure), Val(N), eltype(out), b)
    return out
end

@inline _lap_value(_I, L) = L

# `(L Φ)_I`: the flux balance over cell `I`'s faces, over its measure.
@inline _laplacian_at(Φ, grid, coef, meas, I::CartesianIndex, v::Val, ::Type{S}) where {S} =
    @inbounds S(_flux_sum(Φ, grid, coef, I, S(Φ[I]), v, S) / meas[I])

@inline _flux_sum(Φ, grid, coef, I::CartesianIndex, ΦI, ::Val{0}, ::Type{S}) where {S} = zero(S)
@inline function _flux_sum(Φ, grid, coef, I::CartesianIndex, ΦI, ::Val{d}, ::Type{S}) where {d,S}
    @inbounds begin
        cd = coef[d]
        Fhi = Grids.face_above(grid, I, d)
        here = cd[I] * (_cell_value(Φ, grid, I, d, false, S) - ΦI) +
               cd[Fhi] * (_cell_value(Φ, grid, Fhi, d, true, S) - ΦI)
    end
    return _flux_sum(Φ, grid, coef, I, ΦI, Val(d - 1), S) + here
end

_apply_laplacian!(f::F, out, Φ, grid, msk, coef, meas, v::Val, ::Type{S}, b) where {F,S} =
    _laplacian_cells!(f, out, Φ, grid, msk, coef, meas, v, S, b)

function _laplacian_cells!(f::F, out, Φ, grid, msk, coef, meas, v::Val, ::Type{S}, b) where {F,S}
    _each_cell(Grids.size_tuple(grid), b, Execution.Written(out)) do I
        @inbounds out[I] = msk[I] ? S(f(I, _laplacian_at(Φ, grid, coef, meas, I, v, S))) : zero(S)
        return nothing
    end
    return out
end

# On the host the trailing direction splits into slabs. Along a run of direction 1 whose other indices are
# all interior only the two end cells meet the boundary, and the cells between take `_laplacian_interior`,
# which reads at fixed offsets with no wrap or ghost test: a zero coefficient carries the mask there.
function _apply_laplacian!(f::F, out::Array, Φ::Array, grid, msk, coef, meas, v::Val{N}, ::Type{S},
                           b::Execution._HostBackend) where {F,N,S}
    N == 1 && return _laplacian_cells!(f, out, Φ, grid, msk, coef, meas, v, S, b)
    dims = Grids.size_tuple(grid)
    mid = CartesianIndices(ntuple(e -> dims[e + 1], Val(N - 2)))
    Execution.run_chunks(dims[N], b) do slab
        _laplacian_slab!(f, out, Φ, grid, msk, coef, meas, v, S, slab, dims, mid)
    end
    return out
end

function _laplacian_slab!(f::F, out, Φ, grid, msk, coef, meas, v::Val{N}, ::Type{S}, slab, dims,
                          mid) where {F,N,S}
    n1 = dims[1]
    cf = ntuple(d -> @inbounds(coef[d]), v)
    @inbounds for jN in slab, Im in mid
        rest = (Tuple(Im)..., jN)
        if n1 < 3 || _rest_on_edge(rest, dims)
            for i in 1:n1
                I = CartesianIndex(i, rest...)
                out[I] = msk[I] ? S(f(I, _laplacian_at(Φ, grid, cf, meas, I, v, S))) : zero(S)
            end
        else
            for i in (1, n1)
                I = CartesianIndex(i, rest...)
                out[I] = msk[I] ? S(f(I, _laplacian_at(Φ, grid, cf, meas, I, v, S))) : zero(S)
            end
            for i in 2:(n1 - 1)
                I = CartesianIndex(i, rest...)
                out[I] = msk[I] ? S(f(I, _laplacian_interior(Φ, cf, meas, I, v, S))) : zero(S)
            end
        end
    end
    return nothing
end

@inline _laplacian_interior(Φ, coef, meas, I::CartesianIndex, v::Val, ::Type{S}) where {S} =
    @inbounds S(_flux_interior(Φ, coef, I, S(Φ[I]), v, S) / meas[I])

@inline _flux_interior(Φ, coef, I::CartesianIndex, ΦI, ::Val{0}, ::Type{S}) where {S} = zero(S)
@inline function _flux_interior(Φ, coef, I::CartesianIndex{N}, ΦI, ::Val{d}, ::Type{S}) where {N,d,S}
    @inbounds begin
        e = CartesianIndex(ntuple(i -> ifelse(i == d, 1, 0), Val(N)))
        cd = coef[d]
        Ihi = I + e
        here = cd[I] * (S(Φ[I - e]) - ΦI) + cd[Ihi] * (S(Φ[Ihi]) - ΦI)
    end
    return _flux_interior(Φ, coef, I, ΦI, Val(d - 1), S) + here
end

@inline function _rest_on_edge(rest::NTuple{M,Int}, dims::NTuple{N,Int}) where {M,N}
    @inbounds for e in 1:M
        (rest[e] == 1 || rest[e] == dims[e + 1]) && return true
    end
    return false
end

"""
    project_out_constant!(Φ, c::LaplacianCoefficients; execution = SerialBackend()) -> Φ

Remove from `Φ`, on each closed component of `L`, its measure-weighted mean there: the component of `Φ`
in `L`'s null space.
"""
function project_out_constant!(Φ::AbstractArray, c::LaplacianCoefficients{N,T};
                               execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    c.singular || return Φ
    # The cells of a component are scattered through `Φ`, so the passes stay in this process.
    b = Execution.local_backend(execution)
    ns = c.nullspace
    meas = Execution.on_backend(b, c.measure)
    ns.full && return _project_mean!(Φ, meas, c.total, b, eltype(Φ))
    perm = Execution.on_backend(b, ns.perm)
    S = eltype(Φ)
    for k in eachindex(ns.totals)
        o = ns.offsets[k] - 1
        n = ns.offsets[k + 1] - ns.offsets[k]
        acc = Execution.reduce_indices(+, zero(S), n, b) do i
            @inbounds begin
                j = perm[o + i]
                S(Φ[j] * meas[j])
            end
        end
        m = acc / S(ns.totals[k])
        Execution.run_indices(n, b) do i
            @inbounds Φ[perm[o + i]] -= m
            return nothing
        end
    end
    return Φ
end

function _project_mean!(Φ, meas, total, b, ::Type{S}) where {S}
    iszero(total) && return Φ
    acc = Execution.reduce_indices(+, zero(S), length(Φ), b) do lin
        @inbounds S(Φ[lin] * meas[lin])
    end
    m = acc / S(total)
    Execution.run_indices(length(Φ), b) do lin
        @inbounds iszero(meas[lin]) || (Φ[lin] -= m)
        return nothing
    end
    return Φ
end

# ---- the Galerkin hierarchy -----------------------------------------------------

"""
    coarsen(grid; pinned = false) -> grid

The next grid down: along every direction of at least four cells, coarse cell `i` covers fine cells
`2i − 1` and `2i` (cell `n` alone for the last of an odd count), its centre their mean; a shorter direction
keeps its cells, so a grid with no direction of four cells comes back with the same cells. A coarse cell is
active where any fine cell it covers is, or with `pinned = true` where every one is: on an operator whose
inactive cells hold zero through the coefficients that reach them, a coarse level then keeps that hold as a
coefficient into an inactive coarse cell. Every coarse axis is a `Vector`, whatever the fine axis, so the
grids below a given grid share one type.

On a [`Grids.CurvilinearGrid`](@ref) the coarse mesh keeps every other vertex line and the last one.
"""
function coarsen(grid::Grids.StructuredGrid{T,G,N}; pinned::Bool = false) where {T,G,N}
    dims = Grids.size_tuple(grid)
    axes_c = map((x, n) -> _coarse_axis(x, n), Grids.coordinates(grid), dims)
    cdims = map(length, axes_c)
    return Grids.StructuredGrid(Grids.grid_geometry(grid), axes_c...;
                                mask = _coarse_mask(Grids.mask(grid), cdims, dims, pinned),
                                topology = Grids.topology(grid),
                                period = ntuple(d -> Grids.period(grid, d), Val(N)))
end

function _coarse_axis(x::AbstractVector, n::Int)
    T = float(eltype(x))
    n < 4 && return collect(T, x)
    out = Vector{T}(undef, cld(n, 2))
    @inbounds for i in eachindex(out)
        out[i] = (T(x[2i - 1]) + T(x[min(2i, n)])) / 2
    end
    return out
end

@inline _children(i::Int, n::Int) = n < 4 ? (i:i) : ((2i - 1):min(2i, n))
@inline _parent(j::Int, n::Int) = n < 4 ? j : cld(j, 2)

# The fine face that coarse face `i` sits on: the face below its first child, and for the top face of a
# bounded axis the face above fine cell `n`.
@inline _fine_face(i::Int, n::Int) = n < 4 ? i : min(2i - 1, n + 1)

_coarse_mask(::Grids.AllActive, cdims::NTuple{N,Int}, ::NTuple{N,Int}, ::Bool) where {N} = Grids.AllActive(cdims)
function _coarse_mask(fine, cdims::NTuple{N,Int}, dims::NTuple{N,Int}, pinned::Bool) where {N}
    cmask = falses(cdims)
    @inbounds for I in CartesianIndices(cdims)
        kids = CartesianIndices(ntuple(d -> _children(I[d], dims[d]), Val(N)))
        cmask[I] = pinned ? all(J -> fine[J], kids) : any(J -> fine[J], kids)
    end
    return cmask
end

function coarsen(grid::Grids.CurvilinearGrid{T,G,2}; pinned::Bool = false) where {T,G}
    dims = Grids.size_tuple(grid)
    Grids.has_corners(grid) || throw(ArgumentError(
        "coarsening a CurvilinearGrid reads its cell vertices, which this grid does not hold; build it " *
        "with `corners` given"))
    k = Grids.corners(grid)
    rows = map(_coarse_vertices, dims)
    ck = ntuple(e -> k[e][rows[1], rows[2]], Val(2))
    cdims = map(r -> length(r) - 1, rows)
    centres = ntuple(Val(2)) do e
        q = ck[e]
        return [(q[i, j] + q[i + 1, j] + q[i, j + 1] + q[i + 1, j + 1]) / 4 for i in 1:cdims[1], j in 1:cdims[2]]
    end
    return Grids.CurvilinearGrid(
        Grids.grid_geometry(grid), centres..., _coarse_mask(Grids.mask(grid), cdims, dims, pinned); corners = ck,
        topology = Grids.topology(grid), period = ntuple(d -> Grids.period(grid, d), Val(2)),
    )
end

_coarse_vertices(n::Int) = [_fine_face(i, n) for i in 1:((n < 4 ? n : cld(n, 2)) + 1)]

"""
    galerkin_coefficients(fine::LaplacianCoefficients, cgrid) -> LaplacianCoefficients

`R L P` on `cgrid`, a [`coarsen`](@ref) of `fine.grid`, for injection `P` and its adjoint `R`, the
measure-weighted average of [`restrict!`](@ref): a coarse face's coefficient is the sum of the fine ones
crossing it, a coarse cell's measure the sum of its children's.
"""
function galerkin_coefficients(fine::LaplacianCoefficients{N,T}, cgrid::Grids.FaceIndexedGrid) where {N,T}
    fgrid = fine.grid
    fdims = Grids.size_tuple(fgrid)
    cdims = Grids.size_tuple(cgrid)
    coef = ntuple(d -> zeros(T, Grids.face_dims(cgrid, d)), Val(N))
    @inbounds for d in 1:N
        cd, fd = coef[d], fine.coef[d]
        for Fc in CartesianIndices(cd)
            f = _fine_face(Fc[d], fdims[d])
            acc = zero(T)
            for J in CartesianIndices(ntuple(e -> e == d ? (f:f) : _children(Fc[e], fdims[e]), Val(N)))
                acc += fd[J]
            end
            cd[Fc] = acc
        end
    end
    return LaplacianCoefficients(cgrid, coef, _children_measure(fine, cdims, fdims))
end

"""
    coarse_coefficients(fine::LaplacianCoefficients, cgrid; ghost = nothing) -> LaplacianCoefficients

`L` rediscretized on `cgrid`, a [`coarsen`](@ref) of `fine.grid`: a coarse face's coefficient is the
coupling area of the fine faces it covers over its own gap, `Σ c_f g_f / g_c` with the gaps of
[`face_metrics`](@ref), and an active coarse cell's measure the sum of its children's. The mask and the
boundary condition reach the coarse level through the fine coefficients. Along a
direction a uniform grid halves, this is half the [`galerkin_coefficients`](@ref) there, the factor by
which the Galerkin operator of piecewise-constant transfers overstates the coarse one (Braess,
*Computing* 55 (1995) 379–393).

A fine face that couples a cell to a zero keeps the zero where it is: across it the ratio is the fine
cell's distance to the zero over the coarse cell's, along the direction's coordinate. The zero is an
inactive fine cell beyond an inactive coarse one, or past a bounded end the fine grid's end face, or the
coordinates `ghost[d] = (below, above)` where a condition holds its zeros off the end faces.
"""
function coarse_coefficients(fine::LaplacianCoefficients{N,T}, cgrid::Grids.FaceIndexedGrid;
                             ghost = nothing) where {N,T}
    fgrid = fine.grid
    fdims = Grids.size_tuple(fgrid)
    cdims = Grids.size_tuple(cgrid)
    fgap = _face_gaps(fgrid)
    cgap = _face_gaps(cgrid)
    fmask, cmask = Grids.mask(fgrid), Grids.mask(cgrid)
    coef = ntuple(d -> zeros(T, Grids.face_dims(cgrid, d)), Val(N))
    for d in 1:N
        P = Grids.isperiodic(fgrid, d) ? T(Grids.period(fgrid, d)) : zero(T)
        ends = _end_zeros(fgrid, d, ghost === nothing ? nothing : ghost[d], T)
        _coarse_direction!(coef[d], fine.coef[d], fgap[d], cgap[d], _axis_or_nothing(fgrid, d),
                           _axis_or_nothing(cgrid, d), P, ends, d, fgrid, cgrid, fmask, cmask, fdims, Val(N))
    end
    meas = _children_measure(fine, cdims, fdims)
    @inbounds for Ic in CartesianIndices(cdims)
        cmask[Ic] || (meas[Ic] = zero(T))
    end
    return LaplacianCoefficients(cgrid, coef, meas)
end

# The coefficients `cd` of the coarse faces normal to `d`.
function _coarse_direction!(cd::AbstractArray{T}, fd, gf, gc, xf, xc, P::T, ends, d::Int, fgrid, cgrid, fmask,
                            cmask, fdims, ::Val{N}) where {T,N}
    @inbounds for Fc in CartesianIndices(cd)
        f = _fine_face(Fc[d], fdims[d])
        Clo, Chi = Grids.cell_below(cgrid, Fc, d), Grids.cell_above(cgrid, Fc, d)
        alo = Clo !== nothing && cmask[Clo]
        ahi = Chi !== nothing && cmask[Chi]
        acc = zero(T)
        for J in CartesianIndices(ntuple(e -> e == d ? (f:f) : _children(Fc[e], fdims[e]), Val(N)))
            c = T(fd[J])
            iszero(c) && continue
            acc += c * _coarse_ratio(T(gf[J]), T(gc[Fc]), alo, ahi, Clo, Chi, J, d, fgrid, fmask, xf, xc, P, ends)
        end
        cd[Fc] = acc
    end
    return cd
end

# A rectilinear direction's centre coordinates, or `nothing` where the grid has none (a curvilinear mesh,
# which takes the gap ratio throughout).
_axis_or_nothing(grid::Grids.StructuredGrid, d::Int) = Grids.coordinates(grid, d)
_axis_or_nothing(_, ::Int) = nothing

# The coordinates `(below, above)` of the zeros past a bounded direction's two ends: `ghost` where given,
# and otherwise the fine grid's end faces, which a coarse grid built from centres alone need not share.
function _end_zeros(grid::Grids.StructuredGrid, d::Int, gd, ::Type{T}) where {T}
    Grids.isperiodic(grid, d) && return nothing
    gd === nothing || return (T(gd[1]), T(gd[2]))
    f = Grids.face_coordinates(grid, d)
    return (T(first(f)), T(last(f)))
end
_end_zeros(_, ::Int, _, ::Type) = nothing

# `d_fine / d_coarse` for fine face `J` under a coarse face: between two active coarse cells the gap
# ratio; where one side is a zero, the fine and the coarse cell's distances to that zero.
@inline function _coarse_ratio(gf::T, gc::T, alo::Bool, ahi::Bool, Clo, Chi, J::CartesianIndex, d::Int,
                               fgrid, fmask, xf, xc, P::T, ends) where {T}
    (alo == ahi || xf === nothing) && return gf / gc
    C, Ja = alo ? (Clo, Grids.cell_below(fgrid, J, d)) : (Chi, Grids.cell_above(fgrid, J, d))
    Jz = alo ? Grids.cell_above(fgrid, J, d) : Grids.cell_below(fgrid, J, d)
    (Ja === nothing || (Jz !== nothing && fmask[Jz])) && return gf / gc
    z = if Jz === nothing
        ends === nothing && return gf / gc
        alo ? ends[2] : ends[1]
    else
        T(xf[Jz[d]])
    end
    df = _axis_distance(T(xf[Ja[d]]), z, P)
    dc = _axis_distance(T(xc[C[d]]), z, P)
    return iszero(dc) ? gf / gc : df / dc
end

@inline function _axis_distance(a::T, b::T, P::T) where {T}
    Δ = a - b
    iszero(P) || (Δ -= P * round(Δ / P))
    return abs(Δ)
end

# Each coarse cell's measure, the sum of its children's.
function _children_measure(fine::LaplacianCoefficients{N,T}, cdims::NTuple{N,Int}, fdims::NTuple{N,Int}) where {N,T}
    meas = zeros(T, cdims)
    @inbounds for Ic in CartesianIndices(cdims)
        m = zero(T)
        for J in CartesianIndices(ntuple(e -> _children(Ic[e], fdims[e]), Val(N)))
            m += fine.measure[J]
        end
        meas[Ic] = m
    end
    return meas
end

"""
    restrict!(coarse, fine, fc, cc; execution = SerialBackend()) -> coarse
    prolong_add!(fine, coarse, fc, cc; execution = SerialBackend()) -> fine

The transfers between a level `fc` and the next one down `cc`: `restrict!` writes each coarse cell the
measure-weighted mean of its children, `Σ V_J f_J / Σ V_J`, and `prolong_add!` adds each coarse value to
the active fine cells it covers. Under the cell measures each is the other's adjoint, which
[`galerkin_coefficients`](@ref) assumes.
"""
function restrict!(coarse::AbstractArray, fine::AbstractArray, fc::LaplacianCoefficients{N,T},
                   cc::LaplacianCoefficients{N,T};
                   execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_cells(fine, fc.grid, "fine")
    _check_cells(coarse, cc.grid, "coarse")
    fdims = Grids.size_tuple(fc.grid)
    b = Execution.resolve(execution)
    fmeas, cmeas = Execution.on_backend(b, fc.measure), Execution.on_backend(b, cc.measure)
    S = eltype(coarse)
    _each_cell(Grids.size_tuple(cc.grid), b, Execution.Written(coarse)) do I
        @inbounds begin
            w = cmeas[I]
            if iszero(w)
                coarse[I] = zero(S)
            else
                acc = zero(S)
                for J in CartesianIndices(ntuple(d -> _children(I[d], fdims[d]), Val(N)))
                    acc += S(fmeas[J] * fine[J])
                end
                coarse[I] = acc / S(w)
            end
        end
        return nothing
    end
    return coarse
end

function prolong_add!(fine::AbstractArray, coarse::AbstractArray, fc::LaplacianCoefficients{N,T},
                      cc::LaplacianCoefficients{N,T};
                      execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_cells(fine, fc.grid, "fine")
    _check_cells(coarse, cc.grid, "coarse")
    fdims = Grids.size_tuple(fc.grid)
    b = Execution.resolve(execution)
    fmeas = Execution.on_backend(b, fc.measure)
    _each_cell(fdims, b, Execution.Written(fine)) do J
        @inbounds iszero(fmeas[J]) ||
                  (fine[J] += coarse[CartesianIndex(ntuple(d -> _parent(J[d], fdims[d]), Val(N)))])
        return nothing
    end
    return fine
end

"""
    restrict_residual!(coarse, b, x, fc, cc; execution = SerialBackend()) -> coarse

[`restrict!`](@ref) of the residual `b + L x` on level `fc`, formed child by child inside the restriction:
coarse cell `I` gets `Σ (V_J b_J + V_J (L x)_J) / Σ V_J` over its active children, `V_J (L x)_J` the flux
balance over `J`'s faces.
"""
function restrict_residual!(coarse::AbstractArray, b::AbstractArray, x::AbstractArray,
                            fc::LaplacianCoefficients{N,T}, cc::LaplacianCoefficients{N,T};
                            execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {N,T}
    _check_cells(b, fc.grid, "b")
    _check_cells(x, fc.grid, "x")
    _check_cells(coarse, cc.grid, "coarse")
    fdims = Grids.size_tuple(fc.grid)
    bk = Execution.resolve(execution)
    fgrid = Execution.on_backend(bk, fc.grid)
    fmeas, cmeas = Execution.on_backend(bk, fc.measure), Execution.on_backend(bk, cc.measure)
    coef, msk = Execution.on_backend(bk, fc.coef), Grids.mask(fgrid)
    S = eltype(coarse)
    _each_cell(Grids.size_tuple(cc.grid), bk, Execution.Written(coarse)) do I
        @inbounds begin
            w = cmeas[I]
            if iszero(w)
                coarse[I] = zero(S)
            else
                acc = zero(S)
                for J in CartesianIndices(ntuple(d -> _children(I[d], fdims[d]), Val(N)))
                    msk[J] || continue
                    acc += S(fmeas[J] * b[J]) + _flux_sum(x, fgrid, coef, J, S(x[J]), Val(N), S)
                end
                coarse[I] = acc / S(w)
            end
        end
        return nothing
    end
    return coarse
end
