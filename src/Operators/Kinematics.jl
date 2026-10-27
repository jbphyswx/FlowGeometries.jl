# ---------------------------------------------------------------------------
# Strain rate, curl and divergence of a collocated vector field
# ---------------------------------------------------------------------------
#
# In orthogonal curvilinear coordinates, with `dᵢ = (1/hᵢ) ∂/∂xᵢ` the derivative per unit distance and
# `γᵢₖ = (1/(hᵢ hₖ)) ∂hᵢ/∂xₖ` from `Geometry.log_scale_gradients` (Batchelor 1967, Appendix 2):
#
#     Sᵢᵢ = dᵢuᵢ + Σₖ uₖ γᵢₖ
#     Sᵢⱼ = ½(dⱼuᵢ − uᵢ γᵢⱼ + dᵢuⱼ − uⱼ γⱼᵢ)                  i ≠ j
#     (∇×u)ᵢ = dⱼuₖ + uₖ γₖⱼ − dₖuⱼ − uⱼ γⱼₖ                 (i, j, k) cyclic
#     ∇·u = Σᵢ (dᵢuᵢ + uᵢ Σₖ γₖᵢ)
#
# with `γᵢᵢ = 0`; in two directions the curl is its third component, `h₃ = 1`. The components of `u` lie
# along the local frame — east, north and up on a sphere — and every `dᵢ` comes from one derivative
# source: a structured grid's [`StencilPlans`](@ref), or a least-squares [`GradientPlan`](@ref).

"""
    StencilPlans{N}

The derivative weights of every direction of one structured grid, as [`stencil_plans`](@ref) builds
them: what [`derivative!`](@ref) applies along each direction, and a derivative source for
[`strain_rate!`](@ref), [`curl!`](@ref) and [`divergence!`](@ref).
"""
struct StencilPlans{N,P<:Tuple}
    plans::P
end

Base.show(io::IO, p::StencilPlans{N}) where {N} = print(io, "StencilPlans{", N, "}(", p.plans, ")")

"""
    stencil_plans(grid; order = 1, nodes = order + 1, policy = BlankMasked()) -> StencilPlans

A [`Discretization.stencil_plan`](@ref) along every direction of `grid`, or, on a grid with an inactive
cell, a [`masked_stencil_plan`](@ref) under `policy`. A direction shorter than `nodes` takes the samples
it has; one with fewer than `order + 1` carries no derivative and is written `masked`.

The weights depend on the axes, the mask, the order and the policy, never on a field: one serves every
derivative taken on the grid, and tasks may share it.
"""
function stencil_plans(
    grid::Grids.StructuredGrid{T,G,N}; order::Integer = 1, nodes::Integer = Int(order) + 1,
    policy::AbstractMaskPolicy = BlankMasked(),
) where {T,G,N}
    ord, k = Int(order), Int(nodes)
    k ≥ ord + 1 || throw(ArgumentError("an order-$ord derivative needs at least $(ord + 1) nodes, got $k"))
    mask = Grids.mask(grid)
    plans = map_directions(vd -> _direction_plan(grid, mask, vd, ord, k, policy), Val(N))
    return StencilPlans{N,typeof(plans)}(plans)
end

# A direction of fewer than `ord + 1` samples takes the plan of no rows, which writes `masked`.
function _direction_plan(grid::Grids.StructuredGrid, ::Grids.AllActive, ::Val{d}, ord::Int, k::Int, _policy) where {d}
    x = Execution._host_array(Grids.coordinates(grid)[d])
    n = length(x)
    return Discretization._stencil_plan(Axes.spacing_trait(x), x, ord, n < ord + 1 ? 0 : min(k, n),
                                        Grids._wrap_period(grid, d))
end

function _direction_plan(
    grid::Grids.StructuredGrid{T}, mask::AbstractArray{Bool}, ::Val{d}, ord::Int, k::Int, policy,
) where {T,d}
    n = Grids.size_tuple(grid)[d]
    n < ord + 1 && return _empty_masked_plan(T, _host_mask(mask), d, ord, 0, _along_arc(grid, d, ord))
    return masked_stencil_plan(grid, d; order = ord, nodes = min(k, n), policy = policy)
end

"""
    derivative!(out, field, grid, plans::StencilPlans, dim; masked = zero, execution = SerialBackend()) -> out

[`derivative!`](@ref) along `dim` with the plan [`stencil_plans`](@ref) holds for that direction.
"""
function derivative!(
    out::AbstractArray, field::AbstractArray, grid::Grids.StructuredGrid{T,G,N}, plans::StencilPlans{N},
    dim::Integer; masked = zero(eltype(out)), execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {T,G,N}
    1 ≤ dim ≤ N || throw(ArgumentError("direction $dim is outside 1:$N"))
    return _held_derivative!(out, field, grid, @inbounds(plans.plans[dim]), Int(dim), masked, execution)
end

@inline _held_derivative!(out, field, grid, p::Discretization.AbstractStencilPlan, d::Int, masked, b) =
    derivative!(out, field, grid, p, d; masked = masked, execution = b)
@inline _held_derivative!(out, field, grid, p::MaskedStencilPlan, ::Int, masked, b) =
    derivative!(out, field, grid, p; masked = masked, execution = b)

# `out = d_d f`.
@inline _derivative_along!(out, f, grid::Grids.StructuredGrid, plans::StencilPlans, d::Int, _scratch, masked, b) =
    _held_derivative!(out, f, grid, @inbounds(plans.plans[d]), d, masked, b)

@inline _derivative_along!(out, f, _grid, plan::GradientPlan, d::Int, _scratch, _masked, b) =
    _gradient_component!(out, f, plan, d, b)

# `g[k] = d_k f` for every direction `k`.
@inline function _gradient_into!(
    g::NTuple{N,AbstractArray}, f, grid::Grids.StructuredGrid, plans::StencilPlans{N}, masked, b,
) where {N}
    ntuple(Val(N)) do d
        _held_derivative!(g[d], f, grid, @inbounds(plans.plans[d]), d, masked, b)
        return nothing
    end
    return nothing
end

@inline function _gradient_into!(g::NTuple{N,AbstractArray}, f, _grid, plan::GradientPlan{N}, _masked, b) where {N}
    gradient!(g, f, plan; execution = b)
    return nothing
end

# The component names of a symmetric tensor over the frame's directions `n`, in the order
# `Geometry.tensor_to_local` returns them: the diagonal, then the upper triangle row by row. Generated, so
# the names are constants.
@generated function _sym_keys(::Val{n}) where {n}
    length(n) == 1 && return :($((Symbol(n[1], n[1]),)))
    length(n) == 2 && return :($((Symbol(n[1], n[1]), Symbol(n[2], n[2]), Symbol(n[1], n[2]))))
    return :($((Symbol(n[1], n[1]), Symbol(n[2], n[2]), Symbol(n[3], n[3]),
                Symbol(n[1], n[2]), Symbol(n[1], n[3]), Symbol(n[2], n[3]))))
end

# The arrays of `S` under the names `want`, in that order.
@generated function _components(S::NamedTuple{K}, ::Val{want}) where {K,want}
    ex = Expr(:tuple)
    for k in want
        k in K || return :(throw(ArgumentError(
            $("the strain rate is written into components $want; this NamedTuple has $K"))))
        push!(ex.args, :(getfield(S, $(QuoteNode(k)))))
    end
    return ex
end

# Where `(i, j)` sits in that order.
@inline _sym_slot(i::Int, j::Int, ::Val{N}) where {N} =
    i == j ? i : (N == 2 ? 3 : (min(i, j) == 1 ? max(i, j) + 2 : 6))

@inline _frame_names(grid::Grids.AbstractGrid, ::Val{N}) where {N} =
    Geometry.point_names(Grids.grid_geometry(grid), Val(N))

"""
    strain_components(grid) -> NTuple{K,Symbol}

The names of the rate-of-strain components on `grid`, in order: `(:λλ, :φφ, :λφ)` on a sphere's surface,
`(:xx, :yy, :zz, :xy, :xz, :yz)` in a Cartesian volume — the diagonal, then the upper triangle row by row.
"""
@inline strain_components(grid::Grids.AbstractGrid) =
    _sym_keys(Val(_frame_names(grid, Val(Grids.ncoordinates(grid)))))

function _check_frame(u::Tuple, grid::Grids.AbstractGrid)
    N = Grids.ncoordinates(grid)
    length(u) == N || throw(DimensionMismatch(
        "a field on $(nameof(typeof(grid))) has $N components along the local frame; got $(length(u))",
    ))
    return nothing
end

"""
    strain_rate!(S, u, grid, derivs; masked = NaN, scratch = nothing, execution = SerialBackend()) -> S

The rate of strain `½(∇u + ∇uᵀ)` of the vector field `u`, one array per direction of `grid` along its
local frame, written into the arrays of the `NamedTuple` `S` under the names
[`strain_components`](@ref) gives. `derivs` is [`stencil_plans`](@ref)`(grid)` on a structured grid and a
[`gradient_plan`](@ref) elsewhere.

On a curved geometry each component carries the frame's own terms — on a sphere `S_λλ = d_λu_λ −
u_φ tan φ/R` and `S_λφ = ½(d_φu_λ + d_λu_φ + u_λ tan φ/R)` — from
[`Geometry.log_scale_gradients`](@ref). A cell is written `masked` where it is inactive, where a
derivative it reads does not exist, or where the metric degenerates, as at a pole. `scratch` is one array
per direction for the derivatives of one component; with it a call allocates nothing.
"""
function strain_rate!(
    S::NamedTuple, u::Tuple{Vararg{AbstractArray}}, grid::Grids.AbstractGrid, derivs;
    masked = NaN, scratch = nothing, execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
)
    _check_frame(u, grid)
    N = length(u)
    comps = _components(S, Val(_sym_keys(Val(_frame_names(grid, Val(N))))))
    E = eltype(first(comps))
    g = scratch === nothing ? ntuple(_ -> similar(first(comps)), Val(N)) : scratch
    b = Execution.resolve(execution)
    nan = E(NaN)
    _strain_terms!(comps, u, g, grid, derivs, nan, b, Val(N))
    _frame_pass!(_StrainTerms(), comps, u, grid, convert(E, masked), b)
    return S
end

# The derivative part: `S_cc = d_c u_c`, and each off-diagonal the half sum of its two derivatives. `d_k u_c`
# for `c ≤ k` lands in its component directly; `S_ck` then takes `d_c u_k` in one fused update.
function _strain_terms!(comps, u, g, grid, derivs, nan, b, ::Val{N}) where {N}
    ntuple(Val(N)) do c
        ntuple(Val(N)) do k
            s = comps[_sym_slot(c, k, Val(N))]
            if c ≤ k
                _derivative_along!(s, u[c], grid, derivs, k, g, nan, b)
            else
                _derivative_along!(g[1], u[c], grid, derivs, k, g, nan, b)
                s .= s ./ 2 .+ g[1] ./ 2
            end
            return nothing
        end
        return nothing
    end
    return nothing
end

"""
    strain_rate(u, grid, derivs; masked = NaN, execution = SerialBackend()) -> NamedTuple

[`strain_rate!`](@ref) into fresh arrays, named as [`strain_components`](@ref) gives.
"""
function strain_rate(u::Tuple{Vararg{AbstractArray}}, grid::Grids.AbstractGrid, derivs; kwargs...)
    _check_frame(u, grid)
    keys_ = strain_components(grid)
    S = NamedTuple{keys_}(map(_ -> similar(first(u), float(eltype(first(u)))), keys_))
    return strain_rate!(S, u, grid, derivs; kwargs...)
end

"""
    curl!(ω, u, grid, derivs; masked = NaN, scratch = nothing, execution = SerialBackend()) -> ω

The curl of the vector field `u` on `grid`, its components along the local frame: one array `ω` in two
directions — the vertical vorticity, `d_λu_φ − d_φu_λ + u_λ tan φ/R` on a sphere — and a tuple of three
in three. `derivs` and the masking are as for [`strain_rate!`](@ref); `scratch` is one array per
direction.
"""
function curl!(
    ω::AbstractArray, u::NTuple{2,AbstractArray}, grid::Grids.AbstractGrid, derivs;
    masked = NaN, scratch = nothing, execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
)
    _check_frame(u, grid)
    E = eltype(ω)
    g = scratch === nothing ? (similar(ω), similar(ω)) : scratch
    b = Execution.resolve(execution)
    nan = E(NaN)
    _derivative_along!(ω, u[2], grid, derivs, 1, g, nan, b)
    _derivative_along!(g[1], u[1], grid, derivs, 2, g, nan, b)
    ω .-= g[1]
    _frame_pass!(_CurlTerms(), (ω,), u, grid, convert(E, masked), b)
    return ω
end

function curl!(
    ω::NTuple{3,AbstractArray}, u::NTuple{3,AbstractArray}, grid::Grids.AbstractGrid, derivs;
    masked = NaN, scratch = nothing, execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
)
    _check_frame(u, grid)
    E = eltype(first(ω))
    g = scratch === nothing ? ntuple(_ -> similar(first(ω)), Val(3)) : scratch
    b = Execution.resolve(execution)
    nan = E(NaN)
    foreach(o -> fill!(o, zero(E)), ω)
    # `u_c` enters component `c+1` through `+d_{c+2}` and component `c+2` through `−d_{c+1}`.
    ntuple(Val(3)) do c
        _gradient_into!(g, u[c], grid, derivs, nan, b)
        ω[mod1(c + 1, 3)] .+= g[mod1(c + 2, 3)]
        ω[mod1(c + 2, 3)] .-= g[mod1(c + 1, 3)]
        return nothing
    end
    _frame_pass!(_CurlTerms(), ω, u, grid, convert(E, masked), b)
    return ω
end

"""
    curl(u, grid, derivs; masked = NaN, execution = SerialBackend())

[`curl!`](@ref) into fresh arrays: one in two directions, three in three.
"""
curl(u::NTuple{2,AbstractArray}, grid::Grids.AbstractGrid, derivs; kwargs...) =
    curl!(similar(first(u), float(eltype(first(u)))), u, grid, derivs; kwargs...)
curl(u::NTuple{3,AbstractArray}, grid::Grids.AbstractGrid, derivs; kwargs...) =
    curl!(ntuple(_ -> similar(first(u), float(eltype(first(u)))), Val(3)), u, grid, derivs; kwargs...)

"""
    divergence!(δ, u, grid, derivs; masked = NaN, scratch = nothing, execution = SerialBackend()) -> δ

The divergence of the vector field `u` on `grid`, its components along the local frame: the flux form,
`d_λu_λ + d_φu_φ − u_φ tan φ/R` on a sphere's surface. `derivs` and the masking are as for
[`strain_rate!`](@ref); `scratch` is one array per direction.
"""
function divergence!(
    δ::AbstractArray, u::Tuple{Vararg{AbstractArray}}, grid::Grids.AbstractGrid, derivs;
    masked = NaN, scratch = nothing, execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
)
    _check_frame(u, grid)
    N = length(u)
    E = eltype(δ)
    g = scratch === nothing ? ntuple(_ -> similar(δ), Val(N)) : scratch
    b = Execution.resolve(execution)
    nan = E(NaN)
    _derivative_along!(δ, u[1], grid, derivs, 1, g, nan, b)
    _each_dir(Val(N - 1)) do vc
        c = _val(vc) + 1
        _derivative_along!(g[1], u[c], grid, derivs, c, g, nan, b)
        δ .+= g[1]
        return nothing
    end
    _frame_pass!(_DivergenceTerms(), (δ,), u, grid, convert(E, masked), b)
    return δ
end

"""
    divergence(u, grid, derivs; masked = NaN, execution = SerialBackend())

[`divergence!`](@ref) into a fresh array.
"""
divergence(u::Tuple{Vararg{AbstractArray}}, grid::Grids.AbstractGrid, derivs; kwargs...) =
    divergence!(similar(first(u), float(eltype(first(u)))), u, grid, derivs; kwargs...)

# ---- the frame's terms, and the masking, cell by cell ------------------------

struct _StrainTerms end
struct _CurlTerms end
struct _DivergenceTerms end

# Adds the `γ` terms of `kind` to `outs` at every cell and writes `masked` where the cell is inactive, the
# metric degenerates there, or a derivative read is undefined (NaN). One work item per cell, batch
# elements included, reading the grid in the backend's memory.
_frame_pass!(kind, outs::Tuple, u::Tuple, grid::Grids.AbstractGrid, masked, b) =
    _frame_cells!(kind, outs, u, grid, masked, b)

# On the host, a structured grid whose metric does not vary along direction 1 takes the scale factors and
# `log_scale_gradients` once per direction-1 line.
function _frame_pass!(kind, outs::Tuple, u::Tuple, grid::Grids.StructuredGrid, masked, b::Execution._HostBackend)
    1 in Geometry.metric_invariant_directions(Grids.grid_geometry(grid)) ||
        return _frame_cells!(kind, outs, u, grid, masked, b)
    return _frame_lines!(kind, outs, u, grid, masked, size(first(outs)), b)
end

function _frame_cells!(kind, outs, u, grid, masked, b)
    geo = Grids.grid_geometry(grid)
    g = Execution.on_backend(b, grid)
    msk = Grids.mask(g)
    Execution.run_cells(size(first(outs)), b, map(Execution.Written, outs)...) do I
        _frame_cell!(kind, outs, u, g, geo, msk, masked, I, Val(ndims(msk)))
    end
    return nothing
end

_frame_lines!(kind, outs, u, grid, masked, ::NTuple{1,Int}, b) = _frame_cells!(kind, outs, u, grid, masked, b)

function _frame_lines!(kind, outs, u, grid::Grids.StructuredGrid{T,G,M}, masked, dims::NTuple{D,Int}, b) where {T,G,M,D}
    dims[1] == 0 && return nothing
    geo = Grids.grid_geometry(grid)
    msk = Grids.mask(grid)
    Execution.run_chunks(dims[D], b, map(Execution.Written, outs)...) do slab
        mid = CartesianIndices(ntuple(e -> dims[e + 1], Val(D - 2)))
        @inbounds for jD in slab, Im in mid
            _frame_line!(kind, outs, u, grid, geo, msk, masked, dims[1], (Tuple(Im)..., jD), Val(M))
        end
    end
    return nothing
end

# The direction-1 line whose other indices are `rest`, its metric taken at its first cell.
@inline function _frame_line!(kind, outs, u, grid, geo, msk, masked, n::Int, rest::Tuple, ::Val{M}) where {M}
    @inbounds begin
        grest = ntuple(e -> rest[e], Val(M - 1))
        p = Tuple(Grids.coords(grid, 1, grest...))
        if any(iszero, Geometry.scale_factors(geo, p))
            for i in 1:n
                _write_all!(outs, CartesianIndex(i, rest...), masked)
            end
            return nothing
        end
        γ = Geometry.log_scale_gradients(geo, p)
        for i in 1:n
            I = CartesianIndex(i, rest...)
            msk[i, grest...] ? _frame_terms!(kind, outs, u, masked, I, γ) : _write_all!(outs, I, masked)
        end
    end
    return nothing
end

@inline function _frame_cell!(kind, outs, u, grid, geo, msk, masked, I, ::Val{M}) where {M}
    @inbounds begin
        Is = ntuple(d -> I[d], Val(M))
        if !msk[Is...]
            _write_all!(outs, I, masked)
            return nothing
        end
        p = Tuple(Grids.coords(grid, Is...))
        if any(iszero, Geometry.scale_factors(geo, p))
            _write_all!(outs, I, masked)
            return nothing
        end
        _frame_terms!(kind, outs, u, masked, I, Geometry.log_scale_gradients(geo, p))
    end
    return nothing
end

@inline function _frame_terms!(kind, outs, u, masked, I, γ)
    @inbounds begin
        v = map(c -> c[I], u)
        _add_terms!(kind, outs, I, v, γ)
        any(o -> isnan(o[I]), outs) && _write_all!(outs, I, masked)
    end
    return nothing
end

@inline function _write_all!(outs, I, masked)
    foreach(o -> (@inbounds o[I] = masked), outs)
    return nothing
end

@inline function _add_terms!(::_StrainTerms, S, I, v::NTuple{N}, γ) where {N}
    @inbounds ntuple(Val(N)) do i
        ntuple(Val(N)) do j
            s = S[_sym_slot(i, j, Val(N))]
            if i == j
                s[I] += sum(ntuple(k -> v[k] * γ[i][k], Val(N)))
            elseif i < j
                s[I] -= (v[i] * γ[i][j] + v[j] * γ[j][i]) / 2
            end
            return nothing
        end
        return nothing
    end
    return nothing
end

@inline function _add_terms!(::_CurlTerms, ω, I, v::NTuple{2}, γ)
    @inbounds ω[1][I] += v[2] * γ[2][1] - v[1] * γ[1][2]
    return nothing
end

@inline function _add_terms!(::_CurlTerms, ω, I, v::NTuple{3}, γ)
    @inbounds ntuple(Val(3)) do i
        j, k = mod1(i + 1, 3), mod1(i + 2, 3)
        ω[i][I] += v[k] * γ[k][j] - v[j] * γ[j][k]
        return nothing
    end
    return nothing
end

@inline function _add_terms!(::_DivergenceTerms, δ, I, v::NTuple{N}, γ) where {N}
    @inbounds δ[1][I] += sum(ntuple(i -> v[i] * sum(ntuple(k -> γ[k][i], Val(N))), Val(N)))
    return nothing
end
