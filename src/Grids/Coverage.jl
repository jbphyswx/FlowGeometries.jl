# ---------------------------------------------------------------------------
# Covered measure and integrals
# ---------------------------------------------------------------------------

"""
    covered_area(grid) -> T

The total measure of the active cells: `sum(measure(grid))` with every masked cell left out. `4πR²` on an
unmasked layout tiling the sphere, and in general the area (length, volume) a field on `grid` is known
over.
"""
covered_area(grid::AbstractGrid{G,T}) where {G,T} = _covered(mask(grid), measure(grid), T)

@inline _covered(::AllActive, m, ::Type{T}) where {T} = T(sum(m))
_covered(msk::AbstractArray{Bool}, m, ::Type{T}) where {T} =
    _weighted_sum(_ -> one(T), msk, m, zero(T))

"""
    sky_fraction(grid) -> T

`covered_area(grid) / sum(measure(grid))`: the fraction of the cells' total measure that is active. One
on an unmasked grid; on a partial-sky sphere, `f_sky`.
"""
sky_fraction(grid::AbstractGrid{G,T}) where {G,T} = covered_area(grid) / T(sum(measure(grid)))

"""
    integrate(field, grid; active_only = true, execution = SerialBackend())

`Σᵢ field[i]·measure(grid)[i]` over the cells: the integral of `field` over the region they cover, by the
grid's own measure, which on a grid built from a quadrature sampling is that quadrature. Masked cells are
left out unless `active_only = false`.

A field of the grid's own rank gives a number. One carrying trailing batch axes gives one integral per
element, in the order those axes are laid out; [`integrate!`](@ref) writes them into a caller's vector.
"""
function integrate(
    field::AbstractArray, grid::AbstractGrid{G,T}; active_only::Bool = true,
    execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {G,T}
    nb = _check_integrand(field, grid)
    S = promote_type(eltype(field), T)
    ndims(field) == ndims(grid) &&
        return _integrate_at(field, grid, 0, active_only, S, Execution.resolve(execution))
    return integrate!(Vector{S}(undef, nb), field, grid; active_only = active_only, execution = execution)
end

"""
    integrate!(out, field, grid; active_only = true, execution = SerialBackend()) -> out

[`integrate`](@ref) of a field carrying trailing batch axes, one integral per element written to `out`.
"""
function integrate!(
    out::AbstractVector, field::AbstractArray, grid::AbstractGrid{G,T}; active_only::Bool = true,
    execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {G,T}
    nb = _check_integrand(field, grid)
    length(out) == nb || throw(DimensionMismatch(
        "out holds $(length(out)) values but the field carries $nb batch elements",
    ))
    S = eltype(out)
    b = Execution.resolve(execution)
    n = length(grid)
    @inbounds for k in 1:nb
        out[k] = _integrate_at(field, grid, (k - 1) * n, active_only, S, b)
    end
    return out
end

# The field's leading axes are the grid's; whatever follows is batch. Returns the batch length.
function _check_integrand(field::AbstractArray, grid::AbstractGrid)
    N = ndims(grid)
    ndims(field) ≥ N || throw(DimensionMismatch("field has $(ndims(field)) axes but the grid has $N"))
    ntuple(d -> size(field, d), N) == size(grid) || throw(DimensionMismatch(
        "field's leading $N axes $(ntuple(d -> size(field, d), N)) do not match the grid $(size(grid))",
    ))
    return length(field) ÷ length(grid)
end

# One element, at linear offset `off` into the field. Each arm calls at its own mask type.
_integrate_at(field, grid::AbstractGrid, off::Int, active_only::Bool, ::Type{S}, backend) where {S} =
    active_only ? _integrate_over(field, grid, mask(grid), off, S, backend) :
                  _integrate_over(field, grid, AllActive(size(grid)), off, S, backend)

_integrate_over(field, grid::AbstractGrid, msk, off::Int, ::Type{S}, ::CB.AbstractSerialBackend) where {S} =
    _weighted_sum(i -> S(@inbounds field[off + i]), msk, measure(grid), zero(S))

function _integrate_over(field, grid::AbstractGrid, msk, off::Int, ::Type{S},
                         backend::CB.AbstractExecutionBackend) where {S}
    m = measure(grid)
    return Execution.reduce_indices(+, zero(S), length(grid), backend) do i
        @inbounds _active(msk, i) ? S(field[off + i]) * S(m[i]) : zero(S)
    end
end

@inline _active(::AllActive, ::Int) = true
@inline _active(msk::AbstractArray{Bool}, i::Int) = @inbounds msk[i]

"""
    _weighted_sum(f, mask, measure, zero) -> S

`Σᵢ f(i)·measure[i]` over the active cells `i` (linear indices), each measure container walked in its own
shape: a [`SeparableMeasure`](@ref) as a direction-1 sum per row times the row's factor, a
[`RingwiseVector`](@ref) as a sum per ring times the ring's value, a constant once.
"""
function _weighted_sum(f::F, msk, m::AbstractArray, z::S) where {F,S}
    acc = z
    @inbounds for i in 1:length(m)
        _active(msk, i) && (acc += f(i) * S(m[i]))
    end
    return acc
end

function _weighted_sum(f::F, msk, m::Axes.ConstantVector, z::S) where {F,S}
    acc = z
    @inbounds for i in 1:length(m)
        _active(msk, i) && (acc += f(i))
    end
    return S(m.value) * acc
end

function _weighted_sum(f::F, msk, m::RingwiseVector, z::S) where {F,S}
    acc = z
    @inbounds for r in eachindex(m.value)
        row = z
        for i in (m.offset[r] + 1):m.offset[r + 1]
            _active(msk, i) && (row += f(i))
        end
        acc += S(m.value[r]) * row
    end
    return acc
end

function _weighted_sum(f::F, msk, m::SeparableMeasure{T,N}, z::S) where {F,S,T,N}
    sz = size(m)
    w1 = m.factors[1]
    acc = z
    @inbounds for (q, I) in enumerate(CartesianIndices(ntuple(d -> sz[d + 1], Val(N - 1))))
        wrest = prod(ntuple(d -> S(m.factors[d + 1][I[d]]), Val(N - 1)))
        base = (q - 1) * sz[1]
        row = z
        for i in 1:sz[1]
            _active(msk, base + i) && (row += f(base + i) * S(w1[i]))
        end
        acc += wrest * row
    end
    return acc
end
