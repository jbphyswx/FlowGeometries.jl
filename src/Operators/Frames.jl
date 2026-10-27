# ---------------------------------------------------------------------------
# Fields between the local frame and the ambient Cartesian frame
# ---------------------------------------------------------------------------
#
# The pointwise rotations of `Geometry` (`vector_to_cartesian`, `vector_from_cartesian`,
# `tensor_to_local`) over every cell of a grid, on any execution backend. A field may carry trailing batch
# axes past the grid's own: element `k` sits on cell `mod1(k, ncells)`. The launch reads the grid, or a
# [`FramePlan`](@ref), in the backend's memory.

# The arrays among `xs` a loop writes, as `Execution.Written` declarations.
@inline _written() = ()
@inline _written(::Nothing, xs...) = _written(xs...)
@inline _written(x::AbstractArray, xs...) = (Execution.Written(x), _written(xs...)...)

@inline _cell(cells::CartesianIndices, k::Integer) = @inbounds Tuple(cells[mod1(k, length(cells))])

@inline _lonlat(grid, c::Tuple) = (p = Grids.coords(grid, c...); (p[1], p[2]))

@inline _at(::Nothing, ::Integer, ::Type{T}) where {T} = zero(T)
@inline _at(a::AbstractArray, k::Integer, ::Type{T}) where {T} = @inbounds convert(T, a[k])

@inline _put!(::Nothing, ::Integer, _) = nothing
@inline _put!(a::AbstractArray, k::Integer, v) = (@inbounds a[k] = v; nothing)

"""
    FramePlan

The sines and cosines of longitude and latitude at the cells of a lon–lat grid, with the grid's mask:
what [`to_planetary!`](@ref), [`to_local!`](@ref) and [`tensor_to_local!`](@ref) read in place of the
grid's coordinates, so a call evaluates no trigonometric function. [`frame_plan`](@ref) builds one.
"""
struct FramePlan{L,A<:AbstractArray,M}
    sinλ::A
    cosλ::A
    sinφ::A
    cosφ::A
    mask::M
    layout::L
end

# The tables of a `StructuredGrid` run along its axes, longitude along direction 1 and latitude along
# direction 2; any other grid's are shaped like its mask.
struct _ByAxis end
struct _ByCell end

"""
    frame_plan(grid; execution = SerialBackend()) -> FramePlan

The [`FramePlan`](@ref) of the lon–lat `grid`, built from the grid on the host and held in `execution`'s
memory: a sine and cosine per column and per row of a `StructuredGrid`, per cell of any other grid. The
frame rotations given it in the grid's place write what the grid forms write, bit for bit.
"""
frame_plan(grid::Grids.AbstractGrid{G};
           execution::CB.AbstractExecutionBackend = CB.SerialBackend()) where {G<:Geometry.AbstractLonLatGeometry} =
    Execution.on_backend(Execution.resolve(execution), _frame_tables(grid, Grids.grid_geometry(grid)))

# `sincos` of each value of `x` in `F`, as `Geometry._enu_frame` forms them.
function _sincos_table(::Type{F}, x) where {F}
    s, c = Vector{F}(undef, length(x)), Vector{F}(undef, length(x))
    for (i, v) in enumerate(x)
        s[i], c[i] = sincos(convert(F, v))
    end
    return s, c
end

function _frame_tables(grid::Grids.StructuredGrid, ::Geometry.AbstractGeometry{F}) where {F}
    ax = Grids.coordinates(grid)
    sλ, cλ = _sincos_table(F, Execution._host_array(ax[1]))
    sφ, cφ = _sincos_table(F, Execution._host_array(ax[2]))
    return FramePlan(sλ, cλ, sφ, cφ, Grids.mask(grid), _ByAxis())
end

function _frame_tables(grid::Grids.AbstractGrid, ::Geometry.AbstractGeometry{F}) where {F}
    msk = Grids.mask(grid)
    t = ntuple(_ -> zeros(F, size(msk)), Val(4))
    for c in CartesianIndices(size(msk))
        Grids.isactive(grid, Tuple(c)...) || continue
        λ, φ = _lonlat(grid, Tuple(c))
        t[1][c], t[2][c] = sincos(convert(F, λ))
        t[3][c], t[4][c] = sincos(convert(F, φ))
    end
    return FramePlan(t..., msk, _ByCell())
end

# Where a rotation reads the frame: the grid's coordinates, or a plan's tables.
struct _GridFrames{G,R}
    geometry::G
    grid::R
end

@inline _cells(s::_GridFrames) = CartesianIndices(size(Grids.mask(s.grid)))
@inline _cells(p::FramePlan) = CartesianIndices(size(p.mask))
@inline _frame_float(s::_GridFrames) = _float_of(s.geometry)
@inline _frame_float(p::FramePlan) = eltype(p.sinλ)
@inline _float_of(::Geometry.AbstractGeometry{F}) where {F} = F
@inline _frame_active(s::_GridFrames, c::Tuple) = Grids.isactive(s.grid, c...)
@inline _frame_active(p::FramePlan, c::Tuple) = @inbounds p.mask[c...]
@inline _triad(s::_GridFrames, c::Tuple) = Geometry._enu_frame(s.geometry, _lonlat(s.grid, c)...)
@inline _triad(p::FramePlan{_ByAxis}, c::Tuple) =
    @inbounds Geometry._enu_frame(p.sinλ[c[1]], p.cosλ[c[1]], p.sinφ[c[2]], p.cosφ[c[2]])
@inline _triad(p::FramePlan{_ByCell}, c::Tuple) =
    @inbounds Geometry._enu_frame(p.sinλ[c...], p.cosλ[c...], p.sinφ[c...], p.cosφ[c...])

# `f(k, c)` for every element `k in 1:n` of a field laid out cell after cell over `cells`, `c` the cell
# `mod1(k, length(cells))`. On the host the elements split by direction-1 lines, each line's other indices
# found once; one direction splits by elements, each chunk stepping from its first cell.
function _each_element(f::F, cells::CartesianIndices, n::Int, b::Execution._HostBackend,
                       outs::Execution.Written...) where {F}
    n == 0 && return nothing
    n1 = size(cells, 1)
    rows = CartesianIndices(Base.tail(size(cells)))
    Execution.run_chunks(cld(n, n1), b, outs...) do lines
        @inbounds for ℓ in lines
            r = Tuple(rows[mod1(ℓ, length(rows))])
            k0 = (ℓ - 1) * n1
            for i in 1:min(n1, n - k0)
                @inline f(k0 + i, (i, r...))
            end
        end
        return nothing
    end
    return nothing
end

function _each_element(f::F, cells::CartesianIndices{1}, n::Int, b::Execution._HostBackend,
                       outs::Execution.Written...) where {F}
    Execution.run_chunks(n, b, outs...) do rng
        isempty(rng) && return nothing
        c = @inbounds cells[mod1(first(rng), length(cells))]
        for k in rng
            @inline f(k, Tuple(c))
            nxt = iterate(cells, c)
            c = nxt === nothing ? first(cells) : nxt[1]
        end
        return nothing
    end
    return nothing
end

_each_element(f::F, cells::CartesianIndices, n::Int, b::CB.AbstractExecutionBackend,
              outs::Execution.Written...) where {F} =
    Execution.run_indices(n, b, outs...) do k
        @inline f(k, _cell(cells, k))
    end

_frames(grid::Grids.AbstractGrid, b) = _GridFrames(Grids.grid_geometry(grid), Execution.on_backend(b, grid))
_frames(plan::FramePlan, b) = Execution.on_backend(b, plan)

const _FrameSource = Union{Grids.AbstractGrid{<:Geometry.AbstractLonLatGeometry},FramePlan}

"""
    to_planetary!(X, Y, Z, u, v, w, grid; execution = SerialBackend()) -> nothing
    to_planetary!(X, Y, Z, u, v, w, plan::FramePlan; execution = SerialBackend()) -> nothing

Write the ambient Cartesian components of the vector field whose local components are `u`, `v`, `w` —
along `ê_λ`, `ê_φ`, `ê_r` — at every cell of `grid`: [`Geometry.vector_to_cartesian`](@ref) per cell.
`w = nothing` is a tangent field. An inactive cell gets zero and its inputs are not read. The grid's
[`frame_plan`](@ref) in its place gives the same values with no trigonometry per call.
"""
function to_planetary!(
    X::AbstractArray{T}, Y::AbstractArray{T}, Z::AbstractArray{T},
    u::AbstractArray, v::AbstractArray, w::Union{Nothing,AbstractArray},
    grid::_FrameSource; execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {T<:AbstractFloat}
    b = Execution.resolve(execution)
    src = _frames(grid, b)
    F = _frame_float(src)
    _each_element(_cells(src), length(X), b, _written(X, Y, Z)...) do k, c
        if _frame_active(src, c)
            p = Geometry._to_cartesian(_triad(src, c), convert(F, _at(u, k, T)), convert(F, _at(v, k, T)),
                                       convert(F, _at(w, k, T)))
            @inbounds X[k], Y[k], Z[k] = p.x, p.y, p.z
        else
            @inbounds X[k], Y[k], Z[k] = zero(T), zero(T), zero(T)
        end
    end
    return nothing
end

"""
    to_local!(u, v, w, X, Y, Z, grid; execution = SerialBackend()) -> nothing
    to_local!(u, v, w, X, Y, Z, plan::FramePlan; execution = SerialBackend()) -> nothing

Write the local components, along `ê_λ`, `ê_φ`, `ê_r`, of the vector field whose ambient Cartesian
components are `X`, `Y`, `Z` at every cell of `grid`: [`Geometry.vector_from_cartesian`](@ref) per cell.
`w = nothing` drops the outward component. An inactive cell gets zero. The grid's [`frame_plan`](@ref) in
its place gives the same values with no trigonometry per call.
"""
function to_local!(
    u::AbstractArray{T}, v::AbstractArray{T}, w::Union{Nothing,AbstractArray{T}},
    X::AbstractArray, Y::AbstractArray, Z::AbstractArray,
    grid::_FrameSource; execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {T<:AbstractFloat}
    b = Execution.resolve(execution)
    src = _frames(grid, b)
    F = _frame_float(src)
    _each_element(_cells(src), length(u), b, _written(u, v, w)...) do k, c
        if _frame_active(src, c)
            l = Geometry._from_cartesian(_triad(src, c), convert(F, _at(X, k, T)), convert(F, _at(Y, k, T)),
                                         convert(F, _at(Z, k, T)))
            @inbounds u[k], v[k] = l.λ, l.φ
            _put!(w, k, l.r)
        else
            @inbounds u[k], v[k] = zero(T), zero(T)
            _put!(w, k, zero(T))
        end
    end
    return nothing
end

"""
    tensor_to_local!(λλ, φφ, rr, λφ, λr, φr, xx, yy, zz, xy, xz, yz, grid; execution = SerialBackend()) -> nothing
    tensor_to_local!(λλ, φφ, rr, λφ, λr, φr, xx, yy, zz, xy, xz, yz, plan::FramePlan; execution = SerialBackend()) -> nothing

Write the local components of the symmetric tensor field whose ambient Cartesian components are `xx …
yz` at every cell of `grid`: [`Geometry.tensor_to_local`](@ref) per cell. An output given as `nothing` is
not written. An inactive cell gets zero. The grid's [`frame_plan`](@ref) in its place gives the same values
with no trigonometry per call.
"""
function tensor_to_local!(
    λλ, φφ, rr, λφ, λr, φr,
    xx::AbstractArray, yy::AbstractArray, zz::AbstractArray, xy::AbstractArray, xz::AbstractArray, yz::AbstractArray,
    grid::_FrameSource; execution::CB.AbstractExecutionBackend = CB.SerialBackend(),
)
    T = eltype(xx)
    b = Execution.resolve(execution)
    src = _frames(grid, b)
    F = _frame_float(src)
    _each_element(_cells(src), length(xx), b, _written(λλ, φφ, rr, λφ, λr, φr)...) do k, c
        if _frame_active(src, c)
            t = Geometry._tensor_to_local(_triad(src, c), convert(F, _at(xx, k, T)), convert(F, _at(yy, k, T)),
                                          convert(F, _at(zz, k, T)), convert(F, _at(xy, k, T)),
                                          convert(F, _at(xz, k, T)), convert(F, _at(yz, k, T)))
            _put!(λλ, k, t.λλ); _put!(φφ, k, t.φφ); _put!(rr, k, t.rr)
            _put!(λφ, k, t.λφ); _put!(λr, k, t.λr); _put!(φr, k, t.φr)
        else
            z = zero(T)
            _put!(λλ, k, z); _put!(φφ, k, z); _put!(rr, k, z)
            _put!(λφ, k, z); _put!(λr, k, z); _put!(φr, k, z)
        end
    end
    return nothing
end
