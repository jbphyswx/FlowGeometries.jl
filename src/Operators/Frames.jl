# ---------------------------------------------------------------------------
# Fields between the local frame and the ambient Cartesian frame
# ---------------------------------------------------------------------------
#
# The pointwise rotations of `Geometry` (`vector_to_cartesian`, `vector_from_cartesian`,
# `tensor_to_local`) over every cell of a grid, on any execution backend. A field may carry trailing batch
# axes past the grid's own: element `k` sits on cell `mod1(k, ncells)`.

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
    to_planetary!(X, Y, Z, u, v, w, grid; backend = SerialBackend()) -> nothing

Write the ambient Cartesian components of the vector field whose local components are `u`, `v`, `w` —
along `ê_λ`, `ê_φ`, `ê_r` — at every cell of `grid`: [`Geometry.vector_to_cartesian`](@ref) per cell.
`w = nothing` is a tangent field. An inactive cell gets zero and its inputs are not read.
"""
function to_planetary!(
    X::AbstractArray{T}, Y::AbstractArray{T}, Z::AbstractArray{T},
    u::AbstractArray, v::AbstractArray, w::Union{Nothing,AbstractArray},
    grid::Grids.AbstractGrid{G}; backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {T<:AbstractFloat, G<:Geometry.AbstractLonLatGeometry}
    geo = Grids.grid_geometry(grid)
    cells = CartesianIndices(size(Grids.mask(grid)))
    Execution.run_indices(length(X), backend, _written(X, Y, Z)...) do k
        c = _cell(cells, k)
        if Grids.isactive(grid, c...)
            λ, φ = _lonlat(grid, c)
            p = Geometry.vector_to_cartesian(geo, _at(u, k, T), _at(v, k, T), _at(w, k, T), λ, φ)
            @inbounds X[k], Y[k], Z[k] = p.x, p.y, p.z
        else
            @inbounds X[k], Y[k], Z[k] = zero(T), zero(T), zero(T)
        end
    end
    return nothing
end

"""
    to_local!(u, v, w, X, Y, Z, grid; backend = SerialBackend()) -> nothing

Write the local components, along `ê_λ`, `ê_φ`, `ê_r`, of the vector field whose ambient Cartesian
components are `X`, `Y`, `Z` at every cell of `grid`: [`Geometry.vector_from_cartesian`](@ref) per cell.
`w = nothing` drops the outward component. An inactive cell gets zero.
"""
function to_local!(
    u::AbstractArray{T}, v::AbstractArray{T}, w::Union{Nothing,AbstractArray{T}},
    X::AbstractArray, Y::AbstractArray, Z::AbstractArray,
    grid::Grids.AbstractGrid{G}; backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {T<:AbstractFloat, G<:Geometry.AbstractLonLatGeometry}
    geo = Grids.grid_geometry(grid)
    cells = CartesianIndices(size(Grids.mask(grid)))
    Execution.run_indices(length(u), backend, _written(u, v, w)...) do k
        c = _cell(cells, k)
        if Grids.isactive(grid, c...)
            λ, φ = _lonlat(grid, c)
            l = Geometry.vector_from_cartesian(geo, _at(X, k, T), _at(Y, k, T), _at(Z, k, T), λ, φ)
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
    tensor_to_local!(λλ, φφ, rr, λφ, λr, φr, xx, yy, zz, xy, xz, yz, grid; backend = SerialBackend()) -> nothing

Write the local components of the symmetric tensor field whose ambient Cartesian components are `xx …
yz` at every cell of `grid`: [`Geometry.tensor_to_local`](@ref) per cell. An output given as `nothing` is
not written. An inactive cell gets zero.
"""
function tensor_to_local!(
    λλ, φφ, rr, λφ, λr, φr,
    xx::AbstractArray, yy::AbstractArray, zz::AbstractArray, xy::AbstractArray, xz::AbstractArray, yz::AbstractArray,
    grid::Grids.AbstractGrid{G}; backend::CB.AbstractExecutionBackend = CB.SerialBackend(),
) where {G<:Geometry.AbstractLonLatGeometry}
    T = eltype(xx)
    geo = Grids.grid_geometry(grid)
    cells = CartesianIndices(size(Grids.mask(grid)))
    Execution.run_indices(length(xx), backend, _written(λλ, φφ, rr, λφ, λr, φr)...) do k
        c = _cell(cells, k)
        if Grids.isactive(grid, c...)
            λ, φ = _lonlat(grid, c)
            t = Geometry.tensor_to_local(geo, _at(xx, k, T), _at(yy, k, T), _at(zz, k, T),
                                         _at(xy, k, T), _at(xz, k, T), _at(yz, k, T), λ, φ)
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
