module FlowGeometriesAdaptExt

using Adapt: Adapt
using FlowGeometries.Grids: Grids
using FlowGeometries.Connectivity: Connectivity

# Move a grid's array-backed fields to another storage (a GPU array type, a view wrapper, …). The
# geometry and the periodic/period tuples are plain immutable scalars and travel unchanged.
#
# Every grid type is reconstructed through the inner constructor, so the adapted field types become
# the new type parameters rather than being forced back to `Vector`/`Matrix`.

@inline _adapt_tuple(to, t::Tuple) = map(x -> Adapt.adapt(to, x), t)

# `SeparableMeasure` is the factors; adapting it means adapting each factor, not materializing the
# outer product onto the device. `AllActive` holds only a size, so it is already device-safe.
Adapt.adapt_structure(to, m::Grids.SeparableMeasure) =
    Grids.SeparableMeasure(_adapt_tuple(to, m.factors))
Adapt.adapt_structure(::Any, m::Grids.AllActive) = m

function Adapt.adapt_structure(to, grid::Grids.StructuredGrid{G,T,N}) where {G,T,N}
    coords = _adapt_tuple(to, Grids.coordinates(grid))
    measure = Adapt.adapt(to, Grids.measure(grid))
    mask = Adapt.adapt(to, Grids.mask(grid))
    return Grids.StructuredGrid{G,T,N,typeof(coords),typeof(measure),typeof(mask)}(
        Grids.grid_geometry(grid), coords, measure, mask, getfield(grid, :periodic),
    )
end

function Adapt.adapt_structure(to, grid::Grids.CurvilinearGrid{T,G}) where {T,G}
    coords = _adapt_tuple(to, Grids.coordinates(grid))
    corners = _adapt_tuple(to, getfield(grid, :corners))
    measure = Adapt.adapt(to, Grids.measure(grid))
    mask = Adapt.adapt(to, Grids.mask(grid))
    return Grids.CurvilinearGrid{T,G,typeof(coords),typeof(measure),typeof(mask)}(
        Grids.grid_geometry(grid), coords, corners, measure, mask, getfield(grid, :periodic),
    )
end

function Adapt.adapt_structure(to, grid::Grids.UnstructuredGrid{T,G}) where {T,G}
    coords = _adapt_tuple(to, Grids.coordinates(grid))
    measure = Adapt.adapt(to, Grids.measure(grid))
    mask = Adapt.adapt(to, Grids.mask(grid))
    nbrs = Adapt.adapt(to, getfield(grid, :neighbor_nbrs))
    ptr = Adapt.adapt(to, getfield(grid, :neighbor_ptr))
    return Grids.UnstructuredGrid{
        T,G,typeof(coords),typeof(measure),typeof(mask),typeof(nbrs),typeof(ptr),
    }(
        Grids.grid_geometry(grid), coords, measure, mask, nbrs, ptr,
        getfield(grid, :periodic), getfield(grid, :period),
    )
end

Adapt.adapt_structure(to, conn::Connectivity.CSRConnectivity) =
    Connectivity.CSRConnectivity(Adapt.adapt(to, conn.nbrs), Adapt.adapt(to, conn.ptr))

Adapt.adapt_structure(::Any, t::Connectivity.IndexTopology{N,Nothing}) where {N} = t
Adapt.adapt_structure(to, t::Connectivity.IndexTopology) =
    Connectivity.IndexTopology(t.size, t.periodic, Adapt.adapt(to, t.mask))

end # module
