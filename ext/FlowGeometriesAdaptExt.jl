module FlowGeometriesAdaptExt

using Adapt: Adapt
using FlowGeometries: FlowGeometries
using FlowGeometries.Grids: Grids
using FlowGeometries.Connectivity: Connectivity

# Move a grid's array-backed fields to another storage (a GPU array type, a view wrapper, …). The
# geometry and the periodic/period tuples are plain immutable scalars and travel unchanged.
#
# Each grid is rebuilt through its inner constructor, so the adapted field types become its new type
# parameters.

@inline _adapt_tuple(to, t::Tuple) = map(x -> Adapt.adapt(to, x), t)

# `SeparableMeasure` is its factors, so adapting it adapts each factor and the device receives
# `O(∑ Nᵈ)` numbers. `AllActive` holds only a size, so it is already device-safe.
Adapt.adapt_structure(to, m::Grids.SeparableMeasure) =
    Grids.SeparableMeasure(_adapt_tuple(to, m.factors))

# Likewise a `SlabMeasure`: the coupled pair is one matrix, and the device receives that matrix and the
# per-axis factors, `O(Nφ·Nh + Nλ)` numbers.
Adapt.adapt_structure(to, m::Grids.SlabMeasure) =
    Grids.SlabMeasure(Adapt.adapt(to, m.lead), Adapt.adapt(to, m.slab), _adapt_tuple(to, m.rest))

Adapt.adapt_structure(::Any, m::Grids.AllActive) = m

# A `CellMesh` is four index arrays; adapting it moves those.
Adapt.adapt_structure(to, m::Grids.CellMesh) = Grids.CellMesh(
    Adapt.adapt(to, m.cell_ptr), Adapt.adapt(to, m.cell_nodes),
    Adapt.adapt(to, m.node_ptr), Adapt.adapt(to, m.node_cells),
)

# An unindexed `MetricTopology` is isbits and travels as-is. One carrying a spatial index raises: a k-d
# tree is a host structure, and dropping it leaves the device with a topology that silently scans.
Adapt.adapt_structure(::Any, mt::Connectivity.MetricTopology{N,T,Nothing}) where {N,T} = mt
Adapt.adapt_structure(::Any, mt::Connectivity.MetricTopology) = throw(ArgumentError(
    "a MetricTopology carrying a spatial index cannot be moved to another backend; adapt the grid and " *
    "build the topology there, or use `MetricTopology(grid)` without an index",
))

# `isbits`: no heap reference to move. An analytic axis is its formula's parameters, and those are what
# reach the device; the formula runs there.
Adapt.adapt_structure(::Any, a::FlowGeometries.Axes.UniformAxis) = a
Adapt.adapt_structure(::Any, a::FlowGeometries.Axes.AbstractAnalyticAxis) = a
Adapt.adapt_structure(::Any, c::FlowGeometries.Axes.ConstantVector) = c

# One method for every layout: the array-backed fields are adapted and handed to `Grids.rebuild`, which
# re-derives the grid's type parameters from what they became.
#
# Which fields those are is decided structurally: an array, or a tuple of them. Geometry, topology,
# period, the sampling tag, a resolution parameter and the per-axis reductions are immutable scalars or
# tuples of them, and travel unchanged. So a layout is adapted by what it holds, whether that is a
# coordinate tuple, four ring vectors or a single integer.
@inline _is_adaptable(v) = v isa AbstractArray || v isa Grids.CellMesh
@inline _is_adaptable(v::Tuple) = !isempty(v) && all(x -> x isa AbstractArray, v)

function Adapt.adapt_structure(to, grid::Grids.AbstractGrid)
    changed = NamedTuple(
        n => (v = getfield(grid, n); v isa Tuple ? _adapt_tuple(to, v) : Adapt.adapt(to, v))
        for n in fieldnames(typeof(grid)) if _is_adaptable(getfield(grid, n))
    )
    return Grids.rebuild(grid, changed)
end

Adapt.adapt_structure(to, conn::Connectivity.CSRConnectivity) =
    Connectivity.CSRConnectivity(Adapt.adapt(to, conn.nbrs), Adapt.adapt(to, conn.ptr))

Adapt.adapt_structure(::Any, t::Connectivity.IndexTopology{N,Nothing}) where {N} = t
Adapt.adapt_structure(to, t::Connectivity.IndexTopology) =
    Connectivity.IndexTopology(t.size, t.periodic, Adapt.adapt(to, t.mask))

end # module
