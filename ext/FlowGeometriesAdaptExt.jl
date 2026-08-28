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

# `SeparableMeasure` is the factors; adapting it means adapting each factor, not materializing the
# outer product onto the device. `AllActive` holds only a size, so it is already device-safe.
Adapt.adapt_structure(to, m::Grids.SeparableMeasure) =
    Grids.SeparableMeasure(_adapt_tuple(to, m.factors))
Adapt.adapt_structure(::Any, m::Grids.AllActive) = m

# An unindexed `MetricTopology` is isbits and travels as-is. One carrying a spatial index does not: a
# k-d tree is a host structure, so it is refused rather than dropped, which would leave the device with
# a topology that silently scans.
Adapt.adapt_structure(::Any, mt::Connectivity.MetricTopology{N,T,Nothing}) where {N,T} = mt
Adapt.adapt_structure(::Any, mt::Connectivity.MetricTopology) = throw(ArgumentError(
    "a MetricTopology carrying a spatial index cannot be moved to another backend; adapt the grid and " *
    "build the topology there, or use `MetricTopology(grid)` without an index",
))

# `isbits`: no heap reference to move.
Adapt.adapt_structure(::Any, a::FlowGeometries.Axes.UniformAxis) = a
Adapt.adapt_structure(::Any, c::FlowGeometries.Axes.ConstantVector) = c

# One method for every layout: the array-backed fields are adapted and handed to `Grids.rebuild`, which
# re-derives the grid's type parameters from what they became. Geometry, topology, period, the sampling
# tag and the per-axis reductions are immutable scalars and travel unchanged, so none is named here.
const _ADAPTED = (:coordinates, :corners, :measure, :mask, :neighbor_nbrs, :neighbor_ptr)

function Adapt.adapt_structure(to, grid::Grids.AbstractGrid)
    changed = NamedTuple(
        n => (v = getfield(grid, n); v isa Tuple ? _adapt_tuple(to, v) : Adapt.adapt(to, v))
        for n in _ADAPTED if n in fieldnames(typeof(grid))
    )
    return Grids.rebuild(grid, changed)
end

Adapt.adapt_structure(to, conn::Connectivity.CSRConnectivity) =
    Connectivity.CSRConnectivity(Adapt.adapt(to, conn.nbrs), Adapt.adapt(to, conn.ptr))

Adapt.adapt_structure(::Any, t::Connectivity.IndexTopology{N,Nothing}) where {N} = t
Adapt.adapt_structure(to, t::Connectivity.IndexTopology) =
    Connectivity.IndexTopology(t.size, t.periodic, Adapt.adapt(to, t.mask))

end # module
