module FlowGeometriesAdaptExt

using Adapt: Adapt
using FlowGeometries: FlowGeometries
using FlowGeometries.Grids: Grids
using FlowGeometries.Connectivity: Connectivity
using FlowGeometries.Discretization: Discretization
using FlowGeometries.Operators: Operators
using FlowGeometries.Execution: Execution

# Move a grid's array-backed fields to another storage (a GPU array type, a view wrapper, …). The
# geometry and the periodic/period tuples are plain immutable scalars and travel unchanged.
#
# Each grid is rebuilt through its inner constructor, so the adapted field types become its new type
# parameters.

@inline _adapt_tuple(to, t::Tuple) = map(x -> Adapt.adapt(to, x), t)

# A field that may be a mask. A `BitArray` moves as an `Array{Bool}` to a target that unpacks bits, and
# as itself to any other.
@inline _move(to, v) = Adapt.adapt(to, v)
@inline _move(to, v::BitArray) = Adapt.adapt(to, Execution._unpack_bits(to) ? Array(v) : v)

Execution._unpack_bits(::Any) = false

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

# A cell list is its bucket offsets, point ids and their bins; the query reaches the coordinates through
# the grid, which is adapted on its own.
Adapt.adapt_structure(to, ix::Grids.CellListIndex) = Grids.CellListIndex(
    ix.n, ix.lo, ix.h, ix.nbins, ix.wrap, Adapt.adapt(to, ix.starts), Adapt.adapt(to, ix.items),
    Adapt.adapt(to, ix.bins), ix.embedding, ix.active_only,
)

# A binned index is its bin offsets, its cells' ids, boxes, coordinates and activity in bin order.
Adapt.adapt_structure(to, ix::Grids.BinnedIndex) = Grids.BinnedIndex(
    ix.lo, ix.h, ix.nb, ix.wrap, ix.period, ix.layout, Adapt.adapt(to, ix.order), Adapt.adapt(to, ix.ptr),
    Adapt.adapt(to, ix.bins), _adapt_tuple(to, ix.x), Adapt.adapt(to, ix.active), ix.slack,
    ix.active_only,
)

# An unindexed `MetricTopology` is isbits and travels as-is, and one carrying a cell list or a binned
# index takes it along, so a query there visits the same candidates in the same order. Any other index
# raises: a k-d tree is a host structure, and dropping it leaves the device with a topology that silently
# scans.
Adapt.adapt_structure(::Any, mt::Connectivity.MetricTopology{N,T,Nothing}) where {N,T} = mt
Adapt.adapt_structure(
    to, mt::Connectivity.MetricTopology{N,T,<:Union{Grids.CellListIndex,Grids.BinnedIndex}},
) where {N,T} = Connectivity.MetricTopology(mt.steps, mt.min3, mt.span, Adapt.adapt(to, mt.index))
Adapt.adapt_structure(::Any, mt::Connectivity.MetricTopology) = throw(ArgumentError(
    "a MetricTopology carrying a $(nameof(typeof(mt.index))) cannot be moved to another backend; index " *
    "it with a `cell_list`, which moves, or use `MetricTopology(grid)` without an index",
))

# `isbits`: no heap reference to move. An analytic axis is its formula's parameters, and those are what
# reach the device; the formula runs there.
Adapt.adapt_structure(::Any, a::FlowGeometries.Axes.UniformAxis) = a
Adapt.adapt_structure(::Any, a::FlowGeometries.Axes.PeriodicAxis) = a
Adapt.adapt_structure(::Any, a::FlowGeometries.Axes.AbstractAnalyticAxis) = a
Adapt.adapt_structure(::Any, c::FlowGeometries.Axes.ConstantVector) = c

# One method for every layout: the array-backed fields are adapted and handed to `Grids.rebuild`, which
# re-derives the grid's type parameters from what they became.
#
# Which fields those are is decided by their types: an array, a non-empty tuple of arrays, a cell mesh,
# or a grid (a rotated grid's base). Geometry, topology, period, the sampling tag, a resolution parameter
# and the per-axis reductions are immutable scalars or tuples of them, and travel unchanged. So a layout
# is adapted by what it holds, whether that is a coordinate tuple, four ring vectors or a single integer.
@generated function _adapted_fields(to, grid::G) where {G<:Grids.AbstractGrid}
    names = Symbol[]
    vals = Expr[]
    for (n, FT) in zip(fieldnames(G), fieldtypes(G))
        v = :(getfield(grid, $(QuoteNode(n))))
        if FT <: Union{AbstractArray,Grids.CellMesh,Grids.AbstractGrid}
            push!(names, n)
            push!(vals, :(_move(to, $v)))
        elseif FT <: Tuple && fieldcount(FT) > 0
            arrays = true
            for t in fieldtypes(FT)
                arrays &= t <: AbstractArray
            end
            if arrays
                push!(names, n)
                push!(vals, :(_adapt_tuple(to, $v)))
            end
        end
    end
    return :(NamedTuple{$(Tuple(names))}($(Expr(:tuple, vals...))))
end

Adapt.adapt_structure(to, grid::Grids.AbstractGrid) = Grids.rebuild(grid, _adapted_fields(to, grid))

Adapt.adapt_structure(to, conn::Connectivity.CSRConnectivity) =
    Connectivity.CSRConnectivity(Adapt.adapt(to, conn.nbrs), Adapt.adapt(to, conn.ptr))

Adapt.adapt_structure(to, p::Operators.MaskedStencilPlan) = Operators.MaskedStencilPlan(
    Adapt.adapt(to, p.row), Adapt.adapt(to, p.indices), Adapt.adapt(to, p.weights), p.dim, p.order,
    p.distance, _move(to, p.mask),
)

# A uniform plan's arrays are its centred row and its `O(K²)` shifted end rows.
function Adapt.adapt_structure(to, p::Discretization.UniformStencilPlan{T}) where {T}
    w = Adapt.adapt(to, p.weights)
    ei = Adapt.adapt(to, p.edge_idx)
    ew = Adapt.adapt(to, p.edge_wts)
    return Discretization.UniformStencilPlan{T,typeof(w),typeof(ei),typeof(ew)}(
        w, p.left, p.right, p.n, p.order, p.wrap, ei, ew,
    )
end

Adapt.adapt_structure(to, p::Discretization.TabulatedStencilPlan) =
    Discretization.TabulatedStencilPlan(Adapt.adapt(to, p.indices), Adapt.adapt(to, p.weights), p.order)

Adapt.adapt_structure(to, p::Operators.FramePlan) = Operators.FramePlan(
    Adapt.adapt(to, p.sinλ), Adapt.adapt(to, p.cosλ), Adapt.adapt(to, p.sinφ), Adapt.adapt(to, p.cosφ),
    _move(to, p.mask), p.layout,
)

Adapt.adapt_structure(to, p::Operators.GradientPlan) = Operators.GradientPlan(
    Adapt.adapt(to, p.ptr), Adapt.adapt(to, p.nbr), _adapt_tuple(to, p.coeffs), p.names,
)

# One plan per direction.
function Adapt.adapt_structure(to, p::Operators.StencilPlans{N}) where {N}
    plans = _adapt_tuple(to, p.plans)
    return Operators.StencilPlans{N,typeof(plans)}(plans)
end

# The metric factors a derivative's launch reads: one per direction-1 run, or the coordinates each cell's
# factor is evaluated from.
Adapt.adapt_structure(to, f::Operators._RowFactors) =
    Operators._RowFactors(Adapt.adapt(to, f.inv_h), f.size, f.R)
Adapt.adapt_structure(to, f::Operators._CellFactors) =
    Operators._CellFactors(f.geometry, _adapt_tuple(to, f.coordinates), f.dim, f.order)

# A staggered grid is its centre grid and one face axis per direction.
function Adapt.adapt_structure(to, s::Grids.StaggeredGrid{T,N}) where {T,N}
    c = Adapt.adapt(to, s.center)
    f = _adapt_tuple(to, s.faces)
    return Grids.StaggeredGrid{T,N,typeof(c),typeof(f)}(c, f)
end

# A face metric is its grid and its per-direction and per-pair arrays; `area_flux` is `area_op` itself
# under `Dirichlet`, and stays one array.
function Adapt.adapt_structure(to, m::Operators.FaceMetrics)
    area = _adapt_tuple(to, m.area_op)
    flux = m.area_flux === m.area_op ? area : _adapt_tuple(to, m.area_flux)
    return Operators.FaceMetrics(Adapt.adapt(to, m.grid), m.boundary, area, flux, _adapt_tuple(to, m.gap),
                                 _adapt_tuple(to, m.edge), _adapt_tuple(to, m.corner), Adapt.adapt(to, m.volume),
                                 m.orientation)
end

Adapt.adapt_structure(to, ns::Operators.ComponentNullspace) =
    Operators.ComponentNullspace(ns.full, Adapt.adapt(to, ns.perm), ns.offsets, ns.totals)

function Adapt.adapt_structure(to, c::Operators.LaplacianCoefficients{N,T}) where {N,T}
    g = Adapt.adapt(to, c.grid)
    coef = _adapt_tuple(to, c.coef)
    diag = Adapt.adapt(to, c.diag)
    meas = Adapt.adapt(to, c.measure)
    ns = Adapt.adapt(to, c.nullspace)
    return Operators.LaplacianCoefficients{N,T,typeof(g),typeof(coef),typeof(diag),typeof(meas),typeof(ns)}(
        g, coef, diag, meas, c.total, c.singular, ns)
end

Adapt.adapt_structure(::Any, t::Connectivity.IndexTopology{N,Nothing}) where {N} = t
Adapt.adapt_structure(to, t::Connectivity.IndexTopology) =
    Connectivity.IndexTopology(t.size, t.periodic, _move(to, t.mask))

end # module
