using FlowGeometries: FlowGeometries as FG
using Test: Test

# Every weak dependency is loaded up front so the extensions are exercised by the whole suite.
using Adapt: Adapt
using DelaunayTriangulation: DelaunayTriangulation
using KernelAbstractions: KernelAbstractions
using NearestNeighbors: NearestNeighbors
using Quickhull: Quickhull
using SparseArrays: SparseArrays
using StaticArrays: StaticArrays

# A caller's own stencil shape: forward-only in every direction, radius in the type. `offsets` is the
# whole contract — everything else in Stencils is written in terms of it.
struct Upwind{R} <: FG.Stencils.AbstractStencil end
Upwind(r::Integer) = Upwind{Int(r)}()
FG.Stencils.offsets(::Upwind{R}, ::Val{N}) where {R,N} =
    ntuple(i -> ntuple(d -> d == cld(i, R) ? mod1(i, R) : 0, Val(N)), Val(N * R))

# An axis that records how many elements were read from it. Several claims here are about how much of
# an axis a query touches — "bisects rather than scans", "reads a bounded window" — and a wall-clock
# threshold is a poor way to assert that: it is decided by a GC pause as much as by the algorithm, and
# it encodes machine constants that rot. Counting reads states the claim exactly and is deterministic.
# Wall-clock numbers live in `benchmark/`.
mutable struct CountingAxis{T,V<:AbstractVector{T}} <: AbstractVector{T}
    data::V
    reads::Int
end
CountingAxis(v::AbstractVector) = CountingAxis(v, 0)
Base.size(c::CountingAxis) = size(c.data)
Base.IndexStyle(::Type{<:CountingAxis}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(c::CountingAxis, i::Int)
    c.reads += 1
    return c.data[i]
end
reads(f::F, c::CountingAxis) where {F} = (c.reads = 0; f(); c.reads)

# Element-type entry points, called with NON-constant arguments on purpose. A type given as a keyword
# takes no part in dispatch, so the moment a call cannot be constant-folded end to end the element
# type widens to `DataType` and the result comes back abstract — which is why every one of these
# takes it as a leading positional argument, as `zeros` and `rand` do.
t_gl(::Type{T}, n) where {T}     = FG.SphericalSampling._gauss_legendre_μ(T, n)
t_axes(::Type{T}, s, n) where {T} = FG.SphericalSampling.spherical_axes(T, s, n)
t_quad(::Type{T}, s, n) where {T} = FG.SphericalSampling.spherical_quadrature(T, s, n)
t_wts(::Type{T}, s, n) where {T}  = FG.SphericalSampling.latitude_weights(T, s, n)
t_wts1(::Type{T}, s) where {T}    = FG.SphericalSampling.latitude_weights(T, s)
t_pts(::Type{T}, s, n) where {T}  = FG.SphericalSampling.spherical_points(T, s, n)
t_pts1(::Type{T}, s) where {T}    = FG.SphericalSampling.spherical_points(T, s)
t_pts2(::Type{T}, s, a, b) where {T} = FG.SphericalSampling.spherical_points(T, s, a, b)
t_rlat(::Type{T}, s) where {T}    = FG.SphericalSampling.ring_latitudes(T, s)
t_cube(::Type{T}, n) where {T}    = FG.SphericalSampling.cubed_sphere_points(T, n)
t_icov(::Type{T}, f) where {T}    = FG.SphericalSampling.icosahedral_vertices(T, f)
t_icom(::Type{T}, f) where {T}    = FG.SphericalSampling.icosahedral_mesh(T, f)
t_yy(::Type{T}, a, b) where {T}   = FG.SphericalSampling.yin_yang_panels(T, a, b)
t_ring(::Type{T}, a, b) where {T} = FG.SphericalSampling.ring_info(T, a, b)
t_p2a(::Type{T}, a, b) where {T}  = FG.SphericalSampling.pix2ang(T, a, b)
t_p2v(::Type{T}, a, b) where {T}  = FG.SphericalSampling.pix2vec(T, a, b)
t_scr(::Type{T}, o, k) where {T}  = FG.Discretization.stencil_scratch(T, o, k)
t_sgrid(::Type{T}, s, n) where {T} = FG.Connectivity.structured_grid(T, s, n)

# The HEALPix pixelization as a node set: the layout materialized into dense coordinate vectors and a
# stored CSR graph. `HEALPixGrid` is how the pixelization is used; this is the explicit "as a cloud"
# form, and it is what the unstructured code paths below are exercised against — they need a grid whose
# coordinates and adjacency are DATA, which is exactly what the layout is not.
function healpix_node_grid(nside::Integer; geometry = FG.Geometry.SphericalGeometry())
    g = FG.Grids.HEALPixGrid(geometry, nside)
    λ, φ = FG.Grids.materialize(g)
    conn = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(Int(nside)))
    return FG.Grids.UnstructuredGrid(geometry, λ, φ, collect(FG.Grids.measure(g)),
                                     FG.Grids.AllActive((length(λ),)), conn.nbrs, conn.ptr)
end

# A geometry defined outside the package, supplying only the accessor its hierarchy asks for.
struct OneSphere{T} <: FG.Geometry.AbstractSphericalGeometry{T} end
FG.Geometry.radius(::OneSphere{T}) where {T} = one(T)

# A geometry whose metric varies along direction 1, which the built-in sphere's never does. It exists to
# hold the bulk operators to `metric_invariant_directions` rather than to the sphere's own invariance.
struct TiltedSphere{T} <: FG.Geometry.AbstractSphericalGeometry{T} end
FG.Geometry.radius(::TiltedSphere{T}) where {T} = one(T)
FG.Geometry.scale_factors(::TiltedSphere{T}, p::Tuple{Real,Real}) where {T} =
    (one(T) + T(0.5) * cos(T(p[1])), one(T) + T(0.25) * sin(T(p[2])))

concrete_return(f::F, argtypes) where {F} =
    (t = Base.return_types(f, argtypes)[1]; (isconcretetype(t), t))

# The allocation harness. `discretization`, `connectivity` and `allocations` all measure with it, so it
# sits in the preamble every topic sees. Everything here is an ordinary global: a closure captured
# into `@allocated` measures itself instead of the function under test.

# Fixed arity, and the cell index travels as a tuple that is splatted inside the callee: forwarding
# through `args...` allocates 240 bytes on its own, so a vararg harness would measure itself.
for n in 1:6
    args = [Symbol(:a, i) for i in 1:n]
    @eval function _alloc(f::F, $(args...)) where {F}
        a = typemax(Int)
        for _ in 1:4                       # first calls carry compilation
            a = min(a, @allocated f($(args...)))
        end
        return a
    end
end

# Each entry point is reached through a named top-level function rather than a closure, for the same
# reason: a closure over a testset local boxes what it captures, and a captured `Module` costs a fixed
# 144 bytes, so the harness would measure itself instead of the call.
q_coords(g, I)           = FG.Grids.coords(g, I...)
q_measure(g, I)          = FG.Grids.measure(g, I...)
q_active(g, I)           = FG.Grids.isactive(g, I...)
q_dist(g, I, J)          = FG.Geometry.distance(g, I, J)
q_disp(g, I, J)          = FG.Grids.displacement(g, I, J)
q_nn(g, I)               = FG.Connectivity.nneighbors(g, I...)
q_nbrs!(out, g, I)       = FG.Connectivity.neighbors!(out, g, I...)
q_iter(g, I)             = (t = 0; for j in FG.Grids.neighbors(g, I...); t += j; end; t)
q_top(g)                 = FG.Connectivity.MetricTopology(g)
q_win(g, I, r, mt)       = FG.Connectivity.metric_window(g, I, r, mt)
q_nwithin(g, r, I)       = FG.Connectivity.nneighbors_within(g, I...; ball = r)
q_within!(b, g, r, I)    = FG.Connectivity.neighbors_within!(b, g, I...; ball = r)
q_fold(g, r, I)          = FG.Connectivity.fold_within((a, _, d) -> a + d, 0.0, g, I...; ball = r)
q_knn!(ix, dt, g, k, I)  = FG.Connectivity.k_nearest!(ix, dt, g, I...; k = k)
q_within_top(g, r, I, t) = FG.Connectivity.nneighbors_within(g, I...; ball = r, topology = t)
q_fold_at(g, p, r, t, s) = FG.Connectivity.fold_at((a, _, _) -> a + 1, 0, g, p;
                                                   ball = r, topology = t, scratch = s)
q_locate(g, p)           = FG.Grids.locate(g, p)
q_runreach(m, I, d, i, n, k) = FG.Operators._run_reach(m, I, d, i, n, k, false)
q_locate_top(g, p, t, s) = FG.Grids.locate(g, p; topology = t, scratch = s)
q_gap(g, d, i)           = FG.Grids.local_spacing(g, d, i)
q_tensor_local(geo, t, p) = FG.Geometry.tensor_to_local(geo, t..., p[1], p[2])
q_uvec(p)                = FG.Geometry.unit_vector(Float64, p)
q_s2c(geo, p)            = FG.Geometry.spherical_to_cartesian(geo, p)
q_vfc(geo, v, p)         = FG.Geometry.vector_from_cartesian(geo, v, p[1], p[2])
q_embed(geo, p)          = FG.Geometry.embed(geo, p)
q_proj(geo, c, n)        = FG.Geometry.project_to_tangent_plane(geo, c, n)
q_ltb(geo, p)            = FG.Geometry.local_tangent_basis(geo, p)
q_c2g(geo, v)            = FG.Geometry.cartesian_to_geodetic(geo, v)
q_gradient!(a, b, f, plan) = FG.Operators.gradient!(a, b, f, plan)
q_gradient_t!(outs, f, plan) = FG.Operators.gradient!(outs, f, plan)
q_stag_grad!(outs, f, sg) = FG.Operators.gradient!(outs, f, sg)
q_stag_div!(out, us, sg)  = FG.Operators.divergence!(out, us, sg)
q_stag_curl!(o, u1, u2, sg) = FG.Operators.curl!(o, u1, u2, sg)
q_locdisp(geo, c, n)     = FG.Geometry.local_displacement(geo, c, n)
q_tbl!(o, f, g, iw, d, pol) = FG.Operators.apply_stencil!(o, f, g, iw[1], iw[2], d;
                                                              order = 1, masked = NaN, policy = pol)
q_tbl_sc!(o, f, g, iw, d, sc) = FG.Operators.apply_stencil!(
    o, f, g, iw[1], iw[2], d; order = 1, masked = NaN,
    policy = FG.Operators.ReduceInRun(), scratch = sc)
q_width(g, d, i)         = FG.Grids.cell_width(g, d, i)
q_stencil!(o, f, ix, w, d) = FG.Operators.apply_stencil!(o, f, ix, w, d)
q_stencil_m!(o, f, ix, w, d, msk) = FG.Operators.apply_stencil!(o, f, ix, w, d; mask = msk)
q_area(g, I)             = FG.Grids.area(g, I...)
q_coords!(o, g, I)       = FG.Grids.coords!(o, g, I...)
q_axis(g, d)             = length(FG.Grids.axis(g, d))
q_spacing(g, d)          = FG.Grids.spacing(g, d)
q_origin(g, d)           = FG.Grids.origin(g, d)
q_extent(g, d)           = FG.Grids.extent(g, d)
q_bounds(g, d)           = FG.Grids.bounds(g, d)
q_isper(g, d)            = FG.Grids.isperiodic(g, d)
q_isuni(g, d)            = FG.Grids.isuniform(g, d)
q_period(g, d)           = FG.Grids.period(g, d)
q_minsp(g, d)            = FG.Grids.minimum_spacing(g, d)
q_maxsp(g, d)            = FG.Grids.maximum_spacing(g, d)
q_size(g)                = FG.Grids.size_tuple(g)
q_mask(g)                = FG.Grids.mask(g)
q_names(g)               = FG.Grids.coordinate_names(g)
q_perflags(g)            = FG.Grids.periodic_flags(g)
q_topo(g)                = FG.Grids.topology(g)
q_nlon_in_ring(s, r)     = FG.SphericalSampling.nlon_in_ring(s, r)
q_ring_range(s, r)       = FG.SphericalSampling.ring_range(s, r)
q_npoints(s)             = FG.SphericalSampling.npoints(s)
q_rg_points!(λ, φ, s, sc) = FG.SphericalSampling.spherical_points!(λ, φ, s; scratch = sc)
# The HELD-table form. The other overload builds the table per call — that is what `axis_stencils` is
# for, and it is classified as allocating by contract — so measuring it would measure the table.
q_deriv_held!(o, f, g, iw, d) = FG.Operators.derivative!(o, f, g, iw[1], iw[2], d; order = 1)
q_interp!(o, f, g, p)    = FG.Operators.interpolate!(o, f, g, p)
q_plan!(o, f, pl, d)     = FG.Operators.apply_stencil!(o, f, pl, d)
q_axst!(i, w, x, o, k, s) = FG.Discretization.axis_stencils!(i, w, x, o, k; scratch = s)
q_faces!(o, x)           = FG.Discretization.faces!(o, x)
q_centers!(o, f)         = FG.Discretization.centers!(o, f)
q_lagr!(w, x, v, k)      = FG.Discretization.lagrange_weights!(w, x, v, k)
q_iw(x, v)               = FG.Discretization.interpolation_weights(x, v)
q_conn(g, r, s)          = FG.Connectivity.nneighbors_within(g, 12, 9; ball = r,
                              reach = FG.Connectivity.Connected(), scratch = s)
q_connf(g, r, s)         = FG.Connectivity.nneighbors_within(g, 7; ball = r,
                              reach = FG.Connectivity.Connected(), scratch = s)

# The per-grid-shape sweep. `axes = false` for a layout whose coordinates are a formula: the spacing
# accessors are an axis notion, and such a layout holds no axes to answer them from.
function check_shape(label, g, I; axes::Bool = true)
    r = FG.Grids.grid_geometry(g) isa FG.Geometry.AbstractSphericalGeometry ? 6.371e5 : 2.5
    out = Vector{Int}(undef, 64)
    buf = Vector{Int}(undef, 4096)
    pt = Vector{Float64}(undef, FG.Grids.ncoordinates(g))
    dts = Vector{Float64}(undef, 64)
    mt = FG.Connectivity.MetricTopology(g)
    J = ntuple(d -> I[d] + 1, length(I))
    # `distance` and `displacement` take a CELL, which is an index tuple or a single integer depending
    # on the layout's `cell_address` — the other entry points here take the indices and splat them.
    cI = FG.Grids._cell_named_by(g, I)
    cJ = FG.Grids._cell_named_by(g, J)
    checks = (
        ("coords",             _alloc(q_coords, g, I)),
        ("measure",            _alloc(q_measure, g, I)),
        ("isactive",           _alloc(q_active, g, I)),
        ("distance(g,I,J)",    _alloc(q_dist, g, cI, cJ)),
        ("displacement",       _alloc(q_disp, g, cI, cJ)),
        ("nneighbors",         _alloc(q_nn, g, I)),
        ("neighbors!",         _alloc(q_nbrs!, out, g, I)),
        ("neighbors iterator", _alloc(q_iter, g, I)),
        ("MetricTopology",     _alloc(q_top, g)),
        ("nneighbors_within",  _alloc(q_nwithin, g, r, I)),
        ("neighbors_within!",  _alloc(q_within!, buf, g, r, I)),
        ("fold_within",        _alloc(q_fold, g, r, I)),
        ("area",               _alloc(q_area, g, I)),
        ("k_nearest!",         _alloc(q_knn!, out, dts, g, 6, I)),
        ("coords!",            _alloc(q_coords!, pt, g, I)),
        ("size_tuple",         _alloc(q_size, g)),
        ("mask",               _alloc(q_mask, g)),
        ("coordinate_names",   _alloc(q_names, g)),
        ("periodic_flags",     _alloc(q_perflags, g)),
        ("topology",           _alloc(q_topo, g)),
    )
    for (name, a) in checks
        a == 0 || println("    ", label, " / ", name, " -> ", a, " B")
        Test.@test a == 0
    end
    if g isa FG.Grids.StructuredGrid
        a = _alloc(q_win, g, I, r, mt)
        a == 0 || println("    ", label, " / metric_window -> ", a, " B")
        Test.@test a == 0
    end
    for d in 1:FG.Grids.ncoordinates(g)
        # `spacing` is the constant-spacing accessor and errors on an irregular axis, by design.
        if axes && FG.Grids.isuniform(g, d)
            a = _alloc(q_spacing, g, d)
            a == 0 || println("    ", label, " / spacing(d=", d, ") -> ", a, " B")
            Test.@test a == 0
        end
        span = (("origin", _alloc(q_origin, g, d)), ("extent", _alloc(q_extent, g, d)),
                ("bounds", _alloc(q_bounds, g, d)), ("isperiodic", _alloc(q_isper, g, d)),
                ("period", _alloc(q_period, g, d)))
        for (name, a) in (axes ? (span..., ("isuniform", _alloc(q_isuni, g, d))) : span)
            a == 0 || println("    ", label, " / ", name, "(d=", d, ") -> ", a, " B")
            Test.@test a == 0
        end
        # spacing bounds are a rectilinear notion; a curvilinear grid has no axes to gap
        if g isa FG.Grids.StructuredGrid
            for (name, a) in (("minimum_spacing", _alloc(q_minsp, g, d)),
                              ("maximum_spacing", _alloc(q_maxsp, g, d)))
                a == 0 || println("    ", label, " / ", name, "(d=", d, ") -> ", a, " B")
                Test.@test a == 0
            end
        end
    end
    # A cell-list query enumerates through a fold, so it holds no candidate buffer at all — the property
    # the device path depends on.
    let cl = FG.Connectivity.MetricTopology(g; index = FG.Grids.cell_list(g; ball = r))
        a = _alloc(q_within_top, g, r, I, cl)
        a == 0 || println("    ", label, " / cell-list query -> ", a, " B")
        Test.@test a == 0
        Test.@test q_within_top(g, r, I, cl) == q_nwithin(g, r, I)
    end

    Test.@test (Test.@inferred q_coords(g, I)) isa NamedTuple
    Test.@test (Test.@inferred q_dist(g, cI, cJ)) isa Float64
    Test.@test (Test.@inferred q_nwithin(g, r, I)) isa Int
    Test.@test (Test.@inferred q_top(g)) isa FG.Connectivity.MetricTopology
    return nothing
end

const TOPICS = ["geometry", "axes", "grids", "sampling", "discretization", "connectivity",
                "extensions", "allocations", "api"]

# No argument runs every topic. Naming some — `Pkg.test(test_args = ["axes"])` — runs only those.
Test.@testset "FlowGeometries.jl" begin
    for topic in (isempty(ARGS) ? TOPICS : ARGS)
        include(topic * ".jl")
    end
end
