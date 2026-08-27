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
t_ugrid(::Type{T}, s) where {T}   = FG.Connectivity.unstructured_grid(T, s)

# A geometry defined outside the package, supplying only the accessor its hierarchy asks for.
struct OneSphere{T} <: FG.Geometry.AbstractSphericalGeometry{T} end
FG.Geometry.radius(::OneSphere{T}) where {T} = one(T)

concrete_return(f::F, argtypes) where {F} =
    (t = Base.return_types(f, argtypes)[1]; (isconcretetype(t), t))

const TOPICS = ["geometry", "axes", "grids", "sampling", "discretization", "connectivity",
                "extensions", "allocations", "api"]

# No argument runs every topic. Naming some — `Pkg.test(test_args = ["axes"])` — runs only those.
Test.@testset "FlowGeometries.jl" begin
    for topic in (isempty(ARGS) ? TOPICS : ARGS)
        include(topic * ".jl")
    end
end
