using JLArrays: JLArrays
using GPUArraysCore: GPUArraysCore

# The output type is written inside each kernel: `@allocated` of a call taking a type argument
# dispatches it dynamically.
_sa_to_cartesian(g, p) = FG.Geometry.spherical_to_cartesian(StaticArrays.SVector, g, p)
_sa_to_spherical(g, v) = FG.Geometry.cartesian_to_spherical(StaticArrays.SVector, g, v)
_sa_basis(g, p) = FG.Geometry.local_tangent_basis(StaticArrays.SVector, g, p)
_sa_project(g, p, q) = FG.Geometry.project_to_tangent_plane(StaticArrays.SVector, g, p, q)

Test.@testset "StaticArrays extension" begin
    using StaticArrays: StaticArrays as SA
    GE = FG.Geometry
    sph = GE.SphericalGeometry(6.371e6)
    cart = GE.CartesianGeometry()
    p2, q2 = SA.SVector(0.3, -0.4), SA.SVector(0.3 + 1e-7, -0.4 + 2e-7)
    p3, q3 = SA.SVector(0.3, -0.4, 6.4e6), SA.SVector(2.9, 0.1, 6.5e6)
    v = SA.SVector(-1.0, -2.0, 0.5)
    Test.@test GE.as_ntuple(p2) === (0.3, -0.4)
    # A static vector is read as its tuple, so every result is the tuple path's, bit for bit.
    Test.@test GE.distance(sph, p2, q2) === GE.distance(sph, Tuple(p2), Tuple(q2))
    Test.@test GE.distance(sph, p3, q3) === GE.distance(sph, Tuple(p3), Tuple(q3))
    Test.@test GE.distance(cart, p3, q3) === GE.distance(cart, Tuple(p3), Tuple(q3))
    ê = GE.local_tangent_basis(sph, p2)
    # …and a leading `SVector` builds the result as one.
    x = GE.spherical_to_cartesian(SA.SVector, sph, p2)
    Test.@test x isa SA.SVector{3,Float64}
    Test.@test Tuple(x) === Tuple(GE.spherical_to_cartesian(sph, Tuple(p2)))
    s = GE.cartesian_to_spherical(SA.SVector, sph, v)
    Test.@test s isa SA.SVector{3,Float64}
    Test.@test Tuple(s) === Tuple(GE.cartesian_to_spherical(sph, Tuple(v)))
    Test.@test 0 ≤ s[1] < 2π
    b = GE.local_tangent_basis(SA.SVector, sph, p2)
    Test.@test b.λ isa SA.SVector{3,Float64}
    Test.@test Tuple(b.φ) === ê.φ
    Test.@test GE.local_tangent_basis(SA.SVector, cart, SA.SVector(1.0, 2.0)).x == SA.SVector(1.0, 0.0)
    d = GE.project_to_tangent_plane(SA.SVector, sph, p2, q2)
    Test.@test d isa SA.SVector{2,Float64}
    Test.@test Tuple(d) === Tuple(GE.project_to_tangent_plane(sph, Tuple(p2), Tuple(q2)))
    Test.@test GE.project_to_tangent_plane(SA.SVector, cart, SA.SVector(1.0, 2.0), SA.SVector(1.5, 1.0)) ==
               SA.SVector(0.5, -1.0)
    # Static vectors in, static vectors out, nothing on the heap.
    Test.@test _alloc(GE.distance, sph, p2, q2) == 0
    Test.@test _alloc(GE.distance, sph, p3, q3) == 0
    Test.@test _alloc(_sa_to_cartesian, sph, p2) == 0
    Test.@test _alloc(_sa_to_spherical, sph, v) == 0
    Test.@test _alloc(_sa_basis, sph, p2) == 0
    Test.@test _alloc(_sa_project, sph, p2, q2) == 0
end

Test.@testset "Threading is opt-in and changes no result" begin
    # Chunking must partition exactly — no gaps, no overlap, contiguous, whatever the remainder.
    for (n, k) in ((10, 3), (3, 10), (1, 1))
        rs = FG.Execution.chunk_ranges(n, k)
        Test.@test reduce(vcat, collect.(rs)) == collect(1:n)
        Test.@test all(!isempty, rs) && length(rs) ≤ max(1, min(k, n))
        Test.@test all(FG.Execution._chunk_range(n, length(rs), c) == rs[c] for c in eachindex(rs))
    end
    # More chunks than threads, taken as threads free up: still a partition in order, and an exact
    # reduction is the serial one.
    let bt = FG.Execution.BalancedThreads(4)
        for n in (0, 10_000)
            Test.@test FG.Execution.reduce_indices(i -> i * i, +, 0, n, bt) == sum(i * i for i in 1:n; init = 0)
        end
        Test.@test reduce(vcat, FG.Execution.map_chunks(collect, 1000, bt)) == collect(1:1000)
        hits = zeros(Int, 997)
        FG.Execution.run_indices(i -> (hits[i] += 1; nothing), 997, bt)
        Test.@test all(==(1), hits)
        Test.@test_throws ArgumentError FG.Execution.BalancedThreads(0)
    end
    # A serial backend hands the body the whole range in one call: the serial path adds no partitioning.
    seen = UnitRange{Int}[]
    FG.Execution.run_chunks(17, CB.SerialBackend()) do r; push!(seen, r); end
    Test.@test seen == [1:17]

    # The empty-input contract, which the two shapes answer differently: a write loop has nothing to
    # write, and a reduction still has to produce a value.
    ran = Ref(0)
    FG.Execution.run_chunks(0, CB.SerialBackend()) do r; ran[] += 1; end
    Test.@test ran[] == 0
    Test.@test FG.Execution.map_chunks(r -> length(r), 0, CB.SerialBackend()) == [0]
    Test.@test FG.Execution.map_chunks(r -> length(r), 0, CB.ThreadedBackend()) == [0]

    # AutoBackend is FlowGeometries' own policy: the threaded backend, whose loops run serially on one
    # thread.
    Test.@test FG.Execution.resolve(CB.AutoBackend()) === CB.ThreadedBackend()
    Test.@test FG.Execution.resolve(CB.SerialBackend()) === CB.SerialBackend()
    # A device tag that wraps no KernelAbstractions backend has nothing to launch on.
    Test.@test_throws ArgumentError FG.Execution.run_indices(i -> nothing, 4, CB.GPUBackend(:nodevice))

    Test.@test sum(FG.Execution.map_chunks(r -> sum(r), 1000, CB.ThreadedBackend())) == sum(1:1000)

    # `exclusive_scan!` is exact, so the threaded form has to equal the serial one entry for entry.
    for n in (0, 65_536)
        counts = rand(0:9, n)
        want = FG.Execution.exclusive_scan!(Vector{Int}(undef, n + 1), counts)
        Test.@test FG.Execution.exclusive_scan!(Vector{Int}(undef, n + 1), counts, CB.ThreadedBackend()) == want
    end

    # …and so is an integer reduction, threaded and on the device.
    for n in (0, 10_000), b in (CB.ThreadedBackend(), CB.GPUBackend(KernelAbstractions.CPU()))
        Test.@test FG.Execution.reduce_indices(i -> i * i, +, 0, n, b) == sum(i * i for i in 1:n; init = 0)
    end

    # A cell loop visits every index of its shape once, an empty direction included, and an integer
    # reduction over the cells is exact on every backend.
    for dims in ((0,), (37,), (13, 0), (13, 11), (7, 5, 4)),
        b in (CB.SerialBackend(), CB.ThreadedBackend(), FG.Execution.BalancedThreads(4),
              CB.GPUBackend(KernelAbstractions.CPU()))
        hits = zeros(Int, dims)
        FG.Execution.run_cells(I -> (hits[I] += 1; nothing), dims, b, FG.Execution.Written(hits))
        Test.@test all(==(1), hits)
        lin = LinearIndices(dims)
        Test.@test FG.Execution.reduce_cells(I -> lin[I]^2, +, 0, dims, b) == sum(abs2, 1:prod(dims); init = 0)
    end
    # A floating-point cell sum regroups each line's terms, the same way on every call.
    let x = randn(257, 33)
        s = FG.Execution.reduce_cells(I -> x[I], +, 0.0, size(x), CB.SerialBackend())
        Test.@test s === FG.Execution.reduce_cells(I -> x[I], +, 0.0, size(x), CB.SerialBackend())
        Test.@test abs(s - sum(x)) ≤ length(x) * eps() * sum(abs, x)
    end

    # A CSR build goes through the scan, so it must stay bit-identical under threading.
    let sph = FG.Geometry.SphericalGeometry(), cart = FG.Geometry.CartesianGeometry()
        for gg in (FG.Grids.StructuredGrid(cart, 0.0:1.0:9.0, 0.0:1.0:9.0), FG.Grids.HEALPixGrid(sph, 4),
                   FG.Grids.RingGrid(sph, FG.SphericalSampling.OctahedralGaussianSampling(8)))
            a = FG.Connectivity.build_connectivity(gg)
            b = FG.Connectivity.build_connectivity(gg; execution = CB.ThreadedBackend())
            Test.@test a.ptr == b.ptr
            Test.@test a.nbrs == b.nbrs
        end
        hp = FG.SphericalSampling.HEALPixSampling(4)
        Test.@test FG.Connectivity.build_connectivity(hp).nbrs ==
                   FG.Connectivity.build_connectivity(hp; execution = CB.ThreadedBackend()).nbrs
    end

    # A grid reduction runs per index where the candidates need no buffer, so it reaches a device too.
    let gs = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), 0.0:1.0:19.0, 0.0:1.0:19.0)
        base = FG.Connectivity.mapreduce_within((I, J, d) -> 1, +, 0, gs; ball = 2.5)
        Test.@test base > 0
        for b in (CB.SerialBackend(), CB.ThreadedBackend(), CB.GPUBackend(KernelAbstractions.CPU()))
            Test.@test FG.Connectivity.mapreduce_within((I, J, d) -> 1, +, 0, gs;
                                                        ball = 2.5, execution = b) == base
        end
    end

    # Every threaded kernel is bit-identical to serial.
    geo = FG.Geometry.SphericalGeometry()
    thr = CB.ThreadedBackend()
    for n in (7,)
        a = FG.SphericalSampling.cubed_sphere_points(n)
        b = FG.SphericalSampling.cubed_sphere_points(n; execution = thr)
        c = FG.SphericalSampling.cubed_sphere_points(n; execution = CB.SerialBackend())
        Test.@test a.λ == b.λ == c.λ
        Test.@test a.φ == b.φ == c.φ
        Test.@test a.panel == b.panel == c.panel
        sp = FG.SphericalSampling.spherical_points(FG.SphericalSampling.CubedSphereSampling(), n; execution = thr)
        Test.@test sp.λ == a.λ && sp.φ == a.φ
        d = FG.SphericalSampling.cubed_sphere_points(n; execution = CB.GPUBackend(KernelAbstractions.CPU()))
        Test.@test d.λ == a.λ && d.φ == a.φ && d.panel == a.panel
    end
    for (g, lbl) in ((geo, "spherical"),)
        for n in (3, 40)   # n = 3 gives fewer rows than threads, exercising the short case
            λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
            φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
            m = fill(true, n, n)
            Test.@test FG.Grids.measure(FG.Grids.CurvilinearGrid(g, λ, φ, m)) ==
                       FG.Grids.measure(FG.Grids.CurvilinearGrid(g, λ, φ, m; execution = thr))
        end
    end

    # Connectivity: both cell passes write only slots their own cell owns. The masked and periodic
    # cases carry a varying per-cell degree, so a chunk boundary landing mid-row surfaces as a wrong
    # offset here.
    for (topo, st, ao) in ((FG.Connectivity.IndexTopology((37, 21), (true, true), nothing), FG.Stencils.Axial(1), false),
                           (FG.Connectivity.IndexTopology((31, 29), (true, false),
                                                          [isodd(i * 7 + j * 3) for i in 1:31, j in 1:29]),
                            FG.Stencils.Moore(1), true))
        a = FG.Connectivity.build_connectivity(topo; stencil = st, active_only = ao)
        b = FG.Connectivity.build_connectivity(topo; stencil = st, active_only = ao, execution = thr)
        Test.@test a.ptr == b.ptr
        Test.@test a.nbrs == b.nbrs
    end
    # A tensor-product sampling builds the same graph from its axis lengths alone, on any backend.
    for (s, st, be) in ((FG.SphericalSampling.GaussLegendreSampling(), FG.Stencils.Moore(1), thr),
                        (FG.SphericalSampling.DriscollHealySampling(), FG.Stencils.Axial(1),
                         CB.GPUBackend(KernelAbstractions.CPU())))
        a = FG.Connectivity.build_connectivity(s, 9; stencil = st)
        b = FG.Connectivity.build_connectivity(s, 9; stencil = st, execution = be)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end

    # The metric-ball builder chunks the same two owned-slot passes, and per-cell degrees vary
    # even more (polar rows reach every longitude), so the same chunk-boundary argument applies.
    gball = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6),
                                    FG.Axes.PeriodicUniformAxis(0.0, 2π, 24),
                                    range(-π / 2, π / 2; length = 13))
    a = FG.Connectivity.build_connectivity_within(gball; ball = 2.0e6)
    b = FG.Connectivity.build_connectivity_within(gball; ball = 2.0e6, execution = thr)
    Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs

    # The candidate builder emits and dedups concurrently, so each sampling's `emit!` has to be
    # free of state shared between nodes. nside = 1 and 2 cover the singular pixels, the ones with
    # seven neighbours.
    for s in (FG.SphericalSampling.HEALPixSampling(1), FG.SphericalSampling.HEALPixSampling(8))
        a = FG.Connectivity.build_connectivity(s)
        b = FG.Connectivity.build_connectivity(s; execution = thr)
        Test.@test a.ptr == b.ptr
        Test.@test a.nbrs == b.nbrs
    end
    for (n, st) in ((1, FG.Stencils.Axial(1)), (9, FG.Stencils.Moore(1)))
        a = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), n; stencil = st)
        b = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), n; stencil = st, execution = thr)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
    for (nlon, nlat) in ((7, 5),), st in (FG.Stencils.Moore(1),)
        a = FG.Connectivity.build_connectivity(FG.SphericalSampling.YinYangSampling(), nlon, nlat; stencil = st)
        b = FG.Connectivity.build_connectivity(FG.SphericalSampling.YinYangSampling(), nlon, nlat; stencil = st, execution = thr)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
end

Test.@testset "Grids can be moved to another storage backend" begin
    using Adapt: Adapt, adapt
    # A stand-in device array: a distinct type, so a successful adapt shows in the grid's type.
    struct FakeDev end
    struct DevArr{T,N} <: AbstractArray{T,N}
        a::Array{T,N}
    end
    Base.size(d::DevArr) = size(d.a)
    Base.getindex(d::DevArr, I...) = getindex(d.a, I...)
    Adapt.adapt_storage(::FakeDev, x::Array) = DevArr(x)

    geo = FG.Geometry.SphericalGeometry()
    g = FG.Connectivity.structured_grid(FG.SphericalSampling.Fejer1Sampling(), 9)
    d = adapt(FakeDev(), g)
    # The latitude moves, and the latitude weights of the separable measure with it.
    Test.@test FG.Grids.coordinates(d, 2) isa DevArr
    Test.@test FG.Grids.measure_factors(d)[2] isa DevArr
    Test.@test all(FG.Grids.measure(d)[i, j] == FG.Grids.measure(g)[i, j]
                   for i in 1:size(g, 1), j in 1:size(g, 2))
    Test.@test FG.Grids.isperiodic(d, 1) == FG.Grids.isperiodic(g, 1)

    n = 12
    λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
    φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
    cg0 = FG.Grids.CurvilinearGrid(geo, λ, φ, fill(true, n, n))
    cg = adapt(FakeDev(), cg0)
    Test.@test FG.Grids.coordinates(cg, 1) isa DevArr && FG.Grids.measure(cg) isa DevArr
    Test.@test all(FG.Grids.measure(cg)[i, j] == FG.Grids.measure(cg0)[i, j] for i in 1:n, j in 1:n)

    ug0 = FG.Connectivity.unstructured_grid(FG.SphericalSampling.IcosahedralSampling(3))
    ug = adapt(FakeDev(), ug0)
    Test.@test FG.Grids.coordinates(ug, 1) isa DevArr
    Test.@test getfield(ug, :neighbor_nbrs) isa DevArr
    Test.@test getfield(ug, :neighbor_ptr) isa DevArr
    Test.@test all(FG.Grids.measure(ug)[i] == FG.Grids.measure(ug0)[i] for i in eachindex(FG.Grids.measure(ug0)))

    c = adapt(FakeDev(), FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(2)))
    Test.@test c.nbrs isa DevArr && c.ptr isa DevArr

    # Held plans move by their arrays: a uniform plan's rows, a tabulated plan's table, a gradient plan's
    # CSR and coefficients. A staggered grid moves its centre grid and face axes, a rotated grid its base.
    D = FG.Discretization
    up = adapt(FakeDev(), D.stencil_plan(range(0.0, 1.0; length = 12), 1, 5))
    Test.@test up.weights isa DevArr && up.edge_idx isa DevArr && up.edge_wts isa DevArr
    tp = adapt(FakeDev(), D.stencil_plan(collect(cumsum(1.0 .+ 0.2 .* sin.(1:12))), 1, 3))
    Test.@test tp.indices isa DevArr && tp.weights isa DevArr
    gp = adapt(FakeDev(), FG.Operators.gradient_plan(cg0))
    Test.@test gp.ptr isa DevArr && gp.nbr isa DevArr && all(c -> c isa DevArr, gp.coeffs)
    st = adapt(FakeDev(), FG.Grids.StaggeredGrid(FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(),
                                                                          collect(0.0:1.0:5.0), collect(0.0:1.0:3.0))))
    Test.@test FG.Grids.coordinates(st.center, 1) isa DevArr && all(f -> f isa DevArr, st.faces)
    rg = adapt(FakeDev(), FG.Grids.rotate(FG.Connectivity.structured_grid(FG.SphericalSampling.Fejer1Sampling(), 9),
                                          FG.Geometry.PoleRotation(1.0, 0.7)))
    Test.@test FG.Grids.coordinates(FG.Grids.base_grid(rg), 2) isa DevArr
    # The moved grid's type follows from the grid's field types, so the move infers.
    for gg in (g, cg0, ug0, FG.Grids.YinYangGrid(8, 6))
        Test.@inferred adapt(FakeDev(), gg)
    end

    # A face metric moves its grid and every array; the Laplacian's coefficients move with their
    # null space.
    let mk = trues(6, 5)
        mk[3, 3] = false
        gm = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), collect(0.0:1.0:5.0), collect(0.0:0.5:2.0), mk)
        fm = adapt(FakeDev(), FG.Operators.face_metrics(gm, FG.Operators.Dirichlet()))
        Test.@test all(a -> a isa DevArr, fm.area_op)
        Test.@test all(a -> a isa DevArr, fm.gap) && fm.corner[1] isa DevArr
        Test.@test FG.Grids.coordinates(fm.grid, 1) isa DevArr
        c = adapt(FakeDev(), FG.Operators.laplacian_coefficients(FG.Operators.face_metrics(gm)))
        Test.@test c.diag isa DevArr && c.measure isa DevArr && all(a -> a isa DevArr, c.coef) && c.singular
    end

    t2 = FG.Connectivity.IndexTopology((4, 3), (true, false), fill(true, 4, 3))
    Test.@test adapt(FakeDev(), t2).mask isa DevArr
end

Test.@testset "Equiangular weights by either algorithm agree with the definition" begin
    using FFTW: FFTW          # an FFT for `FTB.Transform()`

    # Literal evaluation of each rule — no recurrence, no transform: Fejér's first rule (a cosine
    # series) on the open nodes, Driscoll–Healy's sine series on the closed ones.
    function direct(family, nlat)
        if family === :open
            return [(2 / nlat) * (1 - 2 * sum((cos(2j * π * (i - 0.5) / nlat) / (4j^2 - 1) for j in 1:(nlat ÷ 2));
                                               init = 0.0))
                    for i in 1:nlat]
        end
        return [(4 / nlat) * sin(π * (i - 1) / nlat) *
                sum(sin((2k + 1) * π * (i - 1) / nlat) / (2k + 1) for k in 0:(nlat ÷ 2 - 1))
                for i in 1:nlat]
    end
    open_counts = (1, 7, 64, 511)
    for nlat in (8, 512)
        dh = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), nlat)
        Test.@test maximum(abs.(dh .- direct(:closed, nlat))) < 1e-13
        Test.@test sum(dh) ≈ 2 atol = 1e-13
    end
    for nlat in open_counts
        fw = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.Fejer1Sampling(), nlat)
        Test.@test maximum(abs.(fw .- direct(:open, nlat))) < 1e-13
        Test.@test sum(fw) ≈ 2 atol = 1e-13
    end
    SS = FG.SphericalSampling
    # Both algorithms, asked for by name, agree on the weights. The sine sums behind them are O(1)
    # while the weights carry a `4/nlat·sinθ` factor, and the recurrence accumulates round-off across
    # its `nterm` steps.
    for (s, fam, counts) in ((SS.Fejer1Sampling(), :open, open_counts),
                             (SS.DriscollHealySampling(), :closed, (8, 512))),
        nlat in counts
        wt = SS.latitude_weights(s, nlat; algorithm = FTB.Transform())
        wr = SS.latitude_weights(s, nlat; algorithm = FTB.Recurrence())
        Test.@test maximum(abs.(wt .- wr)) < 1e-14
        # Each construction independently reproduces the literal defining sum.
        Test.@test maximum(abs.(wt .- direct(fam, nlat))) < 1e-13
        Test.@test maximum(abs.(wr .- direct(fam, nlat))) < 1e-13
    end
    # Gauss–Legendre is not an equiangular family, so asking it for one of these says so.
    Test.@test_throws ArgumentError SS.latitude_weights(
        SS.GaussLegendreSampling(), 8; algorithm = FTB.Recurrence())
end

Test.@testset "Sparse adjacency assembles straight into CSC" begin
    using SparseArrays: SparseArrays
    conn = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(8))
    n, ne = FG.Connectivity.nnodes(conn), FG.Connectivity.nedges(conn)
    A = FG.Connectivity.sparse_adjacency_matrix(conn)
    Test.@test SparseArrays.nnz(A) == ne
    # Same matrix the dense builder produces.
    Test.@test Matrix(A) == FG.Connectivity.adjacency_matrix(conn)
    # Row indices ascending within each column, as CSC requires.
    Test.@test all(issorted(@view SparseArrays.rowvals(A)[SparseArrays.nzrange(A, j)]) for j in 1:n)

    # Caller-owned buffers hold the matrix's storage.
    colptr = Vector{Int}(undef, n + 1)
    rowval = Vector{Int}(undef, ne)
    nzval = Vector{Bool}(undef, ne)
    B = FG.Connectivity.sparse_adjacency_matrix!(colptr, rowval, nzval, conn)
    Test.@test B == A
    Test.@test B.colptr === colptr && B.rowval === rowval && B.nzval === nzval

    # The COO route still agrees with the direct one.
    I = Vector{Int}(undef, ne); J = Vector{Int}(undef, ne)
    Test.@test FG.Connectivity.sparse_adjacency_coo!(I, J, conn) == ne
    Test.@test SparseArrays.sparse(I, J, trues(ne), n, n) == A

    # A symmetric graph's CSR arrays serve as its CSC arrays. On every shape, symmetric or not, entry
    # `(i, j)` is set exactly when `j` is a neighbour of `i`.
    let cart = FG.Geometry.CartesianGeometry{Float64}(),
        xs = collect(0.0:1.0:4.0), ys = collect(0.0:1.0:3.0),
        mk = trues(5, 4)
        mk[3, 2] = false
        gg = FG.Grids.StructuredGrid(cart, xs, ys)
        gm = FG.Grids.StructuredGrid(cart, xs, ys, mk)
        λn = [0.3i for i in 0:19]
        φn = [0.2 * sin(3.0i) for i in 0:19]
        gu = FG.Grids.UnstructuredGrid(FG.Geometry.SphericalGeometry(6.371e6), (λn, φn), trues(20);
                                       k = 3, areas = ones(20))
        cases = (("Axial", gg, (;)),
                 ("Moore", gg, (; stencil = FG.Stencils.Moore(1))),
                 ("Upwind, asymmetric", gg, (; stencil = Upwind(1))),
                 ("Custom forward-only", gg, (; stencil = FG.Stencils.Custom(((1, 0), (0, 1))))),
                 ("masked", gm, (;)),
                 ("stored kNN adjacency", gu, (;)),
                 ("formula neighbours", FG.Grids.HEALPixGrid(2), (;)))
        for (_, g, kw) in cases
            cn = FG.Connectivity.build_connectivity(g; kw...)
            M = FG.Connectivity.sparse_adjacency_matrix(g; kw...)
            Test.@test SparseArrays.nnz(M) == FG.Connectivity.nedges(cn)
            Test.@test all(M[i, Int(cn.nbrs[k])]
                           for i in 1:FG.Connectivity.nnodes(cn)
                           for k in cn.ptr[i]:(cn.ptr[i + 1] - 1))
        end
        # The wrap is the cheaper route for one and the same graph: the transpose allocates a second
        # CSR.
        big = FG.Grids.StructuredGrid(cart, collect(range(0.0, 1.0; length = 120)),
                                      collect(range(0.0, 1.0; length = 120)))
        Test.@test _alloc(q_sparse, big) < _alloc(q_sparse_csr, big)
    end
end

Test.@testset "A formula layout's symmetric adjacency wraps into CSC" begin
    using SparseArrays: SparseArrays
    # Each of these declares `has_symmetric_adjacency`, and every declaration is checked here against
    # the graph the formula builds.
    symmetric = (FG.Grids.HEALPixGrid(4), FG.Grids.HEALPixGrid(8),
                 FG.Grids.CubedSphereGrid(6), FG.Grids.IcosahedralGrid(3),
                 FG.Grids.YinYangGrid(16, 8))
    for g in symmetric
        Test.@test FG.Grids.has_symmetric_adjacency(g)
        conn = FG.Connectivity.build_connectivity(g)
        Test.@test FG.Connectivity.is_symmetric_adjacency(conn)
        A = FG.Connectivity.sparse_adjacency_matrix(g)
        Test.@test A == FG.Connectivity.sparse_adjacency_matrix(conn)
        Test.@test A == permutedims(A)
    end

    # A ring grid's straddling relation is directed where adjacent rings differ in width, so it stays
    # at the default and takes the transpose.
    ring = FG.Grids.RingGrid(FG.SphericalSampling.OctahedralGaussianSampling(8))
    Test.@test !FG.Grids.has_symmetric_adjacency(ring)
    rconn = FG.Connectivity.build_connectivity(ring)
    Test.@test !FG.Connectivity.is_symmetric_adjacency(rconn)
    R = FG.Connectivity.sparse_adjacency_matrix(ring)
    Test.@test R == FG.Connectivity.sparse_adjacency_matrix(rconn)
    Test.@test R != permutedims(R)
    Test.@test all(R[i, Int(rconn.nbrs[k])]
                   for i in 1:FG.Connectivity.nnodes(rconn)
                   for k in rconn.ptr[i]:(rconn.ptr[i + 1] - 1))

    # A mask drops both directions of every edge it touches, so the symmetry the trait declares
    # survives it.
    let n = length(FG.Grids.HEALPixGrid(4)),
        gm = FG.Grids.HEALPixGrid(FG.Geometry.SphericalGeometry(), 4; mask = [isodd(i) for i in 1:n])
        M = FG.Connectivity.sparse_adjacency_matrix(gm)
        Test.@test M == permutedims(M)
        Test.@test M == FG.Connectivity.sparse_adjacency_matrix(FG.Connectivity.build_connectivity(gm))
    end

    # And the wrap is the cheaper route on a formula layout too.
    hp = FG.Grids.HEALPixGrid(16)
    Test.@test _alloc(q_sparse, hp) < _alloc(q_sparse_csr, hp)
end

Test.@testset "Index-parallel loops run as kernels and give the same answer" begin
    D = FG.Discretization
    O = FG.Operators
    C = FG.Connectivity
    GD = FG.Grids
    cpu = CB.GPUBackend(KernelAbstractions.CPU())

    cart = FG.Geometry.CartesianGeometry{Float64}()
    let g = GD.StructuredGrid(cart, collect(0.0:31.0), collect(0.0:31.0); topology = GD.Periodic(), period = (32.0, 0.0))
        a = C.build_connectivity(g; stencil = FG.Stencils.Moore(2))
        b = C.build_connectivity(g; stencil = FG.Stencils.Moore(2), execution = cpu)
        Test.@test a.ptr == b.ptr
        Test.@test a.nbrs == b.nbrs
    end

    acc = zeros(Int, 100)
    FG.Execution.run_indices(i -> (acc[i] = i * i), 100, cpu)
    Test.@test acc == [i * i for i in 1:100]

    # A device reduction is exact on integers at every size: one partial group, a group and a round
    # boundary either side, and more elements than the first stage has work-items.
    for n in (0, 1, 256, 257, 256 * 65_536 + 3)
        Test.@test FG.Execution.reduce_indices(identity, +, 0, n, cpu) == n * (n + 1) ÷ 2
    end
    # It combines in index order, so an associative `op` that does not commute folds as the serial
    # loop does. Composing affine maps `x ↦ a·x + b` (mod p) is such an op, `(1, 0)` its identity.
    let p = 1_000_003, n = 70_001
        compose(u, v) = ((u[1] * v[1]) % p, (u[2] * v[1] + v[2]) % p)
        aff(i) = (i % 7 + 2, i % 11)
        want = foldl(compose, (aff(i) for i in 1:n); init = (1, 0))
        Test.@test FG.Execution.reduce_indices(aff, compose, (1, 0), n, cpu) == want
        Test.@test FG.Execution.reduce_indices(aff, compose, (1, 0), n, CB.SerialBackend()) == want
    end
    # The device scan equals the serial one entry for entry, for either offset width and any `init`.
    for (n, I, init) in ((0, Int, 1), (256, Int32, 0), (257, Int, 5), (300_001, Int32, 1))
        counts = rand(0:9, n)
        want = FG.Execution.exclusive_scan!(Vector{I}(undef, n + 1), counts; init = init)
        Test.@test FG.Execution.exclusive_scan!(Vector{I}(undef, n + 1), counts, cpu; init = init) == want
    end

    # A ball query is device-shaped once the topology carries no index: it reads coordinates and the
    # mask, and allocates nothing, so it runs inside the launch.
    gball = GD.StructuredGrid(cart, collect(0.0:31.0), collect(0.0:31.0))
    function counts(g, r, backend)
        sz = GD.size_tuple(g)
        out = zeros(Int, sz)
        mt = C.MetricTopology(g)
        ci = CartesianIndices(sz)
        FG.Execution.run_indices(length(ci), backend) do lin
            I = Tuple(@inbounds ci[lin])
            @inbounds out[lin] = C.nneighbors_within(g, I...; ball = r, topology = mt)
        end
        return out
    end
    Test.@test counts(gball, 3.0, cpu) == counts(gball, 3.0, CB.SerialBackend())
    Test.@test counts(gball, 3.0, cpu) ==
               [C.nneighbors_within(gball, Tuple(ci)...; ball = 3.0)
                for ci in CartesianIndices(size(GD.mask(gball)))]

    # The sweep follows the same rule: unindexed it is one body per cell, so it launches.
    function sweep_counts(g, r, backend)
        out = zeros(Int, GD.size_tuple(g))
        C.foreach_within(g; ball = r, topology = C.MetricTopology(g), execution = backend) do I, J, d
            @inbounds out[I[1], I[2]] += 1
        end
        return out
    end
    Test.@test sweep_counts(gball, 3.0, cpu) == sweep_counts(gball, 3.0, CB.SerialBackend())
    Test.@test sum(sweep_counts(gball, 3.0, cpu)) ==
               C.mapreduce_within((I, J, d) -> 1, +, 0, gball; ball = 3.0)

    # A chunked body accumulates across its range, so a device backend refuses it outright.
    Test.@test_throws ArgumentError FG.Execution.run_chunks(r -> nothing, 4, cpu)
    Test.@test_throws ArgumentError FG.Execution.map_chunks(r -> 1, 4, cpu)

    # A topology holding a k-d tree refuses to move to a device.
    gu = healpix_node_grid(2)
    Test.@test_throws ArgumentError Adapt.adapt(Array, C.indexed(gu))

    # A cell list moves with its topology, and a launched fold over it visits each cell's candidates in
    # the host's order, so a floating-point sum over them comes back identical.
    gc = GD.CurvilinearGrid(cart, [i + 0.25j for i in 1:24, j in 1:24], [j + 0.02i for i in 1:24, j in 1:24],
                            trues(24, 24))
    mtc = C.default_sweep_topology(gc, 3.0, false)
    moved = Adapt.adapt(Array, mtc)
    Test.@test moved.index.items == mtc.index.items && moved.index.bins == mtc.index.bins
    function dist_sums(g, mt, backend)
        ci = CartesianIndices(GD.size_tuple(g))
        out = zeros(length(ci))
        FG.Execution.run_indices(length(ci), backend) do lin
            I = Tuple(@inbounds ci[lin])
            @inbounds out[lin] = C.fold_within((acc, J, d) -> acc + d, 0.0, g, I...; ball = 3.0, topology = mt)
        end
        return out
    end
    Test.@test dist_sums(gc, moved, cpu) == dist_sums(gc, mtc, CB.SerialBackend())

    # An indexed sweep does stay on the host: each task needs its own candidate buffer, which a
    # launch has nowhere to put.
    Test.@test_throws ArgumentError C.foreach_within(
        (I, J, d) -> nothing, gu; ball = 2.0e6, topology = C.indexed(gu), execution = cpu,
    )
end

Test.@testset "Every layout's cell centres can be indexed, not just the three architectures" begin
    GD = FG.Grids
    C = FG.Connectivity
    # A formula layout streams its centres by arithmetic, and a rotated one evaluates them through its
    # frame; a tree is built over either the same way.
    layouts = (("HEALPix", GD.HEALPixGrid(8)),
               ("ring", GD.RingGrid(FG.SphericalSampling.OctahedralGaussianSampling(16))),
               ("cubed sphere", GD.CubedSphereGrid(8)),
               ("Yin-Yang", GD.YinYangGrid(16, 8)),
               ("icosahedral", GD.IcosahedralGrid(4)),
               ("rotated", GD.rotate(
                    GD.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6),
                                      FG.Axes.PeriodicUniformAxis(0.0, 2π, 24),
                                      range(-1.2, 1.2; length = 13)),
                    FG.Geometry.PoleRotation(0.7, 1.1))))
    for (_, g) in layouts
        Test.@test GD.has_spatial_index(g)
        ix = C.indexed(g)
        sc = C.ball_scratch()
        cell = GD.cell_at(g, first(GD.cells(g)))
        # An index may only over-return, and the distance gate decides membership, so the indexed
        # answer equals the scan's to the cell.
        Test.@test C.nneighbors_within(g, cell...; ball = 2.0e6, topology = ix, scratch = sc) ==
                   C.nneighbors_within(g, cell...; ball = 2.0e6, topology = C.MetricTopology(g))
    end
end

# Rotation about the polar axis, `u = cos φ` eastward on the unit sphere, is `ẑ × r̂ = (−sin λ cos φ,
# cos λ cos φ, 0)` in the ambient frame; a field and its local components round-trip; and an outer
# product rotates to the outer product of the rotated vectors. Over a trailing batch axis, on a lon–lat
# grid and a pixelization, on every backend.
Test.@testset "Fields rotate between the local and the ambient frame" begin
    O = FG.Operators
    geo = FG.Geometry.SphericalGeometry()
    msk = trues(12, 7)
    msk[3:5, 2:3] .= false
    grids = (FG.Grids.StructuredGrid(geo, FG.Axes.PeriodicUniformAxis(0.0, 2π, 12), range(-1.2, 1.2; length = 7), msk),
             FG.Grids.HEALPixGrid(geo, 2))
    for (g, b) in ((grids[1], CB.SerialBackend()), (grids[1], CB.ThreadedBackend()),
                   (grids[2], CB.GPUBackend(KernelAbstractions.CPU())))
        sz = (size(FG.Grids.mask(g))..., 2)
        cells = CartesianIndices(size(FG.Grids.mask(g)))
        λφ = [Tuple(FG.Grids.coords(g, Tuple(cells[mod1(k, length(cells))])...))[1:2] for k in 1:prod(sz)]
        act = reshape([FG.Grids.isactive(g, Tuple(cells[mod1(k, length(cells))])...) for k in 1:prod(sz)], sz)
        u = reshape([cos(p[2]) for p in λφ], sz)
        X, Y, Z = zeros(sz), zeros(sz), zeros(sz)
        O.to_planetary!(X, Y, Z, u, zeros(sz), nothing, g; execution = b)
        Test.@test maximum(abs, vec(X) .- vec(act) .* [-sin(p[1]) * cos(p[2]) for p in λφ]) < 1e-15
        Test.@test maximum(abs, vec(Y) .- vec(act) .* [cos(p[1]) * cos(p[2]) for p in λφ]) < 1e-15
        Test.@test maximum(abs, Z) < 1e-15

        v = reshape(sin.(1:prod(sz)), sz)
        w = reshape(cos.(1:prod(sz)), sz)
        O.to_planetary!(X, Y, Z, u, v, w, g; execution = b)
        u2, v2, w2 = fill(NaN, sz), fill(NaN, sz), fill(NaN, sz)
        O.to_local!(u2, v2, w2, X, Y, Z, g; execution = b)
        Test.@test maximum(abs, u2 .- act .* u) < 1e-15 && maximum(abs, v2 .- act .* v) < 1e-15 &&
                   maximum(abs, w2 .- act .* w) < 1e-15

        t = ntuple(_ -> fill(NaN, sz), 6)
        amb = (X .* X, Y .* Y, Z .* Z, X .* Y, X .* Z, Y .* Z)
        O.tensor_to_local!(t..., amb..., g; execution = b)
        for (q, ref) in zip(t, (u .* u, v .* v, w .* w, u .* v, u .* w, v .* w))
            Test.@test maximum(abs, q .- act .* ref) < 1e-14
        end

        # The grid's frame plan in its place writes the same bits.
        fp = O.frame_plan(g; execution = b)
        P, L, T6 = ntuple(_ -> fill(NaN, sz), 3), ntuple(_ -> fill(NaN, sz), 3), ntuple(_ -> fill(NaN, sz), 6)
        O.to_planetary!(P..., u, v, w, fp; execution = b)
        O.to_local!(L..., X, Y, Z, fp; execution = b)
        O.tensor_to_local!(T6..., amb..., fp; execution = b)
        Test.@test isequal(P, (X, Y, Z)) && isequal(L, (u2, v2, w2)) && isequal(T6, t)
        if b isa CB.SerialBackend
            Test.@test _alloc(q_toplan!, (X, Y, Z), (u, v, w), g) == 0
            Test.@test _alloc(q_toloc!, (u2, v2, w2), (X, Y, Z), g) == 0
            Test.@test _alloc(q_tens!, t, amb, g) == 0
            Test.@test _alloc(q_toplan!, P, (u, v, w), fp) == 0
            Test.@test _alloc(q_toloc!, L, (X, Y, Z), fp) == 0
            Test.@test _alloc(q_tens!, T6, amb, fp) == 0
        end
    end
end

# A KernelAbstractions backend whose arrays are `JLArray`s, which raise on a host read of one element:
# `GPUBackend(KernelAbstractions.CPU())` computes on plain `Array`s, where a host loop over a device array
# still gives the right answer. KernelAbstractions declares `synchronize` with no default and JLArrays
# defines none; each method below takes `NoScalarBackend`, which this file owns, and the kernel-call
# method, KernelAbstractions' extension point for a backend, hands the launch to JLArrays.
struct NoScalarBackend <: KernelAbstractions.GPU end
KernelAbstractions.allocate(::NoScalarBackend, ::Type{T}, dims::Tuple) where {T} =
    JLArrays.JLArray{T}(undef, dims)
KernelAbstractions.synchronize(::NoScalarBackend) = nothing
# `@kernel` builds the host variant for a backend whose `isgpu` is false, the variant JLArrays runs.
KernelAbstractions.isgpu(::NoScalarBackend) = false
function (k::KernelAbstractions.Kernel{NoScalarBackend,W,N,F})(args...; ndrange = nothing,
                                                             workgroupsize = nothing) where {W,N,F}
    jl = KernelAbstractions.Kernel{JLArrays.JLBackend,W,N,F}(JLArrays.JLBackend(), k.f)
    return jl(args...; ndrange = ndrange, workgroupsize = workgroupsize)
end
Adapt.adapt_storage(::NoScalarBackend, a::Array) = JLArrays.JLArray(a)
Adapt.adapt_storage(::NoScalarBackend, a::JLArrays.JLArray) = a

# Every operator against its own serial run, twice: with the grid, metrics and plans held on the host,
# and with them moved to the device first. A launch does the serial arithmetic cell for cell, so the
# values agree to the bit.
Test.@testset "Field operators on a device that refuses host scalar indexing" begin
    O = FG.Operators
    D = FG.Discretization
    GD = FG.Grids
    dev = CB.GPUBackend(NoScalarBackend())
    mv(x) = FG.Execution.on_backend(dev, x)
    same(d, h) = isequal(Array(d), h)

    # A masked plane, stretched along x, and a masked sphere with a stretched latitude axis. The masks are
    # `BitArray`s, which reach the device as `JLArray{Bool}`s.
    xs = collect(cumsum(1.0 .+ 0.3 .* sin.(1:10)))
    ys = collect(range(0.0, 2.1; length = 8))
    mk = trues(10, 8)
    mk[4, 3] = false
    mk[7, 6] = false
    gc = GD.StructuredGrid(FG.Geometry.CartesianGeometry{Float64}(), xs, ys, mk)
    φs = [-1.2 + 2.4 * (t / 8)^1.2 for t in 0:8]
    ms = trues(12, 9)
    ms[5, 4] = false
    gs = GD.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6), FG.Axes.PeriodicUniformAxis(0.0, 2π, 12), φs, ms)
    λs = collect(GD.coordinates(gs, 1))
    fc = [sin(x) * cos(2y) for x in xs, y in ys]
    fs = [sin(2l) * cos(p) for l in λs, p in φs]
    us = [cos(p) * sin(l) for l in λs, p in φs]
    vs = [0.3 * sin(2p) * cos(l) for l in λs, p in φs]

    task_local_storage(:ScalarIndexing, GPUArraysCore.ScalarDisallowed) do
        # The guard is armed: a host read of one element of a device array raises. A moved grid holds its
        # axes and its mask there.
        Test.@test_throws ErrorException JLArrays.JLArray([1.0, 2.0])[1]
        Test.@test GD.coordinates(mv(gc), 1) isa JLArrays.JLArray
        Test.@test GD.mask(mv(gc)) isa JLArrays.JLArray{Bool} && GD.mask(mv(gs)) isa JLArrays.JLArray{Bool}
        Test.@test same(GD.mask(mv(gc)), mk) && same(GD.mask(mv(gs)), ms)

        # `derivative!` from the axes, a held table, a held plan and a held masked plan.
        for (g, f, d) in ((gc, fc, 1), (gs, fs, 2))
            idx, w = D.axis_stencils(g, d; order = 1, nodes = 3)
            pl = D.stencil_plan(g, d; order = 1, nodes = 3)
            mp = O.masked_stencil_plan(g, d; order = 1, nodes = 3, policy = O.ReduceInRun())
            Test.@test mv(mp).mask isa JLArrays.JLArray{Bool}
            ref = (O.derivative!(fill(NaN, size(f)), f, g, d; order = 1, nodes = 3, masked = NaN),
                   O.derivative!(fill(NaN, size(f)), f, g, idx, w, d; order = 1, masked = NaN),
                   O.derivative!(fill(NaN, size(f)), f, g, pl, d; masked = NaN),
                   O.derivative!(fill(NaN, size(f)), f, g, mp; masked = NaN))
            for (gg, t, p, m) in ((g, (idx, w), pl, mp), (mv(g), mv((idx, w)), mv(pl), mv(mp)))
                fd = mv(f)
                Test.@test same(O.derivative!(mv(fill(NaN, size(f))), fd, gg, d; order = 1, nodes = 3,
                                              masked = NaN, execution = dev), ref[1])
                Test.@test same(O.derivative!(mv(fill(NaN, size(f))), fd, gg, t[1], t[2], d; order = 1,
                                              masked = NaN, execution = dev), ref[2])
                Test.@test same(O.derivative!(mv(fill(NaN, size(f))), fd, gg, p, d; masked = NaN,
                                              execution = dev), ref[3])
                Test.@test same(O.derivative!(mv(fill(NaN, size(f))), fd, gg, m; masked = NaN,
                                              execution = dev), ref[4])
            end
        end

        # The strain rate on the sphere, its derivatives from held plans and its frame terms per cell.
        plans = O.stencil_plans(gs; nodes = 3)
        Sref = O.strain_rate((us, vs), gs, plans)
        for (gg, pp) in ((gs, plans), (mv(gs), mv(plans)))
            Sd = O.strain_rate((mv(us), mv(vs)), gg, pp; execution = dev)
            Test.@test all(k -> same(Sd[k], Sref[k]), keys(Sref))
        end

        # A tangent field rotated into the ambient frame on the sphere.
        X, Y, Z = zeros(12, 9), zeros(12, 9), zeros(12, 9)
        O.to_planetary!(X, Y, Z, us, vs, nothing, gs)
        for gg in (gs, mv(gs))
            Xd, Yd, Zd = mv(zeros(12, 9)), mv(zeros(12, 9)), mv(zeros(12, 9))
            O.to_planetary!(Xd, Yd, Zd, mv(us), mv(vs), nothing, gg; execution = dev)
            Test.@test same(Xd, X) && same(Yd, Y) && same(Zd, Z)
        end
        # The same through the grid's frame plan, its tables moved to the device.
        fp = O.frame_plan(gs; execution = dev)
        Test.@test fp.sinλ isa JLArrays.JLArray && fp.mask isa JLArrays.JLArray{Bool}
        Xd, Yd, Zd = mv(zeros(12, 9)), mv(zeros(12, 9)), mv(zeros(12, 9))
        O.to_planetary!(Xd, Yd, Zd, mv(us), mv(vs), nothing, fp; execution = dev)
        Test.@test same(Xd, X) && same(Yd, Y) && same(Zd, Z)

        # The mimetic gradient and divergence, the assembled Laplacian and the restriction to the next
        # level on the masked plane, from metrics and coefficients built on the host.
        fm = O.face_metrics(gc, O.Dirichlet())
        c = O.laplacian_coefficients(fm)
        cg = O.coarsen(gc)
        cc = O.galerkin_coefficients(c, cg)
        χ = [sin(x + 2y) for x in xs, y in ys] .* mk
        gref = O.gradient!(O.allocate_faces(Float64, gc), χ, fm)
        δref = O.divergence!(zeros(10, 8), gref, fm)
        Lref = O.apply_laplacian!(zeros(10, 8), χ, c)
        rref = O.restrict!(zeros(GD.size_tuple(cg)), χ, c, cc)
        for (m, cf, cq) in ((fm, c, cc), (mv(fm), mv(c), mv(cc)))
            gd = O.gradient!(O.allocate_faces(Float64, gc; execution = dev), mv(χ), m; execution = dev)
            Test.@test all(map(same, gd, gref))
            Test.@test same(O.divergence!(mv(zeros(10, 8)), map(mv, gref), m; execution = dev), δref)
            Test.@test same(O.apply_laplacian!(mv(zeros(10, 8)), mv(χ), cf; execution = dev), Lref)
            Test.@test same(O.restrict!(mv(zeros(GD.size_tuple(cg))), mv(χ), cf, cq; execution = dev), rref)
        end
    end
end
