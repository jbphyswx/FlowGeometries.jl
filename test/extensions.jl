Test.@testset "SparseArrays sparse_adjacency_matrix (optional)" begin
    using SparseArrays: SparseArrays as Sp
    geom = FG.Geometry.CartesianGeometry()
    grid = FG.Grids.StructuredGrid(geom, 0.0:1.0:1.0, 0.0:1.0:1.0, trues(2, 2))
    conn = FG.Connectivity.build_connectivity(grid)
    ne = FG.Connectivity.nedges(conn)
    I = Vector{Int}(undef, ne)
    J = Vector{Int}(undef, ne)
    Test.@test FG.Connectivity.sparse_adjacency_coo!(I, J, conn) == ne
    S = FG.Connectivity.sparse_adjacency_matrix(conn)
    Test.@test S isa Sp.SparseMatrixCSC
    Test.@test size(S) == (4, 4)
    Test.@test Sp.nnz(S) == ne
    Test.@test S[1, 2] && S[2, 1]
    Test.@test Matrix(S) == FG.Connectivity.adjacency_matrix(conn)
end

Test.@testset "StaticArrays extension" begin
    using StaticArrays: StaticArrays as SA
    geom = FG.Geometry.SphericalGeometry(1.0)
    p1 = SA.SVector{2,Float64}(0.0, 0.0)
    p2 = SA.SVector{2,Float64}(0.1, 0.2)
    d = FG.Geometry.distance(geom, p1, p2)
    Test.@test d ≈ FG.Geometry.distance(geom, Tuple(p1), Tuple(p2))
    ê = FG.Geometry.local_tangent_basis(geom, p1)
    Test.@test ê.λ isa NTuple
    Test.@test Test.@inferred(FG.Geometry.as_ntuple(p2)) === (0.1, 0.2)
end

Test.@testset "Threading is opt-in and changes no result" begin
    using ComputationalBackends: ComputationalBackends as CB
    Test.@test Base.get_extension(FG, :FlowGeometriesComputationalBackendsExt) !== nothing

    # Chunking must partition exactly — no gaps, no overlap, contiguous, whatever the remainder.
    for (n, k) in ((10, 3), (10, 1), (3, 10), (1, 1), (100, 7))
        rs = FG.Execution.chunk_ranges(n, k)
        Test.@test reduce(vcat, collect.(rs)) == collect(1:n)
        Test.@test all(!isempty, rs) && length(rs) ≤ max(1, min(k, n))
    end
    # `nothing` hands the body the whole range in one call: the serial path adds no partitioning.
    seen = UnitRange{Int}[]
    FG.Execution.run_chunks(17, nothing) do r; push!(seen, r); end
    Test.@test seen == [1:17]

    # The empty-input contract, which the two shapes answer differently on purpose: a write loop has
    # nothing to write, and a reduction still has to produce a value.
    ran = Ref(0)
    FG.Execution.run_chunks(0, nothing) do r; ran[] += 1; end
    Test.@test ran[] == 0
    Test.@test FG.Execution.map_chunks(r -> length(r), 0, nothing) == [0]
    Test.@test FG.Execution.map_chunks(r -> length(r), 0, CB.ThreadedBackend()) == [0]

    # A threaded reduction collects into a CONCRETE vector: the partials are the reduction's own
    # values, and boxing each of them is paid on a path whose serial form allocates nothing.
    let out = FG.Execution.map_chunks(r -> sum(r), 1000, CB.ThreadedBackend())
        Test.@test isconcretetype(eltype(out))
        Test.@test sum(out) == sum(1:1000)
    end

    # `exclusive_scan!` is exact, so the threaded form has to equal the serial one entry for entry.
    for n in (0, 1, 7, 1000, 65_536)
        counts = rand(0:9, n)
        want = FG.Execution.exclusive_scan!(Vector{Int}(undef, n + 1), counts)
        for b in (CB.SerialBackend(), CB.ThreadedBackend())
            Test.@test FG.Execution.exclusive_scan!(Vector{Int}(undef, n + 1), counts, b) == want
        end
    end

    # …and so is an integer reduction, on every policy including the device one.
    for n in (0, 1, 13, 10_000)
        want = sum(i * i for i in 1:n; init = 0)
        for b in (nothing, CB.SerialBackend(), CB.ThreadedBackend(), KernelAbstractions.CPU())
            Test.@test FG.Execution.reduce_indices(i -> i * i, +, 0, n, b) == want
        end
    end

    # A CSR build goes through the scan, so it must stay bit-identical under threading.
    let sph = FG.Geometry.SphericalGeometry(), cart = FG.Geometry.CartesianGeometry()
        for gg in (FG.Grids.StructuredGrid(cart, 0.0:1.0:9.0, 0.0:1.0:9.0),
                   FG.Grids.HEALPixGrid(sph, 4), FG.Grids.CubedSphereGrid(sph, 4),
                   FG.Grids.IcosahedralGrid(sph, 3),
                   FG.Grids.RingGrid(sph, FG.SphericalSampling.OctahedralGaussianSampling(8)))
            a = FG.Connectivity.build_connectivity(gg)
            b = FG.Connectivity.build_connectivity(gg; backend = CB.ThreadedBackend())
            Test.@test a.ptr == b.ptr
            Test.@test a.nbrs == b.nbrs
        end
        hp = FG.SphericalSampling.HEALPixSampling(4)
        Test.@test FG.Connectivity.build_connectivity(hp).nbrs ==
                   FG.Connectivity.build_connectivity(hp; backend = CB.ThreadedBackend()).nbrs
    end

    # A batched sweep is ONE launch over the whole field, batch axes included — so it must agree with
    # differencing each slice on its own, not merely with the host.
    let cpu = KernelAbstractions.CPU(), nx = 12, ny = 8, nb = 3
        ax = range(0.0, 1.0; length = nx)
        fld = reshape(collect(Float64, 1:(nx * ny * nb)), nx, ny, nb) ./ (nx * ny * nb)
        for k in (3, 5, 8), dim in (1, 2)
            iw = FG.Discretization.axis_stencils(dim == 1 ? ax :
                                                 range(0.0, 2.0; length = ny), 1, k)
            dim == 2 && k > ny && continue
            batched = fill(NaN, nx, ny, nb)
            FG.Operators.apply_stencil!(batched, fld, iw..., dim; backend = cpu)
            for b in 1:nb
                slice = fill(NaN, nx, ny)
                FG.Operators.apply_stencil!(slice, fld[:, :, b], iw..., dim; backend = cpu)
                Test.@test isequal(batched[:, :, b], slice)
            end
            # …and the host agrees with it cell for cell.
            host = fill(NaN, nx, ny, nb)
            FG.Operators.apply_stencil!(host, fld, iw..., dim)
            Test.@test isequal(host, batched)
        end
    end

    # A node count above the specialized set still runs, on both paths, and still agrees.
    let cpu = KernelAbstractions.CPU(), n = 20
        ax = range(0.0, 1.0; length = n)
        fld = [sin(3a) * b for a in ax, b in 1:6]
        for k in (8, 10, 11, 13)
            iw = FG.Discretization.axis_stencils(ax, 1, k)
            h = fill(NaN, n, 6)
            d = fill(NaN, n, 6)
            FG.Operators.apply_stencil!(h, fld, iw..., 1)
            FG.Operators.apply_stencil!(d, fld, iw..., 1; backend = cpu)
            Test.@test isequal(h, d)
        end
    end

    # A grid reduction runs per index where the candidates need no buffer, so it reaches a device too.
    let gs = FG.Grids.StructuredGrid(FG.Geometry.CartesianGeometry(), 0.0:1.0:19.0, 0.0:1.0:19.0)
        base = FG.Connectivity.mapreduce_within((I, J, d) -> 1, +, 0, gs; ball = 2.5)
        Test.@test base > 0
        for b in (CB.SerialBackend(), CB.ThreadedBackend(), KernelAbstractions.CPU())
            Test.@test FG.Connectivity.mapreduce_within((I, J, d) -> 1, +, 0, gs;
                                                        ball = 2.5, backend = b) == base
        end
    end

    # Every threaded kernel must be bit-identical to serial, not merely close.
    geo = FG.Geometry.SphericalGeometry()
    thr = CB.ThreadedBackend()
    for n in (7, 64)
        a = FG.SphericalSampling.cubed_sphere_points(n)
        b = FG.SphericalSampling.cubed_sphere_points(n; backend = thr)
        c = FG.SphericalSampling.cubed_sphere_points(n; backend = CB.SerialBackend())
        Test.@test a.λ == b.λ == c.λ
        Test.@test a.φ == b.φ == c.φ
        Test.@test a.panel == b.panel == c.panel
        sp = FG.SphericalSampling.spherical_points(FG.SphericalSampling.CubedSphereSampling(), n; backend = thr)
        Test.@test sp.λ == a.λ && sp.φ == a.φ
    end
    for (g, lbl) in ((geo, "spherical"), (FG.Geometry.CartesianGeometry(), "cartesian"))
        for n in (3, 40)   # n = 3 gives fewer rows than threads, exercising the short case
            λ = [2π * (i - 1) / n for i in 1:n, j in 1:n]
            φ = [asin(2 * (j - 0.5) / n - 1) for i in 1:n, j in 1:n]
            m = fill(true, n, n)
            Test.@test FG.Grids.measure(FG.Grids.CurvilinearGrid(g, λ, φ, m)) ==
                       FG.Grids.measure(FG.Grids.CurvilinearGrid(g, λ, φ, m; backend = thr))
        end
    end

    # Connectivity: both cell passes write only slots their own cell owns. Masked and periodic
    # cases matter most — they make the per-cell degree vary, so a chunk boundary landing
    # mid-row would show up as a wrong offset rather than a wrong count.
    for topo in (FG.Connectivity.IndexTopology((37, 21), (true, false), nothing),
                 FG.Connectivity.IndexTopology((37, 21), (true, true), nothing),
                 FG.Connectivity.IndexTopology((5, 4), (false, false), nothing),
                 FG.Connectivity.IndexTopology((31, 29), (true, false),
                                  [isodd(i * 7 + j * 3) for i in 1:31, j in 1:29]))
        for st in (FG.Stencils.Axial(1), FG.Stencils.Moore(1)), ao in (true, false)
            a = FG.Connectivity.build_connectivity(topo; stencil = st, active_only = ao)
            b = FG.Connectivity.build_connectivity(topo; stencil = st, active_only = ao, backend = thr)
            Test.@test a.ptr == b.ptr
            Test.@test a.nbrs == b.nbrs
        end
    end
    g = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 33)
    Test.@test FG.Connectivity.build_connectivity(g).nbrs ==
               FG.Connectivity.build_connectivity(g; backend = thr).nbrs

    # The metric-ball builder chunks the same two owned-slot passes, and per-cell degrees vary
    # even more (polar rows reach every longitude), so the same chunk-boundary argument applies.
    gball = FG.Grids.StructuredGrid(FG.Geometry.SphericalGeometry(6.371e6),
                                    range(0, 2π; length = 25)[1:24],
                                    range(-π / 2, π / 2; length = 13))
    a = FG.Connectivity.build_connectivity_within(gball; ball = 2.0e6)
    b = FG.Connectivity.build_connectivity_within(gball; ball = 2.0e6, backend = thr)
    Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs

    # The candidate builder emits and dedups concurrently, so each sampling's `emit!` has to be
    # free of state shared between nodes. nside = 1 and 2 cover the singular pixels, where a node
    # has 7 neighbours rather than 8.
    for s in (FG.SphericalSampling.HEALPixSampling(1), FG.SphericalSampling.HEALPixSampling(2), FG.SphericalSampling.HEALPixSampling(8))
        a = FG.Connectivity.build_connectivity(s)
        b = FG.Connectivity.build_connectivity(s; backend = thr)
        Test.@test a.ptr == b.ptr
        Test.@test a.nbrs == b.nbrs
    end
    for n in (1, 2, 9), st in (FG.Stencils.Axial(1), FG.Stencils.Moore(1))
        a = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), n; stencil = st)
        b = FG.Connectivity.build_connectivity(FG.SphericalSampling.CubedSphereSampling(), n; stencil = st, backend = thr)
        Test.@test a.ptr == b.ptr && a.nbrs == b.nbrs
    end
    for (nlon, nlat) in ((1, 1), (7, 5)), st in (FG.Stencils.Axial(1), FG.Stencils.Moore(1))
        a = FG.Connectivity.build_connectivity(FG.SphericalSampling.YinYangSampling(), nlon, nlat; stencil = st)
        b = FG.Connectivity.build_connectivity(FG.SphericalSampling.YinYangSampling(), nlon, nlat; stencil = st, backend = thr)
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
    g = FG.Connectivity.structured_grid(FG.SphericalSampling.ClenshawCurtisSampling(), 9)
    d = adapt(FakeDev(), g)
    Test.@test FG.Grids.coordinates(d, 1) isa DevArr && FG.Grids.coordinates(d, 2) isa DevArr
    # A separable measure must adapt its FACTORS — materializing the outer product onto a device
    # is exactly what the factored form exists to avoid.
    Test.@test FG.Grids.measure(d) isa FG.Grids.SeparableMeasure
    Test.@test all(f -> f isa DevArr, FG.Grids.measure_factors(d))
    Test.@test all(FG.Grids.measure(d)[i, j] == FG.Grids.measure(g)[i, j]
                   for i in 1:size(g, 1), j in 1:size(g, 2))
    Test.@test FG.Grids.mask(d) isa FG.Grids.AllActive           # size only; nothing to move
    Test.@test size(d) == size(g) && FG.Grids.grid_geometry(d) === FG.Grids.grid_geometry(g)
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

    t = FG.Connectivity.IndexTopology((4, 3), (true, false), nothing)
    Test.@test adapt(FakeDev(), t) === t              # no mask, nothing to move
    t2 = FG.Connectivity.IndexTopology((4, 3), (true, false), fill(true, 4, 3))
    Test.@test adapt(FakeDev(), t2).mask isa DevArr
    Test.@test adapt(FakeDev(), FG.Grids.AllActive((5, 5))) isa FG.Grids.AllActive
end

Test.@testset "Equiangular weights: FFT path and recurrence fallback agree with the definition" begin
    using FFTW: FFTW          # loads AbstractFFTs, which fires the extension
    Test.@test Base.get_extension(FG, :FlowGeometriesAbstractFFTsExt) !== nothing

    # Literal evaluation of the defining sum — no recurrence, no transform.
    function direct(family, nlat)
        nterm = (nlat + 1) ÷ 2
        θ(i) = family === :open ? π * (i - 0.5) / nlat : π * (i - 1) / nlat
        return [(4 / nlat) * sin(θ(i)) * sum(sin((2k + 1) * θ(i)) / (2k + 1) for k in 0:(nterm - 1))
                for i in 1:nlat]
    end
    for nlat in (8, 64, 512)
        dh = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.DriscollHealySampling(), nlat)
        cc = FG.SphericalSampling.latitude_weights(FG.SphericalSampling.ClenshawCurtisSampling(), nlat)
        Test.@test maximum(abs.(dh .- direct(:closed, nlat))) < 1e-13
        Test.@test maximum(abs.(cc .- direct(:open, nlat))) < 1e-13
        Test.@test sum(dh) ≈ 2 atol = 1e-13
        Test.@test sum(cc) ≈ 2 atol = 1e-13
    end
    SS = FG.SphericalSampling
    # The transform is the extension's contribution: `src` holds `Recurrence`, the extension holds
    # `Transform`, and the resolution point picks between them by element type.
    Test.@test parentmodule(which(SS._equiangular_sums!,
                                  Tuple{Vector{Float64},SS.ClosedNodes,Int,Int,SS.Transform})) ===
               Base.get_extension(FG, :FlowGeometriesAbstractFFTsExt)
    Test.@test parentmodule(which(SS._equiangular_sums!,
                                  Tuple{Vector{Float64},SS.ClosedNodes,Int,Int,SS.Recurrence})) ===
               SS
    Test.@test SS._equiangular_algorithm(Float64) === SS.Transform()   # a plannable type defaults to it
    Test.@test SS._equiangular_algorithm(BigFloat) === SS.Recurrence() # nothing plans BigFloat

    # Both algorithms are reachable by asking for one, so that correctness does not depend on which
    # extensions happen to be loaded is checked here rather than from a second process.
    #
    # The WEIGHTS are compared, not the sine sums they are built from: the sums are an intermediate
    # whose magnitude is O(1) while the weights carry a `4/nlat·sinθ` factor, and the recurrence
    # accumulates round-off across its `nterm` steps, so the two constructions agree on the returned
    # quantity to a tolerance the intermediate does not meet.
    for (s, fam) in ((SS.ClenshawCurtisSampling(), :open), (SS.DriscollHealySampling(), :closed)),
        nlat in (8, 64, 512)
        wt = SS.latitude_weights(s, nlat; algorithm = SS.Transform())
        wr = SS.latitude_weights(s, nlat; algorithm = SS.Recurrence())
        Test.@test maximum(abs.(wt .- wr)) < 1e-14
        # Each construction independently reproduces the literal defining sum.
        Test.@test maximum(abs.(wt .- direct(fam, nlat))) < 1e-13
        Test.@test maximum(abs.(wr .- direct(fam, nlat))) < 1e-13
        # The default here is the transform, so it must be what an unqualified call runs.
        Test.@test SS.latitude_weights(s, nlat) == wt
    end
    # Gauss–Legendre is not an equiangular family, so asking it for one of these says so.
    Test.@test_throws ArgumentError SS.latitude_weights(
        SS.GaussLegendreSampling(), 8; algorithm = SS.Recurrence())
end

Test.@testset "Sparse adjacency assembles straight into CSC" begin
    using SparseArrays: SparseArrays
    conn = FG.Connectivity.build_connectivity(FG.SphericalSampling.HEALPixSampling(8))
    n, ne = FG.Connectivity.nnodes(conn), FG.Connectivity.nedges(conn)
    A = FG.Connectivity.sparse_adjacency_matrix(conn)
    Test.@test size(A) == (n, n)
    Test.@test SparseArrays.nnz(A) == ne
    # Same matrix the dense builder produces.
    Test.@test Matrix(A) == FG.Connectivity.adjacency_matrix(conn)
    # Row indices ascending within each column, as CSC requires — obtained without a sort.
    Test.@test all(issorted(@view SparseArrays.rowvals(A)[SparseArrays.nzrange(A, j)]) for j in 1:n)

    # A caller-supplied index type, and caller-owned buffers reused with no allocation of the
    # matrix's own storage.
    Test.@test eltype(FG.Connectivity.sparse_adjacency_matrix(conn; Ti = Int32).colptr) === Int32
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
end

Test.@testset "Index-parallel loops run as kernels and give the same answer" begin
    D = FG.Discretization
    O = FG.Operators
    C = FG.Connectivity
    GD = FG.Grids
    cpu = KernelAbstractions.CPU()

    n, m = 48, 33
    x = collect(range(0.0, 2π; length = n))
    y = collect(range(0.0, 1.0; length = m))
    f = [sin(xi) * cos(yj) for xi in x, yj in y]
    ref = similar(f); dev = similar(f)
    for (ax, dim, ord) in ((x, 1, 1), (y, 2, 2))
        O.apply_stencil!(ref, f, ax, dim; order = ord, nodes = 5)
        O.apply_stencil!(dev, f, ax, dim; order = ord, nodes = 5, backend = cpu)
        Test.@test ref == dev
    end
    msk = trues(n, m); msk[10:14, :] .= false
    O.apply_stencil!(ref, f, x, 1; order = 1, nodes = 5, mask = msk)
    O.apply_stencil!(dev, f, x, 1; order = 1, nodes = 5, mask = msk, backend = cpu)
    Test.@test ref == dev

    cart = FG.Geometry.CartesianGeometry{Float64}()
    for g in (GD.StructuredGrid(cart, collect(0.0:31.0), collect(0.0:31.0)),
              GD.StructuredGrid(cart, collect(0.0:31.0), collect(0.0:31.0);
                                periodic = (true, false), period = (32.0, 0.0)))
        for sten in (FG.Stencils.Axial(1), FG.Stencils.Moore(2))
            a = C.build_connectivity(g; stencil = sten)
            b = C.build_connectivity(g; stencil = sten, backend = cpu)
            Test.@test a.ptr == b.ptr
            Test.@test a.nbrs == b.nbrs
        end
    end

    acc = zeros(Int, 100)
    FG.Execution.run_indices(i -> (acc[i] = i * i), 100, cpu)
    Test.@test acc == [i * i for i in 1:100]

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
    Test.@test counts(gball, 3.0, cpu) == counts(gball, 3.0, nothing)
    Test.@test counts(gball, 3.0, cpu) ==
               [C.nneighbors_within(gball, Tuple(ci)...; ball = 3.0)
                for ci in CartesianIndices(size(GD.mask(gball)))]
    Test.@test isbits(C.MetricTopology(gball))

    # The sweep follows the same rule: unindexed it is one body per cell, so it launches.
    function sweep_counts(g, r, backend)
        out = zeros(Int, GD.size_tuple(g))
        C.foreach_within(g; ball = r, topology = C.MetricTopology(g), backend = backend) do I, J, d
            @inbounds out[I[1], I[2]] += 1
        end
        return out
    end
    Test.@test sweep_counts(gball, 3.0, cpu) == sweep_counts(gball, 3.0, nothing)
    Test.@test sum(sweep_counts(gball, 3.0, cpu)) ==
               C.mapreduce_within((I, J, d) -> 1, +, 0, gball; ball = 3.0)

    # A chunked body accumulates across its range, so a device backend refuses it rather than
    # quietly running on the host.
    Test.@test_throws ArgumentError FG.Execution.run_chunks(r -> nothing, 4, cpu)
    Test.@test_throws ArgumentError FG.Execution.map_chunks(r -> 1, 4, cpu)

    # An unindexed topology is device-safe; one holding a k-d tree is refused, not silently dropped.
    gs = GD.StructuredGrid(cart, collect(0.0:7.0), collect(0.0:7.0))
    Test.@test Adapt.adapt(Array, C.MetricTopology(gs)) === C.MetricTopology(gs)
    gu = healpix_node_grid(2)
    Test.@test_throws ArgumentError Adapt.adapt(Array, C.indexed(gu))
    Test.@test !isbits(C.indexed(gu))

    # An indexed sweep does stay on the host: each task needs its own candidate buffer, which a
    # launch has nowhere to put.
    Test.@test_throws ArgumentError C.foreach_within(
        (I, J, d) -> nothing, gu; ball = 2.0e6, topology = C.indexed(gu), backend = cpu,
    )
end
