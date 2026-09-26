module Execution

# Where this package decides *how* a bulk loop runs. Every bulk entry point takes a ComputationalBackends
# tag, `SerialBackend()` by default. Serial and threaded loops run here; a `GPUBackend` wrapping a
# KernelAbstractions backend launches through the KernelAbstractions extension; a `DistributedBackend` or
# an `MPIBackend` runs a share of the indices on each process through the Distributed or MPI extension,
# and gathers what the loop declares it writes.
#
# Two shapes, because bulk loops here come in two: `run_chunks` hands a contiguous range to a body that
# writes across it, and `run_indices` applies a body to one index at a time. Only the second maps to a
# kernel — a chunk body carries loop-carried state a device launch cannot express.

using ComputationalBackends: ComputationalBackends as CB

# The backends whose loops run in this process, over ordinary memory.
const _HostBackend = Union{CB.AbstractSerialBackend, CB.AbstractThreadedBackend}
const _ProcessBackend = Union{CB.AbstractDistributedBackend, CB.AbstractMPIBackend}

"""
    resolve(backend) -> backend

The backend a loop runs under: `backend` itself, and for an `AutoBackend` FlowGeometries' own policy —
`ThreadedBackend()` when Julia runs more than one thread, else `SerialBackend()`.
"""
resolve(b::CB.AbstractExecutionBackend) = b
resolve(::CB.AbstractAutoBackend) = Threads.nthreads() > 1 ? CB.ThreadedBackend() : CB.SerialBackend()

"""
    local_backend(backend) -> backend

The backend a process runs its own loops under: the inner backend of a `DistributedBackend` or an
`MPIBackend`, and [`resolve`](@ref)`(backend)` otherwise. A pass over data every process already holds
whole — a scan of gathered counts, a sum of them — runs under this, with nothing to exchange.
"""
local_backend(b::CB.AbstractExecutionBackend) = resolve(b)
local_backend(b::_ProcessBackend) = CB.local_backend(b)

@noinline _unsupported(b::CB.AbstractGPUBackend) = throw(ArgumentError(
    "$(nameof(typeof(b))) launches through KernelAbstractions: load it (`using KernelAbstractions`) " *
    "and pass `GPUBackend(backend)` with a KernelAbstractions backend."))
@noinline _unsupported(b::CB.AbstractDistributedBackend) = throw(ArgumentError(
    "$(nameof(typeof(b))) runs through the Distributed extension: `using Distributed`, and load " *
    "FlowGeometries on every worker (`Distributed.@everywhere using FlowGeometries`)."))
@noinline _unsupported(b::CB.AbstractMPIBackend) = throw(ArgumentError(
    "$(nameof(typeof(b))) runs through the MPI extension: `using MPI`, and call `MPI.Init()`."))
@noinline _unsupported(b::CB.AbstractExecutionBackend) = throw(ArgumentError(
    "no FlowGeometries loop is defined for $(nameof(typeof(b)))."))

# ---------------------------------------------------------------------------
# What a loop writes
# ---------------------------------------------------------------------------

"""
    Written(array[, span = ByIndex()])

An array a bulk loop writes, declared alongside the loop: the indices `rng` write the linear positions
`span(rng)` of `array`, one contiguous range, and consecutive index ranges write consecutive ranges of
positions. [`ByIndex`](@ref), [`ByBlock`](@ref) and [`ByOffsets`](@ref) are the spans.

Within one process a loop writes `array` in place, and the declaration changes nothing. Under a
`DistributedBackend` or an `MPIBackend` each process runs a share of the indices against its own copy of
what the loop captures, and the declared spans are gathered into `array`: on the calling process for
Distributed, on every rank for MPI. A write the loop does not declare stays on the process that made it.
"""
struct Written{A<:AbstractArray,S}
    array::A
    span::S
end
Written(array::AbstractArray) = Written(array, ByIndex())

"""
    ByIndex()

Index `k` writes linear position `k`.
"""
struct ByIndex end
@inline (::ByIndex)(rng::UnitRange{Int}) = rng

"""
    ByBlock(m)

Index `k` writes the `m` positions `(k-1)m+1 : km`: a column of `m` rows per index, in column-major
storage.
"""
struct ByBlock
    m::Int
end
@inline (s::ByBlock)(rng::UnitRange{Int}) = ((first(rng) - 1) * s.m + 1):(last(rng) * s.m)

"""
    ByOffsets(ptr)

Index `k` writes positions `ptr[k] : ptr[k+1]-1`: row `k` of a CSR array with offsets `ptr`.
"""
struct ByOffsets{P<:AbstractVector{<:Integer}}
    ptr::P
end
@inline function (s::ByOffsets)(rng::UnitRange{Int})
    lo = Int(@inbounds s.ptr[first(rng)])
    return lo:(isempty(rng) ? lo - 1 : Int(@inbounds s.ptr[last(rng) + 1]) - 1)
end

# A span as the host evaluates it: offsets held in device memory are copied to the host once.
_host_span(s) = s
_host_span(s::ByOffsets) = s.ptr isa Array ? s : ByOffsets(Array(s.ptr))

# ---------------------------------------------------------------------------
# The loops
# ---------------------------------------------------------------------------

"""
    chunk_ranges(n, k) -> Vector{UnitRange{Int}}

Partition `1:n` into at most `k` contiguous ranges of near-equal length. Contiguity matters: each chunk
then touches one span of every array the loop indexes.
"""
function chunk_ranges(n::Integer, k::Integer)
    n = Int(n); k = max(1, min(Int(k), max(n, 1)))
    base, rem = divrem(n, k)
    ranges = Vector{UnitRange{Int}}(undef, k)
    lo = 1
    @inbounds for c in 1:k
        len = base + (c ≤ rem ? 1 : 0)
        ranges[c] = lo:(lo + len - 1)
        lo += len
    end
    return ranges
end

"""
    run_chunks(f, n, backend, outputs::Written...)

Apply `f(range)` over a partition of `1:n`, under the execution policy `backend` names. `outputs` are the
arrays `f` writes; see [`Written`](@ref).

`f` must be safe to run on disjoint index ranges concurrently — every write it makes has to be
determined by the index, never accumulated across chunks. A serial backend hands `f` the whole range in
one call, so the serial path adds no partitioning at all; a threaded one runs one chunk per thread.
"""
run_chunks(f::F, n::Integer, ::CB.AbstractSerialBackend, ::Written...) where {F} =
    (n > 0 && f(1:Int(n)); nothing)

# One chunk per thread: the loops this drives are short-bodied, so per-index scheduling costs more than
# the body it schedules.
function run_chunks(f::F, n::Integer, ::CB.AbstractThreadedBackend, ::Written...) where {F}
    n = Int(n)
    n > 0 || return nothing
    nt = Threads.nthreads()
    nt == 1 && return run_chunks(f, n, CB.SerialBackend())
    ranges = chunk_ranges(n, nt)
    Threads.@threads for c in eachindex(ranges)
        @inbounds f(ranges[c])
    end
    return nothing
end

# A chunked body is a host loop by construction — it carries state across the indices in its range — so a
# device backend gets the one-index form or an error, never a silent host fallback.
@noinline run_chunks(f, n::Integer, b::CB.AbstractGPUBackend, ::Written...) = throw(ArgumentError(
    "a chunked loop cannot be launched on $(nameof(typeof(b))): its body accumulates across a range. " *
    "Use the index-parallel entry point, or run this operation on the host."))

run_chunks(f::F, n::Integer, b::CB.AbstractAutoBackend, outs::Written...) where {F} =
    run_chunks(f, n, resolve(b), outs...)
run_chunks(f, n::Integer, b::CB.AbstractExecutionBackend, ::Written...) = _unsupported(b)

"""
    map_chunks(f, n, backend) -> Vector

Apply `f(range)` over a partition of `1:n` and collect one result per chunk, for a bulk operation that
reduces. The caller combines them, so `f` needs no lock and the combining order is the caller's to fix,
which keeps a parallel reduction independent of scheduling for an operation that is associative but not
commutative.

`f` is called at least once, on an empty range when `n == 0`, since a reduction has to produce a value;
[`run_chunks`](@ref) has nothing to write there and does not call `f`. Being callable on an empty range
is part of the contract, and it is how the result's element type is known before the chunks run.
"""
map_chunks(f::F, n::Integer, ::CB.AbstractSerialBackend) where {F} = [f(1:Int(n))]

# Results stay in chunk order, so the caller's combine sees the sequence the serial path produces. The
# output vector is typed from `f` on an empty range, which the contract already requires it to accept.
function map_chunks(f::F, n::Integer, ::CB.AbstractThreadedBackend) where {F}
    n = Int(n)
    n > 0 || return map_chunks(f, 0, CB.SerialBackend())
    nt = Threads.nthreads()
    nt == 1 && return map_chunks(f, n, CB.SerialBackend())
    ranges = chunk_ranges(n, nt)
    out = Vector{typeof(f(1:0))}(undef, length(ranges))
    Threads.@threads for c in eachindex(ranges)
        @inbounds out[c] = f(ranges[c])
    end
    return out
end

@noinline map_chunks(f, n::Integer, b::CB.AbstractGPUBackend) = throw(ArgumentError(
    "a chunk-reducing loop cannot be launched on $(nameof(typeof(b))); reduce with `reduce_indices`, " *
    "or run this operation on a host backend."))

map_chunks(f::F, n::Integer, b::CB.AbstractAutoBackend) where {F} = map_chunks(f, n, resolve(b))
map_chunks(f, n::Integer, b::CB.AbstractExecutionBackend) = _unsupported(b)

"""
    _reduce_chunks(f, op, n, backend)

`f(range)` over a partition of `1:n`, combined left to right with `op`.

Distinct from [`map_chunks`](@ref) because the serial path has nothing to collect: it hands `f` the whole
range and returns that value directly, so a serial reduction allocates nothing.
"""
_reduce_chunks(f::F, ::O, n::Integer, ::CB.AbstractSerialBackend) where {F,O} = f(1:Int(n))

_reduce_chunks(f::F, op::O, n::Integer, backend::CB.AbstractExecutionBackend) where {F,O} =
    reduce(op, map_chunks(f, n, backend))

"""
    run_indices(f, n, backend, outputs::Written...)

Apply `f(i)` for each `i in 1:n`, under the execution policy `backend` names. `outputs` are the arrays
`f` writes; see [`Written`](@ref). `f` must write only what `i` determines, with no accumulation across
indices — the same contract as [`run_chunks`](@ref), stated per index, the form a device launch
expresses.

With `KernelAbstractions` loaded and a `GPUBackend`, `f` becomes the body of a launch over `1:n`.
"""
function run_indices(f::F, n::Integer, ::CB.AbstractSerialBackend, ::Written...) where {F}
    @inbounds for i in 1:Int(n)
        f(i)
    end
    return nothing
end

# Per-index semantics, chunked granularity: these bodies are short enough that per-index scheduling
# dominates the work.
function run_indices(f::F, n::Integer, backend::CB.AbstractThreadedBackend, ::Written...) where {F}
    return run_chunks(n, backend) do rng
        @inbounds for i in rng
            f(i)
        end
    end
end

run_indices(f::F, n::Integer, b::CB.AbstractAutoBackend, outs::Written...) where {F} =
    run_indices(f, n, resolve(b), outs...)
run_indices(f, n::Integer, b::CB.AbstractExecutionBackend, ::Written...) = _unsupported(b)

"""
    reduce_indices(f, op, init, n, backend)

Reduce `f(i)` over `i in 1:n` with `op`, under the execution policy `backend` names.

The per-index counterpart of [`_reduce_chunks`](@ref), and the reduction a device can run: `f` reads
index `i` and nothing else, so the work splits without a chunk body. Every integral, norm and count over
a grid goes through this.

`op` must be associative and `init` its identity. `init` seeds each partial as well as the whole, so
the partials combine in any grouping.

The grouping follows the partition, never the scheduling, so a parallel result is deterministic. It
matches the serial left fold wherever `op` is exactly associative; floating-point `+` is not, so a
parallel sum can differ from the serial one in its last bits.
"""
function reduce_indices(f::F, op::O, init, n::Integer, ::CB.AbstractSerialBackend) where {F,O}
    acc = init
    @inbounds for i in 1:Int(n)
        acc = op(acc, f(i))
    end
    return acc
end

function reduce_indices(f::F, op::O, init, n::Integer, backend::CB.AbstractThreadedBackend) where {F,O}
    n = Int(n)
    n > 0 && return _reduce_chunks(op, n, backend) do rng
        acc = init
        @inbounds for i in rng
            acc = op(acc, f(i))
        end
        return acc
    end
    return init
end

reduce_indices(f::F, op::O, init, n::Integer, b::CB.AbstractAutoBackend) where {F,O} =
    reduce_indices(f, op, init, n, resolve(b))
reduce_indices(f, op, init, n::Integer, b::CB.AbstractExecutionBackend) = _unsupported(b)

"""
    allocate(backend, T, dims...) -> AbstractArray{T}

An uninitialised array of size `dims` the loops running under `backend` can write: an ordinary `Array`
on a host backend and on each process of a distributed one, device memory on a `GPUBackend`.

A CSR build allocates its degree, offset and neighbour arrays through this, so the buffers land where
the passes that fill them run.
"""
allocate(::_HostBackend, ::Type{T}, dims::Integer...) where {T} = Array{T}(undef, Int.(dims)...)
allocate(b::Union{CB.AbstractAutoBackend,_ProcessBackend}, ::Type{T}, dims::Integer...) where {T} =
    allocate(local_backend(b), T, dims...)
allocate(b::CB.AbstractExecutionBackend, ::Type, ::Integer...) = _unsupported(b)

"""
    on_backend(backend, x) -> x, or x in `backend`'s memory

`x` — an array, or a value holding arrays — where `backend`'s loops read it. `x` itself where they read
host memory. On a `GPUBackend` wrapping a KernelAbstractions backend, `Adapt.adapt(backend.backend, x)`:
the array's own package decides what that backend's memory is, and a struct moves by its `Adapt` rules.
"""
on_backend(::_HostBackend, x) = x
on_backend(b::Union{CB.AbstractAutoBackend,_ProcessBackend}, x) = on_backend(local_backend(b), x)
on_backend(b::CB.AbstractExecutionBackend, _) = _unsupported(b)

"""
    exclusive_scan!(out, counts, backend = SerialBackend(); init = 1) -> out

Write the exclusive prefix sums of `counts` into `out`, which is one element longer: `out[1] = init` and
`out[i+1] = out[i] + counts[i]`.

This is a CSR offset array — `out[k]` is where row `k` starts and `out[end] - init` is the total — and
every connectivity builder here calls it between its counting pass and its filling pass. Under a
threaded `backend` it is two passes over `counts` plus a serial scan of one sum per chunk. Under a
distributed one `counts` is whole on every process that holds it, the counting pass having gathered it,
so each process scans it under its [`local_backend`](@ref).
"""
function exclusive_scan!(
    out::AbstractVector, counts::AbstractVector,
    ::CB.AbstractSerialBackend = CB.SerialBackend(); init::Integer = 1,
)
    length(out) == length(counts) + 1 || throw(DimensionMismatch(
        "out must be one longer than counts: got $(length(out)) and $(length(counts))",
    ))
    acc = convert(eltype(out), init)
    @inbounds out[1] = acc
    @inbounds for i in eachindex(counts)
        acc += counts[i]
        out[i + 1] = acc
    end
    return out
end

# Two passes over `counts` with a serial scan of one sum per chunk between them. Each chunk then writes
# only its own span of `out`, from a base the middle pass fixed.
function exclusive_scan!(
    out::AbstractVector, counts::AbstractVector, ::CB.AbstractThreadedBackend; init::Integer = 1,
)
    n = length(counts)
    length(out) == n + 1 || throw(DimensionMismatch(
        "out must be one longer than counts: got $(length(out)) and $n",
    ))
    T = eltype(out)
    nt = Threads.nthreads()
    (n == 0 || nt == 1) && return exclusive_scan!(out, counts, CB.SerialBackend(); init = init)
    ranges = chunk_ranges(n, nt)
    nc = length(ranges)
    sums = Vector{T}(undef, nc)
    Threads.@threads for c in 1:nc
        s = zero(T)
        @inbounds for i in ranges[c]
            s += counts[i]
        end
        @inbounds sums[c] = s
    end
    base = Vector{T}(undef, nc)
    acc = convert(T, init)
    @inbounds for c in 1:nc
        base[c] = acc
        acc += sums[c]
    end
    @inbounds out[n + 1] = acc
    Threads.@threads for c in 1:nc
        @inbounds a = base[c]
        @inbounds for i in ranges[c]
            out[i] = a
            a += counts[i]
        end
    end
    return out
end

exclusive_scan!(out::AbstractVector, counts::AbstractVector,
                b::Union{CB.AbstractAutoBackend,_ProcessBackend}; init::Integer = 1) =
    exclusive_scan!(out, counts, local_backend(b); init = init)
exclusive_scan!(::AbstractVector, ::AbstractVector, b::CB.AbstractExecutionBackend; init::Integer = 1) =
    _unsupported(b)

end # module Execution
