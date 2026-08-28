# ---------------------------------------------------------------------------
# Adjacency that is arithmetic
# ---------------------------------------------------------------------------
#
# A layout with `Grids.FormulaNeighbors` supplies `Grids.formula_neighbors(grid, cell)`, returning a
# fixed-width tuple and a count. Everything below is written against that, so a layout adds its
# arithmetic and gets the whole neighbour API — the buffer form, the count, and the lazy sequence.

"""
    FormulaNeighborSeq{GR,K}

Lazy neighbour sequence of one cell of a layout whose adjacency is arithmetic.

Holds the tuple `Grids.formula_neighbors` returned, so iterating it touches no heap: the counterpart of
[`StencilNeighbors`](@ref) and [`MeshNeighbors`](@ref) for the third kind of adjacency.
"""
struct FormulaNeighborSeq{GR,K}
    grid::GR
    ids::NTuple{K,Int}
    n::Int
    active_only::Bool
end

Base.IteratorSize(::Type{<:FormulaNeighborSeq}) = Base.HasLength()
Base.IteratorEltype(::Type{<:FormulaNeighborSeq}) = Base.HasEltype()
Base.eltype(::Type{<:FormulaNeighborSeq}) = Int

function Base.length(s::FormulaNeighborSeq)
    s.active_only || return s.n
    m = 0
    @inbounds for t in 1:(s.n)
        Grids.isactive(s.grid, s.ids[t]) && (m += 1)
    end
    return m
end

@inline function Base.iterate(s::FormulaNeighborSeq, t::Int = 0)
    @inbounds while t < s.n
        t += 1
        j = s.ids[t]
        s.active_only && !Grids.isactive(s.grid, j) && continue
        return j, t
    end
    return nothing
end

# A masked cell has no neighbours at all, which is the rule the offset walk and the stored graph keep.
@inline function _formula_ids(grid, i::Int, active_only::Bool)
    (active_only && !Grids.isactive(grid, i)) &&
        return (ntuple(_ -> 0, Val(Grids.max_neighbors(grid))), 0)
    return Grids.formula_neighbors(grid, i)
end

@inline function _nneighbors(grid, i::Int, _sten, active_only::Bool, ::Grids.FormulaNeighbors)
    ids, n = _formula_ids(grid, i, active_only)
    active_only || return n
    m = 0
    @inbounds for t in 1:n
        Grids.isactive(grid, ids[t]) && (m += 1)
    end
    return m
end

@inline function _neighbors(grid, i::Int, _sten, active_only::Bool, ::Grids.FormulaNeighbors)
    ids, n = _formula_ids(grid, i, active_only)
    return FormulaNeighborSeq(grid, ids, n, active_only)
end

function _neighbors!(
    out::AbstractVector{<:Integer}, grid, i::Int, _sten, active_only::Bool,
    ::Grids.FormulaNeighbors,
)
    ids, n = _formula_ids(grid, i, active_only)
    m = 0
    @inbounds for t in 1:n
        j = ids[t]
        active_only && !Grids.isactive(grid, j) && continue
        m += 1
        m ≤ length(out) || throw(ArgumentError("out too short (need ≥ $m)"))
        out[m] = j
    end
    return m
end

function build_connectivity end

"""
    build_connectivity(grid; active_only = true) -> CSRConnectivity

The adjacency of a layout whose neighbours are arithmetic, materialized.

Count, scan, fill — the same three passes the index-stencil builder makes, and for the same reason: both
cell passes write only slots their own cell owns, so they carry no running offset and need no
coordination. The adjacency is a per-cell formula, so the whole shape it takes is that formula's.
"""
function build_connectivity(
    grid::Grids.AbstractGrid, ::Grids.FormulaNeighbors; active_only::Bool = true, backend = nothing,
)
    n = length(Grids.mask(grid))
    deg = zeros(Int, n)
    Execution.run_indices(n, backend) do k
        @inbounds deg[k] = _nneighbors(grid, k, nothing, active_only, Grids.FormulaNeighbors())
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + deg[i]
    end
    nbrs = Vector{Int}(undef, ptr[end] - 1)
    Execution.run_indices(n, backend) do k
        @inbounds begin
            ids, m = _formula_ids(grid, k, active_only)
            slot = ptr[k]
            for t in 1:m
                j = ids[t]
                active_only && !Grids.isactive(grid, j) && continue
                nbrs[slot] = j
                slot += 1
            end
        end
    end
    return csr_connectivity(nbrs, ptr; validate = false)
end

# ---- HEALPix ----------------------------------------------------------------

# Eight compass directions, of which the eight pixels at a face corner have only seven.
@inline Grids.max_neighbors(::Grids.HEALPixGrid) = 8

@inline function Grids.formula_neighbors(grid::Grids.HEALPixGrid, i::Integer)
    ids, n = healpix_neighbor_ids(Grids.nside(grid), Int(i) - 1)
    # The pixel walk speaks 0-based ids; a cell here is 1-based.
    return (ntuple(t -> @inbounds(t ≤ n ? ids[t] + 1 : 0), Val(8)), n)
end

# ---- Cubed sphere -----------------------------------------------------------

# Four edge neighbours, on every cell: each of the four offsets crosses at most one panel edge, and every
# panel edge folds onto another panel.
@inline Grids.max_neighbors(::Grids.CubedSphereGrid) = 4

const _CUBED_EDGE_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

"""
    Grids.formula_neighbors(grid::CubedSphereGrid, k)

A cubed-sphere cell's four edge neighbours: the panel-interior offsets where they stay on the panel, and
[`_cubed_neighbor`](@ref)'s exact seam fold where they cross to another. The fold is derived by matching
cube coordinates along the shared edge, so a seam neighbour is symmetric.
"""
@inline function Grids.formula_neighbors(grid::Grids.CubedSphereGrid, k::Integer)
    n = Grids.panel_size(grid)
    f, i, j = SphericalSampling._cubed_unlin(Int(k), n)
    ids = ntuple(_ -> 0, Val(4))
    m = 0
    @inbounds for t in 1:4
        δ = _CUBED_EDGE_OFFSETS[t]
        f2, i2, j2 = _cubed_neighbor(f, i, j, δ[1], δ[2], n)
        f2 == 0 && continue
        m += 1
        ids = Base.setindex(ids, SphericalSampling._cubed_lin(f2, i2, j2, n), m)
    end
    return (ids, m)
end

# ---- Yin–Yang ---------------------------------------------------------------

@inline Grids.max_neighbors(::Grids.YinYangGrid) = 4

"""
    Grids.formula_neighbors(grid::YinYangGrid, k)

A Yin–Yang cell's four edge neighbours WITHIN its own panel. Neither panel wraps and the two are not
cross-linked, so a cell on a panel edge reports fewer — the panels couple through interpolation, which
is the standard Yin–Yang discrete topology.
"""
@inline function Grids.formula_neighbors(grid::Grids.YinYangGrid, k::Integer)
    nlon, nlat = Grids.panel_shape(grid)
    p, i, j = Grids.panel_cell(grid, k)
    base = (p - 1) * nlon * nlat
    ids = ntuple(_ -> 0, Val(4))
    m = 0
    @inbounds for t in 1:4
        δ = _CUBED_EDGE_OFFSETS[t]
        ii = i + δ[1]
        jj = j + δ[2]
        (1 ≤ ii ≤ nlon && 1 ≤ jj ≤ nlat) || continue
        m += 1
        ids = Base.setindex(ids, base + (jj - 1) * nlon + ii, m)
    end
    return (ids, m)
end

# ---- Icosahedral geodesic ---------------------------------------------------

# Six lattice directions. The twelve base corners reach five, which is what makes them pentagons.
@inline Grids.max_neighbors(::Grids.IcosahedralGrid) = 6

const _ICO_LATTICE_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))

"""
    Grids.formula_neighbors(grid::IcosahedralGrid, id)

A geodesic vertex's neighbours: the six barycentric lattice steps, taken on every face the vertex sits
on and resolved back to global ids.

A vertex on a macro-edge or at a corner sits on several faces, and the steps along a shared edge land on
the same vertex from each of them, so the duplicates are dropped. What survives is five neighbours at a
corner and six everywhere else.
"""
@inline function Grids.formula_neighbors(grid::Grids.IcosahedralGrid, id::Integer)
    ν = Grids.frequency(grid)
    k = Int(id)
    occ, nocc = SphericalSampling._ico_occurrences(k, ν)
    ids = ntuple(_ -> 0, Val(6))
    n = 0
    @inbounds for t in 1:nocc
        fc, i, j = occ[t]
        for s in 1:6
            δ = _ICO_LATTICE_STEPS[s]
            a = i + δ[1]
            b = j + δ[2]
            (a ≥ 0 && b ≥ 0 && a + b ≤ ν) || continue
            ids, n = _push_unique(ids, n, k, SphericalSampling._ico_lattice_id(fc, a, b, ν))
        end
    end
    return (ids, n)
end

# ---- Ring grids -------------------------------------------------------------

# Two along the ring and two on each adjacent one.
@inline Grids.max_neighbors(::Grids.RingGrid) = 6

# Appends `id` unless it is the cell itself or already present. The tuple is six wide and lives on the
# stack, so the scan is six compares and no memory.
@inline function _push_unique(ids::NTuple{K,Int}, n::Int, self::Int, id::Int) where {K}
    id == self && return (ids, n)
    @inbounds for t in 1:n
        ids[t] == id && return (ids, n)
    end
    return (Base.setindex(ids, id, n + 1), n + 1)
end

"""
    Grids.formula_neighbors(grid::RingGrid, i)

A ring grid's adjacency: the two points either side along the ring, wrapping in longitude, and on each
adjacent ring the two points whose longitudes straddle this one's.

The straddling pair comes from proportional position, `⌊(j−1)·nlon[r′]/nlon[r]⌋` and its successor. Where
adjacent rings differ in width this relation is directed — a wide ring's point can straddle a narrow
ring's point that does not straddle it back — so the graph is not symmetric in general. It is symmetric
whenever the two rings have equal counts, which is most of a Gaussian grid's interior.

Duplicates are dropped, so a ring holding one or two points reports fewer than the six a wide ring does.
"""
@inline function Grids.formula_neighbors(grid::Grids.RingGrid, i::Integer)
    ic = Int(i)
    r = Grids.ring_of(grid, ic)
    nring = Grids.nrings(grid)
    m = Grids.nlon_in_ring(grid, r)
    base = first(Grids.ring_range(grid, r)) - 1
    j = ic - base                                    # 1-based position along the ring

    ids = ntuple(_ -> 0, Val(6))
    n = 0
    # Along the ring, wrapping: `mod1` folds both ends onto the seam.
    ids, n = _push_unique(ids, n, ic, base + mod1(j - 1, m))
    ids, n = _push_unique(ids, n, ic, base + mod1(j + 1, m))
    # The straddling pair on each adjacent ring.
    for rr in (r - 1, r + 1)
        (1 ≤ rr ≤ nring) || continue
        mm = Grids.nlon_in_ring(grid, rr)
        bb = first(Grids.ring_range(grid, rr)) - 1
        p = fld((j - 1) * mm, m)                     # 0-based position on the adjacent ring
        ids, n = _push_unique(ids, n, ic, bb + mod1(p + 1, mm))
        ids, n = _push_unique(ids, n, ic, bb + mod1(p + 2, mm))
    end
    return (ids, n)
end
