# FlowGeometries.jl — actionable fixes

Problems that cost memory, allocations, or correctness, and what to change.
Ordered by priority.

---

## P0 — Correctness

### P0.1 Quickhull spherical Voronoi calls a missing function

**Problem:** `ext/FlowGeometriesQuickhullExt.jl` calls `Grids._sph_triangle_area`, which is not defined in `Grids.jl`. Loading Quickhull and requesting spherical Voronoi areas errors.

**Fix:** Implement `_sph_triangle_area(geo, p1, p2, p3)` (spherical triangle area via excess / existing `_tri_excess` on unit vectors), or delete/disable the extension until it works. Add a test that `using Quickhull` + `_voronoi_areas` on a tiny spherical point set succeeds.

---

## P1 — Construction memory (structured grids)

### P1.1 Dense cell measure always allocated

**Problem:** `StructuredGrid` computes separable 1D measure factors, then stores the full outer product (`wx .* transpose(wy)`). A 2000×2000 `Float64` grid pays ~30.5 MiB for measure alone. `_measure_factors` documents separability, but factors are discarded.

**Fix:**
- Store factors (or scalar Δ for uniform `Range` axes) on the grid.
- `measure(grid, I...) = prod(factor[d][I[d]])`.
- Add `materialize_measure!(out, grid)` / `measure_array(grid)` for callers that need a dense buffer.
- Update `_measure_factors` docstring to match stored fields.

### P1.2 Dense active mask always required

**Problem:** Every grid constructor requires a full `Bool` array. All-active grids still allocate `trues(Nx, Ny)` (~3.8 MiB at 2000×2000). `size`/`length`/`axes` are tied to the mask. (`isactive` docstring “e.g. land” is wrong package semantics — use “inactive cell/node” only.)

**Fix:**
- Allow `mask = nothing` (or an always-active sentinel) with no storage.
- `isactive` → `true` in that case; derive `size` from coordinate extents.
- Allocate `Bool`/`BitArray` only when the caller passes a mask.
- Fix the `isactive` docstring.

---

## P2 — Bang contracts and icosahedral allocs

### P2.1 `icosahedral_vertices!` is not in-place

**Problem:** It calls `icosahedral_mesh` (allocates verts, edges, λ, φ), then `copyto!` into caller buffers. Benches show ~4×10⁵ allocs at frequency 32.

**Fix:** Write vertices (XYZ or lon/lat) directly into caller buffers. Do not build edges unless requested. Hardcode the base icosahedron (12 verts, 20 faces, 30 edges); remove `_icosahedron_faces` Set/`Vector{Int}[]` recovery.

### P2.2 Icosahedral connectivity rebuilds the full mesh

**Problem:** `build_connectivity(IcosahedralSampling)` calls `icosahedral_mesh` again. Points + connectivity = two full builds.

**Fix:** One mesh build shared by points and CSR (e.g. `icosahedral_mesh!` → verts + edges → CSR), or build CSR during the face lattice walk (count + fill) without a second mesh pass. Prefer topological edge emission over `sort!`+`unique!` of a redundant edge list when numbering already shares entities.

### P2.3 Other fake / incomplete bangs

| API | Problem | Fix |
|-----|---------|-----|
| `spherical_axes!(…, GaussLegendre)` | Allocates μ/weight scratch inside bang | Pass scratch buffers, or document algorithm scratch separately from output bang |
| `spherical_points!(TP, …)` | Allocates axes inside bang | Accept axes or scratch axis buffers as arguments |
| `spherical_points!(CubedSphere, …)` | Allocates panel vector then drops it | Optional `panel` out-arg, or fill without allocating panel |
| `_gauss_legendre_μ!` | Golub–Welsch → O(n²) eigenvectors | Prefer Newton/asymptotic Legendre roots (O(n) memory) for SHT nodes |

### P2.4 Alloc tests miss the failing paths

**Problem:** Tests check `icosahedral_mesh` allocs but not `icosahedral_vertices!` or `build_connectivity(IcosahedralSampling)`.

**Fix:** Add alloc bounds for those two entry points.

---

## P3 — Curvilinear construction tax

### P3.1 Corners + exact areas always

**Problem:** Construction always stores/reconstructs `(Nx+1)²` corners and always computes dense areas (spherical: full-domain unit-vector array + Van Oosterom excess per cell). Hundreds of ms and ~38 MiB at 1000×1000 even when the caller only needs centers, or already has areas. Docstring says L'Huilier; code is Van Oosterom. `_quad_area` is unused.

**Fix:** Explicit policy argument, defaulting to cheap:

| Policy | Behavior |
|--------|----------|
| `:centers_only` | centers (+ mask/periodic); no corners/areas |
| `:supplied_areas` | use caller areas; corners optional |
| `:center_spacing` | approx from neighboring centers |
| `:shoelace` / `:spherical_excess` | compute from corners (opt-in) |

If exact excess remains: sliding 2-row XYZ buffer instead of full-domain `dirs`. Fix docstring; use or delete `_quad_area`.

---

## P4 — Connectivity API/docs (structured)

### P4.1 Docs call CSR “native connectivity”

**Problem:** README: “Native connectivity is CSR via `build_connectivity` / `neighbors!`.” On structured/curvilinear grids, native connectivity is the index stencil; CSR is an export. That encourages `build_connectivity` (~46 MiB CSR for 1000×1000 face) before neighbor queries that `neighbors!` already answers with no graph storage.

**Fix:** Document per architecture:

- Structured/curvilinear: query via `neighbors!`; CSR/dense/sparse are exports.
- Unstructured: CSR on the grid.
- Mesh samplings: build CSR (or equivalent) from mesh tables once.

### P4.2 `sparse_adjacency_matrix(grid)` always builds CSR first

**Problem:** Dense `adjacency_matrix!(A, grid)` fills from the stencil. Sparse-from-grid calls `build_connectivity` then CSC — double materialization on structured grids.

**Fix:** Stencil→CSC path for structured/curvilinear (mirror dense), or document that sparse-from-grid materializes CSR as an intermediate.

### P4.3 `neighbors` / `StencilNeighbors` docstring

**Problem:** Docstring claims the lazy iterator “allocates nothing at all.” In-loop iteration can be 0-alloc (tests); escaping/`collect`/constructor microbenches allocate.

**Fix:** Docstring: hot path is `neighbors!`; convenience `neighbors` may allocate if the iterator escapes. Optionally make the iterator isbits or return `NTuple{maxdeg,Int}` + count.

---

## P5 — Docs / naming defects

| Location | Fix |
|----------|-----|
| README `to_planetary_cartesian` / `from_planetary_cartesian` | Rename to `vector_to_cartesian` / `vector_from_cartesian` |
| `isactive` “e.g. land” | Remove geophysical example |
| Curvilinear “L'Huilier” | Say Van Oosterom (or whatever the code uses) |
| `_measure_factors` “separability available to callers” | True only after P1.1 stores factors |
| Empty `docs/src/assets/{Project.toml,make.jl}` | Wire Documenter or remove stubs |

---

## P6 — Secondary performance

| Item | Fix |
|------|-----|
| `period::Union{Nothing,Real}` in width kernels | Split periodic / non-periodic methods; branch once at construction |
| Default CSR index `Int` (64-bit) | Offer `I::Type{<:Integer}=Int32` on large-mesh builders |
| HEALPix `_csr_from_candidates` sorts every row | Emit unique/sorted neighbors from RING tables; skip generic `sort!` |
| `_periodic_flags(StructuredGrid)` rebuilds tuple | Return `grid.periodic` |
| `stencil::Symbol` on hot API | Expose `Val` overloads publicly |
| `getproperty` via `findfirst` on names | Kernels use `coordinates(grid, d)`; keep properties convenience-only |
| NamedTuple as core point return | Core `NTuple{N,T}`; NamedTuple convenience wrapper; low-level helpers return tuples |
| Equiangular weights O(n²) | Document; DCT weights if it becomes hot |
| `unstructured_grid` default areas `4πR²/N` | Document as uniform placeholder (wrong for non-equal-area meshes unless supplied) |
| NearestNeighbors builds `Vector` of NTuples | Preallocate `3×N` (or `2×N`) point matrix if NN allows |
| Delaunay Voronoi forces `Float64` | Document promotion |

---

## Implementation order

1. P0.1 Quickhull (correctness)
2. P1.1 Separable measure
3. P1.2 Optional mask
4. P2.1–P2.4 Icosahedral bangs + tests
5. P3.1 Curvilinear policy
6. P4.1–P4.3 Connectivity docs + sparse-from-grid + `neighbors` docstring
7. P5 Docstring/README cleanup
8. P6 as needed

---

## Done when

- Structured all-active grid construction is O(axes) memory by default (no dense measure/mask unless requested).
- `icosahedral_vertices!` and icosahedral `build_connectivity` allocs are O(1) in count (not O(ν²) heap objects); shared build when both needed.
- Quickhull spherical Voronoi tested green.
- Curvilinear default does not run spherical excess unless opted in.
- README/docs describe stencil vs CSR correctly; `sparse_adjacency_matrix(grid)` does not force a silent CSR on structured grids (or says it does).
- No `f!` whose undocumented behavior is “allocate a full mesh then copy.”
