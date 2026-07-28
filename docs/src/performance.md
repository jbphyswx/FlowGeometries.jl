```@meta
CurrentModule = FlowGeometries
```

# [Performance](@id performance-page)

The design rules, and the measurements behind them.

## Rules

1. No superlinear algorithm where a linear one exists.
2. Intrinsic data is `O(∑ Nᵈ)`; anything `O(∏ Nᵈ)` is a materialization — make it opt-in.
3. Compute nothing twice, across calls as well as within one.
4. One algorithm, one implementation.
5. No transcendental in an inner loop that a recurrence, `sincos`, or a hoist can remove.
6. Never compute at runtime what is a mathematical constant.
7. An API must not be able to request a terabyte by accident.

## Scaling

Every public entry point is linear or better in its problem size. Measured as `cost ~ nᵖ` across a
size step:

| entry point | MiB | allocs | `t ~ nᵖ` |
|---|---|---|---|
| `latitude_weights` Gauss–Legendre | 0.01 | 3 | 0.99 |
| `spherical_axes` Gauss–Legendre | 0.02 | 6 | 0.96 |
| `spherical_quadrature` Gauss–Legendre | 0.03 | 9 | 0.97 |
| `latitude_weights` Driscoll–Healy | 0.02 | 11 | 0.34 |
| `latitude_weights` Clenshaw–Curtis | 0.02 | 11 | 1.14 |
| `UnstructuredGrid` k-d tree, 40k | 6.23 | 52 | 1.06 |
| `build_connectivity` sampling | 91.48 | 9 | 0.93 |
| `unstructured_grid` cubed sphere | 1.79 | 28 | 1.04 |
| `unstructured_grid` icosahedral | 4.00 | 80 | 1.04 |
| `unstructured_grid` Yin–Yang | 3.76 | 41 | 0.98 |
| `unstructured_grid` HEALPix | 1.22 | 18 | 1.06 |

Allocation counts are flat in `n` everywhere. `build_connectivity`'s 91 MiB is the CSR output itself
for ~2M nodes, not overhead.

## Quadrature

Gauss–Legendre uses asymptotic expansions (Bogaert 2014; Hale & Townsend 2013) above `n = 60`: each
node sits near `j_k/(n+½)` for `j_k` a zero of `J₀`, with corrections in powers of `(n+½)⁻²`. Nothing
iterates and no root depends on its neighbours, so it is `O(1)` per node and allocation-free.

| `n = 2048` | Golub–Welsch | now |
|---|---|---|
| time | 279.4 ms | **0.05 ms** |
| memory | 33.4 MiB | 32 KiB |

Accuracy against a 256-bit reference is ~6e-16 relative weight error at every `n` from 64 to 4096.

Below `n = 60`, and for element types wider than `Float64`, the solve falls back to Newton on the
Bonnet recurrence: the expansion's coefficients are a fixed `Float64` set and cannot exceed that
precision, while Newton converges to `eps(T)` — `BigFloat` at 256 bits gives `|Σw − 2| = 1.7e-77`.

Equiangular (DH/CC) weights are `O(n log n)` with an FFT loaded and `O(n²)` without, with the two
agreeing to 1.4e-14.

## Memory

The two big materializations are gone:

- **Cell measure** — stored as per-axis factors. At 2000²: **61.0 MiB → 0.046 MiB**.
- **Curvilinear corner directions** — two rows live at a time instead of the whole field. At 1000²:
  **61.3 MiB → 23.1 MiB**, with bit-identical areas.

The factored measure is also *faster* to read, which was not the expectation. At 2000² the dense
array is 61 MiB and DRAM-bound while the factors stay in cache:

| access | factored vs dense |
|---|---|
| full sweep | 0.91× |
| strided, radius 8 | 0.43× |
| strided, radius 32 | 0.40× |
| `sum` | ~6000× |

!!! warning "Measure at realistic sizes"
    An earlier version of this comparison found factored and dense within 1.0× and concluded the
    factoring was memory-only. That measurement was taken at a size where the dense array still fit
    in cache. Benchmarks of memory-bound code at toy sizes measure the cache, not the code.

## Threading

Opt-in through `ComputationalBackends` tags, serial by default, bit-identical results:

| kernel | speedup (8 threads) |
|---|---|
| `cubed_sphere_points` | 6.2× |
| curvilinear corner areas | 4.9× |
| index-topology connectivity | 3.7–4.3× |
| candidate CSR builder (HEALPix / cubed sphere / Yin–Yang) | 1.8–3.7× |

```julia
using ComputationalBackends: ThreadedBackend
FG.build_connectivity(grid; backend = ThreadedBackend())
```

Chunks are contiguous, so each thread touches one span of every array rather than striding across
all of them. Passing `nothing` (the default) hands the loop body the whole range in one call, so the
serial path adds no partitioning at all.

Two passes are deliberately **not** threaded, and the reasons are structural rather than pending
work. The CSR compacting move can have row `j`'s destination fall inside row `i`'s source block for
`i < j`, so concurrent rows would overwrite unread candidates. And the prefix scan between the count
and fill passes is inherently sequential — it is `O(n)` against the `O(n·stencil)` passes it
separates.

## Things measured and found *not* to be problems

Recorded so they are not re-litigated:

- **`getproperty` coordinate-name lookup is free** — 0.065 vs 0.077 ms per 10⁶ reads; it const-folds.
- **The dense-mask branch is free** in a predictable loop.
- **Cross-point SIMD is unavailable** — `@simd` over 10⁶ haversines is 1.01×; LLVM cannot vectorize
  `sin`/`cos`/`atan`. Eliminating trig is the only lever, which is why the geometry kernels use
  `sincos` and hoist `atan` out of loops.
- **Vector-axis construction is not superlinear** — the real finding was a 3.8× constant gap against
  range axes.
