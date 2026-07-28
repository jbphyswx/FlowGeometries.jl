```@meta
CurrentModule = FlowGeometries.SphericalSampling
```

# [Spherical Sampling](@id sampling-page)

A sampling answers: **where are the points?** It is independent of the metric and of how the data is
stored.

## The families

| sampling | layout | points | notes |
|---|---|---|---|
| `GaussLegendreSampling` | tensor product | `(2N−1) × N` | exact quadrature to `lmax = N−1` |
| `DriscollHealySampling` | tensor product | `2N × N` | equiangular, includes the north pole |
| `DriscollHealyEqualSampling` | tensor product | `N × N` | the square DH1 layout |
| `ClenshawCurtisSampling` | tensor product | `(2N−1) × N` | open — no polar points |
| `McEwenWiauxSampling` | tensor product | `(2L−1) × L` | ~half the samples of classical DH |
| `LatLonSampling` | tensor product | arbitrary | regional or global; no spectral claim |
| `HEALPixSampling` | iso-latitude rings | `12·nside²` | equal area by construction |
| `CubedSphereSampling` | 6 panels | `6n²` | quasi-uniform |
| `IcosahedralSampling` | geodesic | `10ν²+2` | quasi-uniform; hexagonal dual |
| `YinYangSampling` | 2 overlapping panels | `2·nlon·nlat` | no polar singularity |
| `ScatteredSphericalSampling` | arbitrary | any | for NUFFT-style paths |


![Six spherical samplings](assets/samplings.png)

Gauss–Legendre and Clenshaw–Curtis put their points on iso-latitude rings that crowd toward the
poles; HEALPix, the cubed sphere and the icosahedral geodesic are quasi-uniform. Yin–Yang is two
overlapping panels, which is why it has no polar singularity.

## Traits

Rather than testing types, ask:

```julia
FG.is_tensor_product(s)                       # fits an (nlon × nlat) structured grid?
FG.is_iso_latitude(s)                          # points lie on rings of constant φ?
FG.is_equal_area(s)                            # every cell the same area?
FG.admits_exact_bandlimited_quadrature(s)      # weights integrate PRODUCTS exactly at `bandlimit`?
```

That last one is deliberately strict. Spectral analysis forms products of two degree-`lmax` fields,
so it needs exactness to degree `2·lmax`, not `lmax`:

| sampling | exact for a single `Pₗ` | `bandlimit` | needs | exact? |
|---|---|---|---|---|
| Gauss–Legendre | `2N−1` | `N−1` | `2N−2` | yes |
| Driscoll–Healy | `N−1` | `N/2−1` | `N−2` | yes |
| Clenshaw–Curtis | `N−1` | `N−1` | `2N−2` | **no** |

Clenshaw–Curtis's band limit describes what its grid can *represent*; its quadrature only supports
analysis to about `(N−1)/2`. Use Gauss–Legendre when the analysis has to be exact.

## Sizing

```julia
FG.bandlimit(s, nlat)              # lmax this grid resolves
FG.nlat_for_bandlimit(s, lmax)     # the inverse
FG.nlon_for_nlat(s, nlat)          # the matching longitude count
FG.axes_lengths(s, nlat)           # (; nlon, nlat)
FG.npoints(s, args...)             # total points
```

## Axes, points and weights

Tensor-product samplings give separable axes; the rest give point lists.

```julia
ax = FG.spherical_axes(FG.GaussLegendreSampling(), 64)      # (; λ, φ) — 127 and 64 long
w  = FG.latitude_weights(FG.GaussLegendreSampling(), 64)    # 64 weights, Σw = 2
```

**If you need both, ask for both.** Gauss–Legendre nodes and weights come out of one root solve;
requesting them separately solves twice:

```julia
q = FG.spherical_quadrature(FG.GaussLegendreSampling(), 64) # (; λ, φ, w) — one solve
```

Every one of these has an in-place form that writes into your buffers and allocates nothing beyond
its return value: `spherical_axes!`, `latitude_weights!`, `spherical_quadrature!`,
`spherical_points!`.

```julia
λ = Vector{Float64}(undef, 127); φ = Vector{Float64}(undef, 64); w = similar(φ)
FG.spherical_quadrature!(λ, φ, w, FG.GaussLegendreSampling(), 64)
```

Weights are normalized so `Σw = ∫₀^π sinθ dθ = 2` for every sampling that has them. McEwen–Wiaux
deliberately has none: its quadrature is built on an extension of the sphere to a torus, and applying
the sine-series rule to its nodes is not exact even at `l = 0`, so asking for them raises rather than
returning a plausible wrong answer.

## Sampling-specific constructors

```julia
FG.cubed_sphere_points(n)              # (; λ, φ, panel) — 6n² cell centres
FG.icosahedral_mesh(ν)                 # (; λ, φ, edges, triangles, verts)
FG.icosahedral_vertices(ν)             # just the points; skips building the topology
FG.yin_yang_panels(nlon, nlat)         # (; yin, yang)
FG.healpix_npix(nside), FG.healpix_nring(nside), FG.healpix_pixel_area(nside)
```

`yin_yang_panels` returns asymmetric shapes on purpose. The yin panel is a separable lat–lon patch in
its own frame, so it is a pair of **axes**; yang is that panel rotated onto the sphere, which is not
separable in global lon/lat, so it is a pair of `nlon × nlat` **fields**. The shapes differ because
the geometry does.

`icosahedral_mesh(ν; topology = false)` skips the edge and triangle lists when you only want points —
which is what `icosahedral_vertices` does, and is several times faster at large ν.

## Cell centres, not panel edges

Every sampling here places points at **cell centres**. This matters more than it sounds: sampling the
panel edges instead makes adjacent panels emit coincident points while the connectivity treats those
same edges as folding onto a different panel. Points and topology then disagree, and any grid built
from them carries duplicate nodes and zero-area cells. Cell centres also make the degenerate sizes
(`n = 1`, `nlon = 1`) fall out of the formula instead of needing special cases.
