"""
    FlowGeometries

Coordinate metrics, spherical samplings, and grid types.

```
AbstractGeometry
├── AbstractCartesianGeometry  ← CartesianGeometry
└── AbstractSphericalGeometry  ← SphericalGeometry

AbstractSphericalSampling
├── … spectral / lat-lon / HEALPix / cubed-sphere / icosahedral / Yin–Yang / scattered …

AbstractGrid
├── AbstractStructuredGrid     ← StructuredGrid
├── AbstractCurvilinearGrid    ← CurvilinearGrid
└── AbstractUnstructuredGrid   ← UnstructuredGrid
```

Use qualified calls — do not bare-`using` the API into your namespace:

```julia
using FlowGeometries: FlowGeometries as FG
FG.CartesianGeometry(1.0, 1.0)
FG.coords(grid, 2, 3)            # (x=, y=) or (λ=, φ=)
FG.Geometry.distance(geo, p1, p2)
```

Submodules: `FG.Geometry`, `FG.SphericalSampling`, `FG.Grids`, `FG.Connectivity`.
"""
module FlowGeometries

using LinearAlgebra: LinearAlgebra

include("Geometry.jl")
using .Geometry: Geometry

include("SphericalSampling.jl")
using .SphericalSampling: SphericalSampling

include("Grids.jl")
using .Grids: Grids

include("Connectivity.jl")
using .Connectivity: Connectivity

# Bind the public API on this module for `FG.coords`-style calls.
# Nothing is `export`ed — callers use `using FlowGeometries: FlowGeometries as FG`.
using .Geometry:
    AbstractGeometry, AbstractCartesianGeometry, AbstractSphericalGeometry,
    CartesianGeometry, SphericalGeometry,
    distance, area_element, volume_element,
    to_planetary_cartesian, from_planetary_cartesian,
    nonuniform_first_derivative,
    local_tangent_basis, project_to_tangent_plane

using .SphericalSampling:
    AbstractSphericalSampling,
    AbstractTensorProductSphericalSampling, AbstractSpectralQuadratureSampling,
    AbstractGaussLegendreSampling, AbstractDriscollHealySampling,
    AbstractClenshawCurtisSampling, AbstractMcEwenWiauxSampling,
    AbstractLatLonSampling,
    AbstractEqualAreaSphericalSampling, AbstractHEALPixSampling,
    AbstractCubedSphereSampling, AbstractIcosahedralSampling,
    AbstractYinYangSampling, AbstractScatteredSphericalSampling,
    GaussLegendreSampling, DriscollHealySampling, DriscollHealyEqualSampling,
    ClenshawCurtisSampling, McEwenWiauxSampling, LatLonSampling,
    HEALPixSampling, CubedSphereSampling, IcosahedralSampling,
    YinYangSampling, ScatteredSphericalSampling,
    is_tensor_product, is_iso_latitude, is_equal_area,
    admits_exact_bandlimited_quadrature,
    bandlimit, nlat_for_bandlimit, nlon_for_nlat,
    axes_lengths, npoints, icosahedral_nvertices,
    spherical_axes, spherical_axes!, spherical_points, spherical_points!,
    latitude_weights, latitude_weights!,
    colatitude, geographic_latitude,
    healpix_npix, healpix_nring, healpix_pixel_area,
    cubed_sphere_points, cubed_sphere_points!,
    yin_yang_axes, yin_yang_axes!, icosahedral_vertices, icosahedral_vertices!,
    icosahedral_mesh

using .Grids:
    AbstractGrid, AbstractStructuredGrid, AbstractCurvilinearGrid, AbstractUnstructuredGrid,
    StructuredGrid, CurvilinearGrid, UnstructuredGrid,
    coords, coords!, area, isactive, grid_geometry, size_tuple, isperiodic, neighbors

using .Connectivity:
    CSRConnectivity, csr_connectivity, empty_csr,
    build_connectivity, adjacency_matrix, adjacency_matrix!,
    sparse_adjacency_matrix, sparse_adjacency_coo!,
    nneighbors, neighbors!, nnodes, nedges,
    linear_index, cartesian_index,
    structured_grid, unstructured_grid,
    healpix_neighbors!, healpix_neighbors

end # module FlowGeometries
