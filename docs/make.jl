using Documenter
using FlowGeometries

DocMeta.setdocmeta!(
    FlowGeometries,
    :DocTestSetup,
    :(using FlowGeometries: FlowGeometries as FG);
    recursive = true,
)

const MODULES = [
    FlowGeometries,
    FlowGeometries.Axes,
    FlowGeometries.Geometry,
    FlowGeometries.Stencils,
    FlowGeometries.Discretization,
    FlowGeometries.SphericalSampling,
    FlowGeometries.Grids,
    FlowGeometries.Connectivity,
    FlowGeometries.Execution,
]

makedocs(;
    modules = MODULES,
    authors = "Jordan Benjamin",
    sitename = "FlowGeometries.jl",
    format = Documenter.HTML(;
        canonical = "https://jbphyswx.github.io/FlowGeometries.jl",
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = String[],
        size_threshold = 400 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Geometry" => "geometry.md",
        "Axes" => "axes.md",
        "Spherical Sampling" => "sampling.md",
        "Grids" => "grids.md",
        "Stencils & Connectivity" => "connectivity.md",
        "Discretization" => "discretization.md",
        "Extensions" => "extensions.md",
        "Performance" => "performance.md",
        "API Reference" => "api.md",
    ],
    # Every example in the docs is executed, and a docstring that names a signature the code does not
    # have is an error. Without this, documented examples drift from the code silently.
    doctest = true,
    warnonly = [:missing_docs, :cross_references],
    checkdocs = :none,
)

deploydocs(;
    repo = "github.com/jbphyswx/FlowGeometries.jl",
    devbranch = "main",
    push_preview = true,
)
