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
    FlowGeometries.Geometry,
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
        "Spherical Sampling" => "sampling.md",
        "Grids" => "grids.md",
        "Connectivity" => "connectivity.md",
        "Extensions" => "extensions.md",
        "Performance" => "performance.md",
        "API Reference" => "api.md",
    ],
    warnonly = true,
    checkdocs = :none,
)

deploydocs(;
    repo = "github.com/jbphyswx/FlowGeometries.jl",
    devbranch = "main",
    push_preview = true,
)
