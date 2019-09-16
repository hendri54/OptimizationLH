using Documenter, OptimizationLH

makedocs(
    modules = [OptimizationLH],
    format = :html,
    checkdocs = :exports,
    sitename = "OptimizationLH.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/hendri54/OptimizationLH.jl.git",
)
