using Documenter
using MixedModelsPermutations

makedocs(
    root = joinpath(dirname(pathof(MixedModelsPermutations)), "..", "docs"),
    sitename = "MixedModelsPermutations",
    doctest = true,
    pages = [
        "index.md",
    ],
)

deploydocs(repo = "github.com/palday/MixedModelsPermutations.jl.git", push_preview = true)
