using Documenter
using MixedModelsPermutations

makedocs(
    sitename = "MixedModelsPermutations",
    doctest = true,
    pages = [
        "index.md",
    ],
)

deploydocs(repo = "github.com/palday/MixedModelsPermutations.jl.git", push_preview=true, devbranch="main")
