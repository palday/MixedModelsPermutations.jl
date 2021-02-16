module MixedModelsPermutations

using LinearAlgebra
using MixedModels # we add several methods
using Random
using SparseArrays
using StaticArrays
using Statistics
using StatsBase
using Tables

using MixedModels: MixedModelBootstrap
using MixedModels: fixef!, stderror!
using MixedModels: getθ!, updateL! # setθ! is imported for extending
using MixedModels: unscaledre!

import MixedModels: allpars, coefpvalues, issingular, setθ!, tidyβ, tidyσs

export MixedModelPermutation,
       nonparametricbootstrap,
       olsranef,
       permutation,
       permutationtest

include("mixedmodelpermutation.jl")
include("nonparametricbootstrap.jl")
include("permutation.jl")

end
