module MixedModelsPermutations

using LinearAlgebra
using MixedModels # we add several methods
using Random
using SparseArrays
using StaticArrays
using Statistics
using StatsBase

using MixedModels: MixedModelBootstrap
using MixedModels: fixef!, stderror!
using MixedModels: getθ!, updateL! # setθ! is imported for extending
using MixedModels: unscaledre!

import MixedModels: allpars, coefpvalues, issingular, setθ!, tidyβ, tidyσs

export MixedModelPermutation,
       nonparametricbootstrap,
       resample!

include("mixedmodelpermutation.jl")
include("nonparametricbootstrap.jl")

end
