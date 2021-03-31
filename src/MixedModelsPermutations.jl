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
using MixedModels: replicate
using MixedModels: unscaledre!

import MixedModels: allpars, coefpvalues, issingular, setθ!, tidyβ, tidyσs, residuals

export MixedModelPermutation,
       nonparametricbootstrap,
       olsranef,
       permutation,
       permutationtest,
       residuals

include("mixedmodelpermutation.jl")
include("inflation.jl")
include("nonparametricbootstrap.jl")
include("permutation.jl")
include("ols.jl")

end
