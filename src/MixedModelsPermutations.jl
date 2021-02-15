module MixedModelsPermutations

using LinearAlgebra
using MixedModels
using MixedModels: MixedModelBootstrap
using Random
using StaticArrays
using StatsBase

export MixedModelPermutation,
       nonparametricbootstrap

include("mixedmodelpermutation.jl")
include("nonparametricbootstrap.jl")

end
