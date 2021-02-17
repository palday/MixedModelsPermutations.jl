using MixedModelsPermutations
using Test

@testset "nonparametricbootstrap" begin
    include("nonparametricbootstrap.jl")
end

@testset "permutation" begin
    include("permutation.jl")
end

@testset "ols" begin
    include("ols.jl")
end
