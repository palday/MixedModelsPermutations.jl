using DataFrames
using MixedModels
using MixedModelsPermutations
using StableRNGs
using Test

# TODO: add in model cache for the sleepstudy model
isdefined(@__MODULE__, :io) || const io = IOBuffer()

sleepstudy = MixedModels.dataset(:sleepstudy)
m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy)

# TODO: add in testset showing that the OLS ranef match OLS fit to each participant
@testset "olsranef" begin
    # for a balanced design with one blocking variable, these methods should
    # give the same results
    @test all(only(olsranef(m1, :stratum)) .≈ only(olsranef(m1, :simultaneous)))
end


@testset "residuals" begin
    @test residuals(m1) ≈ residuals(m1, ranef(m1))
end
