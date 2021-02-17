using DataFrames
using MixedModels
using MixedModelsPermutations
using StableRNGs
using Test

# TODO: add in model cache for the sleepstudy model
isdefined(@__MODULE__, :io) || const io = IOBuffer()

# TODO: add in testset showing that the OLS ranef match OLS fit to each participant
@testset "olsranef" begin

end


@testset "residuals" begin
    sleepstudy = MixedModels.dataset(:sleepstudy)
    m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy)

    @test residuals(m1) ≈ residuals(m1, ranef(m1))
end
