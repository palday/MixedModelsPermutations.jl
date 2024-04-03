using DataFrames
using MixedModels
using MixedModelsPermutations
using StableRNGs
using Test

# TODO: add in model cache for the sleepstudy model
isdefined(@__MODULE__, :io) || const io = IOBuffer()

sleepstudy = MixedModels.dataset(:sleepstudy)
m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy;
         progress=false)

# TODO: add in testset showing that the OLS ranef match OLS fit to each participant
@testset "olsranef" begin
    # for a balanced design with one blocking variable, these methods should
    # give the same results
    stratum = only(olsranef(m1, :stratum))
    simultaneous = only(olsranef(m1, :simultaneous))
    inflated_identity = only(olsranef(m1, :inflated_identity))
    @test all(isapprox.(stratum, simultaneous))
    @test all(isapprox.(stratum, inflated_identity; atol=1e-4))
end

@testset "residuals" begin
    @test residuals(m1) ≈ residuals(m1, ranef(m1))
end
