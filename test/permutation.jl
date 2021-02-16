using DataFrames
using MixedModels
using MixedModelsPermutations
using MixedModelsPermutations: permute!
using StableRNGs
using Statistics
using Test

isdefined(@__MODULE__, :io) || const io = IOBuffer()

@testset "LMM" begin
    sleepstudy = MixedModels.dataset(:sleepstudy)
    m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy)
    rm1 = permute!(StableRNG(42), deepcopy(m1); residual_method=:signflip)
    rm1 = permute!(StableRNG(42), deepcopy(m1); residual_method=:shuffle)
    # test marking as not fit
    @test_logs (:warn,) show(io, rm1)
    @test_throws ArgumentError permute!(deepcopy(m1); residual_method=:bad)

    H0 = coef(m1)
    H0[2] = 0.0 # slope of days is 0


    perm = permutation(StableRNG(42),1000, m1; β=H0)
    @test perm isa MixedModelPermutation

    df = combine(groupby(DataFrame(perm.allpars), [:type, :group, :names]),
                :value => shortestcovint => :interval)

    let days = filter(df) do row
            return row.type == "β" && row.names == "days"
        end

        lower, upper = only(days.interval)

        @test coef(m1)[2] > upper
        @test coef(m1)[2] > -lower

    end

    @testset "permutationtest" begin
        # the total area above and below the observed value should equal 1
        @test all(map(sum,
                  zip(permutationtest(perm, m1, :lesser),
                      permutationtest(perm, m1, :greater))) .== 1)

        @test_throws ArgumentError permutationtest(perm, m1, :bad)

    end

end

@testset "GLMM" begin
    cbpp = MixedModels.dataset(:cbpp)
    gm1 = fit(MixedModel, @formula((incid/hsz) ~ 1 + period + (1|herd)),
              cbpp, Binomial(); wts=cbpp.hsz, fast=true)

    @test_throws MethodError permutation(1, gm1)
    @test_throws MethodError permute!(gm1)
end
