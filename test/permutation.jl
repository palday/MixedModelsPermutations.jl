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
    m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy, progress=false)
    rm1 = permute!(StableRNG(42), deepcopy(m1); residual_permutation=:signflip)
    rm1 = permute!(StableRNG(42), deepcopy(m1); residual_permutation=:shuffle)
    # test marking as not fit
    @test_logs (:warn,) show(io, rm1)
    @test_throws ArgumentError permute!(deepcopy(m1); residual_permutation=:bad)

    H0 = coef(m1)
    H0[2] = 0.0 # slope of days is 0


    perm = permutation(StableRNG(42),1000, m1; β=H0)
    @test perm isa MixedModelPermutation
    @test perm isa MixedModels.MixedModelFitCollection

    df = combine(groupby(DataFrame(perm.allpars), [:type, :group, :names]),
                :value => shortestcovint => :interval)

    # our original slope should lie well outside of the null distribution
    let days = filter(df) do row
            return row.type == "β" && row.names == "days"
        end

        lower, upper = only(days.interval)

        @test coef(m1)[2] > upper
        @test coef(m1)[2] > -lower
    end

    # our original intercept should lie within the range of
    # the intercepts generated via permutation
    let est = filter(df) do row
            return row.type == "β" && row.names == "(Intercept)"
        end

        lower, upper = only(est.interval)
        @test lower < first(H0) < upper
    end

    @testset "permutationtest" begin
        # the total area above and below the observed value should equal 1
        @test all(map(sum,
                  zip(permutationtest(perm, m1, type=:lesser),
                      permutationtest(perm, m1, type=:greater))) .== 1)

        # we should have a p-value near 1 since our null distribution
        # was generated from value being tested...
        @test first(permutationtest(perm, m1)) > 0.95
        # we should have a p-value near 0 since this effect is clear
        @test last(permutationtest(perm, m1)) <= 1 / 1000
        @test_throws ArgumentError permutationtest(perm, m1, type=:bad)
    end

    @testset "olsranef" begin
        permols = permutation(StableRNG(42),1000, m1; β=H0, blup_method=olsranef)
        @test permols isa MixedModelPermutation
        @test last(permutationtest(perm, m1, :greater)) ≈ last(permutationtest(permols, m1, :greater)) atol=0.01
        @test last(permutationtest(perm, m1, :lesser)) ≈ last(permutationtest(permols, m1, :lesser)) atol=0.01
    end

end

@testset "GLMM" begin
    cbpp = MixedModels.dataset(:cbpp)
    gm1 = fit(MixedModel, @formula((incid/hsz) ~ 1 + period + (1|herd)),
              cbpp, Binomial(); wts=cbpp.hsz, fast=true, progress=false)

    @test_throws MethodError permutation(1, gm1)
    @test_throws MethodError permute!(gm1)
end
