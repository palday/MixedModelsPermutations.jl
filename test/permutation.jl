using DataFrames
using MixedModels
using MixedModelsPermutations
using MixedModelsPermutations: permute!
using StableRNGs
using Test

isdefined(@__MODULE__, :io) || const io = IOBuffer()

@testset "LMM" begin
    sleepstudy = MixedModels.dataset(:sleepstudy)
    m1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days|subj)), sleepstudy)
    rm1 = permute!(StableRNG(42), deepcopy(m1))
    # test marking as not fit
    @test_logs (:warn,) show(io, rm1)

    perm = permutation(StableRNG(42),1000, m1)
    @test perm isa MixedModelPermutation

    df = combine(groupby(DataFrame(perm.allpars), [:type, :group, :names]),
                :value => shortestcovint => :interval)

    # let rho = filter(:type => ==("ρ"), df)
    #     violations = mapreduce(+, rho[:, :interval]) do (a,b)
    #         return !(-1 <= a <= b <= +1)
    #     end

    #     @test violations == 0
    # end

    # let beta = filter(:type => ==("β"), df)
    #     violations = mapreduce(+, zip(fixef(m1), beta[:, :interval])) do (b, (lower,upper))
    #         return !(lower <= b <= upper)
    #     end
    #     @test violations == 0
    # end

    # let sigma = filter(:type => ==("σ"), df)
    #     sigs =[ MixedModels.σs(m1).subj...; m1.σ ]

    #     violations = mapreduce(+, zip(sigs, sigma[:, :interval])) do (b, (lower,upper))
    #         return !(0 <= lower <= b <= upper)
    #     end
    #     @test violations == 0
    # end
end

@testset "GLMM" begin
    cbpp = MixedModels.dataset(:cbpp)
    gm1 = fit(MixedModel, @formula((incid/hsz) ~ 1 + period + (1|herd)),
              cbpp, Binomial(); wts=cbpp.hsz, fast=true)

    @test_throws MethodError permutation(1, gm1)
    @test_throws MethodError permute!(gm1)
end
