using DrWatson
@quickactivate "unfoldjl_dev"

using Random
include("test/sim_utilities.jl")


##---
paramList = Dict(
    "f" => @formula(dv ~ 1 + condition  + (1+condition|subj)),
            #@formula dv ~ 1 + age * condition  + (1+condition|item) + (1+condition|subj)],
    "θ" => [[1., 1., 1.]],
    "σ" => 1.,
    "β" => [[0., 0.]],
    "blup_method" => ["ranef_scaled","olsranefjf", "olsranef"],
    "residual_method" => [:signflip,:shuffle],
    "nRep" => 1000,
    "nPerm"=> 1000,
)

##--
dl = dict_list(paramList)[1]
simMod = sim_model(dl["f"])
res = run_permutationtest(MersenneTwister(1),simMod,dl["nPerm"],dl["β"],dl["σ"],dl["θ"])

##---
nWorkers=20
for dl = dict_list(paramList)
    
    res = run_permutationtest_distributed(nWorkers,dl["nRep"],simMod,dl["nPerm"],dl["β"],dl["σ"],dl["θ"],dl["residual_method"],dl["blup_method"])

end
using DataFrames, AlgebraOfGraphics

df = DataFrame(z:=>[],:β=>[],:h1=>[],ranef=>[])
df = DataFrame(:z=>res[1][:],:β=>res[2][:],:h1=>[repeat(["1"],size(res[1],1)); repeat(["0"],size(res[1],1))])

data(df) * mapping(:z,layout_y=:h1) * visual(Hist,bins=0:0.01:1)  |>draw()