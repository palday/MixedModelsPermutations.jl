
#------------ setup parallel
using Distributed
using SharedArrays
addprocs(30,exeflags="--project")
@everywhere using Pkg
@everywhere Pkg.activate(".")

@everywhere using MixedModelsSim
@everywhere using Random
@everywhere using MixedModels
@everywhere using MixedModelsPermutations

@everywhere function run_perm(mres,k,nPerm,β,σ,θ)
    simMod = deepcopy(mres)
    simMod = simulate!(MersenneTwister(k), simMod, β = β, σ = σ, θ = θ)
    refit!(simMod)
    H0 = coef(simMod)
    H0[2] = 0.0 

    perm = permutation(MersenneTwister(k),nPerm,simMod,use_threads=false;β=H0); 

    return values(permutationtest(perm,simMod))
    

end
# ------------ define model etc.
# put between-subject factors in a Dict
subj_btwn = Dict("age" => ["O", "Y"])

# there are no between-item factors in this design so you can omit it or set it to nothing
item_btwn = Dict("stimType" => ["I","II"])

# put within-subject/item factors in a Dict
both_win = Dict("condition" => ["A", "B"])

# simulate data
dat = simdat_crossed(15, 30,
                    subj_btwn = subj_btwn, 
                    item_btwn = item_btwn, 
                    both_win = both_win);


#    f1 = @formula dv ~ 1 + age * condition  + (1+condition|item) + (1+condition|subj);
f1 = @formula dv ~ 1 + condition  + (1+condition|item) + (1+condition|subj);
mres = fit(MixedModel, f1, dat)


rngseed = 1
β = repeat([0],length(mres.β))
σ = 0.1
θ = repeat([1],length(mres.θ))

θ = [1,0,1,1,0,1.]./σ

# ---------------- run jobs


nPerm = 100
nRep = 100
permResult = SharedArray{Float64}(nRep,length(β))

a = @distributed for k =1:nRep
    permResult[k,:] .= run_perm(mres,k,nPerm,β,σ,θ)
end
