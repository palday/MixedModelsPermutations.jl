
#------------ setup parallel
using Distributed
using ProgressMeter
using SharedArrays
addprocs(10,exeflags="--project",enable_threaded_blas = true)
#@everywhere using BLAS
#@everywhere BLAS.set_num_threads(4)
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
    #H0[1] = 0.0
    perm = permutation(MersenneTwister(k+1),nPerm,simMod,use_threads=false;β=H0); 
    p_β = values(permutationtest(perm,simMod;statistic=:β ))
    p_z = values(permutationtest(perm,simMod;statistic=:z ))
    return (p_β,p_z)
    

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
mres2 = MixedModels.fit(MixedModel, f1, dat)


rngseed = 1
β = repeat([0],length(mres.β))
σ = 0.1
θ = repeat([1],length(mres.θ))

θ = [1,0,1,1,0,1.]./σ

# ---------------- run jobs

refit!(updateL!(setθ!(mres2,Vector(θ))))

nPerm = 500
nRep = 200
β_permResult = SharedArray{Float64}(nRep,length(β))
z_permResult = SharedArray{Float64}(nRep,length(β))

@showprogress 0.5 @distributed for k =1:nRep
    res = run_perm(mres,k,nPerm,β,σ,θ)
    β_permResult[k,:] .= res[1]
    z_permResult[k,:] .= res[2]
end

#------------

subj_btwn = Dict("age" => ["O", "Y"])
item_btwn = Dict("stimType" => ["I","II"])
both_win = Dict("condition" => ["A", "B"])

dat_sim = simdat_crossed(30, 30,subj_btwn = subj_btwn, item_btwn = item_btwn, both_win = both_win);

f1 = @formula dv ~ 1 + condition  + (1+condition|item) + (1+condition|subj);
mres1 = MixedModels.fit(MixedModel, f1, dat_sim)
β = repeat([0],length(mres1.β))
σ = 0.1


θ = [1,0,1,1,0,1.]

mres1 = simulate!(MersenneTwister(1), mres1, β = β, σ = σ, θ = θ)
f_applied = apply_schema(f1,schema(f1,dat_sim),LinearMixedModel)
tmp, Xs = modelcols(f_applied, dat)

mres2 = MixedModels.LinearMixedModel(mres1.y, Xs,f_applied,[])
updateL!(setθ!(mres2,Vector(θ)))
permutation(MersenneTwister(1),2,mres2)
