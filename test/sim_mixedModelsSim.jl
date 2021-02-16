using MixedModelsSim


# put between-subject factors in a Dict
subj_btwn = Dict("age" => ["O", "Y"])

# there are no between-item factors in this design so you can omit it or set it to nothing
item_btwn = Dict("stimType" => ["I","II"])

# put within-subject/item factors in a Dict
both_win = Dict("condition" => ["A", "B"])

# simulate data
dat = simdat_crossed(10, 30, 
                     subj_btwn = subj_btwn, 
                     item_btwn = item_btwn, 
                     both_win = both_win);


f1 = @formula dv ~ 1 + age * condition  + (1|item) + (1|subj);
mod = fit(MixedModel, f1, dat)


rngseed = 1
β = repeat([0],length(fixef(mod)))
σ = 1
θ = repeat([1],length(ranef(mod)))
simulate!(Random.MersenneTwister(rngseed), mod, β = β, σ = σ, θ = θ)
