using CairoMakie
CairoMakie.activate!()

using MixedModelsSim
using MixedModelsPermutations
using MixedModels
using DataFrames
using StableRNGs
using LinearAlgebra
using ProgressMeter
using Statistics

seed = 666
rng = StableRNG(seed)
nsub = 30
nitem = 30

f =  @formula(dv ~ 1 + condition + (1 + condition | subj))  ;

subj_btwn = Dict("age" => ["O", "Y"])
# there are no between-item factors in this design so you can omit it or set it to nothing
item_btwn = Dict("stimType" => ["I", "II"])

# put within-subject/item factors in a Dict
both_win = Dict("condition" => ["A", "B"])

# simulate data
dat = simdat_crossed(rng,
    nsub,
    nitem,
    subj_btwn = subj_btwn,
    item_btwn = item_btwn,
    both_win = both_win,
)

# generate simulated "y" vector
β = [1., 1.];
σs = [1, 1, 0.2];
simMod = fit(MixedModel, f, dat)
simMod = MixedModelsSim.update!(simMod,[create_re(x...) for x in σs]...)
simMod = simulate!(rng, simMod; β, σ = 1.)

# fit a LMM mod to the y (separate models to maybe have separate formulas later
datSim = DataFrame(dat)
datSim[!, :dv] .= response(simMod)
morig = fit(MixedModel, f, datSim; REML=false)
let
    inflate_names = Dict("ols" => "inflation_factor from OLS blups",
                           "yes" => "inflation_factor from original blups/residuals",
                            "no" => "identity")

    global function simulate(model, inflate, niter=100;
                             sample_func=MixedModelsPermutations.permute!)
        inflate in keys(inflate_names) || error("Invalid inflation strategy")
        samps = DataFrame()
        @showprogress for p in 1:niter
            m_new = deepcopy(model)

            blups = olsranef(m_new)
            resids = residuals(m_new, blups)
            if inflate == "ols"
                scalings = MixedModelsPermutations.inflation_factor(m_new, blups, resids)
            elseif inflate == "yes"
                scalings = MixedModelsPermutations.inflation_factor(m_new)
            else
                # assumes that we have exactly 2 blocking vars
                scalings = [I, I, 1]
            end
            sample_func(rng, m_new; model.β, blups, resids, scalings)

            refit!(m_new; progress=false)
            push!(samps, (; iter=p, σ=m_new.σ, β=m_new.β ))
        end
        return samps
    end
end

MixedModels.stderror(v; corrected=false) = std(v; corrected) / sqrt(length(v))

let
    fig = Figure()
    fig[1,1] = ax = Axis(fig)
    xs = Observable([1.0])
    ys = Observable([1.0])
    mean_σ = Observable(1.0)
    model_σ = Observable(1.0)
    # lower = Observable(1.0)
    # upper = Observable(1.0)

    lines!(ax, xs, ys)
    hlines!(ax, 1, color=:black)
    hlines!(ax, mean_σ, linetype=:dotted, label="mean resampled σ")
    hlines!(ax, model_σ, linetype=:dashed, label="model σ")
    # band!(ax, xs, lower, upper; label="mean resampled σ")
    axislegend(ax; orientation=:horizontal)

    global function simulate_and_plot(model, args...; kwargs...)
        dat = simulate(model, args...; kwargs...)

        resize!(xs[], nrow(dat))
        resize!(ys[], nrow(dat))
        xs[] = dat.iter
        ys[] = dat.σ
        mean_σ[] = mean(dat.σ)
        # sem = 1.96 * stderror(dat.σ)
        #lower[] .= mean(dat.σ) - sem
        #upper[] .=  mean(dat.σ) + sem
        model_σ[] = model.σ
        autolimits!(ax)

        return fig
    end
end

simulate_and_plot(morig, "yes", 200; sample_func=MixedModelsPermutations.resample!)
simulate_and_plot(morig, "yes", 200; sample_func=MixedModelsPermutations.permute!)