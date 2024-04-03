### A Pluto.jl notebook ###
# v0.18.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try
            Base.loaded_modules[Base.PkgId(
                Base.UUID("6e696c72-6542-2067-7265-42206c756150"),
                "AbstractPlutoDingetjes",
            )].Bonds.initial_value
        catch
            b -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ f92470d8-c28e-4057-87bd-12383b43c158
begin
    using Pkg
    Pkg.activate(; temp = true)
    Pkg.add("Revise")
    using Revise
    Pkg.add(path = "..")
    Pkg.add(["PlutoUI", "MixedModelsSim", "MixedModels", "DataFrames", "StableRNGs"])
    using PlutoUI
    using MixedModelsSim
    using MixedModelsPermutations
    using MixedModels
    using DataFrames
    using StableRNGs
    using LinearAlgebra
    using Statistics
end

# ╔═╡ f5830c4e-a8d0-47f2-baca-bc734720b1e0
pwd()

# ╔═╡ 2dce0f9f-0e00-46ee-8a6b-b1441e69f4f5
md"## Calculate Single Inflation factor"

# ╔═╡ fe1305ba-f6bf-47ee-a89e-d7326a8dab1a
md"### Steering committee"

# ╔═╡ 55e97ef0-0f23-48a6-8229-7f1b6ca3d044
md"seed $(@bind seed_val Slider(1:100; default=1., show_value=true))"

# ╔═╡ c7dfaa8d-dcdf-4431-98b8-c6bb9d8fac52
rng = StableRNG(Int(seed_val))

# ╔═╡ 2141edd4-c8ed-4d1b-a5ec-9913b19a2d4a
md"Subjects: $(@bind nsub Slider(2:2:50; default=30, show_value=true))"


# ╔═╡ bac32d2f-fb1f-4f02-b2f5-49df029b7b92
md"Items per Subject: $(@bind nitem Slider(2:2:50; default=30, show_value=true))"

# ╔═╡ 461f6470-359e-406a-b32d-5455647e36b8
@bind inflate Radio(
    [
        "ols" => "inflation_factor from OLS blups",
        "yes" => "inflation_factor from original blups/residuals",
        "no" => "identity",
    ],
    "ols",
)

# ╔═╡ a44e5f9f-5a23-46b6-adbd-228b893ac81f
md"### Calculate 50 permutations"

# ╔═╡ 2a59f2be-0037-4a8f-bee6-cce32ee80407
md"### How the saussage is simulated"

# ╔═╡ 0cbf776c-8e56-4d32-8a55-da22f8bf1940
f = @formula(dv ~ 1 + condition + (1 + condition | subj));

# ╔═╡ 72f5e910-e40d-11eb-3725-f19a95c9b3c1
begin
    subj_btwn = Dict("age" => ["O", "Y"])
    # there are no between-item factors in this design so you can omit it or set it to nothing
    item_btwn = Dict("stimType" => ["I", "II"])

    # put within-subject/item factors in a Dict
    both_win = Dict("condition" => ["A", "B"])

    # simulate data
    dat = simdat_crossed(
        rng,
        nsub,
        nitem,
        subj_btwn = subj_btwn,
        item_btwn = item_btwn,
        both_win = both_win,
    )
end

# ╔═╡ cd9b3fb7-06c7-4c44-81ac-d8a4a3e0c22e
# generate simulated "y" vector
begin
    β = [1.0, 1.0]
    σs = [1, 1, 0.2]
    simMod = fit(MixedModel, f, dat)
    simMod = MixedModelsSim.update!(simMod, [create_re(x...) for x in σs]...)
    simMod = simulate!(rng, simMod; β, σ = 1.0)
end;

# ╔═╡ cfd225bb-1eec-41b1-b6c1-ebc036b7e1d2
begin
    # fit a LMM mod to the y (separate models to maybe have separate formulas later
    datSim = DataFrame(dat)
    datSim[!, :dv] .= response(simMod)
    morig = fit(MixedModel, f, datSim; REML = false)
end;

# ╔═╡ a26a8327-8342-472e-841f-a9a68e5b15a8
MixedModelsPermutations.inflation_factor(morig)

# ╔═╡ 58fa0d39-65e0-4ec7-aeb0-e8bde381379a
begin
    samps = DataFrame()
    seed = Int(seed_val)

    for p = 1:50
        local m_new = deepcopy(morig)

        sample_func = MixedModelsPermutations.permute!


        sample_rng = StableRNG(seed + p)
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
        sample_func(sample_rng, m_new; β, blups, resids, scalings)

        refit!(m_new)
        push!(samps, (; σ = m_new.σ, β = m_new.β))

    end
end

# ╔═╡ 64d2509b-d94e-4e48-8f81-fa13d5f9e9be
let# convenience tuple :-)
    (
        "non parametric resampling" => samps.σ,
        "emprical σ" => morig.σ,
        "mean of resampled σ" => mean(samps.σ),
    )
end

# ╔═╡ Cell order:
# ╠═f5830c4e-a8d0-47f2-baca-bc734720b1e0
# ╠═f92470d8-c28e-4057-87bd-12383b43c158
# ╟─c7dfaa8d-dcdf-4431-98b8-c6bb9d8fac52
# ╟─2dce0f9f-0e00-46ee-8a6b-b1441e69f4f5
# ╠═a26a8327-8342-472e-841f-a9a68e5b15a8
# ╟─fe1305ba-f6bf-47ee-a89e-d7326a8dab1a
# ╟─55e97ef0-0f23-48a6-8229-7f1b6ca3d044
# ╟─2141edd4-c8ed-4d1b-a5ec-9913b19a2d4a
# ╟─bac32d2f-fb1f-4f02-b2f5-49df029b7b92
# ╠═461f6470-359e-406a-b32d-5455647e36b8
# ╠═64d2509b-d94e-4e48-8f81-fa13d5f9e9be
# ╟─a44e5f9f-5a23-46b6-adbd-228b893ac81f
# ╠═58fa0d39-65e0-4ec7-aeb0-e8bde381379a
# ╟─2a59f2be-0037-4a8f-bee6-cce32ee80407
# ╠═0cbf776c-8e56-4d32-8a55-da22f8bf1940
# ╠═cfd225bb-1eec-41b1-b6c1-ebc036b7e1d2
# ╠═cd9b3fb7-06c7-4c44-81ac-d8a4a3e0c22e
# ╠═72f5e910-e40d-11eb-3725-f19a95c9b3c1
