### A Pluto.jl notebook ###
# v0.17.3

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
    Pkg.add([
        "PlutoUI",
        "MixedModelsSim",
        "MixedModels",
        "DataFrames",
        "Plots",
        "StableRNGs",
    ])
end

# ╔═╡ b7ac3efc-8bda-41d9-b3d5-147591bac4b4
begin
    using PlutoUI
    using MixedModelsSim
    using MixedModelsPermutations
    using MixedModels
    using DataFrames
    using Plots
    using StableRNGs
    using LinearAlgebra
    using Statistics
end

# ╔═╡ 8ad1fcc6-2e49-4a67-9fae-a146e1baccce
pwd()

# ╔═╡ 0eaf2824-dd65-4dfb-905e-96d8a1754e08
begin
    contrasts = Dict(
        :age => EffectsCoding,
        :stimType => EffectsCoding(),
        :condition => EffectsCoding(),
    )
end

# ╔═╡ 0cbf776c-8e56-4d32-8a55-da22f8bf1940
f = @formula(dv ~ 1 + condition + (1 + condition | subj));

# ╔═╡ 3f4afdc8-04af-45b6-aadf-a72ca1ff158c
blup_method = olsranef;

# ╔═╡ 6917ddd6-6ef3-43a5-83f1-ef6583be54f3
residual_method = :signflip;

# ╔═╡ cbecd90f-b262-4bbd-8a55-177f3e40ae50
β = [1.0, 1.0]

# ╔═╡ 55e97ef0-0f23-48a6-8229-7f1b6ca3d044
md"seed $(@bind seed_val Slider(1:100; default=1., show_value=true))"

# ╔═╡ c7dfaa8d-dcdf-4431-98b8-c6bb9d8fac52
rng = StableRNG(Int(seed_val))

# ╔═╡ 2141edd4-c8ed-4d1b-a5ec-9913b19a2d4a
md"Subjects: $(@bind nsub Slider(2:2:50; default=30, show_value=true))"


# ╔═╡ bac32d2f-fb1f-4f02-b2f5-49df029b7b92
md"Items per Subject: $(@bind nitem Slider(2:2:50; default=30, show_value=true))"

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

# ╔═╡ eccf3c09-aa58-4c7c-8e43-364309df6885
@bind bootPerm Radio(["0" => "Bootstrap", "1" => "Permutation"], "1")

# ╔═╡ e132bb53-9b6c-4d61-bb45-d5d9209ff4da
@bind reml Radio([
    "0" => "ML",
    "1" => "REML",
    #"3"=>"experimental"
], "1")

# ╔═╡ acc913dd-b17c-4137-83f4-7df179bdc3a1
@bind res Radio(["1" => "residuals", "2" => "residuals_from_blups"], "1")

# ╔═╡ 968e3f53-055c-4dc0-b4e3-1217e511b3b4
# radio button to function
begin
    if res == "1"
        residual_function = residuals
    elseif res == "2"
        residual_function = MixedModelsPermutations.residuals_from_blups
    else
        error("wtf, mate")
    end
end;

# ╔═╡ d35213d3-75e4-45fb-880c-713d8caaf9d7
@bind ols Radio(
    [
        "stratum" => "stratum",
        "simultaneous" => "simultaneous",
        "inflated_identity" => "inflated_identity",
    ],
    "stratum",
)

# ╔═╡ 461f6470-359e-406a-b32d-5455647e36b8
@bind inflate Radio(
    [
        "ols" => "inflation_factor from OLS blups",
        "yes" => "inflation_factor from original blups/residuals",
        "no" => "identity",
    ],
    "ols",
)

# ╔═╡ 2776e7cf-ee8f-421e-98ed-fa03ecdbee68
md"σ-intercept: $(@bind σ1 Slider(0:0.1:5; default=0., show_value=true))"

# ╔═╡ 629d660c-6a13-422c-8acc-6c3b8d37acaa
md"σ-test: $(@bind σ2 Slider(0:0.1:10; default=4., show_value=true))"

# ╔═╡ 21664a88-30fd-4829-a70d-e727a5f5b66b
σs = [σ1, σ2, 0.2]

# ╔═╡ cd9b3fb7-06c7-4c44-81ac-d8a4a3e0c22e
# generate simulated "y" vector
begin
    simMod = fit(MixedModel, f, dat; contrasts)
    simMod = MixedModelsSim.update!(simMod, [create_re(x...) for x in σs]...)
    simMod = simulate!(rng, simMod; β, σ = 1.0)
end;

# ╔═╡ cfd225bb-1eec-41b1-b6c1-ebc036b7e1d2
begin
    # fit a LMM mod to the y (separate models to maybe have separate formulas later
    datSim = DataFrame(dat)
    datSim[!, :dv] .= response(simMod)
    mlfit = fit(MixedModel, f, datSim; contrasts, REML = false)
    remlfit = fit(MixedModel, f, datSim; contrasts, REML = true)
end;

# ╔═╡ 35ce6722-4a07-4bc1-a7ec-9ab2562edc1a
simMod2 = reml == "0" ? mlfit : remlfit

# ╔═╡ dc9c416c-855c-4894-8e38-cda1c3297472
if reml == "0"
    ifactor = MixedModelsPermutations.inflation_factor(simMod2)
else
    # ???
    ifactor = [I, I, simMod2.σ]
end

# ╔═╡ ac176d08-e147-4e37-9904-beffd9813a91
[ll * ll' for ll in ifactor[1:end-1]]

# ╔═╡ 58fa0d39-65e0-4ec7-aeb0-e8bde381379a
begin
    morig = simMod2
    m_new = deepcopy(morig)
    samps = DataFrame()
    seed = Int(seed_val)

    for p = 1:100
        # this could be made more efficient
        m_new = deepcopy(morig)
        if bootPerm == "0"
            sample_func = MixedModelsPermutations.resample!
        else
            sample_func = MixedModelsPermutations.permute!
        end

        sample_rng = StableRNG(seed + p)
        blups = olsranef(m_new, Symbol(ols))
        resids = residual_function(m_new, blups)
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
        # copyto!(m_new.y, morig.y)
        # MixedModels.updateL!(MixedModels.setθ!(m_new, morig.optsum.final))
    end
end

# ╔═╡ 64d2509b-d94e-4e48-8f81-fa13d5f9e9be
let
    Plots.plot(samps.σ, label = "non parametric resampling")
    hline!([1.0], label = "theoretical σ")
    hline!([simMod2.σ], label = "empirical σ")
    hline!([mean(samps.σ)], label = "mean of resampled σ")
    ylims!((1.0 * 0.8, 1.0 * 1.2))
    ylabel!("residual σ")
    xlabel!("iteration")
end

# ╔═╡ a26a8327-8342-472e-841f-a9a68e5b15a8
last(MixedModelsPermutations.inflation_factor(remlfit))

# ╔═╡ 4d6a2335-5e63-491f-be9c-1072d61291f9
remlfit.σ / mlfit.σ

# ╔═╡ Cell order:
# ╠═8ad1fcc6-2e49-4a67-9fae-a146e1baccce
# ╠═f92470d8-c28e-4057-87bd-12383b43c158
# ╟─b7ac3efc-8bda-41d9-b3d5-147591bac4b4
# ╟─c7dfaa8d-dcdf-4431-98b8-c6bb9d8fac52
# ╟─72f5e910-e40d-11eb-3725-f19a95c9b3c1
# ╟─0eaf2824-dd65-4dfb-905e-96d8a1754e08
# ╟─cd9b3fb7-06c7-4c44-81ac-d8a4a3e0c22e
# ╠═cfd225bb-1eec-41b1-b6c1-ebc036b7e1d2
# ╠═35ce6722-4a07-4bc1-a7ec-9ab2562edc1a
# ╠═a26a8327-8342-472e-841f-a9a68e5b15a8
# ╠═4d6a2335-5e63-491f-be9c-1072d61291f9
# ╠═dc9c416c-855c-4894-8e38-cda1c3297472
# ╠═ac176d08-e147-4e37-9904-beffd9813a91
# ╠═58fa0d39-65e0-4ec7-aeb0-e8bde381379a
# ╠═968e3f53-055c-4dc0-b4e3-1217e511b3b4
# ╠═0cbf776c-8e56-4d32-8a55-da22f8bf1940
# ╠═3f4afdc8-04af-45b6-aadf-a72ca1ff158c
# ╠═6917ddd6-6ef3-43a5-83f1-ef6583be54f3
# ╠═cbecd90f-b262-4bbd-8a55-177f3e40ae50
# ╠═21664a88-30fd-4829-a70d-e727a5f5b66b
# ╟─55e97ef0-0f23-48a6-8229-7f1b6ca3d044
# ╟─2141edd4-c8ed-4d1b-a5ec-9913b19a2d4a
# ╟─bac32d2f-fb1f-4f02-b2f5-49df029b7b92
# ╟─eccf3c09-aa58-4c7c-8e43-364309df6885
# ╟─e132bb53-9b6c-4d61-bb45-d5d9209ff4da
# ╟─acc913dd-b17c-4137-83f4-7df179bdc3a1
# ╟─d35213d3-75e4-45fb-880c-713d8caaf9d7
# ╠═461f6470-359e-406a-b32d-5455647e36b8
# ╠═64d2509b-d94e-4e48-8f81-fa13d5f9e9be
# ╟─2776e7cf-ee8f-421e-98ed-fa03ecdbee68
# ╟─629d660c-6a13-422c-8acc-6c3b8d37acaa
