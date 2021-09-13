using Distributed
using MixedModelsSim, MixedModels,MixedModelsPermutations
using ProgressMeter
using SharedArrays
function sim_model(f)
    subj_btwn = Dict("age" => ["O", "Y"])

    # there are no between-item factors in this design so you can omit it or set it to nothing
    item_btwn = Dict("stimType" => ["I", "II"])

    # put within-subject/item factors in a Dict
    both_win = Dict("condition" => ["A", "B"])

    # simulate data
    dat = simdat_crossed(
        30,
        30,
        subj_btwn = subj_btwn,
        item_btwn = item_btwn,
        both_win = both_win,
    )


    simMod = MixedModels.fit(MixedModel, f, dat)

    return simMod

end
function run_permutationtest_distributed(n_workers, nRep, simMod,args...)

    if nworkers() < n_workers
        # open as many as necessary
        println("Starting Workers, this might take some time")
        addprocs(
            n_workers - nworkers() + 1,
            exeflags = "--project",
            enable_threaded_blas = true,
        )
    end

    # activate environment
    eval(macroexpand(Distributed, quote
        @everywhere using Pkg
    end))

    @everywhere Pkg.activate(".")
    # load packages on distributed
    eval(
        macroexpand(
            Distributed,
            quote
                @everywhere using MixedModelsSim,
                    Random, MixedModels, MixedModelsPermutations
            end,
        ),
    )
    β_permResult = SharedArray{Float64}(nRep, length(β))
    z_permResult = SharedArray{Float64}(nRep, length(β))

    @everywhere include("test/sim_utilities.jl")

    println("starting @distributed")
    # parallel loop
    @showprogress @distributed for k = 1:nRep
        #println("Thread "*string(Threads.threadid()) * "\t Running "*string(k))
        #res = [1,1.]#
        res = run_permutationtest(MersenneTwister(k), deepcopy(simMod),args...)
        β_permResult[k, :] .= res[1]
        z_permResult[k, :] .= res[2]

    end
    return β_permResult, z_permResult

end
function run_permutationtest(rng, simMod, nPerm, β, σ, θ)
    simMod = simulate!(rng, simMod, β = β, σ = σ, θ = θ)
    refit!(simMod)
    H0 = coef(simMod)
    H0[2] = 0.0
    #H0[1] = 0.0
    perm = permutation(rng, nPerm, simMod, use_threads = false; β = H0)
    p_β = values(permutationtest(perm, simMod; statistic = :β))
    p_z = values(permutationtest(perm, simMod; statistic = :z))
    return (p_β, p_z)
end


function fitsignal(formula, data, signal, contrasts)
    # fit the MixedModel

    fits = Array{Any}(undef, size(signal)[2])
    model = Array{Any}(undef, 1)
    cdata = copy(data)

    for i = 1:(size(signal)[2])
        println(i)
        if i == 1
            cdata[:, formula.lhs.sym] = (signal[:, i])
            model[1] = MixedModels.fit(MixedModel, formula, cdata, contrasts = contrasts)
        else
            model[1] = refit!(model[1], signal[:, i])
        end
        fits[i] = deepcopy(model[1])
    end
    return fits
end




function circulant(x)
    # returns a symmetric matrix where X was circ-shifted.
    lx = length(x)
    ids = [1:1:(lx-1);]
    a = Array{Float64,2}(undef, lx,lx)
    for i = 1:length(x)
        if i==1
            a[i,:] = x
        else
            a[i,:] = vcat(x[i],a[i-1,ids])
        end
    end
    return Symmetric(a)
end



function circulant(x)
    # returns a symmetric matrix where X was circ-shifted.
    lx = length(x)
    ids = [1:1:(lx-1);]
    a = Array{Float64,2}(undef, lx, lx)
    for i = 1:length(x)
        if i == 1
            a[i, :] = x
        else
            a[i, :] = vcat(x[i], a[i-1, ids])
        end
    end
    return Symmetric(a)
#endurn reduce(vcat,a)
end


function exponentialCorrelation(x; nu = 1, length_ratio = 1)
    # generate exponential function
    R = length(x) * length_ratio
    return exp.(-3 * (x / R) .^ nu)
end

function expandgrid(df1, df2)
    # get all combinations of df1&df2

    a = Array{Any}(undef, nrow(df1))
    for i = 1:nrow(df1)
        a[i] = hcat(repeat(df1[[i], :], nrow(df2)), df2)
    end
    return reduce(vcat, a)
end
