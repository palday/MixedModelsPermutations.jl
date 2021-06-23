"""
    olsranef(model::LinearMixedModel, method=:simultaneous)

Compute the group-level estimates using ordinary least squares.

This is somewhat similar to the conditional modes / BLUPs computed without
shrinkage.

There is no shrinkage of the group-level estimates with this approach, which
means singular estimates can be avoided. However, this also means the random
effects design matrix must not be singular.

Two methods are provided:
1. (default) OLS estimates computed for all strata (blocking variables)
   simultaneously with `method=simultaneous`. This pools the variance
   across estimates but does not shrink the estimates.
2. OLS estimates computed within each stratum with `method=stratum`. This
   method is equivalent for example to computing each subject-level and each
   item-level regression separately.

For fully balanced designs with a single blocking variable, these methods will
give the same results.

!!! warning
    If the design matrix for the random effects is rank deficient (e.g., through
    the use of `MixedModels.fulldummy` or missing cells in the data), then only
    method will fail.
"""
function olsranef(model::LinearMixedModel{T}, method=:simultaneous) where {T}
    fixef_res = copy(response(model))
    # what's not explained by the fixed effects has to be explained by the RE
    X = model.X
    β = model.β # (X'X) \ (X'fixef_res)
    mul!(fixef_res, X, β, -one(T), one(T))

    return olsranef(model, fixef_res, Val(method))
end

function olsranef(model::LinearMixedModel{T}, fixef_res, ::Val{:stratum}) where {T}

    blups = Vector{Matrix{T}}()
    for trm in model.reterms
        re = []

        z = trm.z
        refs = unique(trm.refs)
        ngrps = length(refs)
        npreds = length(trm.cnames)

        j = 1
        for r in refs
            i = trm.refs .== r
            X = trm[i, j:(j+npreds-1)]
            b = (X'X) \ (X'fixef_res[i])
            push!(re, b)
            j += npreds
        end

        push!(blups, hcat(re...))
    end

    return blups
end


function olsranef(model::LinearMixedModel{T}, fixef_res, ::Val{:simultaneous}) where {T}
    l = size(model.reterms)[1]

    mat = Array{Any}(undef, l);
    code = Array{Any}(undef, l);
    ### I get the contrasts
    for i in 1:l
        trm = model.reterms[i];
        dim = size(trm.z)[1];
        cd = StatsModels.ContrastsMatrix(EffectsCoding(), trm.levels).matrix;
        cd = kron(cd,I(dim));
        code[i] = cd;
        mat[i] = trm*cd;
    end
    mat
    X = hcat(mat...)
    X1 = hcat(ones(size(X)[1]), X)
    fixef_res = response(model) - model.X*model.β;
    flatblups = X1'X1 \ X1'fixef_res;
    flatblups = deleteat!(flatblups,1)

    code_all = BlockDiagonal([x for x in code])

    flatblups = code_all*flatblups

    blups = Vector{Matrix{T}}()

    offset = 1
    for trm in model.reterms
        chunksize = size(trm, 2)
        ngrps = length(trm.levels)
        npreds = length(trm.cnames)
        re = Matrix{T}(reshape(view(flatblups, offset:(offset+chunksize-1)),
                               npreds, ngrps))
        offset += chunksize
        push!(blups, re)
    end
    return blups
end

"""
    residuals(model::LinearMixedModel, blups)

Compute the residuals of a mixed model using the specified group-level BLUPs/predictors.

This is useful for, e.g., comparing the residuals from a mixed-effects model with shrunken
group-level predictors against a non-shrunken classical OLS model fit within each group.
"""
function MixedModels.residuals(model::LinearMixedModel{T}, blups::Vector{<:AbstractMatrix{T}}) where T
    # XXX This is kinda type piracy, if it weren't developed by one of the MixedModels.jl devs....

    y = response(model) # we are now modifying the model

    ŷ = zeros(T, length(y))

    for (re, trm) in zip(blups, model.reterms)
        # our RE are actually already scaled, but this method (of unscaledre!)
        # isn't dependent on the scaling (only the RNG methods are)
        MixedModels.unscaledre!(ŷ, trm, re)
    end

    mul!(ŷ, model.X, model.β, one(T), one(T))

    # TODO: do this inplace to avoid an allocation
    return y .- ŷ
end
