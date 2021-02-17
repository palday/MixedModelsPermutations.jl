"""
    olsranef(model::LinearMixedModel)

Compute the group-level estimates using ordinary least squares.

This is somewhat similar to the conditional modes / BLUPs computed without shrinkage.

Optionally, instead of using the shrunken random effects from `ranef`, within-group OLS
estimates can be computed and used instead with [`olsranef`](@ref). There is no shrinkage
of the group-level estimates with this approach, which means singular estimates can be
avoided. However,

!!! warning
    If the design matrix for the random effects is rank deficient (e.g., through the use of
    `MixedModels.fulldummy` or missing cells in the data), then this method will fail.
"""
function olsranef(model::LinearMixedModel{T}) where {T}
    fixef_res = copy(response(model))
    # what's not explained by the fixed effects has to be explained by the RE
    X = model.X
    β = model.β # (X'X) \ (X'fixef_res)
    mul!(fixef_res, X, β, one(T), -one(T))

    blups = Vector{Matrix{Float64}}()
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
