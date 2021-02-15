function fitsignal(formula, data, signal, contrasts)
    fits = Array{Any}(undef, size(signal)[2])
    mod = Array{Any}(undef,1)
    cdata = copy(data)

    for i = 1:(size(signal)[2])
        println(i)
        if i==1
            cdata[:,formula.lhs.sym] = (signal[:,i])
            mod[1] = MixedModels.fit(MixedModel, formula, cdata, contrasts = contrasts)
        else
            mod[1] = refit!(mod[1],signal[:,i])
        end
        fits[i] = deepcopy(mod[1])
    end
    return fits
end
