function fitsignal(formula, data, signal, contrasts)
    # fit the MixedModel

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



function exponentialCorrelation(x; nu = 1, length_ratio = 1)
    # generate exponential function
    R = length(x)*length_ratio
    return exp.(-3*(x/R).^nu)
end

function expandgrid(df1,df2)
    # get all combinations of df1&df2

    a = Array{Any}(undef, nrow(df1))
    for i = 1:nrow(df1)
        a[i] = hcat(repeat(df1[[i],:],nrow(df2)),df2)
    end
return reduce(vcat,a)
end
