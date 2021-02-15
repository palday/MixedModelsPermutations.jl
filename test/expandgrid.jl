function expandgrid(df1,df2)
    a = Array{Any}(undef, nrow(df1))
    for i = 1:nrow(df1)
        a[i] = hcat(repeat(df1[[i],:],nrow(df2)),df2)
    end
return reduce(vcat,a)
end
