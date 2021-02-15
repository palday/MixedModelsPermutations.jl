function exponentialCorrelation(x; nu = 1, length_ratio = 1)
    R = length(x)*length_ratio
    return exp.(-3*(x/R).^nu)
end
