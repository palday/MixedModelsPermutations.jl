using MixedModels: fixef!, stderror!
using MixedModels: getθ!, setθ!, updateL!
using MixedModels: unscaledre!
using Statistics

"""
    permutation([rng::AbstractRNG,] nsamp::Integer, m::LinearMixedModel;
                           use_threads=false)

Perform `nsamp` nonparametric bootstrap replication fits of `m`, returning a `MixedModelBootstrap`.

The default random number generator is `Random.GLOBAL_RNG`.

`GeneralizedLinearMixedModel` is currently unsupported.

# Named Arguments
`use_threads` determines whether or not to use thread-based parallelism.

Note that `use_threads=true` may not offer a performance boost and may even
decrease peformance if multithreaded linear algebra (BLAS) routines are available.
In this case, threads at the level of the linear algebra may already occupy all
processors/processor cores. There are plans to provide better support in coordinating
Julia- and BLAS-level threads in the future.

`blup_method` provides options for how/which group-level effects are passed for permutation.
The default `ranef` uses the shrunken conditional modes / BLUPs. Unshrunken estimates from
ordinary least squares (OLS) can be used with `olsranef`. There is no shrinkage of the
group-level estimates with this approach, which means singular estimates can be avoided.
However, if the design matrix for the random effects is rank deficient (e.g., through the use
of `MixedModels.fulldummy` or missing cells in the data), then this method will fail.
See [`olsranef`](@ref) and `MixedModels.ranef` for more information.

!!! note
    The permutation (test) generates samples from H0, from which
    it is possible to compute p-values. The bootstrap typically generates
    samples from H1, which are convenient for generating coverage/confidence
    intervals. Of course, confidence intervals and p-values are duals of each
    other, so it is possible to convert from one to the other.

# Method

The method implemented here is based on the approach given in:

ter Braak C.J.F. (1992) Permutation Versus Bootstrap Significance Tests in
Multiple Regression and Anova. In: Jöckel KH., Rothe G., Sendler W. (eds)
Bootstrapping and Related Techniques. Lecture Notes in Economics and Mathematical
Systems, vol 376. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-642-48850-4_10

and

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014).
Permutation inference for the general linear model. NeuroImage, 92, 381–397.
https://doi.org/10.1016/j.neuroimage.2014.01.060

!!! warning
    This method has serious limitations for singular models because sign-flipping a zero
    is not an effective randomization technique.
"""
function permutation(
    rng::AbstractRNG,
    n::Integer,
    morig::LinearMixedModel{T};
    use_threads::Bool=false,
    β::AbstractVector{T}=coef(morig),
    residual_method=:signflip,
    blup_method=ranef,
) where {T}
    βsc, θsc = similar(morig.β), similar(morig.θ)
    p, k = length(βsc), length(θsc)
    m = deepcopy(morig)

    β_names = (Symbol.(fixefnames(morig))..., )
    rank = length(β_names)

    blups = blup_method(morig)
    reterms = morig.reterms

    # we need arrays of these for in-place operations to work across threads
    m_threads = [m]
    βsc_threads = [βsc]
    θsc_threads = [θsc]

    if use_threads
        Threads.resize_nthreads!(m_threads)
        Threads.resize_nthreads!(βsc_threads)
        Threads.resize_nthreads!(θsc_threads)
    end
    # we use locks to guarantee thread-safety, but there might be better ways to do this for some RNGs
    # see https://docs.julialang.org/en/v1.3/manual/parallel-computing/#Side-effects-and-mutable-function-arguments-1
    # see https://docs.julialang.org/en/v1/stdlib/Future/index.html
    rnglock = ReentrantLock()
    samp = replicate(n, use_threads=use_threads) do
        model = m_threads[Threads.threadid()]

        local βsc = βsc_threads[Threads.threadid()]
        local θsc = θsc_threads[Threads.threadid()]
        lock(rnglock)
        model = permute!(rng, model, blups, reterms;
                       β=β, residual_method=residual_method)
        unlock(rnglock)
        refit!(model)
        (
         objective = model.objective,
         σ = model.σ,
         β = NamedTuple{β_names}(fixef!(βsc, model)),
         se = SVector{p,T}(stderror!(βsc, model)),
         θ = SVector{k,T}(getθ!(θsc, model)),
        )
    end
    MixedModelPermutation(
        samp,
        deepcopy(morig.λ),
        getfield.(morig.reterms, :inds),
        copy(morig.optsum.lowerbd),
        NamedTuple{Symbol.(fnames(morig))}(map(t -> (t.cnames...,), morig.reterms)),
    )
end

function permutation(nsamp::Integer, m::LinearMixedModel; kwargs...)
    return permutation(Random.GLOBAL_RNG, nsamp, m; kwargs...)
end

function permutation(rng::AbstractRNG, n::Integer,
                                morig::GeneralizedLinearMixedModel;
                                use_threads::Bool=false) where {T}

    throw(ArgumentError("GLMM support is not yet implemented"))
end

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


permute!(model::LinearMixedModel, blups=ranef(model), reterms=model.reterms; kwargs...) =
    permute!(Random.GLOBAL_RNG, model, blups, reterms; kwargs...)

"""
    permute!([rng::AbstractRNG,] model::LinearMixedModel,
              blups=ranef(model), reterms=model.reterms;
              β=coef(model), residual_method=:signflip)

Simulate and install a new response via permutation of the residuals
at the observational level and sign-flipping of the conditional modes at group level.

Generally, permutations are used to test a particular (null) hypothesis. This
hypothesis is specified via by setting `β` argument to match the hypothesis. For
example, the null hypothesis that the first coefficient is zero would expressed as

```julialang
julia> hypothesis = coef(model);
julia> hypothesis[1] = 0.0;
```

Permutation at the level of residuals can be accomplished either via sign
flipping (`residual_method=:signflip`) or via classical
permutation/shuffling (`residual_method=:shuffle`).

Sign-flipped permutation of the residuals is similar to permuting the
(fixed-effects) design matrix; shuffling the residuals is the same as permuting the
(fixed-effects) design matrix. Sign-flipping the random effects
preserves the correlation structure of the random effects, while also being equivalent to
permutation via swapped labels for categorical variables.

!!! warning
    This method has serious limitations for singular models because sign-flipping a zero
    is not an effective randomization technique.

Optionally, instead of using the shrunken random effects from `ranef`, within-group OLS
estimates can be computed and used instead with [`olsranef`](@ref). There is no shrinkage
of the group-level estimates with this approach, which means singular estimates can be
avoided. However, if the design matrix for the random effects is rank deficient (e.g.,
through the use of `MixedModels.fulldummy` or missing cells in the data), then this method
will fail.

See also [`permutation`](@ref), [`nonparametricbootstrap`](@ref) and [`resample!`](@ref).

The functions `coef` and `coefnames` from `MixedModels` may also be useful.

# Reference
The method implemented here is based on the approach given in:

ter Braak C.J.F. (1992) Permutation Versus Bootstrap Significance Tests in
Multiple Regression and Anova. In: Jöckel KH., Rothe G., Sendler W. (eds)
Bootstrapping and Related Techniques. Lecture Notes in Economics and Mathematical
Systems, vol 376. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/978-3-642-48850-4_10

and

Winkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014).
Permutation inference for the general linear model. NeuroImage, 92, 381–397.
https://doi.org/10.1016/j.neuroimage.2014.01.060
"""
function permute!(rng::AbstractRNG, model::LinearMixedModel{T},
                   blups=ranef(model),
                   reterms=model.reterms;
                   β::AbstractVector{T}=coef(model),
                   residual_method=:signflip) where {T}

    y = response(model) # we are now modifying the model
    copy!(y, residuals(model))

    if residual_method == :shuffle
        shuffle!(rng, y)
    elseif  residual_method == :signflip
        y .*= rand(rng, (-1,1), length(y))
    else
        throw(ArgumentError("Invalid: residual permutation method: $(residual_method)"))
    end

    for (re, trm) in zip(blups, reterms)
        npreds, ngrps = size(re)
        # sign flipping
        newre = re * diagm(rand(rng, (-1,1), ngrps))

        # our RE are actually already scaled, but this method (of unscaledre!)
        # isn't dependent on the scaling (only the RNG methods are)
        MixedModels.unscaledre!(y, trm, newre)
    end

    mul!(y, model.X, β, one(T), one(T))

    # mark model as unfitted
    model.optsum.feval = -1

    return model
end


permutationtest(perm::MixedModelPermutation, model::LinearMixedModel) = permutationtest(perm::MixedModelPermutation, model, :twosided)

"""
    permutationtest(perm::MixedModelPermutation, model, type=:greater)

Perform a permutation using the already computed permutation and given the observed values.

The `type` parameter specifies use of a two-sided test (`:twosided`) or the directionality of a one-sided test
(either `:lesser` or `:greater`, depending on the hypothesized difference to the null hypothesis).

See also [`permutation`](@ref).

"""
function permutationtest(perm::MixedModelPermutation, model, type::Symbol=:twosided)
    if type == :greater || type  == :twosided
        comp = >
    elseif type == :lesser
        comp = <
    else
        throw(ArgumentError("Comparison type $(type) unsupported"))
    end

    ests = Dict(Symbol(k) => v for (k,v) in zip(coefnames(model), coef(model)))
    perms = columntable(perm.β)

    dd = Dict{Symbol, Vector}()

    for k in Symbol.(coefnames(model))
        dd[k] = perms.β[perms.coefname .== k]
        if type == :twosided
              μ = mean(dd[k])
              dd[k] .= abs.(dd[k] .- μ)
              ests[k] = abs(ests[k].- μ)
        end
    end
    results = (; (k => mean(comp(ests[k]), v) for (k,v) in dd)...)

    return results
end
