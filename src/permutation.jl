using MixedModels: fixef!, stderror!
using MixedModels: getθ!, setθ!, updateL!
using LinearAlgebra
using Statistics

"""
    permutation([rng::AbstractRNG,] nsamp::Integer, m::LinearMixedModel;
                use_threads::Bool=false,
                β=zeros(length(coef(morig))),
                residual_permutation=:signflip,
                blup_method=ranef,
                residual_method=residuals)

Perform `nsamp` nonparametric bootstrap replication fits of `m`, returning a `MixedModelBootstrap`.

The default random number generator is `Random.GLOBAL_RNG`.

`GeneralizedLinearMixedModel` is currently unsupported.

# Named Arguments

`progress=false` can be used to disable the progress bar. Note that the progress
bar is automatically disabled for non-interactive (i.e. logging) contexts.

Permutation at the level of residuals can be accomplished either via sign
flipping (`residual_permutation=:signflip`) or via classical
permutation/shuffling (`residual_permutation=:shuffle`).

`blup_method` provides options for how/which group-level effects are passed for permutation.
The default `ranef` uses the shrunken conditional modes / BLUPs. Unshrunken estimates from
ordinary least squares (OLS) can be used with `olsranef`. There is no shrinkage of the
group-level estimates with this approach, which means singular estimates can be avoided.
However, if the design matrix for the random effects is rank deficient (e.g., through the use
of `MixedModels.fulldummy` or missing cells in the data), then this method will fail.
See [`olsranef`](@ref) and `MixedModels.ranef` for more information.

`residual_method` provides options for how observation-level residuals are passed for permutation.
This should be a two-argument function, taking the model and the BLUPs (as computed with `blup_method`)
as arguments. If you wish to ignore the BLUPs as computed with `blup_method`, then you still need
the second argument, but you can simply not use it in your function.

`inflation_method` is a three-argument function (model, BLUPs as computed by `blup_method`,
residuals computed by `residual_method`) for computing the inflation factor passed onto [`permute!`](@ref).

Generally, permutations are used to test a particular (null) hypothesis. This
hypothesis is specified via by setting `β` argument to match the hypothesis. For
example, the null hypothesis that the first coefficient is zero would expressed as

```julialang
julia> hypothesis = coef(model);
julia> hypothesis[1] = 0.0;
```
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
    progress=true,
    β::AbstractVector{T}=zeros(T, length(coef(morig))),
    residual_permutation=:signflip,
    residual_method=residuals,
    blup_method=ranef,
    inflation_method=inflation_factor,
) where {T}
    # XXX instead of straight zeros,
    #     should we use 1-0s for intercept only?
    βsc, θsc = similar(morig.β), similar(morig.θ)
    p, k = length(βsc), length(θsc)
    model = deepcopy(morig)

    β_names = (Symbol.(fixefnames(morig))..., )
    rank = length(β_names)

    blups = blup_method(morig)
    resids = residual_method(morig, blups)
    reterms = morig.reterms
    scalings = inflation_method(morig, blups, resids)
    samp = replicate(n; progress) do
        model = permute!(rng, model; β=β, blups=blups, resids=resids,
                         residual_permutation=residual_permutation, scalings=scalings)
        refit!(model; progress=false)
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

function permutation(nsamp::Integer, m::LinearMixedModel, args...; kwargs...)
    return permutation(Random.GLOBAL_RNG, nsamp, m, args...; kwargs...)
end

function permutation(::AbstractRNG, ::Integer, ::GeneralizedLinearMixedModel; kwargs...)
    throw(ArgumentError("GLMM support is not yet implemented"))
end


permute!(model::LinearMixedModel, args...; kwargs...) =
    permute!(Random.GLOBAL_RNG, model, args...; kwargs...)

"""
    permute!([rng::AbstractRNG,] model::LinearMixedModel;
             β=zeros(length(coef(model))),
             blups=ranef(model),
             resids=residuals(model,blups),
             residual_permutation=:signflip,
             scalings=inflation_factor(model))

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
flipping (`residual_permutation=:signflip`) or via classical
permutation/shuffling (`residual_permutation=:shuffle`).

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

In addition to the permutation step, there is also an inflation step. Due to the
shrinkage associated with the random effects in a mixed model, the variance of the
conditional modes / BLUPs / random intercepts and slopes is less than the variance
estimated by the model and displayed in the model summary or via `MixedModels.VarCorr`.
This shrinkage also impacts the observational level residuals. To compensate for this,
the resampled residuals and groups are scale-inflated so that their standard deviation
matches that of the estimates in the original model. The default inflation factor is
computed using [`inflation_factor`](@ref) on the model.

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
function permute!(rng::AbstractRNG, model::LinearMixedModel{T};
                  β::AbstractVector{T}=zeros(T, length(coef(model))),
                  blups=ranef(model),
                  resids=residuals(model,blups),
                  residual_permutation=:signflip,
                  scalings=inflation_factor(model)) where {T}

    reterms = model.reterms
    y = response(model) # we are now modifying the model
    copy!(y, resids)

    # inflate these to be on the same scale as the empirical variation instead of the MLE
    y .*= last(scalings)

    if residual_permutation == :shuffle
        shuffle!(rng, y)
    elseif  residual_permutation == :signflip
        y .*= rand(rng, (-1,1), length(y))
    else
        throw(ArgumentError("Invalid: residual permutation method: $(residual_permutation)"))
    end

    for (inflation, re, trm) in zip(scalings, blups, reterms)
        npreds, ngrps = size(re)
        # sign flipping
        newre = re * diagm(rand(rng, (-1,1), ngrps))

        # this just multiplies the Z matrices by the BLUPs
        # and add that to y
        mul!(y, trm, inflation*newre, one(T), one(T))
        # XXX inflation is resampling invariant -- should we move it out?
    end

    mul!(y, model.X, β, one(T), one(T))

    # mark model as unfitted
    model.optsum.feval = -1

    return model
end


"""
    permutationtest(perm::MixedModelPermutation, model, type=:greater)

Perform a permutation using the already computed permutation and given the observed values.

The `type` parameter specifies use of a two-sided test (`:twosided`) or the directionality of a one-sided test
(either `:lesser` or `:greater`, depending on the hypothesized difference to the null hypothesis).

See also [`permutation`](@ref).

To account for finite permutations, we implemented the conservative method from Phipson & Smyth 2010:
 Permutation P-values Should Never Be Zero:Calculating Exact P-values When Permutations Are Randomly Drawn
 http://www.statsci.org/webguide/smyth/pubs/permp.pdf 

"""
function permutationtest(perm::MixedModelPermutation, model; type::Symbol=:twosided,β::AbstractVector=zeros(length(coef(model))), statistic=:z)
    #@warn """This method is known not to be fully correct.
    #         The interface for this functionality will likely change drastically in the near future."""
    # removed due to distributed run

    if type == :greater || type  == :twosided
        comp = >=
    elseif type == :lesser
        comp = <=
    else
        throw(ArgumentError("Comparison type $(type) unsupported"))
    end
    if statistic == :z
        x = coeftable(model)
        ests = Dict(Symbol(k) => v for (k,v) in zip(coefnames(model), x.cols[x.teststatcol]))
    elseif statistic == :β
        ests = Dict(Symbol(k) => v for (k,v) in zip(coefnames(model), coef(model)))
    else
        error("statistic not implemented yet")
    end

    perms = columntable(perm.coefpvalues)

    dd = Dict{Symbol, Vector}()

    for (ix,k) in enumerate(Symbol.(coefnames(model)))
        dd[k] = perms[statistic][perms.coefname .== k]


        push!(dd[k],ests[k]) # simplest approximation to ensure p is never 0 (impossible for permutation test)
        if type == :twosided
            # in case of testing the betas, H0 might be not β==0, therefore we have to remove it here first before we can abs
            # the "z's" are already symmetric around 0 regardless of hypothesis.
            if statistic == :β
                #println(β[ix])
                dd[k]  .= dd[k]  .- β[ix]
                ests[k] = ests[k] - β[ix]
            end

              dd[k]  .= abs.(dd[k])
              ests[k] = abs(ests[k])
        end


    end

    # short way to calculate:
    # b = sum.(abs.(permDist).>=abs.(testValue)); (twosided)
    # Includes the conservative correction for approximate permutation tests
    # p_t = (b+1)/(nperm+1);

    # (with comp being <=) Note that sum(<=(ests),v) does the same as  sum(v .<=ests) (thus "reversed" arguments in the first bracket)
    results = (; (k=> sum(comp(ests[k]),v)/length(v) for (k,v) in dd)...)
    #results = (; (k => (1+sum(comp(ests[k]),v))/(1+length(v)) for (k,v) in dd)...)

    return results
end
