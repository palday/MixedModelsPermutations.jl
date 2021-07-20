using MixedModels: fixef!, stderror!
using MixedModels: getθ!, setθ!, updateL!
using LinearAlgebra
using Statistics

"""
    permutation([rng::AbstractRNG,] nsamp::Integer, m::LinearMixedModel;
                use_threads::Bool=false,
                β=zeros(length(coef(morig))),
                residual_method=:signflip,
                blup_method=ranef)

Perform `nsamp` nonparametric bootstrap replication fits of `m`, returning a `MixedModelBootstrap`.

The default random number generator is `Random.GLOBAL_RNG`.

`GeneralizedLinearMixedModel` is currently unsupported.

# Named Arguments
`use_threads` determines whether or not to use thread-based parallelism.

!!! note
    Note that `use_threads=true` may not offer a performance boost and may even
    decrease peformance if multithreaded linear algebra (BLAS) routines are available.
    In this case, threads at the level of the linear algebra may already occupy all
    processors/processor cores. There are plans to provide better support in coordinating
    Julia- and BLAS-level threads in the future.

!!! warning
    The PRNG shared between threads is locked using [`Threads.SpinLock`](@ref), which
    should not be used recursively. Do not wrap `permutation` in an outer `SpinLock`.

`hide_progress` can be used to disable the progress bar. Note that the progress
bar is automatically disabled for non-interactive (i.e. logging) contexts.

Permutation at the level of residuals can be accomplished either via sign
flipping (`residual_method=:signflip`) or via classical
permutation/shuffling (`residual_method=:shuffle`).

`blup_method` provides options for how/which group-level effects are passed for permutation.
The default `ranef` uses the shrunken conditional modes / BLUPs. Unshrunken estimates from
ordinary least squares (OLS) can be used with `olsranef`. There is no shrinkage of the
group-level estimates with this approach, which means singular estimates can be avoided.
However, if the design matrix for the random effects is rank deficient (e.g., through the use
of `MixedModels.fulldummy` or missing cells in the data), then this method will fail.
See [`olsranef`](@ref) and `MixedModels.ranef` for more information.

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
    use_threads::Bool=false,
    hide_progress=false,
    β::AbstractVector{T}=zeros(T, length(coef(morig))),
    residual_method=:signflip,
    blup_method=ranef,
) where {T}
    # XXX instead of straight zeros,
    #     should we use 1-0s for intercept only?
    βsc, θsc = similar(morig.β), similar(morig.θ)
    p, k = length(βsc), length(θsc)
    m = deepcopy(morig)

    β_names = (Symbol.(fixefnames(morig))..., )
    rank = length(β_names)

    blups = blup_method(morig)
    resids = residuals(morig)#, blups)
    reterms = morig.reterms
    scalings = inflation_factor(morig, blups, resids)
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
    rnglock = Threads.SpinLock()
    samp = replicate(n; use_threads=use_threads, hide_progress=hide_progress) do
        tidx = use_threads ? Threads.threadid() : 1
        model = m_threads[tidx]
        local βsc = βsc_threads[tidx]
        local θsc = θsc_threads[tidx]
        lock(rnglock)
        model = permute!(rng, model; β=β, blups=blups, resids=resids,
                         residual_method=residual_method, scalings=scalings)
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

function permutation(nsamp::Integer, m::LinearMixedModel, args...; kwargs...)
    return permutation(Random.GLOBAL_RNG, nsamp, m, args...; kwargs...)
end

function permutation(rng::AbstractRNG, n::Integer,
                                morig::GeneralizedLinearMixedModel;
                                use_threads::Bool=false) where {T}

    throw(ArgumentError("GLMM support is not yet implemented"))
end


permute!(model::LinearMixedModel, args...; kwargs...) =
    permute!(Random.GLOBAL_RNG, model, args...; kwargs...)

"""
    permute!([rng::AbstractRNG,] model::LinearMixedModel;
             β=zeros(length(coef(model))),
             blups=ranef(model),
             resids=residuals(model,blups),
             residual_method=:signflip,
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
                  residual_method=:signflip,
                  scalings=inflation_factor(model)) where {T}

    reterms = model.reterms
    y = response(model) # we are now modifying the model
    copy!(y, resids)

    # inflate these to be on the same scale as the empirical variation instead of the MLE
    y .*= last(scalings)

    if residual_method == :shuffle
        shuffle!(rng, y)
    elseif  residual_method == :signflip
        y .*= rand(rng, (-1,1), length(y))
    else
        throw(ArgumentError("Invalid: residual permutation method: $(residual_method)"))
    end

    for (inflation, re, trm) in zip(scalings, blups, reterms)
        npreds, ngrps = size(re)
        # sign flipping
        newre = re * diagm(rand(rng, (-1,1), ngrps))

        # this just multiplies the Z matrices by the BLUPs
        # and add that to y
        mul!(y, trm, lmul!(inflation, newre), one(T), one(T))
        # XXX inflation is resampling invariant -- should we move it out?
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
    @warn """This method is known not to be fully correct.
             The interface for this functionality will likely change drastically in the near future."""

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
