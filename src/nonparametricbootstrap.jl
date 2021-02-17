"""
    nonparametricbootstrap([rng::AbstractRNG,] nsamp::Integer, m::LinearMixedModel;
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


`blup_method` provides options for how/which group-level effects are passed for resampling.
The default `ranef` uses the shrunken conditional modes / BLUPs. Unshrunken estimates from
ordinary least squares (OLS) can be used with `olsranef`. There is no shrinkage of the
group-level estimates with this approach, which means singular estimates can be avoided.
However, if the design matrix for the random effects is rank deficient (e.g., through the use
of `MixedModels.fulldummy` or missing cells in the data), then this method will fail.
See [`olsranef`](@ref) and `MixedModels.ranef` for more information.

# Method

The method implemented here is based on the approach given in Section 3.2 of:
Carpenter, J.R., Goldstein, H. and Rasbash, J. (2003).
A novel bootstrap procedure for assessing the relationship between class size and achievement.
Journal of the Royal Statistical Society: Series C (Applied Statistics), 52: 431-443.
https://doi.org/10.1111/1467-9876.00415
"""
function nonparametricbootstrap(
    rng::AbstractRNG,
    n::Integer,
    morig::LinearMixedModel{T};
    use_threads::Bool=false,
    blup_method=ranef,
) where {T}
    # XXX should we allow specifying betas and blups?
    #     if so, should we use residuals computed based on those or the observed ones?
    βsc, θsc = similar(morig.β), similar(morig.θ)
    p, k = length(βsc), length(θsc)
    m = deepcopy(morig)

    β_names = (Symbol.(fixefnames(morig))..., )
    rank = length(β_names)

    blups = blup_method(morig)
    reterms = morig.reterms
    yorig = copy(response(morig))
    θorig = morig.θ
    scalings = inflation_factor(morig)
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
    samp = replicate(n; use_threads=use_threads) do
        mod = m_threads[Threads.threadid()]
        #copy!(mod.y, yorig)
        #updateL!(setθ!(mod, θorig))
        # XXX
        refit!(mod, yorig)
        local βsc = βsc_threads[Threads.threadid()]
        local θsc = θsc_threads[Threads.threadid()]
        lock(rnglock)
        mod = resample!(rng, mod, blups, reterms, scalings)
        unlock(rnglock)
        refit!(mod)
        (
         objective = mod.objective,
         σ = mod.σ,
         β = NamedTuple{β_names}(fixef!(βsc, mod)),
         se = SVector{p,T}(stderror!(βsc, mod)),
         θ = SVector{k,T}(getθ!(θsc, mod)),
        )
    end
    MixedModelBootstrap(
        samp,
        deepcopy(morig.λ),
        getfield.(morig.reterms, :inds),
        copy(morig.optsum.lowerbd),
        NamedTuple{Symbol.(fnames(morig))}(map(t -> (t.cnames...,), morig.reterms)),
    )
end

function nonparametricbootstrap(nsamp::Integer, m::LinearMixedModel; kwargs...)
    return nonparametricbootstrap(Random.GLOBAL_RNG, nsamp, m; kwargs...)
end

function nonparametricbootstrap(rng::AbstractRNG, n::Integer,
                                morig::GeneralizedLinearMixedModel;
                                use_threads::Bool=false) where {T}

    throw(ArgumentError("GLMM support is not yet implemented"))
end

resample!(mod::LinearMixedModel, blups=ranef(mod), reterms=mod.reterms) =
    resample!(Random.GLOBAL_RNG, mod, blups, reterms)

"""
    resample!([rng::AbstractRNG,] mod::LinearMixedModel,
              blups=ranef(mod), reterms=mod.reterms)

Simulate and install a new response using resampling at the observational and group level.

At both levels, resampling is done with replacement. At the observational level,
this is resampling of the residuals, i.e. comparable to the step in the classical
nonparametric bootstrap for OLS regression. At the group level, samples are done
jointly for all  terms ("random intercept and random slopes", conditional modes)
associated with a particular level of a blocking factor. For example, the
predicted slope and intercept for a single participant or item are kept together.
This clumping in the resampling procedure is necessary to preserve the original
correlation structure of the slopes and intercepts.

In addition to this resampling step, there is also an inflation step. Due to the
shrinkage associated with the random effects in a mixed model, the variance of the
conditional modes / BLUPs / random intercepts and slopes is less than the variance
estimated by the model and displayed in the model summary or via `MixedModels.VarCorr`.
This shrinkage also impacts the observational level residuals. To compensate for this,
the resampled residuals and groups are scale-inflated so that their standard deviation
matches that of the estimates in the original model.

See also [`nonparametricbootstrap`](@ref) and `MixedModels.simulate!`.

!!! warning
    This method has serious limitations for singular models because resampling from
    a distribution with many zeros (e.g. the random effects components with zero variance)
    will often generate new data with even less variance.

# Reference
The method implemented here is based on the approach given in Section 3.2 of:
Carpenter, J.R., Goldstein, H. and Rasbash, J. (2003).
A novel bootstrap procedure for assessing the relationship between class size and achievement.
Journal of the Royal Statistical Society: Series C (Applied Statistics), 52: 431-443.
https://doi.org/10.1111/1467-9876.00415
"""
function resample!(rng::AbstractRNG, mod::LinearMixedModel{T},
                   blups=ranef(mod),
                   reterms=mod.reterms,
                   scalings=inflation_factor(mod)) where {T}
    β = coef(mod)
    y = response(mod) # we are now modifying the model
    res = residuals(mod, blups)

    sample!(rng, res, y; replace=true)
    # inflate these to be on the same scale as the empirical variation instead of the MLE
    y .*= first(scalings)

    for (inflation, re, trm) in zip(scalings[2:end], blups, reterms)
        npreds, ngrps = size(re)
        samp = sample(rng, 1:ngrps, ngrps; replace=true)

        newre = view(re, :, samp)
        # our RE are actually already scaled, but this method (of unscaledre!)
        # isn't dependent on the scaling (only the RNG methods are)
        # this just multiplies the Z matrices by the BLUPs
        # and add that to y
        MixedModels.unscaledre!(y, trm, inflation * newre)
    end

    mul!(y, mod.X, β, one(T), one(T))

    # mark model as unfitted
    mod.optsum.feval = -1

    return mod
end

function inflation_factor(m::LinearMixedModel)
    σ = sdest(m)
    σres = std(residuals(m); corrected=false)
    inflation = map(zip(m.reterms, ranef(m))) do (trm, re)
        # inflation
        λmle =  trm.λ * σ                               # L_R in CGR
        λemp = cholesky(cov(re'; corrected=false)).L    # L_S in CGR
        # no transpose because the RE are transposed relativ to CGR
        λmle / λemp
    end

    return [σ / σres; inflation]
end
