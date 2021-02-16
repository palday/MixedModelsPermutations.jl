using MixedModels: fixef!, stderror!
using MixedModels: getθ!, setθ!, updateL!
using MixedModels: unscaledre!
using Statistics

"""
    nonparametricbootstrap([rng::AbstractRNG,] nsamp::Integer, m::LinearMixedModel;
                           use_threads=false)

Perform `nsamp` nonparametric bootstrap replication fits of `m`, returning a `MixedModelBootstrap`.

The default random number generator is `Random.GLOBAL_RNG`.

# Named Arguments

`β`, `σ`, and `θ` are the values of `m`'s parameters for simulating the responses.
`σ` is only valid for `LinearMixedModel`.
`GeneralizedLinearMixedModel` is currently unsupported.
`use_threads` determines whether or not to use thread-based parallelism.

Note that `use_threads=true` may not offer a performance boost and may even
decrease peformance if multithreaded linear algebra (BLAS) routines are available.
In this case, threads at the level of the linear algebra may already occupy all
processors/processor cores. There are plans to provide better support in coordinating
Julia- and BLAS-level threads in the future.

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
) where {T}
    β::AbstractVector=coef(morig)
    θ = morig.θ

    βsc, θsc, p, k = similar(β), similar(θ), length(β), length(θ)
    m = deepcopy(morig)

    β_names = (Symbol.(fixefnames(morig))..., )
    rank = length(β_names)

    blups = ranef(morig; uscale=false)
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
        mod = m_threads[Threads.threadid()]

        local βsc = βsc_threads[Threads.threadid()]
        local θsc = θsc_threads[Threads.threadid()]
        lock(rnglock)
        mod = resample!(rng, mod, blups, reterms)
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
    resample!(GLOBAL_RNG, mod, blups, reterms)

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

# Reference
The method implemented here is based on the approach given in Section 3.2 of:
Carpenter, J.R., Goldstein, H. and Rasbash, J. (2003).
A novel bootstrap procedure for assessing the relationship between class size and achievement.
Journal of the Royal Statistical Society: Series C (Applied Statistics), 52: 431-443.
https://doi.org/10.1111/1467-9876.00415

"""
function resample!(rng::AbstractRNG, mod::LinearMixedModel,
                   blups=ranef(mod),
                   reterms=mod.reterms)
    β = coef(mod)
    y = response(mod) # we are now modifying the model
    res = residuals(mod)

    sample!(rng, res, y; replace=true)
    # inflate these to be on the same scale as the empirical variation instead of the MLE
    y .*= sdest(mod) / std(y; corrected=false)
    # sign flipping
    # y .*= rand(rng, (-1,1), length(y))

    σ = sdest(mod)

    for (re, trm) in zip(blups, reterms)
        npreds, ngrps = size(re)
        samp = sample(rng, 1:ngrps, ngrps; replace=true)

        newre = view(re, :, samp)
        # sign flipping
        # newre *= diagm(rand(rng, (-1,1), ngrps))

        # inflation
        λmle =  trm.λ * σ                               # L_R in CGR
        λemp = cholesky(cov(newre'; corrected=false)).L # L_S in CGR
        # no transpose because the RE are transposed relativ to CGR
        inflation = λmle / λemp

        # our RE are actually already scaled, but this method (of unscaledre!)
        # isn't dependent on the scaling (only the RNG methods are)
        MixedModels.unscaledre!(y, trm, inflation * newre)
    end

    # TODO: convert to inplace ops with mul!(y, mod.X, β, one(T), one(T))
    y .+= mod.X * β

    # mark model as unfitted
    mod.optsum.feval = -1

    return mod
end
