using MixedModels: fixef!, stderror!
using MixedModels: getθ!, setθ!, updateL!
using MixedModels: unscaledre!

"""
    nonparametricbootstrap(rng::AbstractRNG, nsamp::Integer, m::MixedModel;
        β = coef(m), σ = m.σ, θ = m.θ, use_threads=false)
    nonparametricbootstrap(nsamp::Integer, m::MixedModel;
        β = coef(m), σ = m.σ, θ = m.θ, use_threads=false)

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

function resample!(rng::AbstractRNG, mod::LinearMixedModel{T},
                   blups=ranef(mod),
                   reterms=mod.reterms) where {T}
    β = coef(mod)
    y = response(mod) # we are now modifying the model
    res = residuals(mod)
    # TODO: inflate these
    sample!(rng, res, y; replace=true)
    #y .= res

    for (re, trm) in zip(blups, reterms)
         # TODO: inflate re
         nre = size(re,2)
         samp = sample(rng, 1:nre, nre; replace=true)
         # our RE are actually already scaled, but this method
         # isn't dependent on the scaling (only the RNG methods are)
         MixedModels.unscaledre!(y, trm, re[:,samp])
    end

    y .+= mod.X * β

    # mark model as unfitted
    mod.optsum.feval = 0

    return mod
end
