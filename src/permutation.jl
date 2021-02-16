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

# Method

The method implemented here is based on the approach given in
"""
function permutation(
    rng::AbstractRNG,
    n::Integer,
    morig::LinearMixedModel{T};
    use_threads::Bool=false,
) where {T}
    βsc, θsc = similar(morig.β), similar(morig.θ)
    p, k = length(βsc), length(θsc)
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
        mod = permute!(rng, mod, blups, reterms)
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

permute!(mod::LinearMixedModel, blups=ranef(mod), reterms=mod.reterms) =
    permute!(GLOBAL_RNG, mod, blups, reterms)

"""
    permute!([rng::AbstractRNG,] mod::LinearMixedModel,
              blups=ranef(mod), reterms=mod.reterms)

Simulate and install a new response via sign-flipped permutation of the residuals
at the observational and sign-flipping of the conditional modes at group level.

Sign-flipped permutation of the residuals is similar to permuting the
(fixed-effects) design matrix. Sign-flipping the conditional modes (random effects)
preserves the correlation structure of the random effects, while also being equivalent to
permutation via swapped labels for categorical variables.

!!! note
    This method has serious limitations for singular models because permuting zeros

See also [`permutation`](@ref), [`nonparametricbootstrap`](@ref) and [`resample!`](@ref).

# Reference
ter Braak?

"""
function permute!(rng::AbstractRNG, mod::LinearMixedModel,
                   blups=ranef(mod),
                   reterms=mod.reterms)
    β = coef(mod)
    y = response(mod) # we are now modifying the model
    copy!(y, residuals(mod))

    shuffle!(rng, y)
    # inflate these to be on the same scale as the empirical variation instead of the MLE
    # y .*= sdest(mod) / std(y; corrected=false)
    # sign flipping
    y .*= rand(rng, (-1,1), length(y))

    # σ = sdest(mod)

    for (re, trm) in zip(blups, reterms)
        npreds, ngrps = size(re)
        samp = sample(rng, 1:ngrps, ngrps; replace=false)

        # use a view to avoid copying
        newre = view(re, :, samp)
        # sign flipping
        newre *= diagm(rand(rng, (-1,1), ngrps))

        # inflation
        #λmle =  trm.λ * σ                               # L_R in CGR
        #λemp = cholesky(cov(newre'; corrected=false)).L # L_S in CGR
        # no transpose because the RE are transposed relativ to CGR
        #inflation = λmle / λemp

        # our RE are actually already scaled, but this method (of unscaledre!)
        # isn't dependent on the scaling (only the RNG methods are)
        MixedModels.unscaledre!(y, trm, newre)
    end

    # TODO: convert to inplace ops with mul!(y, mod.X, β, one(T), one(T))
    y .+= mod.X * β

    # mark model as unfitted
    mod.optsum.feval = -1

    return mod
end
