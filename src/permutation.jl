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
    residual_method=:signflip
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
        mod = permute!(rng, mod, blups, reterms;
                       β=β, residual_method=residual_method)
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

permute!(mod::LinearMixedModel, blups=ranef(mod), reterms=mod.reterms; kwargs...) =
    permute!(Random.GLOBAL_RNG, mod, blups, reterms; kwargs...)

"""
    permute!([rng::AbstractRNG,] mod::LinearMixedModel,
              blups=ranef(mod), reterms=mod.reterms;
              β=coef(mod), residual_method=:signflip)

Simulate and install a new response via permutation of the residuals
at the observational level and sign-flipping of the conditional modes at group level.

Permutation at the level of residuals can be accomplished either via sign
flipping (`residual_method=:signflip`) or via classical
permutation/shuffling (`residual_method=:shuffle`).

Generally, permutations are used to test a particular (null) hypothesis. This
hypothesis is specified via by setting `β` argument to match the hypothesis. For
example, the null hypothesis that the first coefficient is zero would expressed as

```julialang
julia> hypothesis = coef(model);
julia> hypothesis[1] = 0.0;
```

Sign-flipped permutation of the residuals is similar to permuting the
(fixed-effects) design matrix; shuffling the residuals is the same as permuting the
(fixed-effects) design matrix. Sign-flipping the conditional modes (random effects)
preserves the correlation structure of the random effects, while also being equivalent to
permutation via swapped labels for categorical variables.

!!! warning
    This method has serious limitations for singular models because sign-flipping a zero
    is not an effective randomization technique.

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
function permute!(rng::AbstractRNG, mod::LinearMixedModel{T},
                   blups=ranef(mod),
                   reterms=mod.reterms;
                   β::AbstractVector{T}=coef(mod),
                   residual_method=:signflip) where {T}
    y = response(mod) # we are now modifying the model
    copy!(y, residuals(mod))

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

    mul!(y, mod.X, β, one(T), one(T))

    # mark model as unfitted
    mod.optsum.feval = -1

    return mod
end


permutationtest(perm::MixedModelPermutation, model::LinearMixedModel) = permutationtest(perm::MixedModelPermutation, model, :twosided)

"""
    permutationtest(perm::MixedModelPermutation, model, type=:greater)

Perform a permutation using the already computed permutation and given the observed values.

The `type` parameter specifies the directionality of a one-sided test
(either `lesser` or `greater`, depending on the hypothesized difference to the null hypothesis).

See also [`permutation`](@ref).

"""
function permutationtest(perm::MixedModelPermutation, model, type::Symbol=:greater)
    if type == :greater
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
        # if type == :twosided
        #      dd[k] .= abs.(dd[k] .- ests[k])
        # end
    end
    results = (; (k => mean(comp(ests[k]), v) for (k,v) in dd)...)

    return results
end
