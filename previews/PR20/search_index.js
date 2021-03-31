var documenterSearchIndex = {"docs":
[{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"CurrentModule = MixedModelsPermutations\nDocTestSetup = quote\n    using MixedModelsPermutations\nend\nDocTestFilters = [r\"([a-z]*) => \\1\", r\"getfield\\(.*##[0-9]+#[0-9]+\"]","category":"page"},{"location":"api/#MixedModelsPermutations.jl-API","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"","category":"section"},{"location":"api/#Nonparametric-Bootstrap","page":"MixedModelsPermutations.jl API","title":"Nonparametric Bootstrap","text":"","category":"section"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"nonparametricbootstrap","category":"page"},{"location":"api/#MixedModelsPermutations.nonparametricbootstrap","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.nonparametricbootstrap","text":"nonparametricbootstrap([rng::AbstractRNG,] nsamp::Integer, m::LinearMixedModel;\n                       use_threads=false, blup_method=ranef, β=coef(morig))\n\nPerform nsamp nonparametric bootstrap replication fits of m, returning a MixedModelBootstrap.\n\nThe default random number generator is Random.GLOBAL_RNG.\n\nGeneralizedLinearMixedModel is currently unsupported.\n\nNamed Arguments\n\nuse_threads determines whether or not to use thread-based parallelism.\n\nnote: Note\nNote that use_threads=true may not offer a performance boost and may even decrease peformance if multithreaded linear algebra (BLAS) routines are available. In this case, threads at the level of the linear algebra may already occupy all processors/processor cores. There are plans to provide better support in coordinating Julia- and BLAS-level threads in the future.\n\nwarning: Warning\nThe PRNG shared between threads is locked using Threads.SpinLock, which should not be used recursively. Do not wrap nonparametricbootstrap in an outer SpinLock.\n\nhide_progress can be used to disable the progress bar. Note that the progress bar is automatically disabled for non-interactive (i.e. logging) contexts.\n\nblup_method provides options for how/which group-level effects are passed for resampling. The default ranef uses the shrunken conditional modes / BLUPs. Unshrunken estimates from ordinary least squares (OLS) can be used with olsranef. There is no shrinkage of the group-level estimates with this approach, which means singular estimates can be avoided. However, if the design matrix for the random effects is rank deficient (e.g., through the use of MixedModels.fulldummy or missing cells in the data), then this method will fail. See olsranef and MixedModels.ranef for more information.\n\nMethod\n\nThe method implemented here is based on the approach given in Section 3.2 of: Carpenter, J.R., Goldstein, H. and Rasbash, J. (2003). A novel bootstrap procedure for assessing the relationship between class size and achievement. Journal of the Royal Statistical Society: Series C (Applied Statistics), 52: 431-443. https://doi.org/10.1111/1467-9876.00415\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"Note that this method is not exported to match permute!.","category":"page"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"MixedModelsPermutations.resample!","category":"page"},{"location":"api/#MixedModelsPermutations.resample!","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.resample!","text":"resample!([rng::AbstractRNG,] model::LinearMixedModel;\n          β=coef(model), blups=ranef(model), resids=residuals(model,blups)\n          scalings=inflation_factor(model))\n\nSimulate and install a new response using resampling at the observational and group level.\n\nAt both levels, resampling is done with replacement. At the observational level, this is resampling of the residuals, i.e. comparable to the step in the classical nonparametric bootstrap for OLS regression. At the group level, samples are done jointly for all  terms (\"random intercept and random slopes\", conditional modes) associated with a particular level of a blocking factor. For example, the predicted slope and intercept for a single participant or item are kept together. This clumping in the resampling procedure is necessary to preserve the original correlation structure of the slopes and intercepts.\n\nIn addition to this resampling step, there is also an inflation step. Due to the shrinkage associated with the random effects in a mixed model, the variance of the conditional modes / BLUPs / random intercepts and slopes is less than the variance estimated by the model and displayed in the model summary or via MixedModels.VarCorr. This shrinkage also impacts the observational level residuals. To compensate for this, the resampled residuals and groups are scale-inflated so that their standard deviation matches that of the estimates in the original model. The default inflation factor is computed using inflation_factor on the model.\n\nSee also nonparametricbootstrap and MixedModels.simulate!.\n\nwarning: Warning\nThis method has serious limitations for singular models because resampling from a distribution with many zeros (e.g. the random effects components with zero variance) will often generate new data with even less variance.\n\nReference\n\nThe method implemented here is based on the approach given in Section 3.2 of: Carpenter, J.R., Goldstein, H. and Rasbash, J. (2003). A novel bootstrap procedure for assessing the relationship between class size and achievement. Journal of the Royal Statistical Society: Series C (Applied Statistics), 52: 431-443. https://doi.org/10.1111/1467-9876.00415\n\n\n\n\n\n","category":"function"},{"location":"api/#Permutation-Testing","page":"MixedModelsPermutations.jl API","title":"Permutation Testing","text":"","category":"section"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"permutation","category":"page"},{"location":"api/#MixedModelsPermutations.permutation","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.permutation","text":"permutation([rng::AbstractRNG,] nsamp::Integer, m::LinearMixedModel;\n            use_threads::Bool=false,\n            β=zeros(length(coef(morig))),\n            residual_method=:signflip,\n            blup_method=ranef)\n\nPerform nsamp nonparametric bootstrap replication fits of m, returning a MixedModelBootstrap.\n\nThe default random number generator is Random.GLOBAL_RNG.\n\nGeneralizedLinearMixedModel is currently unsupported.\n\nNamed Arguments\n\nuse_threads determines whether or not to use thread-based parallelism.\n\nnote: Note\nNote that use_threads=true may not offer a performance boost and may even decrease peformance if multithreaded linear algebra (BLAS) routines are available. In this case, threads at the level of the linear algebra may already occupy all processors/processor cores. There are plans to provide better support in coordinating Julia- and BLAS-level threads in the future.\n\nwarning: Warning\nThe PRNG shared between threads is locked using Threads.SpinLock, which should not be used recursively. Do not wrap permutation in an outer SpinLock.\n\nhide_progress can be used to disable the progress bar. Note that the progress bar is automatically disabled for non-interactive (i.e. logging) contexts.\n\nPermutation at the level of residuals can be accomplished either via sign flipping (residual_method=:signflip) or via classical permutation/shuffling (residual_method=:shuffle).\n\nblup_method provides options for how/which group-level effects are passed for permutation. The default ranef uses the shrunken conditional modes / BLUPs. Unshrunken estimates from ordinary least squares (OLS) can be used with olsranef. There is no shrinkage of the group-level estimates with this approach, which means singular estimates can be avoided. However, if the design matrix for the random effects is rank deficient (e.g., through the use of MixedModels.fulldummy or missing cells in the data), then this method will fail. See olsranef and MixedModels.ranef for more information.\n\nGenerally, permutations are used to test a particular (null) hypothesis. This hypothesis is specified via by setting β argument to match the hypothesis. For example, the null hypothesis that the first coefficient is zero would expressed as\n\njulia> hypothesis = coef(model);\njulia> hypothesis[1] = 0.0;\n\nnote: Note\nThe permutation (test) generates samples from H0, from which it is possible to compute p-values. The bootstrap typically generates samples from H1, which are convenient for generating coverage/confidence intervals. Of course, confidence intervals and p-values are duals of each other, so it is possible to convert from one to the other.\n\nMethod\n\nThe method implemented here is based on the approach given in:\n\nter Braak C.J.F. (1992) Permutation Versus Bootstrap Significance Tests in Multiple Regression and Anova. In: Jöckel KH., Rothe G., Sendler W. (eds) Bootstrapping and Related Techniques. Lecture Notes in Economics and Mathematical Systems, vol 376. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-48850-4_10\n\nand\n\nWinkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the general linear model. NeuroImage, 92, 381–397. https://doi.org/10.1016/j.neuroimage.2014.01.060\n\nwarning: Warning\nThis method has serious limitations for singular models because sign-flipping a zero is not an effective randomization technique.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"olsranef","category":"page"},{"location":"api/#MixedModelsPermutations.olsranef","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.olsranef","text":"olsranef(model::LinearMixedModel, method=:simultaneous)\n\nCompute the group-level estimates using ordinary least squares.\n\nThis is somewhat similar to the conditional modes / BLUPs computed without shrinkage.\n\nThere is no shrinkage of the group-level estimates with this approach, which means singular estimates can be avoided. However, this also means the random effects design matrix must not be singular.\n\nTwo methods are provided:\n\n(default) OLS estimates computed for all strata (blocking variables) simultaneously with method=simultaneous. This pools the variance across estimates but does not shrink the estimates.\nOLS estimates computed within each stratum with method=stratum. This method is equivalent for example to computing each subject-level and each item-level regression separately.\n\nFor fully balanced designs with a single blocking variable, these methods will give the same results.\n\nwarning: Warning\nIf the design matrix for the random effects is rank deficient (e.g., through the use of MixedModels.fulldummy or missing cells in the data), then this method will fail.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"Note that this method is not exported to avoid a name collision with Base.permute!.","category":"page"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"MixedModelsPermutations.permute!","category":"page"},{"location":"api/#MixedModelsPermutations.permute!","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.permute!","text":"permute!([rng::AbstractRNG,] model::LinearMixedModel;\n         β=zeros(length(coef(model))),\n         blups=ranef(model),\n         resids=residuals(model,blups),\n         residual_method=:signflip,\n         scalings=inflation_factor(model))\n\nSimulate and install a new response via permutation of the residuals at the observational level and sign-flipping of the conditional modes at group level.\n\nGenerally, permutations are used to test a particular (null) hypothesis. This hypothesis is specified via by setting β argument to match the hypothesis. For example, the null hypothesis that the first coefficient is zero would expressed as\n\njulia> hypothesis = coef(model);\njulia> hypothesis[1] = 0.0;\n\nPermutation at the level of residuals can be accomplished either via sign flipping (residual_method=:signflip) or via classical permutation/shuffling (residual_method=:shuffle).\n\nSign-flipped permutation of the residuals is similar to permuting the (fixed-effects) design matrix; shuffling the residuals is the same as permuting the (fixed-effects) design matrix. Sign-flipping the random effects preserves the correlation structure of the random effects, while also being equivalent to permutation via swapped labels for categorical variables.\n\nwarning: Warning\nThis method has serious limitations for singular models because sign-flipping a zero is not an effective randomization technique.\n\nOptionally, instead of using the shrunken random effects from ranef, within-group OLS estimates can be computed and used instead with olsranef. There is no shrinkage of the group-level estimates with this approach, which means singular estimates can be avoided. However, if the design matrix for the random effects is rank deficient (e.g., through the use of MixedModels.fulldummy or missing cells in the data), then this method will fail.\n\nIn addition to the permutation step, there is also an inflation step. Due to the shrinkage associated with the random effects in a mixed model, the variance of the conditional modes / BLUPs / random intercepts and slopes is less than the variance estimated by the model and displayed in the model summary or via MixedModels.VarCorr. This shrinkage also impacts the observational level residuals. To compensate for this, the resampled residuals and groups are scale-inflated so that their standard deviation matches that of the estimates in the original model. The default inflation factor is computed using inflation_factor on the model.\n\nSee also permutation, nonparametricbootstrap and resample!.\n\nThe functions coef and coefnames from MixedModels may also be useful.\n\nReference\n\nThe method implemented here is based on the approach given in:\n\nter Braak C.J.F. (1992) Permutation Versus Bootstrap Significance Tests in Multiple Regression and Anova. In: Jöckel KH., Rothe G., Sendler W. (eds) Bootstrapping and Related Techniques. Lecture Notes in Economics and Mathematical Systems, vol 376. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-48850-4_10\n\nand\n\nWinkler, A. M., Ridgway, G. R., Webster, M. A., Smith, S. M., & Nichols, T. E. (2014). Permutation inference for the general linear model. NeuroImage, 92, 381–397. https://doi.org/10.1016/j.neuroimage.2014.01.060\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"permutationtest","category":"page"},{"location":"api/#MixedModelsPermutations.permutationtest","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.permutationtest","text":"permutationtest(perm::MixedModelPermutation, model, type=:greater)\n\nPerform a permutation using the already computed permutation and given the observed values.\n\nThe type parameter specifies use of a two-sided test (:twosided) or the directionality of a one-sided test (either :lesser or :greater, depending on the hypothesized difference to the null hypothesis).\n\nSee also permutation.\n\n\n\n\n\n","category":"function"},{"location":"api/#Scale-Inflation","page":"MixedModelsPermutations.jl API","title":"Scale Inflation","text":"","category":"section"},{"location":"api/","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.jl API","text":"MixedModelsPermutations.inflation_factor","category":"page"},{"location":"api/#MixedModelsPermutations.inflation_factor","page":"MixedModelsPermutations.jl API","title":"MixedModelsPermutations.inflation_factor","text":"inflation_factor(m::LinearMixedModel, blups=ranef(m), resids=residuals(m)))\n\nCompute how much the standard deviation of the BLUPs/conditional modes and residuals needs to be inflated in order to match the (restricted) maximum-likelihood estimate.\n\nDue to the shrinkage associated with the random effects in a mixed model, the variance of the conditional modes / BLUPs / random intercepts and slopes is less than the variance estimated by the model and displayed in the model summary or via MixedModels.VarCorr. This shrinkage also impacts the observational level residuals. To compensate for this, the resampled residuals and groups need to be scale-inflated so that their standard deviation matches that of the estimates in the original model.\n\nThe factor for scale inflation is returned as a Vector of inflation factors for each of the random-effect strata (blocking variables) and the observation-level variability. The factor is on the standard deviation scale (lower Cholesky factor in the case of vector-valued random effects).\n\n\n\n\n\n","category":"function"},{"location":"#MixedModelsPermutations.jl-Documentation","page":"MixedModelsPermutations.jl Documentation","title":"MixedModelsPermutations.jl Documentation","text":"","category":"section"},{"location":"","page":"MixedModelsPermutations.jl Documentation","title":"MixedModelsPermutations.jl Documentation","text":"CurrentModule = MixedModelsPermutations","category":"page"},{"location":"","page":"MixedModelsPermutations.jl Documentation","title":"MixedModelsPermutations.jl Documentation","text":"MixedModelsPermutations.jl is a Julia package providing extra capabilities for models fit with MixedModels.jl.","category":"page"},{"location":"","page":"MixedModelsPermutations.jl Documentation","title":"MixedModelsPermutations.jl Documentation","text":"Pages = [\n        \"api.md\",\n]\nDepth = 1","category":"page"}]
}
