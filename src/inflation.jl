"""
    inflation_factor(m::LinearMixedModel, blups=ranef(m), resids=residuals(m)))

Compute how much the standard deviation of the BLUPs/conditional modes and residuals
needs to be inflated in order to match the (restricted) maximum-likelihood estimate.

Due to the shrinkage associated with the random effects in a mixed model, the
variance of the conditional modes / BLUPs / random intercepts and slopes is less
than the variance estimated by the model and displayed in the model summary or
via `MixedModels.VarCorr`. This shrinkage also impacts the observational level
residuals. To compensate for this, the resampled residuals and groups need to be
scale-inflated so that their standard deviation matches that of the estimates in
the original model.

The factor for scale inflation is returned as a `Vector` of inflation factors
for each of the random-effect strata (blocking variables) and the observation-level
variability. The factor is on the standard deviation scale (lower Cholesky factor
in the case of vector-valued random effects).
"""
function inflation_factor(m::LinearMixedModel, blups=ranef(m), resids=residuals(m))
# FIXME I'm not sure this is correct
#       the nonparametric bootstrap underestimates variance components
#       compared to the parametricbootstrap

    σ = sdest(m)
    σres = std(resids; corrected=false)
    inflation = map(zip(m.reterms, blups)) do (trm, re)
        # inflation
        λmle =  trm.λ * σ                               # L_R in CGR
        cov_mat = cov(re'; corrected=false)

        chol = cholesky(cov_mat, Val(true); check=false)

        # ATTEMPT 0
        # L = chol.L[invperm(chol.p), invperm(chol.p)]

        # ATTEMPT 1
        # L = if chol.rank != size(cov_mat, 1)
        #     pivoted_out = (chol.rank + 1):lastindex(chol.p)
        #     ip = invperm(chol.p)
        #     L = chol.L[ip, ip]
        #     L[pivoted_out, pivoted_out] .+= 1e-5
        #     L
        # else
        #     chol.L
        # end

        #  ATTEMPT 2
         while chol.rank != size(cov_mat, 1)
            idx = chol.p[(chol.rank+1):end]
            cov_mat[idx, idx] .+= 1e-5
            chol = cholesky(cov_mat, Val(true); check=false)
        end
        L = chol.L

        if !istril(L)
            println("L = ")
            display(L)
            error("Tell @palday that the pivoting isn't working")
        end
        λemp = LowerTriangular(L)    # L_S in CGR
        # no transpose because the RE are transposed relativ to CGR
        return λmle / λemp
    end

    return [(σ / σres) * inflation; σ / σres]
end
