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
    σ = sdest(m)
    σres = std(resids; corrected=false)
      inflation = map(zip(m.reterms, blups)) do (trm, re)
        # inflation
        λmle =  trm.λ * σ                              # L_R in CGR

        cov_emp = StatsBase.cov(re'; corrected=false)

        chol = cholesky(cov_emp, RowMaximum(); check=false,tol=10^-5)

        #  ATTEMPT 2
         while chol.rank != size(cov_emp, 1)
             #@info "rep"
            idx = chol.p[(chol.rank+1):end]
            cov_emp[idx, idx] .+= 1e-6
            chol = cholesky(cov_emp, RowMaximum(); check=false,tol=10^-5)
        end

        L = chol.L[invperm(chol.p),:]
        cov_emp = L * L'
        cov_mle = λmle * λmle'

        return cov_mle / cov_emp
    end
    return [inflation; σ / σres]
end
