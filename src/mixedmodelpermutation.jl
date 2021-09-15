"""
    MixedModelPermutation{T<:AbstractFloat}

Object returned by the various permutation methods with fields
- `fits`: the parameter estimates from the permutation replicates as a vector of named tuples.
- `λ`: `Vector{LowerTriangular{T,Matrix{T}}}` containing copies of the λ field from `ReMat` model terms
- `inds`: `Vector{Vector{Int}}` containing copies of the `inds` field from `ReMat` model terms
- `lowerbd`: `Vector{T}` containing the vector of lower bounds (corresponds to the identically named field of [`OptSummary`](@ref))
- `fcnames`: NamedTuple whose keys are the grouping factor names and whose values are the column names

The schema of `fits` is, by default,
```
Tables.Schema:
 :objective  T
 :σ          T
 :β          NamedTuple{β_names}{NTuple{p,T}}
 :se         StaticArrays.SArray{Tuple{p},T,1,p}
 :θ          StaticArrays.SArray{Tuple{k},T,1,k}
```
where the sizes, `p` and `k`, of the `β` and `θ` elements are determined by the model.

Characteristics of the permutation replicates can be extracted as properties.  The `σs` and
`σρs` properties unravel the `σ` and `θ` estimates into estimates of the standard deviations
and correlations of the random-effects terms.
"""
struct MixedModelPermutation{T<:AbstractFloat} <: MixedModels.MixedModelFitCollection{T}
    fits::Vector
    λ::Vector{LowerTriangular{T,Matrix{T}}}
    inds::Vector{Vector{Int}}
    lowerbd::Vector{T}
    fcnames::NamedTuple
end

"""
_perm2boot(p::MixedModelPermutation)

Non copying conversion to MixedModels.MixedModelBootstrap.
"""
_perm2boot(p::MixedModelPermutation) = MixedModelBootstrap(p.fits, p.λ, p.inds, p.lowerbd, p.fcnames)
