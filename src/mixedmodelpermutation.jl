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
struct MixedModelPermutation{T<:AbstractFloat}
    fits::Vector
    λ::Vector{LowerTriangular{T,Matrix{T}}}
    inds::Vector{Vector{Int}}
    lowerbd::Vector{T}
    fcnames::NamedTuple
end


#####
##### Convert and Delegate Methods to MixedModelBootstrap
#####

# XXX This can be removed when MixedModels.jl introduces the abstract fit
#     collection type and broadens their method signatures

# non copying converstion to a bootstrap object.
_perm2boot(p::MixedModelPermutation) = MixedModelBootstrap(p.fits, p.λ, p.inds, p.lowerbd, p.fcnames)

# delegate methods
for f in (:allpars, :coefpvalues, :issingular, :setθ!, :tidyβ, :tidyσs)

    @eval begin
        MixedModels.$f(p::MixedModelPermutation) = MixedModels.$f(_perm2boot(p))
    end
end


Base.propertynames(p::MixedModelPermutation) = propertynames(_perm2boot(p))


function Base.getproperty(p::MixedModelPermutation, s::Symbol)
    if s ∈ [:objective, :σ, :θ, :se]
        getproperty.(getfield(p, :fits), s)
    elseif s == :β
        tidyβ(p)
    elseif s == :coefpvalues
        coefpvalues(p)
    elseif s == :σs
        tidyσs(p)
    elseif s == :allpars
        allpars(p)
    else
        getfield(p, s)
    end
end
