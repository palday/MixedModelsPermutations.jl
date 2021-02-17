```@meta
CurrentModule = MixedModelsPermutations
DocTestSetup = quote
    using MixedModelsPermutations
end
DocTestFilters = [r"([a-z]*) => \1", r"getfield\(.*##[0-9]+#[0-9]+"]
```

# MixedModelsPermutations.jl API

## Nonparametric Bootstrap
```@docs
nonparametricbootstrap
```

Note that this method is not exported to match `permute!`.
```@docs
MixedModelsPermutations.resample!
```

## Permutation Testing
```@docs
permutation
```

```@docs
olsranef
```

Note that this method is not exported to avoid a name collision with `Base.permute!`.
```@docs
MixedModelsPermutations.permute!
```

```@docs
permutationtest
```
## Scale Inflation
```@docs
MixedModelsPermutations.inflation_factor
```
