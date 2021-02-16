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
MixedModels.resample!
```

## Permutation Testing
```@docs
permutation
```

Note that this method is not exported to avoid a name collision with `Base.permute!`.
```@docs
MixedModelsPermutations.permute!
```
