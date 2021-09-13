# MixedModelsPermutations

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Stable Docs][docs-stable-img]][docs-stable-url]
[![Dev Docs][docs-dev-img]][docs-dev-url]
[![Codecov](https://codecov.io/gh/palday/MixedModelsPermutations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/palday/MixedModelsPermutations.jl)
[![DOI](https://zenodo.org/badge/337080334.svg)](https://zenodo.org/badge/latestdoi/337080334)

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://palday.github.io/MixedModelsPermutations.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://palday.github.io/MixedModelsPermutations.jl/stable


`MixedModelsPermutations.jl` is a Julia package providing permutation and other resampling-based methods for[`MixedModels.jl`](https://juliastats.org/MixedModels.jl/stable/).

This package is alpha software in early development and results may not be accurate.
Nonetheless, it is registered in the Julia General Registry.
You can install it like this:
```julia
julia>]
pkg> add MixedModelsPermutations.jl
```

To get the bleeding edge development version, you can install like this:
You can install it like this:
```julia
julia>]
pkg> add MixedModelsPermutations.jl#main
```

If you find inaccurate results, please try the development version to see if the underlying problem has already been fixed before filing an issue.

<!-- Note that plotting functionality is planned for inclusion in separate packages. -->

[`MixedModelsSim.jl`](https://github.com/RePsychLing/MixedModelsSim.jl/) provides additional functionality for data/power simulation and may also be of interest.
