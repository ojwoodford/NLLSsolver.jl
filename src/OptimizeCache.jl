# OptimizeCache: the NLLSsolver 5 reusable workspace. A function NLLSsolver owns (NLLSModel has no
# `OptimizeCache` name, so a method here is not type piracy, rule 8) that returns a NLLSModel `ModelWorkspace`.
# A `ModelWorkspace` reuses the model's linearization setup (`optimvars`) as well as the optimizer's state, so
# a repeated cached call with a typed workspace is allocation-free again, as in NLLSsolver 5 (D5); the earlier
# wrapper returned a bare `OptimizationWorkspace` and rebuilt the setup each call.

"""
    OptimizeCache() -> ModelWorkspace
    OptimizeCache(model::Model, options = NLLSOptions(), unfixed = nothing; formulation = :hessian, kwargs...) -> ModelWorkspace

A reusable optimizer workspace. **Deprecated**; use `ModelWorkspace` from NLLSModel.jl, passed as
`optimize!(model, options; workspace)`.

`OptimizeCache()` is an empty workspace that adapts to any problem. The second form builds a workspace
typed for `model`, reusing both the model's linearization setup and the optimizer's state, so a repeated
call allocates nothing while the model's structure and the keywords are unchanged. For the setup to be
reused rather than rebuilt, the **same keywords** (`unfixed`, `formulation`, `reduction`, `storage`,
`precision`, …) must be given to `OptimizeCache` and to the later `optimize!` call. `empty!` and `isempty`
(NLLSModel.jl) clear and test it.
"""
OptimizeCache() = ModelWorkspace()
OptimizeCache(model::Model, options::OptimizationOptions=OptimizationOptions(), unfixed=nothing; kwargs...) =
    ModelWorkspace(model, options; unfixed, kwargs...)

export OptimizeCache
