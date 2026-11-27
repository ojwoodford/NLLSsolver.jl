# OptimizeCache: the NLLSsolver 5 workspace. It is a function NLLSsolver owns (a constructor method on
# SecondOrderOptimization's OptimizationWorkspace for NLLSModel's Model would be type piracy, rule 8) that
# returns a SecondOrderOptimization.OptimizationWorkspace. Unlike NLLSsolver 5's cache it holds only the
# optimizer's state (its linear system, factorization and iterator state); the model's linearization setup
# (`optimvars`) is rebuilt on every `optimize!` call, so a cached call is no longer allocation-free (D5).

# Wraps the Jacobian evaluator so it writes into a pre-zeroed Jacobian, mirroring how the model's front door
# builds the objective. A struct (not a closure) so the objective's type is concrete.
struct JacobianResJacCache{RJ}
    resjac!::RJ
end
(w::JacobianResJacCache)(r, J, x) = w.resjac!(r, J, x; prezeroed=true)

"""
    OptimizeCache() -> OptimizationWorkspace
    OptimizeCache(model::Model, options = NLLSOptions(), unfixed = nothing; formulation = :hessian, kwargs...) -> OptimizationWorkspace

A reusable optimizer workspace. **Deprecated**; use `OptimizationWorkspace` from
SecondOrderOptimization.jl, passed as `optimize!(model, options; workspace)`.

`OptimizeCache()` is an empty workspace that adapts to any problem. The second form builds a workspace
typed for `model` by linearizing it once (`optimvars` for `formulation = :hessian`, `optimvars_nls` for
`:jacobian`) and wrapping the result in a `HessianObjective` or `JacobianObjective`. For the workspace to be
reused rather than rebuilt, the **same keywords** (`unfixed`, `formulation`, `reduction`, `storage`,
`precision`, …) must be given to `OptimizeCache` and to the later `optimize!` call.

Only the optimizer's state is reused: its linear system, factorization and iterator state. The model's
linearization setup is rebuilt on every call, so unlike NLLSsolver 5 a cached call is not allocation-free.
`empty!` and `isempty` (SecondOrderOptimization.jl) clear and test it.
"""
OptimizeCache() = OptimizationWorkspace()

function OptimizeCache(model::Model, options::OptimizationOptions=OptimizationOptions(), unfixed=nothing; formulation::Symbol=:hessian, kwargs...)
    if formulation === :hessian
        f, fdf!, x0, H = optimvars(model; unfixed, kwargs...)
        return OptimizationWorkspace(x0, HessianObjective(f, fdf!; H), options)
    elseif formulation === :jacobian
        f, _, resjac!, x0, J = optimvars_nls(model; unfixed, kwargs...)
        return OptimizationWorkspace(x0, JacobianObjective(f, JacobianResJacCache(resjac!), J), options)
    else
        throw(ArgumentError("Unknown formulation $(repr(formulation)); expected :hessian or :jacobian."))
    end
end

export OptimizeCache
