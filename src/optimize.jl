# The optimizer entry points NLLSsolver owns. They forward to SecondOrderOptimization; NLLSsolver holds no
# optimizer code of its own. `optimize!`/`optimize` are functions NLLSsolver owns (not methods added to
# SecondOrderOptimization's, which would be type piracy, rule 8), so `using NLLSsolver, SecondOrderOptimization`
# together makes the name ambiguous and it must be qualified.

"""
    optimize!(model::Model, options = NLLSOptions(); kwargs...) -> NLLSResult
    optimize!(model::Model, options, unfixed, callback = nullcallback, cache = nothing; kwargs...) -> NLLSResult

Solve `model` in place and return the result. **Deprecated** in favour of
`optimize!(model, options; ...)` from NLLSModel + SecondOrderOptimization; see the README guide.

With two positional arguments (or fewer) the call forwards unchanged to
`SecondOrderOptimization.optimize!`, so every keyword of the new entry point is available here:
`unfixed`, `numthreads`, `formulation`, `reduction`, `storage`, `precision`, `max_dense_bytes`,
`min_work`, `min_work_cost`, `callback` and `workspace`. `optimize!(model, rec)` for an
`OptimizationRecommendation`, and `optimize!(x0, HessianObjective(…), options; …)`, forward the
same way.

The three-or-more-argument form is the NLLSsolver 5 call `optimize!(problem, options, unfixed,
callback, cache)`. `unfixed` and `cache` become the `unfixed` and `workspace` keywords; the
positional `callback` has the 5.x signature `(cost, problem, data, iteratedata)` and is wrapped so
that it receives `model` as its second argument (see "Behaviour that changed" in the README: in the
all-variables-unfixed case the model holds the last accepted point, not the trial point, at the
callback). Giving `unfixed`, `callback` or `workspace` both positionally and by keyword is an error.
Any other keyword is forwarded untouched, so new features reach old-style calls.
"""
function optimize! end

"""
    optimize(model::Model, options = NLLSOptions(); kwargs...) -> (model_copy, NLLSResult)
    optimize(kernel::AbstractAdaptiveRobustifier, squarederrors, maxiters = 10) -> kernel

**Deprecated.** The first form runs `optimize!` on a copy of `model` and leaves it unchanged
(forwards to `SecondOrderOptimization.optimize`). The second is the NLLSsolver 5 adaptive-kernel
fit; use `optimizerobustifier(kernel, squarederrors; maxiters)` (NLLSModel.jl) instead.
"""
function optimize end

export optimize!, optimize

# The pass-through: everything new reaches NLLSsolver through here. The callee spells its keywords out
# (rule 19), so a misspelt keyword is still an error.
optimize!(args...; kwargs...) = SecondOrderOptimization.optimize!(args...; kwargs...)
optimize(args...; kwargs...) = SecondOrderOptimization.optimize(args...; kwargs...)

# The NLLSsolver 5 positional form. Dispatches only for three or more positional arguments (with one or
# two, the call is already the new form and takes the pass-through above).
function optimize!(problem::Model, options::OptimizationOptions, unfixed, callback=nullcallback, cache=nothing; kwargs...)
    for key in (:unfixed, :callback, :workspace)
        key in keys(kwargs) && throw(ArgumentError("`$key` was given both positionally and by keyword; give it once."))
    end
    return SecondOrderOptimization.optimize!(problem, options; unfixed, callback=adapt(callback, problem), workspace=cache, kwargs...)
end

# The NLLSsolver 5 adaptive-kernel fit (unexported in 5.x; VisualGeometryOptimization's examples call it).
optimize(kernel::AbstractAdaptiveRobustifier, squarederrors::AbstractVector, maxiters::Integer=10) =
    optimizerobustifier(kernel, squarederrors; maxiters)

# Adapts a NLLSsolver 5 callback `(cost, problem, data, iteratedata)` to SecondOrderOptimization's
# `(cost, x_trial, data, iteratedata)` by substituting the model for the trial point. A struct (not a
# closure) so the wrapped type is concrete. `nullcallback` is passed straight through. SecondOrderOptimization's
# own `printoutcallback` and `storecostscallback` ignore their second argument, so they work unwrapped too.
struct ProblemCallback{CB, P}
    callback::CB
    problem::P
end
(w::ProblemCallback)(cost, x_trial, data, iteratedata) = w.callback(cost, w.problem, data, iteratedata)
adapt(::typeof(nullcallback), problem) = nullcallback
adapt(callback, problem) = ProblemCallback(callback, problem)
