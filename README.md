# NLLSsolver

> [!WARNING]
> **`NLLSsolver.jl` is deprecated.** It has been split into two packages that replace it:
> - [**`NLLSModel.jl`**](https://github.com/ojwoodford/NLLSModel.jl) — the model: variables, residuals, robust and adaptive kernels, automatic and analytic differentiation, cost/gradient/Hessian assembly.
> - [**`SecondOrderOptimization.jl`**](https://github.com/ojwoodford/SecondOrderOptimization.jl) — the optimizer: Levenberg–Marquardt, Dogleg and Newton iterators, step control, callbacks, stats and workspace.
>
> Version 6 is a thin compatibility layer over the two and the **last major version**; it receives compatibility fixes only. New code should depend on `NLLSModel.jl` and `SecondOrderOptimization.jl` directly. See their documentation: [NLLSModel](https://ojwoodford.github.io/NLLSModel.jl/), [SecondOrderOptimization](https://ojwoodford.github.io/SecondOrderOptimization.jl/).

[![Build Status](https://github.com/ojwoodford/NLLSsolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/NLLSsolver.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/NLLSsolver.jl/branch/main/graph/badge.svg?token=1556CDMEGH)](https://codecov.io/gh/ojwoodford/NLLSsolver.jl)

## What NLLSsolver 6 is

Every call forwards to `NLLSModel.jl` and `SecondOrderOptimization.jl`; NLLSsolver holds no modelling or optimizer code of its own. It re-exports every name those packages export, so their new features — Schur elimination, `formulation = :jacobian`, `precision = Float32`, `recommendoptimizer`, user-defined iterators, and the rest — are reachable through NLLSsolver without it naming them. Code written against the NLLSsolver 5 API keeps working, except for the breaks listed below.

## Translation guide

| NLLSsolver 5 | NLLSModel + SecondOrderOptimization | Works in 6 |
|---|---|---|
| `using NLLSsolver` | `using NLLSModel, SecondOrderOptimization` (and `OptimizationVariableInterface` to define variables) | ✔ |
| `NLLSsolver.nvars(::MyVar)`, `NLLSsolver.update(…)` | methods of `OptimizationVariableInterface.nvars`, `update` | ✔ |
| `NLLSsolver.ndeps`, `nres`, `varindices`, `getvars`, `computeresidual`, `computeresjac`, `robustkernel`, … | the same names in `NLLSModel` | ✔ |
| `NLLSProblem(VarTypes, CostTypes)` | `Model(VarTypes, CostTypes)` | ✔ |
| `NLLSProblem()`, `NLLSProblem(VarTypes)` (costs of type `Any`) | not available: give the cost types | ✘ |
| `EuclideanVector{N, T}` | `SVector{N, T}` | ✔ |
| `NLLSOptions(…)` | `OptimizationOptions(…)` | ✔ |
| `NLLSOptions(maxfails = n)` | `maxstall = n` (and `stalltol`) | ✘ |
| `NLLSOptions(numthreads = n)` | `optimize!(…; numthreads = n)` | ✘ |
| `iterator = NLLSsolver.dogleg` | `iterator = :dogleg` | ✔ |
| `iterator = :gradientdescent` | define it: see `examples/gradientdescent.jl` and SecondOrderOptimization's tutorial | ✘ |
| `optimize!(problem, options, unfixed, callback)` | `optimize!(model, options; unfixed, callback)` | ✔ |
| callback signature `(cost, problem, data, iteratedata)` | `(cost, x_trial, data, iteratedata)`: the 2nd argument is the trial point, not the model, and `data` is `OptimizationInternal`, not `NLLSInternal` | ✘ |
| `problem.varnext`, `problem.varbest` | gone; the trial point is `x_trial` | ✘ |
| `cache = OptimizeCache()`; `optimize!(problem, options, unfixed, callback, cache)` | `workspace = OptimizationWorkspace()`; `optimize!(model, options; workspace)` | ✔ |
| `NLLSsolver.optimizesingles!(problem, options, T)` | a loop: `subproblem` and `optimize!(sub, options; unfixed = i)` | ✔ |
| `NLLSResult`, `result.bestcost`, … | `OptimizationResult`, same fields | ✔ |
| `result.iterator` (enum) | a `Symbol` | ✘ |
| `result.termination` bit values | `isconverged(result)` and the `TERMINATION_*` masks | ✘ |
| `CostTrajectory().trajectory` | `.steps` | ✘ |
| `ContaminatedGaussian(s1, s2, w)` | `ContaminatedGaussian(s1, s2, w, static(d))`, `d` the residual length | ✘ |
| `NLLSsolver.optimize(kernel, squarederrors)` | `optimizerobustifier(kernel, squarederrors)` | ✔ |
| `updatevarcostmap!` | not needed | ✘ |
| *(new in 6)* | `reduction`, `storage`, `formulation = :jacobian`, `precision = Float32`, `linsolver`, `recommendoptimizer`, `collectstats`, user-defined iterators | ✔ |

### A worked example

The same Rosenbrock problem, solved through NLLSsolver 6 (the NLLSsolver 5 API):

```julia
using NLLSsolver, StaticArrays, Static

struct Rosenbrock <: AbstractResidual
    a::Float64
    b::Float64
end
NLLSsolver.ndeps(::Rosenbrock) = static(1)
NLLSsolver.nres(::Rosenbrock) = static(2)
NLLSsolver.varindices(::Rosenbrock) = SVector(1)
NLLSsolver.getvars(::Rosenbrock, vars::Vector) = (vars[1]::EuclideanVector{2, Float64},)
NLLSsolver.computeresidual(r::Rosenbrock, x) = SVector(r.a * (1 - x[1]), r.b * (x[1]^2 - x[2]))
Base.eltype(::Rosenbrock) = Float64

problem = NLLSProblem(EuclideanVector{2, Float64}, Rosenbrock)
addvariable!(problem, EuclideanVector(-1.2, 1.0))
addcost!(problem, Rosenbrock(1.0, 10.0))
result = optimize!(problem, NLLSOptions(), nothing, nullcallback)
cost(problem)   # ≈ 0 at the minimum (1, 1)
```

and directly on the new packages:

```julia
using NLLSModel, SecondOrderOptimization, StaticArrays, Static

struct Rosenbrock2 <: AbstractResidual
    a::Float64
    b::Float64
end
NLLSModel.ndeps(::Rosenbrock2) = static(1)
NLLSModel.nres(::Rosenbrock2) = static(2)
NLLSModel.varindices(::Rosenbrock2) = SVector(1)
NLLSModel.getvars(::Rosenbrock2, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSModel.computeresidual(r::Rosenbrock2, x) = SVector(r.a * (1 - x[1]), r.b * (x[1]^2 - x[2]))
Base.eltype(::Rosenbrock2) = Float64

model = Model(SVector{2, Float64}, Rosenbrock2)
addvariable!(model, SVector(-1.2, 1.0))
addcost!(model, Rosenbrock2(1.0, 10.0))
result = optimize!(model, OptimizationOptions(); callback = nullcallback)
cost(model)   # ≈ 0, the same minimum
```

### Behaviour that changed

- **Convergence defaults.** `reldcost` defaults to `1e-12` (was `1e-15`), so a run may stop an iteration or two sooner. The other tolerances are unchanged.
- **Stalling, not failures.** `maxfails` is gone. A run that stops making progress stops on `maxstall` consecutive non-improving iterations within a `stalltol` band (`TERMINATION_STALLED`); test it with `isconverged(result)`. Cost-change termination now applies to increases as well as decreases.
- **`result.iterator`** is a `Symbol`, and `result.termination` is read with `isconverged` and the exported `TERMINATION_*` masks; a callback's termination flags occupy bits 16 and up.
- **`linearsolves`** counts every linear solve, including those of rejected steps.
- **The callback signature changed (breaking).** SecondOrderOptimization calls `callback(cost, x_trial, data, iteratedata)`: the second argument is the **trial point** `x_trial`, not the `NLLSProblem`, and `data` is `OptimizationInternal`, not `NLLSInternal`. The *positional* `optimize!(problem, options, unfixed, callback)` form adapts a NLLSsolver 5 `(cost, problem, data, iteratedata)` callback by passing the model as the second argument, so callbacks that use only the `cost` argument — including `printoutcallback` and `storecostscallback` — keep working unchanged. A callback that reads the second argument or `data` must be updated: there is no `problem.varnext`/`varbest` (the trial is `x_trial`), and because the model holds the *last accepted* point (not the trial) at the callback when **every** variable is unfixed, a callback that recomputes `cost(problem)` is reliable only when a subset of variables is unfixed. A callback passed by keyword (`callback = …`) is the new signature and is not adapted.
- **Adaptive-kernel cost** is now exactly `-2 log p`, so an adaptive kernel's costs shift and can be negative.
- **A cached call is no longer allocation-free:** `OptimizeCache` reuses only the optimizer's workspace, and the model's linearization is rebuilt each call.

### Removed

`NLLSProblem` with untyped costs; `problem.varnext`/`problem.varbest`; the `NLLSOptions` keywords `maxfails` and `numthreads`; `iterator = :gradientdescent` and the `NLLSsolver.gradientdescent` enum member; `updatevarcostmap!`; and every other unexported NLLSsolver 5 name (`SR`, `sqnorm`, `@elapsed_ns`, `block_sizes_indices`, `reordercostsforschur!`, …) — these were never part of the public API.

## Examples

- [`examples/rosenbrock.jl`](examples/rosenbrock.jl) — optimizes the 2D Rosenbrock function with each iterator and visualises the trajectories; also shows gradient descent as a user-defined iterator.
- [`examples/robustifiers.jl`](examples/robustifiers.jl) — plots the robust kernels and their derivatives.
- [`examples/adaptivekernel.jl`](examples/adaptivekernel.jl) — fits a `ContaminatedGaussian` adaptive kernel to data.
