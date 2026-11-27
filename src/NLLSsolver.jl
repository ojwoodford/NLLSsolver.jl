"""
    NLLSsolver

**Deprecated.** NLLSsolver has been split into two packages:

- [NLLSModel.jl](https://github.com/ojwoodford/NLLSModel.jl) — the model: variables, residuals,
  robust and adaptive kernels, automatic and analytic differentiation, cost/gradient/Hessian assembly.
- [SecondOrderOptimization.jl](https://github.com/ojwoodford/SecondOrderOptimization.jl) — the optimizer:
  iterators (Levenberg–Marquardt, Dogleg, Newton), step control, the loop, callbacks, stats and workspace.

NLLSsolver 6 is a thin compatibility layer over the two: every call forwards to them, and every name they
export is re-exported here, so their features are reachable through NLLSsolver without it naming them. It is
the last major version and receives compatibility fixes only. New code should depend on NLLSModel.jl and
SecondOrderOptimization.jl directly.

See the README for the translation guide and the behaviour that changed from NLLSsolver 5.
"""
module NLLSsolver

import NLLSModel
import SecondOrderOptimization
import OptimizationVariableInterface
using StaticArrays: SVector

# --- Names NLLSsolver owns (exported with the 5.3.4 spelling; each names its replacement) -------------

"""
    NLLSProblem(VariableTypes, CostTypes) -> Model

Deprecated alias of [`Model`](@ref) (NLLSModel.jl). Build a model with
`Model(VariableTypes, CostTypes)`. The no-argument and variables-only forms of NLLSsolver 5
(costs of type `Any`) are not available: give the cost types.
"""
const NLLSProblem = NLLSModel.Model

"""
    NLLSOptions(; kwargs...) -> OptimizationOptions

Deprecated alias of `OptimizationOptions` (SecondOrderOptimization.jl). The 5.x keywords
`maxfails` and `numthreads` are gone: use `maxstall`/`stalltol`, and pass `numthreads` to
`optimize!`. The 5.x defaults are not restored (e.g. `reldcost` is `1e-12`).
"""
const NLLSOptions = SecondOrderOptimization.OptimizationOptions

"""
    NLLSResult

Deprecated alias of `OptimizationResult` (SecondOrderOptimization.jl). Same fields;
`result.iterator` is now a `Symbol`, and the termination field is read with `isconverged` and
the `TERMINATION_*` masks.
"""
const NLLSResult = SecondOrderOptimization.OptimizationResult

"""
    EuclideanVector{N, T} = SVector{N, T}

Deprecated alias of `StaticArrays.SVector`. Use `SVector{N, T}` directly.
"""
const EuclideanVector{N, T} = SVector{N, T}

"""
    NLLSsolver.DynamicVector{T} = Vector{T}

Deprecated (unexported) alias of `Vector`. Use `Vector{T}` directly.
"""
const DynamicVector{T} = Vector{T}

"""
    NLLSIterator

Deprecated enum of iterator names, kept so that `NLLSOptions(iterator = NLLSsolver.dogleg)` still
works. New code passes the iterator as a `Symbol` (`iterator = :dogleg`). The members
(`NLLSsolver.newton`, `NLLSsolver.levenbergmarquardt`, `NLLSsolver.dogleg`,
`NLLSsolver.levenbergmarquardt_dtd`) are unexported, as in NLLSsolver 5. Gradient descent is no
longer a built-in iterator: see `examples/gradientdescent.jl`.
"""
@enum NLLSIterator newton levenbergmarquardt dogleg levenbergmarquardt_dtd
export NLLSProblem, NLLSOptions, NLLSResult, NLLSIterator, EuclideanVector

# --- Re-export every exported name of the three packages (D7) -----------------------------------------
# One short loop, no explicit lists, so a new upstream export passes through. The module's own name and
# the owned `optimize`/`optimize!` are skipped; a name already bound here (an owned alias, or `syncvars!`
# which NLLSModel and OptimizationVariableInterface export as the same object) is skipped too.
for mod in (NLLSModel, SecondOrderOptimization, OptimizationVariableInterface)
    for name in names(mod)
        (name === nameof(mod) || name === :optimize || name === :optimize!) && continue
        isdefined(@__MODULE__, name) && continue
        @eval const $name = $(getfield(mod, name))
        @eval export $name
    end
end

# --- Owned entry points and compatibility shims -------------------------------------------------------
# `optimizesingles!` is owned by NLLSModel now (D6) and re-exported by the loop above, so there is no file
# for it here; `OptimizeCache` is the one 5.3.4 name the new packages have no equivalent of.
include("optimize.jl")
include("OptimizeCache.jl")

end # module NLLSsolver
