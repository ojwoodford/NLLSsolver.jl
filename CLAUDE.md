# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```
julia --project=. -e 'using Pkg; Pkg.test()'
```
or from within Julia REPL:
```julia
] test
```

**Run a single test file:**
```
julia --project=. -e 'include("test/functional.jl")'
```
Note: individual test files use `using Test` and `using NLLSsolver` - you may need to add those manually if running standalone.

**Start Julia with the package loaded:**
```
julia --project=.
```

**Run an example:**
```
julia --project=. examples/rosenbrock.jl
```
(Examples require GLMakie; install with `] add GLMakie` or use the `examples` target.)

## Architecture

### Core Abstractions

The package models optimization problems as a set of **variable blocks** and **cost blocks**:

- **`NLLSProblem{VarTypes, CostTypes}`** (`src/problem.jl`): The top-level container. Holds `variables::Vector` and `costs::CostStruct` (a `VectorRepo`). Also maintains `varnext`/`varbest` for optimizer bookkeeping, and a sparse `varcostmap` mapping variables to costs.

- **`VectorRepo{T}`** (`src/VectorRepo.jl`): A `Dict{DataType, Vector}` that stores heterogeneous typed collections. Costs of each concrete type are stored as a separate `Vector{T}`. Iteration over `VectorRepo` dispatches statically when the union type `T` is known, enabling zero-overhead dispatch across cost types.

### Variable Interface (`src/variable.jl`)
User-defined variables must implement:
- `nvars(var)::Union{Int, StaticInt}` — intrinsic DoF (prefer `StaticInt` for performance)
- `update(var, updatevec, start=1)` — returns new variable given a step vector

Built-in variable types: `EuclideanVector` (alias for `SVector`), `DynamicVector`, `ZeroToInfScalar`, `ZeroToOneScalar`.

### Cost Block Interface
Cost blocks are typed structs that implement the abstract types:

- **`AbstractResidual`** — for robustified least-squares blocks. Must implement `ndeps(::StaticInt)`, `nres(::Union{Int,StaticInt})`, `varindices()`, `getvars()`, `computeresidual()`. Optionally: `computeresjac()` (falls back to ForwardDiff autodiff), `robustkernel()`.

- **`AbstractCost`** — for general scalar costs. Must implement `ndeps()`, `varindices()`, `getvars()`, `computecost()`. Optionally: `computecostgradhess()`.

- **`AbstractAdaptiveResidual`** — like `AbstractResidual`, but the robustifier is a variable returned as the first element of `getvars()`.

### Linear System Types (`src/linearsystem.jl`)
The optimizer builds and solves a linear system each iteration:
- **`UniVariateLSstatic{N,N2}`** / **`UniVariateLSdynamic`**: For single-variable optimization; uses `MMatrix`/`Matrix` for the Hessian.
- **`MultiVariateLSsparse`**: Block sparse Hessian using `BlockSparseMatrix` + LDL factorization.
- **`MultiVariateLSdense`**: Block dense Hessian using `BlockDenseMatrix`.

The choice between sparse and dense is automatic based on problem size (threshold: total DoF ≥ 40).

### Optimization Flow (`src/optimize.jl`, `src/iterators.jl`)
`optimize!(problem, options, unfixed, callback)` dispatches to `optimizeinternal!`, which:
1. Calls `preoptimization` (iterator-specific setup)
2. Builds the linear system via `costgradhess!` (`src/cost.jl`)
3. Iterates: solve linear system → update variables → check termination
4. Restores best variables on exit

**Available iterators** (set via `NLLSOptions(iterator=...)`):
- `levenbergmarquardt` (default) — adaptive damping
- `newton` — undamped Newton steps
- `dogleg` — trust region with Cauchy/Newton steps
- `gradientdescent` — steepest descent with line search

### Robustifiers (`src/robust.jl`, `src/robustadaptive.jl`)
Standard: `NoRobust`, `Scaled`, `HuberKernel`, `Huber2oKernel`, `GemanMcclureKernel`.
Adaptive: `ContaminatedGaussian` — robustifier parameters are variables optimized jointly.

### Key Constants (`src/NLLSsolver.jl`)
- `MAX_ARGS = 10`: maximum number of variable dependencies per cost block
- `MAX_BLOCK_SZ = 32`: maximum DoF for a single variable/residual
- `MAX_STATIC_VAR = 64`: threshold for switching to static-sized autodiff

### Performance Notes
- Use `StaticInt` returns from `nvars()` and `nres()` wherever possible — enables compile-time unrolling and static dispatch.
- The `@unroll` macro (`src/unroll.jl`) unrolls loops up to `MAX_ARGS` iterations at compile time for inner loops over variable dependencies.
- `LoopVectorization` and `OhMyThreads` are used for SIMD and multi-threaded cost accumulation (`src/multithreading.jl`).
- Autodiff (`src/autodiff.jl`) uses ForwardDiff with static dual numbers when variable size ≤ `MAX_STATIC_VAR`.
