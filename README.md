# NLLSsolver

[![Build Status](https://github.com/ojwoodford/NLLSsolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/NLLSsolver.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/NLLSsolver.jl/branch/main/graph/badge.svg?token=1556CDMEGH)](https://codecov.io/gh/ojwoodford/NLLSsolver.jl)

A package for optimizing robustified Non-Linear Least Squares (NLLS) problems, with the following features:
- **Simple interface**: Large scale problems can be defined with relatively little code. Derivatives are computed automatically (though can be user-provided if desired).
- **Robust**: Residual blocks can be robustified easily.
- **Non-Euclidean variables**: Variables do not need to exist in a Euclidean space. For example, 3D rotations can be represented as a 9 parameter SO(3) matrix, yet retain a minimal 3DoF update parameterization.

The package is heavily inspired by, and very similar to, the [Ceres-Solver C++ library](http://ceres-solver.org/), in terms of interface and functionality. Ceres is used to optimize a large number of academic and commercial [problems](http://ceres-solver.org/users.html), mostly in the field of multi-view geometry.

Features not currently supported:
- **Constraints**: Bounds on variables are not explicitly supported. However, bounded variables can be implemented using special, non-Euclidean parameterizations.

## Usage

### Problem definition
Each NLLS problem is defined using two types of data structure:
- **Variable blocks**, which contain the parameters to be optimized.
- **Cost blocks**, which contain the data that defines the NLLS cost function to be minimized w.r.t. the variables. A single cost block generates a scalar cost as a function of a fixed (at compile-time) number of variable blocks, currently up to 10.

Each instance of these two types must implement a standard API, as follows.

#### Variable blocks
- **`N::Union{Int, StaticInt} = nvars(::MyVar)`** returns the intrinsic dimensionality, N, of the variable block. This must be fixed for the duration of the optimization. Returning a `::StaticInt{N}` will improve performance, so do so where possible.
- **`newvar::MyVar = update(oldvar::MyVar, updatevec::AbstractVector, start::Int=1)`** updates a variable, using the values in update vector `updatevec`, starting at index `start`. 

#### Cost blocks
- **`::StaticInt{N} = ndeps(::MyCost)`** returns the number of variable blocks the cost block depends on. This must be a compile-time constant of type `StaticInt`.
- **`varind::SVector{N, Int} = varindices(costblock::MyCost)`** returns the indices of the variable blocks (stored in problem) that this cost block depends on. These values are assumed to remain fixed for the duration of an optimization.
- **`blockvars::Tuple = getvars(costblock::MyCost, allvars::Vector)`** returns a tuple containing the variables the cost block depends on.

Cost blocks can either be robustified least-squares blocks (an instance of an `AbstractResidual`), or a standard scalar cost block (an instance of an `AbstractCost`).

`AbstractResidual` blocks should implement the following functions:
- **`M::Union{Int, StaticInt} = nres(::MyCost)`** (required for least squares costs only) returns the number of scalar residuals in the block. This must be fixed for the duration of the optimization. Returning a `::StaticInt{M}` will improve performance, so do so where possible.
- **`res::SVector{M, Float} = computeresidual(costblock::MyCost, blockvars...)`** returns the computed residual block.
- **`res::SVector{M, Float}, jac::SMatrix{M, P, Float} = computeresjac(::StaticInt{varflags}, res::MyCost, blockvars...)`** (*optional*) returns the computed residual block and its Jacobian, for all the variables whose corresponding bit in `varflags` is set. If this function isn't provided, the Jacobian is computed using ForwardDiff auto-differentiation. `P` is the total dimensionality of the unfixed variable blocks.
- **`robker::AbstractRobustifier = robustkernel(res::MyCost)`** (*optional*) returns the robust kernel data structure for the residual block. If this function isn't provided, the cost is the squared norm of the residual block.

A subtype of `AbstractResidual`, `AdaptiveAbstractResidual`, also exists. Instances of this abstract type should return a `robker::AbstractAdaptiveRobustifier` as the first value returned by the `getvars()` method, rather implementing the `robustkernel()` method. Otherwise, the API is similar to above. See examples/adaptivekernel.jl for an example.

`AbstractCost` blocks should implement the following functions:
- **`cost::Float = computecost(costblock::MyCost, blockvars...)`** returns the computed residual block.
- **`cost::Float, grad::SVector{P, Float}, hess::SMatrix{P, P, Float} = computecostgradhess(::StaticInt{varflags}, costblock::MyCost, blockvars...)`** (*optional*) returns the computed cost and its gradient and Hessian, for all the variables whose corresponding bit in `varflags` is set. If this function isn't provided, the derivatives are computed using ForwardDiff auto-differentiation. `P` is the total dimensionality of the unfixed variable blocks.

A problem is then defined by creating an `NLLSProblem` object, and adding variable blocks and cost blocks to it using the `addvariable!` and `addcost!` methods respectively.

### Optimization
Optimization is done as follows:
```
    result::NLLSResult = optimize!(problem::NLLSProblem, options::NLLSOptions)
```
Various optimizer `options` can be defined. During optimization, the optimizer updates variable blocks (stored in `problem`) in-place. Information about the optimization is retruned in `result`.

## Examples
The following examples of problem definition, creation and optimization are included:
- **Rosenbrock function** (examples/rosenbrock.jl): Visualizes optimization of the Rosenbrock function using some of the  available optimizers. Click on the parameter space to interactively select a new start point.

## Future work & collaboration
- **Allow residuals to dynamically change the variables they depend on** to broaden the types of problems that can be optimized.
- **Allow residual blocks to depend on a dynamic number of variables** to further broaden the types of problems that can be optimized.
- **Add additional solvers**
- **Add constraints**, such as equality and inequality constraints on variables.
- **Improve code coverage of tests**
- **Add separate docs pages** with much more detail

Help is needed to improve both the functionality of the package, and also the documentation and test framework. If you wish to collaborate on this project, please raise an issue, indicating what you would like to help with.
