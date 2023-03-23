# NLLSsolver

[![Build Status](https://github.com/ojwoodford/NLLSsolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/NLLSsolver.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ojwoodford/NLLSsolver.jl/branch/main/graph/badge.svg?token=1556CDMEGH)](https://codecov.io/gh/ojwoodford/NLLSsolver.jl)

A package for optimizing robustified Non-Linear Least Squares (NLLS) problems, with the following features:
- **Simple interface**: Large scale problems can be defined with relatively little code. Derivatives are computed automatically (though can be user-provided if desired).
- **Robust**: Residual blocks can be robustified easily.
- **Non-Euclidean variables**: Variables do not need to exist in a Euclidean space. For example, 3D rotations can be represented as a 9 parameter SO(3) matrix, yet retain a minimal 3DoF update parameterization.

Features not currently supported:
- **Constraints**: Bounds on variables are not explicitly supported. However, bounded variables can be implemented using special, non-Euclidean parameterizations.

## Usage

### Problem definition
Each NLLS problem is defined using two types of data structure:
- **Variable blocks**, which contain the parameters to be optimized.
- **Residual blocks**, which contain the data that defines the NLLS function to be minimized w.r.t. the variables.

Each instance of these two types must implement a standard API, as follows.

#### Variable blocks
- **`N::Int = nvars(::MyVar)`** returns the intrinsic dimensionality, N, of the variable block.
- **`newvar::MyVar = update(oldvar::MyVar, updatevec)`** updates a variable, given an update vector of length N.

#### Residual blocks
- **`N::Int = nvars(::MyRes)`** returns the number of variable blocks the residual block depends on.
- **`M::Int = nres(::MyRes)`** returns the number of scalar residuals in the block.
- **`varind::SVector{N, Int} = varindices(res::MyRes)`** returns the indices of the variable blocks (stored in problem) that this residual block depends on. These values are assumed to remain fixed for the duration of an optimization.
- **`resvars::Tuple = getvars(res::MyRes, allvars::Vector)`** returns a tuple containing the variables the residual block depends on.
- **`res::SVector{M, Float} = computeresidual(res::MyRes, resvars...)`** returns the computed residual block.
- **`res::SVector{M, Float}, jac::SMatrix{M, P, Float} = computeresjac(::Val{varflags}, res::MyRes, resvars...)`** (*optional*) returns the computed residual block and its Jacobian, for all the variables whose corresponding bit in `varflags` is set. If this function isn't provided, the Jacobian is computed using ForwardDiff auto-differentiation.
- **`robker = robustkernel(res::MyRes)`** (*optional*) returns the robust kernel data structure for the residual block. If this function isn't provided, the cost is the squared norm of the residual block.

A problem is then defined by creating an `NLLSProblem` object, and adding variables and residuals to it using the `addvariable!` and `addresidual!` methods respectively.

### Optimization
Optimization is done as follows:
```
    result::NLLSResult = optimize!(problem::NLLSProblem, options::NLLSOptions)
```
Various optimizer `options` can be defined. During optimization, the optimizer updates variable blocks (stored in `problem`) in-place. Information about the optimization is retruned in `result`.

## Examples
The following examples of problem definition, creation and optimization are included:
- **Rosenbrock function** (examples/rosenbrock.jl): Visualizes optimization of the Rosenbrock function using some of the  available optimizers. Click on the parameter space to interactively select a new start point.
- **Bundle adjustment** (examples/bundleadjustment.jl): Optimization of large scale [Bundle Adjustement in the Large](https://grail.cs.washington.edu/projects/bal/) problems, with non-Euclidean variables.

## Future work & collaboration
- **Add Schur complement** to speed up optimization of bipartite problems.
- **Add Variable Projection method** for solving bipartite problems.
- **Implement reduced memory Variable Projection** for solving very large scale bipartite problems.
- **Allow residuals to dynamically change the variables they depend on** to broaden the types of problems that can be optimized.
- **Add additional solvers**
- **Add constraints**, such as equality and inequality constraints on variables.
- **Improve code coverage of tests**
- **Add separate docs pages** with much more detail

Help is needed to improve both the functionality of the package, and also the documentation and test framework. If you wish to collaborate on this project, please raise an issue, indicating what you would like to help with.
