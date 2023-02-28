# NLLSsolver

[![Build Status](https://github.com/ojwoodford/NLLSsolver.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ojwoodford/NLLSsolver.jl/actions/workflows/CI.yml?query=branch%3Amain)

A package for optimizing robustified Non-Linear Least Squares problems, with the following features:
- **Simple interface**: Large scale problems can be defined with relatively little code. Derivatives are computed automatically (though can be user-provided if desired).
- **Efficient**: Designed with computational efficiency in mind, the package also supports Schur complement reduction and the Variable Projection method.
- **Non-Euclidean variables**: Variables do not need to exist in a Euclidean space. For example, 3D rotations can be represented as a 9 parameter SO(3) matrix, yet retain a minimal 3DoF update parameterization.
- **Robust**: Residuals can be robustified easily.
- **Flexible**: A callback called once per iteration can be used to (amongst other things) display visualizations, dynamically change both the assignment of variables to residuals, and which variables are fixed, as well as terminate the optimization.

Features not currently supported:
- **Constraints**: Bounds on variables are not explicitly supported. However, bounded variables can be implemented using special, non-Euclidean parameterizations.

## Examples to run:
- **Rosenbrock function** (examples/rosenbrock.jl): Visualizes optimization of the Rosenbrock function using all available optimizers. Click on the parameter space to interactively select a new start point.
- **Bundle adjustment** (examples/bundleadjustment.jl): Optimization of large scale [Bundle Adjustement in the Large](https://grail.cs.washington.edu/projects/bal/) problems.
