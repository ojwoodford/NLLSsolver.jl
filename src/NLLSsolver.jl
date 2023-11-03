module NLLSsolver
# Export the API
# Types
export AbstractCost, AbstractResidual, AbstractRobustifier # Abstract problem definition types
export NLLSProblem, NLLSOptions, NLLSResult, NLLSIterator # Concrete problem & solver types
export AbstractRobustifier, NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel # Robustifiers
export ContaminatedGaussian # Adaptive robustifiers
export EuclideanVector, ZeroToInfScalar, ZeroToOneScalar # Concrete general variables
export SimpleError2, SimpleError3, SimpleError4 # Measurement error residuals depending on 2, 3, and 4 variables
export CostTrajectory # Object for storing opimization trajectory and costs
# Functions
export addcost!, addvariable!, updatevarcostmap!, subproblem, subproblem!, nvars, nres # Construct a problem
export update, nvars # Variable interface
export nres, ndeps, varindices, getvars, generatemeasurement # Residual interface
export robustkernel, robustify, robustifydcost, robustifydkernel # Robustifier interface
export cost, computeresidual, computeresjac, computecost, computecostgradhess # Compute the objective
export optimize!  # Optimize the objective
export nullcallback, printoutcallback, storecostscallback # Callbacks to be called once per optimization iteration

# Exported abstract types
abstract type AbstractCost end  # Standard (non-squared) cost
abstract type AbstractResidual <: AbstractCost end # Squared (or robustified squared) cost
abstract type AbstractAdaptiveResidual <: AbstractResidual end # Squared cost with adaptive robustifier
abstract type AbstractRobustifier end # Robustifier with fixed parameters
abstract type AbstractAdaptiveRobustifier <: AbstractRobustifier end # Robustifier with variable parameters

# Constants
const MAX_ARGS = 10 # Maximum number of variables a residual can depend on
const MAX_BLOCK_SZ = 32 # Maximum DoF of a variable and also maximum length of a residual allowed for various operations, such as marginalization
const MAX_STATIC_VAR = 64 # Maximum static variable size for static sized autodiff to be used

# Helper functions
include("utils.jl")
include("unroll.jl")

# Problem definition
include("VectorRepo.jl")
include("problem.jl")

# Variables
include("variable.jl")

# Types
include("BlockSparseMatrix.jl")
include("BlockDenseMatrix.jl")
include("linearsystem.jl")
include("structs.jl")

# Optimization
include("residual.jl")
include("robust.jl")
include("robustadaptive.jl")
include("autodiff.jl")
include("cost.jl")
include("linearsolver.jl")
include("callbacks.jl")
include("iterators.jl")
include("optimize.jl")
end
