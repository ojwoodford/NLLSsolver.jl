module NLLSsolver

using BlockMatrices
using NLLSModels
using OptimizationVariableInterface
using SecondOrderOptimization
using Static, StaticArrays, SparseArrays

# Module aliases
const SecondOrderOptimizer = SecondOrderOptimization
const SecodOrderOptimizer = SecondOrderOptimization

import SecondOrderOptimization:
    optimize, optimize!,
    OptimizationOptions, OptimizationResult, OptimizationWorkspace, OptimizationInternal,
    OptimizationIterator, newton, levenbergmarquardt, dogleg, gradientdescent, levenbergmarquardt_dtd, getiterator,
    NewtonData, NewtonDataRelinearize, DoglegData, LevenbergData, LevenbergDataTrial, MarquardtData, MarquardtDataTrial, GradientDescentData,
    OptimizationLinearSolver, auto_solver, cholesky_solver, ldl_solver, qr_solver,
    CostTrajectory,
    nullcallback, printoutcallback, storecostscallback,
    reset!, construct

import OptimizationVariableInterface:
    nvars, update, update!, syncvars!,
    ZeroToInfScalar, ZeroToOneScalar

import BlockMatrices:
    BlockSparseMatrix, BlockDenseMatrix,
    block, zero!, copy_structure, uniformscaling!,
    symmetrifyfull, computestartindices

import NLLSModels:
    NLLSProblem,
    AbstractCost, AbstractResidual, AbstractAdaptiveResidual, AbstractRobustifier, AbstractAdaptiveRobustifier,
    NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel, GemanMcclure2oKernel, ContaminatedGaussian,
    SimpleError2, SimpleError3, SimpleError4,
    VectorRepo,
    addvariable!, addcost!, addresidual!, updatevarcostmap!, subproblem, subproblem!,
    cost, costgradhess!, computeresidual, computeresjac, computecost, computecostgradhess,
    optimvars,
    getvars, varindices, generatemeasurement, ndeps, nres,
    robustkernel, robustify, robustifydcost, robustifydkernel,
    countcosts, costnum,
    reordercostsforschur!,
    SR,
    @unroll, @bitiset,
    LinearSystem, LinearSystemShared,
    UniVariateLSstatic_x, UniVariateLSstatic_ls, UniVariateLSdynamic_x, UniVariateLSdynamic_ls,
    MultiVariateLSdense, MultiVariateLSsparse,
    block_sizes_indices, getgrad, getA, getb, gethessian

# Backward-compatibility aliases
const EuclideanVector{N, T} = SVector{N, T}
const DynamicVector{T}      = Vector{T}
const NLLSOptions   = SecondOrderOptimization.OptimizationOptions
const NLLSResult    = SecondOrderOptimization.OptimizationResult
const NLLSIterator  = SecondOrderOptimization.OptimizationIterator
const NLLSInternal  = SecondOrderOptimization.OptimizationInternal

include("optimize.jl")

# Export the API
# Types
export AbstractCost, AbstractResidual, AbstractRobustifier # Abstract problem definition types
export NLLSProblem, NLLSOptions, NLLSResult, NLLSIterator, OptimizeCache # Concrete problem & solver types
export AbstractRobustifier, NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel, GemanMcclure2oKernel # Robustifiers
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

# Also import any remaining unexported symbols from component modules for full internal test compatibility
for mod in (SecondOrderOptimization, NLLSModels, BlockMatrices, OptimizationVariableInterface)
    for sym in names(mod; all=true)
        s = string(sym)
        if !startswith(s, "#") && !startswith(s, "@") && !isdefined(@__MODULE__, sym) && sym !== :eval && sym !== :include
            try
                val = getfield(mod, sym)
                @eval const $sym = $val
            catch
            end
        end
    end
end

# Backward-compatibility method extensions
SecondOrderOptimization.construct(iter::StaticSymbol, problem, ls, cbchangesstate) = construct(iter, ls, cbchangesstate)
SecondOrderOptimization.optimize(kernel::ContaminatedGaussian{T}, squarederrors::Vector{T}, maxiters=10) where T =
    NLLSModels.optimize(kernel, squarederrors, maxiters)
BlockMatrices.zero!(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.zero!(ls)
BlockMatrices.copy_structure(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.copy_structure(ls)
BlockMatrices.uniformscaling!(ls::SecondOrderOptimization.AbstractLinearSystem, k) = SecondOrderOptimization.uniformscaling!(ls, k)
SecondOrderOptimization.copy_structure(x) = BlockMatrices.copy_structure(x)
SecondOrderOptimization.getx(x) = NLLSModels.getx(x)
SecondOrderOptimization.getA(x) = NLLSModels.getA(x)
SecondOrderOptimization.getb(x) = NLLSModels.getb(x)
SecondOrderOptimization.getgrad(x) = NLLSModels.getgrad(x)
SecondOrderOptimization.gethessian(x) = NLLSModels.gethessian(x)
SecondOrderOptimization.gethessgrad(x) = NLLSModels.gethessgrad(x)
NLLSModels.getx(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.getx(ls)
NLLSModels.getA(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.getA(ls)
NLLSModels.getb(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.getb(ls)
NLLSModels.getgrad(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.getgrad(ls)
NLLSModels.gethessian(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.gethessian(ls)
NLLSModels.gethessgrad(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.gethessgrad(ls)

NLLSModels.getoffsets(block, ::SecondOrderOptimization.StaticLinearSystem) = SVector(UInt(1))
function NLLSModels.updatesymlinearsystem!(linsystem::SecondOrderOptimization.StaticLinearSystem, g, H, unusedargs...)
    getb(linsystem) .+= g
    getA(linsystem) .+= H
end
function NLLSModels.costgradhess!(linsystem::SecondOrderOptimization.DenseLinearSystem, vars::AbstractVector, costs::NLLSModels.CostStruct)
    zero!(linsystem)
    n = length(vars)
    blocksizes, blockindices = block_sizes_indices(vars, trues(n), n)
    boffsets = computestartindices(blocksizes)
    push!(boffsets, boffsets[end] + blocksizes[end])
    bdm = BlockDenseMatrix{Float64}(linsystem.A, boffsets)
    ls = LinearSystem(NLLSModels.MultiVariateLSdense_x(linsystem.x, blockindices, boffsets), NLLSModels.MultiVariateLSdense_ls(bdm, linsystem.b))
    c = NLLSModels.costgradhess_nozero!(ls, vars, costs)
    symmetrifyfull(bdm)
    return c
end

end # module NLLSsolver
