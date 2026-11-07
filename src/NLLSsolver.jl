module NLLSsolver

using BlockLinearSystems
using NLLSModel
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

import BlockLinearSystems:
    SparseBlockLinearSystem, DenseBlockLinearSystem,
    block, zero!, copy_structure, uniformscaling!,
    symmetrifyfull, symmetrifyfull!, computestartindices

import NLLSModel:
    Model,
    AbstractCost, AbstractResidual, AbstractAdaptiveResidual, AbstractRobustifier, AbstractAdaptiveRobustifier,
    NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel, GemanMcclure2oKernel, ContaminatedGaussian,
    SimpleError2, SimpleError3, SimpleError4,
    CostStruct,
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
const NLLSProblem           = Model
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
export Model, NLLSProblem, NLLSOptions, NLLSResult, NLLSIterator, OptimizeCache # Concrete problem & solver types
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
for mod in (SecondOrderOptimization, NLLSModel, BlockLinearSystems, OptimizationVariableInterface)
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
    NLLSModel.optimize(kernel, squarederrors, maxiters)
BlockLinearSystems.zero!(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.zero!(ls)
BlockLinearSystems.copy_structure(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.copy_structure(ls)
BlockLinearSystems.uniformscaling!(ls::SecondOrderOptimization.AbstractLinearSystem, k) = SecondOrderOptimization.uniformscaling!(ls, k)
SecondOrderOptimization.copy_structure(x) = BlockLinearSystems.copy_structure(x)
SecondOrderOptimization.getx(x) = NLLSModel.getx(x)
SecondOrderOptimization.getA(x) = NLLSModel.getA(x)
SecondOrderOptimization.getb(x) = NLLSModel.getb(x)
SecondOrderOptimization.getgrad(x) = NLLSModel.getgrad(x)
SecondOrderOptimization.gethessian(x) = NLLSModel.gethessian(x)
SecondOrderOptimization.gethessgrad(x) = NLLSModel.gethessgrad(x)
NLLSModel.getx(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.getx(ls)
NLLSModel.getA(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.getA(ls)
NLLSModel.getb(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.getb(ls)
NLLSModel.getgrad(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.getgrad(ls)
NLLSModel.gethessian(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.gethessian(ls)
NLLSModel.gethessgrad(ls::SecondOrderOptimization.AbstractLinearSystem) = SecondOrderOptimization.gethessgrad(ls)

NLLSModel.getoffsets(block, ::SecondOrderOptimization.StaticLinearSystem) = SVector(UInt(1))
function NLLSModel.updatesymlinearsystem!(linsystem::SecondOrderOptimization.StaticLinearSystem, g, H, unusedargs...)
    getb(linsystem) .+= g
    getA(linsystem) .+= H
end
function NLLSModel.costgradhess!(linsystem::SecondOrderOptimization.DenseLinearSystem, vars::AbstractVector, costs::NLLSModel.CostStruct)
    zero!(linsystem)
    n = length(vars)
    blocksizes, blockindices = block_sizes_indices(vars, trues(n), n)
    boffsets = computestartindices(blocksizes)
    push!(boffsets, boffsets[end] + blocksizes[end])
    bdm = DenseBlockLinearSystem{Float64}(linsystem.A, boffsets)
    ls = LinearSystem(NLLSModel.MultiVariateLSdense_x(linsystem.x, blockindices, boffsets), NLLSModel.MultiVariateLSdense_ls(bdm, linsystem.b))
    c = NLLSModel.costgradhess_nozero!(ls, vars, costs)
    symmetrifyfull!(bdm)
    return c
end

end # module NLLSsolver
