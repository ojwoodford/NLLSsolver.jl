module NLLSsolver

using SecondOrderOptimization
using NLLSModels
using Static, StaticArrays, SparseArrays, LinearAlgebra

import SecondOrderOptimization:
    optimize,
    OptimizationOptions, OptimizationResult, OptimizationWorkspace, OptimizationInternal,
    OptimizationIterator, newton, levenbergmarquardt, dogleg, gradientdescent, levenbergmarquardt_dtd, getiterator,
    NewtonData, NewtonDataRelinearize, DoglegData, LevenbergData, LevenbergDataTrial, MarquardtData, MarquardtDataTrial, GradientDescentData,
    OptimizationLinearSolver, auto_solver, cholesky_solver, ldl_solver, qr_solver,
    CostTrajectory,
    nullcallback, printoutcallback, storecostscallback,
    nvars, update, update!, zero!, solve!,
    copy_structure, getgrad, getA, getb, gethessian, sharex,
    reset!, construct, iterate!, linearize!,
    timestring, bytesstring,
    @elapsed_ns, @stats

import NLLSModels:
    NLLSProblem,
    AbstractCost, AbstractResidual, AbstractAdaptiveResidual, AbstractRobustifier, AbstractAdaptiveRobustifier,
    NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel, GemanMcclure2oKernel, ContaminatedGaussian,
    EuclideanVector, DynamicVector, ZeroToInfScalar, ZeroToOneScalar,
    SimpleError2, SimpleError3, SimpleError4,
    BlockSparseMatrix, BlockDenseMatrix, VectorRepo,
    addvariable!, addcost!, updatevarcostmap!, subproblem, subproblem!,
    cost, costgradhess!, computeresidual, computeresjac, computecost, computecostgradhess,
    optimize!, optimizesingles!,
    getvars, varindices, generatemeasurement, ndeps, nres,
    robustkernel, robustify, robustifydcost, robustifydkernel,
    OptimizeCache, copyvars, isbitsvartype,
    NLLSOptions, NLLSResult, NLLSIterator, NLLSInternal,
    countcosts, costnum, costdeps, resnum,
    reordercostsforschur!, makesparseindices, block_sparsity,
    SR, valuedispatch, bindleadingargs, runlengthencodesortedints, fast_bAb,
    @unroll, @bitiset,
    LinearSystem, LinearSystemShared,
    UniVariateLSstatic_x, UniVariateLSstatic_ls, UniVariateLSdynamic_x, UniVariateLSdynamic_ls,
    MultiVariateLSdense, MultiVariateLSsparse,
    block_sizes_indices, sparse_dense_decision

# Re-export and extend public API

# Export the public API
export optimize, OptimizationOptions, OptimizationResult, OptimizationWorkspace, OptimizationInternal
export OptimizationIterator, newton, levenbergmarquardt, dogleg, gradientdescent, levenbergmarquardt_dtd, getiterator
export NewtonData, NewtonDataRelinearize, DoglegData, LevenbergData, LevenbergDataTrial, MarquardtData, MarquardtDataTrial, GradientDescentData
export OptimizationLinearSolver, auto_solver, cholesky_solver, ldl_solver, qr_solver
export CostTrajectory, nullcallback, printoutcallback, storecostscallback
export nvars, update, update!, zero!, solve!

export NLLSProblem
export AbstractCost, AbstractResidual, AbstractAdaptiveResidual, AbstractRobustifier, AbstractAdaptiveRobustifier
export NoRobust, Scaled, HuberKernel, Huber2oKernel, GemanMcclureKernel, GemanMcclure2oKernel, ContaminatedGaussian
export EuclideanVector, DynamicVector, ZeroToInfScalar, ZeroToOneScalar
export SimpleError2, SimpleError3, SimpleError4
export BlockSparseMatrix, BlockDenseMatrix, VectorRepo
export addvariable!, addcost!, updatevarcostmap!, subproblem, subproblem!
export cost, costgradhess!, computeresidual, computeresjac, computecost, computecostgradhess
export optimize!, optimizesingles!
export getvars, varindices, generatemeasurement, ndeps, nres
export robustkernel, robustify, robustifydcost, robustifydkernel
export OptimizeCache, copyvars, isbitsvartype
export NLLSOptions, NLLSResult, NLLSIterator, NLLSInternal
export @elapsed_ns, @stats, @unroll, @bitiset

# Also import any remaining unexported symbols from both modules for full internal test compatibility
for mod in (SecondOrderOptimization, NLLSModels)
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

end # module NLLSsolver
