module NLLSsolver

using BlockMatrices
using FastLinAlg
using MicroInfo
using NLLSModels
using OptimizationVariableInterface
using SecondOrderOptimization
using ValueDispatch
using Static, StaticArrays, SparseArrays, LinearAlgebra

import SecondOrderOptimization:
    optimize,
    OptimizationOptions, OptimizationResult, OptimizationWorkspace, OptimizationInternal,
    OptimizationIterator, newton, levenbergmarquardt, dogleg, gradientdescent, levenbergmarquardt_dtd, getiterator,
    NewtonData, NewtonDataRelinearize, DoglegData, LevenbergData, LevenbergDataTrial, MarquardtData, MarquardtDataTrial, GradientDescentData,
    OptimizationLinearSolver, auto_solver, cholesky_solver, ldl_solver, qr_solver,
    CostTrajectory,
    nullcallback, printoutcallback, storecostscallback,
    solve!, sharex,
    reset!, construct, iterate!, linearize!

import MicroInfo:
    TimeCounter, StatsCounter, Stats,
    timestring, bytesstring,
    @elapsed_ns, @stats

import FastLinAlg:
    sqnorm, fast_dot, fast_norm, fast_bAb, cumsum!

import ValueDispatch:
    valuedispatch

import OptimizationVariableInterface:
    nvars, update, update!, copyvars, syncvars!, isbitsvartype,
    EuclideanVector, DynamicVector, ZeroToInfScalar, ZeroToOneScalar

import BlockMatrices:
    BlockSparseMatrix, BlockDenseMatrix,
    block, validblock, zero!, copy_structure, uniformscaling!,
    symmetrifyfull, symmetrifyfull!, symmetrifysparse,
    makesparseindices, cacheindices, clearindicescache, updatelastrowsym!,
    sparse_dense_decision, block_sparse_nnz, computestartindices, find_diag_inds

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
    countcosts, costnum, costdeps, resnum,
    reordercostsforschur!, block_sparsity,
    SR, bindleadingargs, runlengthencodesortedints,
    @unroll, @bitiset,
    LinearSystem, LinearSystemShared,
    UniVariateLSstatic_x, UniVariateLSstatic_ls, UniVariateLSdynamic_x, UniVariateLSdynamic_ls,
    MultiVariateLSdense, MultiVariateLSsparse,
    block_sizes_indices, getgrad, getA, getb, gethessian

# Backward-compatibility aliases
const NLLSOptions   = SecondOrderOptimization.OptimizationOptions
const NLLSResult    = SecondOrderOptimization.OptimizationResult
const NLLSIterator  = SecondOrderOptimization.OptimizationIterator
const NLLSInternal  = SecondOrderOptimization.OptimizationInternal

include("optimize.jl")

# Export the public API
export optimize, optimvars, OptimizationOptions, OptimizationResult, OptimizationWorkspace, OptimizationInternal
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
export addvariable!, addcost!, addresidual!, updatevarcostmap!, subproblem, subproblem!
export cost, costgradhess!, computeresidual, computeresjac, computecost, computecostgradhess
export optimize!, optimizesingles!
export getvars, varindices, generatemeasurement, ndeps, nres
export robustkernel, robustify, robustifydcost, robustifydkernel
export OptimizeCache, copyvars, syncvars!, isbitsvartype
export NLLSOptions, NLLSResult, NLLSIterator, NLLSInternal
export @elapsed_ns, @stats, @unroll, @bitiset

# Also import any remaining unexported symbols from component modules for full internal test compatibility
for mod in (SecondOrderOptimization, NLLSModels, BlockMatrices, FastLinAlg, MicroInfo, OptimizationVariableInterface, ValueDispatch)
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
