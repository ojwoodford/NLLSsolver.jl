using Test, NLLSsolver
import NLLSModel, SecondOrderOptimization, OptimizationVariableInterface

@testset "exports.jl" begin
    # names(NLLSsolver) is exactly the owned names plus every exported name of the three re-exported packages
    # (minus their module names, and minus optimize/optimize!, which NLLSsolver owns).
    owned = Set([:NLLSsolver, :optimize!, :optimize, :NLLSProblem, :NLLSOptions, :NLLSResult, :NLLSIterator, :EuclideanVector, :OptimizeCache])
    reexported = union((Set(filter(n -> n !== nameof(m), names(m))) for m in (NLLSModel, SecondOrderOptimization, OptimizationVariableInterface))...)
    expected = union(owned, setdiff(reexported, Set([:optimize, :optimize!])))
    @test Set(names(NLLSsolver)) == expected

    # Every name NLLSsolver 5.3.4 exported is still defined, except the accepted breaks.
    names_534 = [:AbstractCost, :AbstractResidual, :AbstractRobustifier,
                 :NLLSProblem, :NLLSOptions, :NLLSResult, :NLLSIterator, :OptimizeCache,
                 :NoRobust, :Scaled, :HuberKernel, :Huber2oKernel, :GemanMcclureKernel, :GemanMcclure2oKernel,
                 :ContaminatedGaussian, :EuclideanVector, :ZeroToInfScalar, :ZeroToOneScalar,
                 :SimpleError2, :SimpleError3, :SimpleError4, :CostTrajectory,
                 :addcost!, :addvariable!, :subproblem, :subproblem!, :nvars, :nres,
                 :update, :ndeps, :varindices, :getvars, :generatemeasurement,
                 :robustkernel, :robustify, :robustifydcost, :robustifydkernel,
                 :cost, :computeresidual, :computeresjac, :computecost, :computecostgradhess,
                 :optimize!, :nullcallback, :printoutcallback, :storecostscallback]
    for n in names_534
        @test isdefined(NLLSsolver, n)
    end
    # Accepted break: updatevarcostmap! was exported by 5.3.4 and is gone.
    @test !isdefined(NLLSsolver, :updatevarcostmap!)

    # Each owned name has a docstring.
    hasdoc(n) = haskey(Base.Docs.meta(NLLSsolver), Base.Docs.Binding(NLLSsolver, n))
    for n in (:optimize!, :optimize, :NLLSProblem, :NLLSOptions, :NLLSResult, :NLLSIterator, :EuclideanVector, :OptimizeCache)
        @test hasdoc(n)
    end
end
