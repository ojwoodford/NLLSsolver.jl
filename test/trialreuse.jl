using Test, StaticArrays, Static, NLLSsolver

# Fit y = a exp(b x), with the residual and its Jacobian counted separately
const TRIAL_COUNTS = [0, 0] # computeresidual calls, computeresjac calls
struct ExpFit{V} <: AbstractResidual
    x::Float64
    y::Float64
end
NLLSsolver.ndeps(::ExpFit{SVector{2, Float64}}) = static(1)
NLLSsolver.ndeps(::ExpFit{Float64}) = static(2)
NLLSsolver.nres(::ExpFit) = static(1)
NLLSsolver.varindices(::ExpFit{SVector{2, Float64}}) = SVector(1)
NLLSsolver.varindices(::ExpFit{Float64}) = SVector(1, 2)
NLLSsolver.getvars(::ExpFit{SVector{2, Float64}}, vars::Vector) = (vars[1]::SVector{2, Float64},)
NLLSsolver.getvars(::ExpFit{Float64}, vars::Vector) = (vars[1]::Float64, vars[2]::Float64)
Base.eltype(::ExpFit) = Float64
expfit(r, a, b) = SVector(a * exp(b * r.x) - r.y)
NLLSsolver.computeresidual(r::ExpFit, ab) = (TRIAL_COUNTS[1] += 1; expfit(r, ab[1], ab[2]))
NLLSsolver.computeresidual(r::ExpFit, a, b) = (TRIAL_COUNTS[1] += 1; expfit(r, a, b))
function NLLSsolver.computeresjac(::StaticInt, r::ExpFit, ab)
    TRIAL_COUNTS[2] += 1
    e = exp(ab[2] * r.x)
    return expfit(r, ab[1], ab[2]), SMatrix{1, 2}(e, ab[1] * r.x * e)
end
function NLLSsolver.computeresjac(::StaticInt{3}, r::ExpFit, a, b)
    TRIAL_COUNTS[2] += 1
    e = exp(b * r.x)
    return expfit(r, a, b), SMatrix{1, 2}(e, a * r.x * e)
end

function expfitproblem(::Type{V}) where V
    problem = NLLSProblem(V, ExpFit{V})
    if V <: SVector
        addvariable!(problem, SVector(1.0, 0.0))
    else
        addvariable!(problem, 1.0)
        addvariable!(problem, 0.0)
    end
    for k in 0:20
        x = k / 10
        addcost!(problem, ExpFit{V}(x, 2.0 * exp(-1.3 * x) + 0.01 * sin(7k)))
    end
    return problem
end

blockdata(A) = A isa NLLSsolver.BlockDenseMatrix ? A.data : A

# Check, each iteration, that the trial linear system the iterator filled holds exactly the cost,
# gradient and Hessian at the point it returned
struct CheckTrial
    checked::Vector{Int}
end
function (cb::CheckTrial)(cost, problem, data, iteratedata)
    if iteratedata.trialready
        ref = NLLSsolver.copy_structure(iteratedata.trial)
        refcost = NLLSsolver.costgradhess!(ref, problem.varnext, problem.costs)
        @test refcost == cost
        @test NLLSsolver.getgrad(ref) == NLLSsolver.getgrad(iteratedata.trial)
        @test blockdata(NLLSsolver.getA(ref)) == blockdata(NLLSsolver.getA(iteratedata.trial))
        cb.checked[1] += 1
    end
    return cost, 0
end

@testset "trialreuse.jl" begin
    for V in (SVector{2, Float64}, Float64), iterator in (:levenbergmarquardt, :levenbergmarquardt_dtd)
        options = NLLSOptions(iterator=iterator, reldcost=1e-12)
        # Reference: the gradient recomputed after every accepted step (a callback that clears the iterator's
        # trialready stops the optimizer from swapping in the trial linear system)
        problem = expfitproblem(V)
        fill!(TRIAL_COUNTS, 0)
        refresult = optimize!(problem, options, nothing, (cost, problem, data, iteratedata) -> (iteratedata.trialready = false; (cost, 0)))
        refcounts = copy(TRIAL_COUNTS)
        refvars = copy(problem.variables)
        # Reuse
        problem = expfitproblem(V)
        fill!(TRIAL_COUNTS, 0)
        result = optimize!(problem, options)
        # The same iterates, with fewer residual evaluations: an accepted trial point's residuals are not recomputed
        @test problem.variables == refvars
        @test result.bestcost == refresult.bestcost
        @test result.niterations == refresult.niterations
        @test result.gradientcomputations < refresult.gradientcomputations
        @test TRIAL_COUNTS[1] == refcounts[1]
        @test TRIAL_COUNTS[2] < refcounts[2]
        @test isapprox(cost(problem), 0.0; atol=0.01)
        # The trial linear system is exactly the one at the trial point
        problem = expfitproblem(V)
        cb = CheckTrial([0])
        optimize!(problem, options, nothing, cb)
        @test cb.checked[1] > 0
        @test problem.variables == refvars
    end
    # A callback that changes the cost: the trial system is not used, the gradient is recomputed
    for iter in (:levenbergmarquardt, :levenbergmarquardt_dtd)
        problem = expfitproblem(SVector{2, Float64})
        result = optimize!(problem, NLLSOptions(iterator=iter, callbackchangesstate=true), nothing, (cost, unusedargs...) -> (nextfloat(cost), 0))
        @test result.gradientcomputations >= result.niterations
    end
    # Safety assertion when callback changes cost without callbackchangesstate=true
    problem = expfitproblem(Float64)
    @test_throws AssertionError optimize!(problem, NLLSOptions(callbackchangesstate=false), nothing, (cost, args...) -> (nextfloat(cost), 0))

    # Newton optimization with and without callbackchangesstate
    p1 = expfitproblem(SVector{2, Float64})
    res1 = optimize!(p1, NLLSOptions(iterator=:newton, callbackchangesstate=false))
    p2 = expfitproblem(SVector{2, Float64})
    res2 = optimize!(p2, NLLSOptions(iterator=:newton, callbackchangesstate=true))
    @test p1.variables ≈ p2.variables
    @test res1.bestcost ≈ res2.bestcost
    @test res1.costcomputations == 0
    @test res2.costcomputations > 0
end
