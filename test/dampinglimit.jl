using Test, StaticArrays, Static, NLLSsolver

# r(x) = 1e100 at x = 0 and 2e100 elsewhere, with a Jacobian of 1: every step from x = 0 raises the cost. The
# Levenberg-Marquardt inner loop raises lambda after each failed trial, and the steps, -1e100 / (1 + lambda), stay
# above dstep until lambda exceeds 1e115, some 30 trials. The loop must stop at its bound on lambda (1e16, about
# 12 trials) with no step, rather than return a trial that raises the cost.
struct StepUp <: AbstractResidual end
NLLSsolver.ndeps(::StepUp) = static(1)
NLLSsolver.nres(::StepUp) = static(1)
NLLSsolver.varindices(::StepUp) = SVector(1)
NLLSsolver.getvars(::StepUp, vars::Vector) = (vars[1]::Float64,)
Base.eltype(::StepUp) = Float64
NLLSsolver.computeresidual(::StepUp, x) = SVector(x == 0.0 ? 1e100 : 2e100)
NLLSsolver.computeresjac(::StaticInt, r::StepUp, x) = (NLLSsolver.computeresidual(r, x), @SMatrix [1.0])

@testset "dampinglimit.jl" begin
    for iterator in (NLLSsolver.levenbergmarquardt, NLLSsolver.levenbergmarquardt_dtd)
        problem = NLLSProblem(Float64, StepUp)
        addvariable!(problem, 0.0)
        addcost!(problem, StepUp())
        costs = Vector{Float64}()
        result = optimize!(problem, NLLSOptions(iterator = iterator, numthreads = static(1)), nothing, storecostscallback(costs))
        @test result.linearsolves < 20               # the inner loop stopped at its bound
        @test result.termination & (1 << 6) != 0     # with no step
        @test costs == [result.startcost]            # one iteration, which did not raise the cost
        @test problem.variables[1] == 0.0
        @test result.bestcost == result.startcost
    end
end
