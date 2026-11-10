using Test, StaticArrays, Static, NLLSsolver

# r(x) = [acos(min(x, 1)), x - 2]: the cost is finite everywhere, but for x >= 1 the derivative of the first residual
# is acos'(1) * 0 = -Inf * 0 = NaN, so the gradient and Hessian there are NaN. From x = 0 the first step crosses x = 1
# and lowers the cost, so it is accepted, and every step computed at that point is NaN. The iterator must return
# (keeping the best, finite point) rather than raising lambda forever.
struct ClampedAcos <: AbstractResidual end
NLLSsolver.ndeps(::ClampedAcos) = static(1)
NLLSsolver.nres(::ClampedAcos) = static(2)
NLLSsolver.varindices(::ClampedAcos) = SVector(1)
NLLSsolver.getvars(::ClampedAcos, vars::Vector) = (vars[1]::Float64,)
Base.eltype(::ClampedAcos) = Float64
NLLSsolver.computeresidual(::ClampedAcos, x) = SVector(acos(min(x, 1.0)), x - 2.0)

@testset "nanlinearization.jl" begin
    for iterator in (NLLSsolver.levenbergmarquardt, NLLSsolver.levenbergmarquardt_dtd)
        problem = NLLSProblem(Float64, ClampedAcos)
        addvariable!(problem, 0.0)
        addcost!(problem, ClampedAcos())
        result = optimize!(problem, NLLSOptions(iterator = iterator, numthreads = static(1)))   # returns, rather than spinning
        @test result.termination & (1 << 5) != 0          # stopped on the NaN step
        @test isfinite(result.bestcost) && result.bestcost < result.startcost
        @test result.bestcost == cost(problem)             # the variables are the best ones found
    end
end
