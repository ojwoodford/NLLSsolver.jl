using Test, StaticArrays, Static, NLLSsolver

# f(x) = x - 3, NaN past a wall at x = 0.5 (the minimizer x = 3 is behind it). A trial whose cost is NaN
# must be rejected like a higher one: the optimization continues from the last finite point and ends at
# the wall, rather than stopping at the start on the first NaN trial.
struct NaNWall <: AbstractResidual end
NLLSsolver.ndeps(::NaNWall) = static(1)
NLLSsolver.nres(::NaNWall) = static(1)
NLLSsolver.varindices(::NaNWall) = SVector(1)
NLLSsolver.getvars(::NaNWall, vars::Vector) = (vars[1]::Float64,)
Base.eltype(::NaNWall) = Float64
NLLSsolver.computeresidual(::NaNWall, x) = SVector(x > 0.5 ? NaN : x - 3.0)
NLLSsolver.computeresjac(::StaticInt, r::NaNWall, x) = (NLLSsolver.computeresidual(r, x), @SMatrix [1.0])

@testset "nantrial.jl" begin
    for iterator in (NLLSsolver.levenbergmarquardt, NLLSsolver.levenbergmarquardt_dtd, NLLSsolver.dogleg)
        problem = NLLSProblem(Float64, NaNWall)
        addvariable!(problem, 0.0)
        addcost!(problem, NaNWall())
        result = optimize!(problem, NLLSOptions(iterator = iterator, numthreads = static(1)))
        x = problem.variables[1]
        @test 0.4 < x <= 0.5                     # advanced to the wall, not stuck at the start
        @test isfinite(result.bestcost) && result.bestcost < 0.5 * 9.0
        @test result.bestcost == cost(problem)
    end
end
