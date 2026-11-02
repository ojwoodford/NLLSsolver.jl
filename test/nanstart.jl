using Test, StaticArrays, Static, NLLSsolver

# A start point outside the residuals' domain. A trial is accepted only when its cost is not greater than the best, and
# every comparison with a NaN best cost is false, so without a check at the start the iterators never return.
# NaNStart: the residual is NaN at the start (x = 1), finite and minimized at x = 3 further on.
# NaNJacobianStart: the residual is finite everywhere, its Jacobian NaN at the start.
struct NaNStart <: AbstractResidual end
struct NaNJacobianStart <: AbstractResidual end
for R in (NaNStart, NaNJacobianStart)
    @eval NLLSsolver.ndeps(::$R) = static(1)
    @eval NLLSsolver.nres(::$R) = static(1)
    @eval NLLSsolver.varindices(::$R) = SVector(1)
    @eval NLLSsolver.getvars(::$R, vars::Vector) = (vars[1]::Float64,)
    @eval Base.eltype(::$R) = Float64
end
NLLSsolver.computeresidual(::NaNStart, x) = SVector(x < 2.0 ? NaN : x - 3.0)
NLLSsolver.computeresjac(::StaticInt, r::NaNStart, x) = (NLLSsolver.computeresidual(r, x), @SMatrix [1.0])
NLLSsolver.computeresidual(::NaNJacobianStart, x) = SVector(x - 3.0)
NLLSsolver.computeresjac(::StaticInt, r::NaNJacobianStart, x) = (NLLSsolver.computeresidual(r, x), @SMatrix [x < 2.0 ? NaN : 1.0])

@testset "nanstart.jl" begin
    for R in (NaNStart, NaNJacobianStart),
        iterator in (NLLSsolver.levenbergmarquardt, NLLSsolver.levenbergmarquardt_dtd, NLLSsolver.dogleg,
                     NLLSsolver.newton, NLLSsolver.gradientdescent)
        problem = NLLSProblem(Float64, R)
        addvariable!(problem, 1.0)
        addcost!(problem, R())
        result = optimize!(problem, NLLSOptions(iterator = iterator, numthreads = static(1)))   # returns, rather than spinning
        @test result.termination & (1 << 11) != 0      # stopped at the start point
        @test result.niterations == 0
        @test problem.variables[1] == 1.0              # the variables are left as they were
        @test (result.termination & (1 << 1) != 0) == (R === NaNStart)   # the NaN-cost flag only where the cost is NaN
    end
end
