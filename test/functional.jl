using Test, StaticArrays, SparseArrays, Static
import NLLSsolver

# Define the Rosenbrock cost function
struct RosenbrockA <: NLLSsolver.AbstractResidual
    a::Float64
end
NLLSsolver.ndeps(::RosenbrockA) = static(1)
NLLSsolver.nres(::RosenbrockA) = 1
NLLSsolver.varindices(::RosenbrockA) = SVector(1)
NLLSsolver.getvars(::RosenbrockA, vars::Vector) = (vars[1]::Float64,)
NLLSsolver.computeresidual(res::RosenbrockA, x) = res.a - x
Base.eltype(::RosenbrockA) = Float64
const rosenbrockrobustifier = NLLSsolver.Scaled(NLLSsolver.Huber2oKernel(1.6), 1.0)
NLLSsolver.robustkernel(::RosenbrockA) = rosenbrockrobustifier

struct RosenbrockB <: NLLSsolver.AbstractResidual
    b::Float64
end
NLLSsolver.ndeps(::RosenbrockB) = static(2)
NLLSsolver.nres(::RosenbrockB) = static(1)
NLLSsolver.varindices(::RosenbrockB) = SVector(1, 2)
NLLSsolver.getvars(::RosenbrockB, vars::Vector) = (vars[1]::Float64, vars[2]::Float64)
NLLSsolver.computeresidual(res::RosenbrockB, x, y) = SVector(res.b * (x ^ 2 - y))
Base.eltype(::RosenbrockB) = Float64

@testset "functional.jl" begin
    # Create the problem
    problem = NLLSsolver.NLLSProblem(Float64)
    @test NLLSsolver.addvariable!(problem, 0.) == 1
    @test NLLSsolver.addvariable!(problem, 0.) == 2
    NLLSsolver.addcost!(problem, RosenbrockA(1.0))
    @test NLLSsolver.countcosts(NLLSsolver.costnum, problem.costs) == 1
    @test NLLSsolver.countcosts(NLLSsolver.resnum, problem.costs) == 1
    NLLSsolver.addcost!(problem, RosenbrockB(10.))
    @test NLLSsolver.countcosts(NLLSsolver.costnum, problem.costs) == 2
    @test NLLSsolver.countcosts(NLLSsolver.resnum, problem.costs) == 2
    @test NLLSsolver.cost(problem) == 0.5
    @test problem.varcostmapvalid == false
    NLLSsolver.updatevarcostmap!(problem)
    @test problem.varcostmapvalid == true
    @test vec(sum(Matrix(problem.varcostmap); dims=2)) == [2; 1]

    # Create a subproblem
    @test NLLSsolver.countcosts(NLLSsolver.resnum, NLLSsolver.subproblem(problem, trues(2)).costs) == 2
    subprob = NLLSsolver.subproblem(problem, 2)
    @test NLLSsolver.countcosts(NLLSsolver.resnum, subprob.costs) == 1
    @test NLLSsolver.cost(subprob) == 0.

    # Check callback termination
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(callback=(cost, unusedargs...)->(cost, 13)))
    @test result.termination == (13 << 9)
    @test result.niterations == 1

    # Test optimization
    for iter in [NLLSsolver.newton, NLLSsolver.levenbergmarquardt, NLLSsolver.dogleg]
        # Set the start
        problem.variables[1] = -0.5
        problem.variables[2] = 2.5

        # Optimize the cost
        result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=iter))

        # Check the result
        @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
        @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)
    end

    # Test standard gradient descent (a worse optimizer, so needs a closer starting point)
    problem.variables[1] = 1.0 - 1.e-5
    problem.variables[2] = 1.0
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.gradientdescent))
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-4)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-4)
end
