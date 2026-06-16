using Test, StaticArrays, SparseArrays, Static, NLLSsolver

# Define the Rosenbrock cost function
struct RosenbrockA <: AbstractResidual
    a::Float64
end
NLLSsolver.ndeps(::RosenbrockA) = static(1)
NLLSsolver.nres(::RosenbrockA) = 1
NLLSsolver.varindices(::RosenbrockA) = SVector(1)
NLLSsolver.getvars(::RosenbrockA, vars::Vector) = (vars[1]::Float64,)
NLLSsolver.computeresidual(res::RosenbrockA, x) = res.a * (1 - x)
Base.eltype(::RosenbrockA) = Float64
const rosenbrockrobustifier = NLLSsolver.Scaled(Huber2oKernel(1.6), 1.0)
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
    problem = NLLSProblem(Float64)
    @test NLLSsolver.countcosts(NLLSsolver.costnum, problem.costs) == 0
    @test NLLSsolver.countcosts(NLLSsolver.costdeps, problem.costs) == 0
    @test NLLSsolver.countcosts(NLLSsolver.resnum, problem.costs) == 0
    @test addvariable!(problem, 0.) == 1
    addcost!(problem, RosenbrockA(1.0))
    @test NLLSsolver.countcosts(NLLSsolver.costnum, problem.costs) == 1
    @test NLLSsolver.countcosts(NLLSsolver.costdeps, problem.costs) == 1
    @test NLLSsolver.countcosts(NLLSsolver.resnum, problem.costs) == 1
    @test addvariable!(problem, 0.) == 2
    addcost!(problem, RosenbrockB(10.))
    @test NLLSsolver.countcosts(NLLSsolver.costnum, problem.costs) == 2
    @test NLLSsolver.countcosts(NLLSsolver.costdeps, problem.costs) == 3
    @test NLLSsolver.countcosts(NLLSsolver.resnum, problem.costs) == 2
    @test cost(problem) == 0.5
    @test problem.varcostmapvalid == false
    NLLSsolver.updatevarcostmap!(problem)
    @test problem.varcostmapvalid == true
    @test vec(sum(Matrix(problem.varcostmap); dims=2)) == [2; 1]
    # Create a subproblem
    @test NLLSsolver.countcosts(NLLSsolver.resnum, NLLSsolver.subproblem(problem, trues(2)).costs) == 2
    subprob = NLLSsolver.subproblem(problem, 2)
    @test NLLSsolver.countcosts(NLLSsolver.resnum, subprob.costs) == 1
    @test NLLSsolver.cost(subprob) == 0.

    # Optimize a subset of variables, such that some of the costs have no free variables
    @test_logs (:warn, "A cost for which all variables are fixed is being optimized. Remove such cost functions from the cost to avoid redundant computation, using the subproblem() method.") optimize!(problem, NLLSOptions(), 2)

    # Check callback and max time termination
    problem.variables[1] = -0.5
    problem.variables[2] = 2.5
    result = optimize!(problem, NLLSOptions(maxtime=0.0), nothing, (cost, unusedargs...)->(cost, 13))
    @test cost(problem) == result.bestcost
    @test result.termination == (1 << 9) | (13 << 16)
    @test result.niterations == 1

    # Optimize using Newton
    display(problem)
    result = optimize!(problem, NLLSOptions(iterator=NLLSsolver.newton), nothing, printoutcallback)
    display(result)
    @test NLLSsolver.cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)

    # Optimize using Levenberg-Marquardt
    problem.variables[1] = -0.5
    problem.variables[2] = 2.5
    ct = NLLSsolver.CostTrajectory()
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.levenbergmarquardt), nothing, storecostscallback(ct))
    @test NLLSsolver.cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)
    # Check callback results
    len = length(ct.costs)
    @test length(ct.times_ns) == len
    @test length(ct.trajectory) == len
    @test all(diff(ct.costs) .<= 0.0) # Check costs decrease
    @test all(diff(ct.times_ns) .>= 0.0) # Check costs increase
    @test all(x -> length(x) == 2, ct.trajectory) # Check the trajectory lengths

    # Optimize using dogleg
    problem.variables[1] = -0.5
    problem.variables[2] = 2.5
    empty!(ct)
    result = optimize!(problem, NLLSOptions(iterator=NLLSsolver.dogleg), nothing, storecostscallback(ct.costs))
    @test cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)
    @test all(diff(ct.costs) .<= 0.0) # Check costs decrease

    # Test standard gradient descent (a worse optimizer, so needs a closer starting point)
    problem.variables[1] = 1.0 - 1.e-5
    problem.variables[2] = 1.0
    display(problem)
    result = optimize!(problem, NLLSOptions(iterator=NLLSsolver.gradientdescent), nothing, printoutcallback)
    display(result)
    @test cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-5)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-5)
end
