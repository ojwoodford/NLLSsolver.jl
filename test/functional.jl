using Test, StaticArrays, SparseArrays, Static, NLLSsolver

# Define the Rosenbrock cost function
struct RosenbrockA <: AbstractResidual
    a::Float64
end
NLLSsolver.ndeps(::RosenbrockA) = static(1)
NLLSsolver.nres(::RosenbrockA) = static(1)
NLLSsolver.varindices(::RosenbrockA) = SVector(1)
NLLSsolver.getvars(::RosenbrockA, vars::Vector) = (vars[1]::Float64,)
NLLSsolver.computeresidual(res::RosenbrockA, x) = SVector(res.a * (1 - x))
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
    problem = NLLSProblem(Float64, Union{RosenbrockA, RosenbrockB})
    addvariable!(problem, 0.)
    addcost!(problem, RosenbrockA(1.0))
    addvariable!(problem, 0.)
    addcost!(problem, RosenbrockB(10.))
    @test cost(problem) == 0.5

    # Optimize a subset of variables, such that some of the costs have no free variables (a warning).
    @test_logs (:warn, "A cost for which all variables are fixed is being optimized. Remove such cost functions from the cost to avoid redundant computation, using the subproblem() method.") match_mode=:any optimize!(problem, NLLSOptions(), 2)

    # Check callback and max time termination. The positional callback has the NLLSsolver 5 signature
    # (cost, problem, data, iteratedata) and asks to stop with flag 13; user flags occupy bits 16 and up.
    problem.variables[1] = -0.5
    problem.variables[2] = 2.5
    result = optimize!(problem, NLLSOptions(maxtime=0.0), nothing, (cost, unusedargs...)->(cost, 13))
    @test cost(problem) == result.bestcost
    @test result.termination == TERMINATION_MAX_TIME | (13 << 16)
    @test result.niterations == 1

    # Optimize using Newton
    display(problem)
    result = optimize!(problem, NLLSOptions(iterator=:newton), nothing, printoutcallback)
    display(result)
    @test NLLSsolver.cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)

    # Optimize using Levenberg-Marquardt (enum iterator member still accepted via NLLSOptions)
    problem.variables[1] = -0.5
    problem.variables[2] = 2.5
    ct = NLLSsolver.CostTrajectory()
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.levenbergmarquardt), nothing, storecostscallback(ct))
    @test NLLSsolver.cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)
    # Check callback results. CostTrajectory's field is `steps` (was `trajectory` in NLLSsolver 5).
    len = length(ct.costs)
    @test length(ct.times_ns) == len
    @test length(ct.steps) == len
    @test all(diff(ct.costs) .<= 0.0) # Check costs decrease
    @test all(diff(ct.times_ns) .>= 0.0) # Check times increase
    @test all(x -> length(x) == 2, ct.steps) # Check the step lengths

    # Optimize using Levenberg-Marquardt with D'D (running-max Marquardt) scaling
    problem.variables[1] = -0.5
    problem.variables[2] = 2.5
    empty!(ct)
    result = optimize!(problem, NLLSOptions(iterator=NLLSsolver.levenbergmarquardt_dtd), nothing, storecostscallback(ct.costs))
    @test cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)
    @test all(diff(ct.costs) .<= 0.0) # Check costs decrease

    # Optimize using dogleg
    problem.variables[1] = -0.5
    problem.variables[2] = 2.5
    empty!(ct)
    result = optimize!(problem, NLLSOptions(iterator=:dogleg), nothing, storecostscallback(ct.costs))
    @test cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)
    @test all(diff(ct.costs) .<= 0.0) # Check costs decrease

    # A positional callback receives the Model. With a single unfixed variable (and with a subset), the model
    # holds the trial point, so cost(problem) equals the cost the callback is given. (When every variable is
    # unfixed the model holds the last accepted point instead; see the README "Behaviour that changed".)
    problem.variables[1] = -0.5
    problem.variables[2] = 2.5
    gotmodel = Ref(false)
    maxdiff = Ref(0.0)
    cb = function (c, prob, data, iteratedata)
        gotmodel[] = prob isa NLLSProblem
        maxdiff[] = max(maxdiff[], abs(NLLSsolver.cost(prob) - c))
        return (c, 0)
    end
    optimize!(problem, NLLSOptions(iterator=:levenbergmarquardt), 2, cb)
    @test gotmodel[]
    @test maxdiff[] < 1.e-10

    # Gradient descent is no longer a built-in iterator: it is defined in examples/gradientdescent.jl as a
    # user-defined iterator. Load it, check it conforms, then run it.
    include(joinpath(pkgdir(NLLSsolver), "examples", "gradientdescent.jl"))
    @test check_iterator(:gradientdescent)
    problem.variables[1] = 1.0 - 1.e-5
    problem.variables[2] = 1.0
    display(problem)
    result = optimize!(problem, NLLSOptions(iterator=:gradientdescent), nothing, printoutcallback)
    display(result)
    @test cost(problem) == result.bestcost
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-5)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-5)
end
