using Test, StaticArrays
import NLLSsolver

# Define the Rosenbrock cost function
struct RosenbrockA <: NLLSsolver.AbstractResidual
    a::Float64
end
NLLSsolver.nvars(::RosenbrockA) = 1
NLLSsolver.varindices(::RosenbrockA) = SVector(1)
function NLLSsolver.getvars(::RosenbrockA, vars::Vector)
    return (vars[1]::Float64,)
end
function NLLSsolver.computeresidual(res::RosenbrockA, x)
    return res.a - x
end
Base.eltype(::RosenbrockA) = Float64

struct RosenbrockB <: NLLSsolver.AbstractResidual
    b::Float64
end
NLLSsolver.nvars(::RosenbrockB) = 2
NLLSsolver.varindices(::RosenbrockB) = SVector(1, 2)
function NLLSsolver.getvars(::RosenbrockB, vars::Vector)
    return (vars[1]::Float64, vars[2]::Float64)
end
function NLLSsolver.computeresidual(res::RosenbrockB, x, y)
    return SVector(res.b * (x ^ 2 - y))
end
Base.eltype(::RosenbrockB) = Float64

@testset "functional.jl" begin
    # Create the problem
    problem = NLLSsolver.NLLSProblem{Float64}()
    NLLSsolver.addvariable!(problem, 0.)
    NLLSsolver.addvariable!(problem, 0.)
    NLLSsolver.addresidual!(problem, RosenbrockA(1.0))
    NLLSsolver.addresidual!(problem, RosenbrockB(10.))

    for (ind, iter) in enumerate([NLLSsolver.gaussnewton, NLLSsolver.levenbergmarquardt, NLLSsolver.dogleg])
        # Set the start
        problem.variables[1] = -0.5
        problem.variables[2] = 2.5

        # Optimize the cost
        result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=iter))

        # Check the result
        @test isapprox(problem.variables[1], 1.0; rtol=1.e-10)
        @test isapprox(problem.variables[2], 1.0; rtol=1.e-10)
    end
end