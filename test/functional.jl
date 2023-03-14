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
    return SVector(res.a - x)
end

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

@testset "functional.jl" begin
    # Create the problem
    problem = NLLSsolver.NLLSProblem{Float64}()
    NLLSsolver.addvariable!(problem, -0.5)
    NLLSsolver.addvariable!(problem, 2.5)
    NLLSsolver.addresidual!(problem, RosenbrockA(1.0))
    NLLSsolver.addresidual!(problem, RosenbrockB(10.))

    # Optimize
    NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.gaussnewton))

    # Check the result
    @test isapprox(problem.variables[1], 1.0; rtol=1.e-15)
    @test isapprox(problem.variables[2], 1.0; rtol=1.e-15)
end