using Test, StaticArrays, SparseArrays, Static
import NLLSsolver

# Define the Rosenbrock cost function
struct RosenbrockA <: NLLSsolver.AbstractResidual
    a::Float64
end
NLLSsolver.ndeps(::RosenbrockA) = static(1)
NLLSsolver.nres(::RosenbrockA) = 1
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
NLLSsolver.ndeps(::RosenbrockB) = static(2)
NLLSsolver.nres(::RosenbrockB) = static(1)
NLLSsolver.varindices(::RosenbrockB) = SVector(1, 2)
function NLLSsolver.getvars(::RosenbrockB, vars::Vector)
    return (vars[1]::Float64, vars[2]::Float64)
end
function NLLSsolver.computeresidual(res::RosenbrockB, x, y)
    return SVector(res.b * (x ^ 2 - y))
end
Base.eltype(::RosenbrockB) = Float64
const rosenbrockrobustifier = NLLSsolver.Huber2oKernel(2.0, 1.)
NLLSsolver.robustkernel(::RosenbrockB) = rosenbrockrobustifier

@testset "functional.jl" begin
    # Create the problem
    problem = NLLSsolver.NLLSProblem{Float64}()
    @test NLLSsolver.addvariable!(problem, 0.) == 1
    @test NLLSsolver.addvariable!(problem, 0.) == 2
    NLLSsolver.addresidual!(problem, RosenbrockA(1.0))
    @test NLLSsolver.lengthresiduals(problem.residuals) == 1
    @test NLLSsolver.numresiduals(problem.residuals) == 1
    NLLSsolver.addresidual!(problem, RosenbrockB(10.))
    @test NLLSsolver.lengthresiduals(problem.residuals) == 2
    @test NLLSsolver.numresiduals(problem.residuals) == 2
    @test NLLSsolver.cost(problem) == 1.
    varresmap = spzeros(Bool, 2, 2)
    NLLSsolver.updatevarresmap!(varresmap, problem)
    fill!(varresmap.nzval, true)
    @test vec(sum(Matrix(varresmap); dims=2)) == [2; 1]

    # Create a subproblem
    @test NLLSsolver.numresiduals(NLLSsolver.subproblem(problem, trues(2)).residuals) == 2
    subprob = NLLSsolver.subproblem(problem, 2)
    @test NLLSsolver.numresiduals(subprob.residuals) == 1
    @test NLLSsolver.cost(subprob) == 0.

    # Test optimization
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