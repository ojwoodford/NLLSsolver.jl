using NLLSsolver, Test, Static, StaticArrays, Random

struct SimpleResidual <: NLLSsolver.AbstractAdaptiveResidual
    data::Float64
end
NLLSsolver.ndeps(::SimpleResidual) = static(2) # Residual depends on 2 variables
NLLSsolver.nres(::SimpleResidual) = static(1) # Residual has length 1
NLLSsolver.varindices(::SimpleResidual) = SVector(1, 2)
NLLSsolver.getvars(::SimpleResidual, vars::Vector) = (vars[1]::NLLSsolver.ContaminatedGaussian{Float64}, vars[2]::Float64)
NLLSsolver.computeresidual(res::SimpleResidual, mean) = mean - res.data
NLLSsolver.computeresjac(varflags, res::SimpleResidual, mean) = mean - res.data, one(mean)
Base.eltype(::SimpleResidual) = Float64

@testset "adaptivecost.jl" begin
    # Create the problem
    problem = NLLSsolver.NLLSProblem(Union{NLLSsolver.ContaminatedGaussian{Float64}, Float64}, SimpleResidual)
    NLLSsolver.addvariable!(problem, ContaminatedGaussian(0.5, 5.0, 0.6))
    NLLSsolver.addvariable!(problem, 0.)
    Random.seed!(1)
    points = vcat(randn(800), randn(200) * 10.0) .+ 1
    for p in points
        NLLSsolver.addcost!(problem, SimpleResidual(p))
    end

    # Optimize the cost
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.levenbergmarquardt))

    # Check the result
    @test isapprox(NLLSsolver.params(problem.variables[1]), SVector(1.0, 10.0, 0.8); rtol=0.1)
    @test isapprox(problem.variables[2], 1.0; rtol=0.1)
end
