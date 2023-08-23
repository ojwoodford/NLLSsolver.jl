using NLLSsolver, Test, Static, StaticArrays, Random

struct SimpleResidual <: NLLSsolver.AbstractAdaptiveResidual
    data::Float64
    varind::Int
end
NLLSsolver.ndeps(::SimpleResidual) = static(2) # Residual depends on 2 variables
NLLSsolver.nres(::SimpleResidual) = static(1) # Residual has length 1
NLLSsolver.varindices(res::SimpleResidual) = SVector(1, res.varind)
NLLSsolver.getvars(res::SimpleResidual, vars::Vector) = (vars[1]::NLLSsolver.ContaminatedGaussian{Float64}, vars[res.varind]::Float64)
NLLSsolver.computeresidual(res::SimpleResidual, mean) = mean - res.data
NLLSsolver.computeresjac(varflags, res::SimpleResidual, mean) = mean - res.data, SVector(one(mean))'
Base.eltype(::SimpleResidual) = Float64

function emcallback(cost, problem, data)
    # Compute the squared errors
    squarederrors = [NLLSsolver.computeresidual(res, problem.varnext[res.varind]) ^ 2 for res in problem.costs.data[SimpleResidual]]
    # Optimize the kernel parameters
    problem.varnext[1] = NLLSsolver.optimize(problem.varnext[1], squarederrors)
    # Recompute the cost
    data.timecost += @elapsed newcost = NLLSsolver.cost(problem.varnext, problem.costs)
    data.costcomputations += 1
    @assert newcost <= cost
    # Return cost and do not trigger termination
    return newcost, 0
end

@testset "adaptivecost.jl" begin
    # Create the problem
    problem = NLLSsolver.NLLSProblem(Union{NLLSsolver.ContaminatedGaussian{Float64}, Float64}, SimpleResidual)
    NLLSsolver.addvariable!(problem, ContaminatedGaussian(0.5, 5.0, 0.6))
    NLLSsolver.addvariable!(problem, 0.)
    NLLSsolver.addvariable!(problem, 0.)
    Random.seed!(1)
    points = vcat(randn(800), randn(200) * 10.0)
    for p in points
        NLLSsolver.addcost!(problem, SimpleResidual(p - 1, 2))
        NLLSsolver.addcost!(problem, SimpleResidual(p + 1, 3))
    end

    # Optimize the cost
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.levenbergmarquardt))

    # Check the result
    @test isapprox(NLLSsolver.params(problem.variables[1]), SVector(1.0, 10.0, 0.8); rtol=0.1)
    @test isapprox(problem.variables[2], -1.0; rtol=0.1)
    @test isapprox(problem.variables[3], 1.0; rtol=0.1)
    
    # Reset the variable starting values
    problem.variables[1] = ContaminatedGaussian(0.5, 5.0, 0.6)
    problem.variables[2] = 0.
    problem.variables[3] = 0.

    # Optimize the cost by alternating EM & optimizing the mean
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.newton, callback=emcallback), SVector(1, 2, 3) .> 1)

    # Check the result
    @test isapprox(NLLSsolver.params(problem.variables[1]), SVector(1.0, 10.0, 0.8); rtol=0.1)
    @test isapprox(problem.variables[2], -1.0; rtol=0.1)
    @test isapprox(problem.variables[3], 1.0; rtol=0.1)
end
