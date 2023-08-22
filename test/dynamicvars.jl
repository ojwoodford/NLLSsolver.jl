using NLLSsolver, Test, Static, Random, LinearAlgebra

struct LinearResidual <: NLLSsolver.AbstractResidual
    y::Float64
    X::Vector{Float64}
end
NLLSsolver.ndeps(::LinearResidual) = static(1) # Residual depends on 1 variables
NLLSsolver.nres(::LinearResidual) = static(1) # Residual has length 1
NLLSsolver.varindices(::LinearResidual) = 1
NLLSsolver.getvars(::LinearResidual, vars::Vector) = (vars[1]::NLLSsolver.DynamicVector{Float64},)
NLLSsolver.computeresidual(res::LinearResidual, w) = res.X' * w - res.y
Base.eltype(::LinearResidual) = Float64

struct NormResidual <: NLLSsolver.AbstractResidual
    len::Int
end
NLLSsolver.ndeps(::NormResidual) = static(1) # Residual depends on 1 variables
NLLSsolver.nres(res::NormResidual) = res.len
NLLSsolver.varindices(::NormResidual) = 1
NLLSsolver.getvars(::NormResidual, vars::Vector) = (vars[1]::NLLSsolver.DynamicVector{Float64},)
NLLSsolver.computeresidual(::NormResidual, w) = w
Base.eltype(::NormResidual) = Float64

@testset "dynamicvars.jl" begin
    # Generate some test data
    Random.seed!(1)
    X = normalize(randn(Int(ceil((1.0 + rand()) * 50))))

    # Create the problem
    problem = NLLSsolver.NLLSProblem(NLLSsolver.DynamicVector{Float64})
    NLLSsolver.addvariable!(problem, zeros(length(X)))
    NLLSsolver.addcost!(problem, LinearResidual(1.0, X))
    NLLSsolver.addcost!(problem, NormResidual(length(X)))

    # Optimize
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.gaussnewton, storecosts=true))

    # Check the result is collinear to X
    Y = problem.variables[1]
    @test X' * Y ≈ norm(Y)
end
