using NLLSsolver, Test, Static

struct LinearResidual{T} <: NLLSsolver.AbstractResidual
    y::T
    X::Vector{T}
end
NLLSsolver.ndeps(::LinearResidual) = static(1) # Residual depends on 1 variables
NLLSsolver.nres(::LinearResidual) = static(1) # Residual has length 1
NLLSsolver.varindices(::LinearResidual) = 1
NLLSsolver.getvars(::LinearResidual{T}, vars::Vector) where T = (vars[1]::NLLSsolver.DynamicVector{T},)
NLLSsolver.computeresidual(res::LinearResidual, w) = res.X' * w - res.y
Base.eltype(::LinearResidual{T}) where T = T

struct NormResidual{T} <: NLLSsolver.AbstractResidual
    len::Int
end
NLLSsolver.ndeps(::NormResidual) = static(1) # Residual depends on 1 variables
NLLSsolver.nres(res::NormResidual) = res.len
NLLSsolver.varindices(::NormResidual) = 1
NLLSsolver.getvars(::NormResidual{T}, vars::Vector) where T = (vars[1]::NLLSsolver.DynamicVector{T},)
NLLSsolver.computeresidual(::NormResidual, w) = w
Base.eltype(::NormResidual{T}) where T = T

@testset "dynamicvars.jl" begin
    # Generate some test data
    X = randn(Int(ceil((1.0+rand())*50.0)))
    X = X .+ sign.(X) * 0.2

    # Create the problem
    problem = NLLSsolver.NLLSProblem(NLLSsolver.DynamicVector{Float64})
    NLLSsolver.addvariable!(problem, zeros(length(X)))
    NLLSsolver.addresidual!(problem, LinearResidual(1.0, X))
    NLLSsolver.addresidual!(problem, NormResidual{Float64}(length(X)))

    # Optimize
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.gaussnewton, storecosts=true))

    # Check the result is collinear to X
    X = X ./ problem.variables[1]
    meanx = mean(X)
    @test all(x -> isapprox(x, meanx), X)
end
