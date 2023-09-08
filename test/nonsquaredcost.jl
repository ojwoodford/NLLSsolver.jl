using NLLSsolver, Test, Static, StaticArrays, LinearAlgebra

const NDIMS = 3
struct LinearResidualStatic <: NLLSsolver.AbstractResidual
    y::SVector{NDIMS, Float64}
    X::SMatrix{NDIMS, NDIMS, Float64, NDIMS*NDIMS}
    varind::Int
end
NLLSsolver.ndeps(::LinearResidualStatic) = static(1) # Residual depends on 1 variables
NLLSsolver.nres(::LinearResidualStatic) = static(NDIMS) # Residual has length NDIMS
NLLSsolver.varindices(res::LinearResidualStatic) = res.varind
NLLSsolver.getvars(res::LinearResidualStatic, vars::Vector) = (vars[res.varind]::NLLSsolver.EuclideanVector{NDIMS, Float64},)
NLLSsolver.computeresidual(res::LinearResidualStatic, w) = res.X * w - res.y
Base.eltype(::LinearResidualStatic) = Float64

struct LinearResidualDynamic <: NLLSsolver.AbstractResidual
    y::Vector{Float64}
    X::Matrix{Float64}
    varind::Int
end
NLLSsolver.ndeps(::LinearResidualDynamic) = static(1) # Residual depends on 1 variables
NLLSsolver.nres(res::LinearResidualDynamic) = length(res.y) # Residual has dynamic length
NLLSsolver.varindices(res::LinearResidualDynamic) = res.varind
NLLSsolver.getvars(res::LinearResidualDynamic, vars::Vector) = (vars[res.varind]::NLLSsolver.DynamicVector{Float64},)
NLLSsolver.computeresidual(res::LinearResidualDynamic, w) = res.X * w - res.y
Base.eltype(::LinearResidualDynamic) = Float64

struct LinearCostStatic <: NLLSsolver.AbstractCost
    y::SVector{NDIMS, Float64}
    varind::Int
end
NLLSsolver.ndeps(::LinearCostStatic) = static(1) # Cost depends on 1 variables
NLLSsolver.varindices(cost::LinearCostStatic) = cost.varind
NLLSsolver.getvars(cost::LinearCostStatic, vars::Vector) = (vars[cost.varind]::NLLSsolver.EuclideanVector{NDIMS, Float64},)
NLLSsolver.computecost(cost::LinearCostStatic, w) = cost.y' * w
Base.eltype(::LinearCostStatic) = Float64

struct LinearCostDynamic <: NLLSsolver.AbstractCost
    y::Vector{Float64}
    varind::Int
end
NLLSsolver.ndeps(::LinearCostDynamic) = static(1) # Residual depends on 1 variables
NLLSsolver.varindices(cost::LinearCostDynamic) = cost.varind
NLLSsolver.getvars(cost::LinearCostDynamic, vars::Vector) = (vars[cost.varind]::NLLSsolver.DynamicVector{Float64},)
NLLSsolver.computecost(cost::LinearCostDynamic, w) = cost.y' * w
Base.eltype(::LinearCostDynamic) = Float64

@testset "nonsquaredcost.jl" begin
    # Generate some test data
    X = randn(SMatrix{NDIMS, NDIMS, Float64, NDIMS*NDIMS})
    y = randn(SVector{NDIMS, Float64})
    solution = (X' * X) \ ((X' - I) * y)

    # Create the problem
    problem = NLLSsolver.NLLSProblem()
    NLLSsolver.addvariable!(problem, zeros(NLLSsolver.EuclideanVector{NDIMS, Float64}))
    NLLSsolver.addcost!(problem, LinearResidualStatic(y, X, 1))
    NLLSsolver.addcost!(problem, LinearCostStatic(y, 1))
    NLLSsolver.addvariable!(problem, zeros(NDIMS))
    NLLSsolver.addcost!(problem, LinearResidualDynamic(Vector(y), Matrix(X), 2))
    NLLSsolver.addcost!(problem, LinearCostDynamic(Vector(y), 2))

    # Optimize
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.newton))

    # Check the result
    @test problem.variables[1] ≈ solution
    @test problem.variables[2] ≈ solution
end