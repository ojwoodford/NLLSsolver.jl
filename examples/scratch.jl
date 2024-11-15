struct Rosenbrock <: NLLSsolver.AbstractResidual
    a::Float64
    b::Float64
end
Rosenbrock() = Rosenbrock(1.0, 10.0)
Base.eltype(::Rosenbrock) = Float64
NLLSsolver.ndeps(::Rosenbrock) = static(1) # Residual depends on 1 variable
NLLSsolver.nres(::Rosenbrock) = static(2) # Residual has length 2
NLLSsolver.varindices(::Rosenbrock) = SVector(1) # There's only one variable
NLLSsolver.getvars(::Rosenbrock, vars::Vector) = (vars[1]::NLLSsolver.EuclideanVector{2, Float64},)
NLLSsolver.computeresidual(res::Rosenbrock, x) = SVector(res.a * (1 - x[1]), res.b * (x[1] ^ 2 - x[2]))

struct RosenbrockB <: NLLSsolver.AbstractResidual
    a::Float64
    b::Float64
end
RosenbrockB() = RosenbrockB(1.0, 10.0)
Base.eltype(::RosenbrockB) = Float64
NLLSsolver.ndeps(::RosenbrockB) = 1 # Residual depends on 1 variable
NLLSsolver.nres(::RosenbrockB) = static(2) # Residual has length 2
NLLSsolver.varindices(::RosenbrockB) = SVector(1) # There's only one variable
NLLSsolver.getvars(::RosenbrockB, vars::Vector) = (vars[1]::NLLSsolver.EuclideanVector{2, Float64},)
NLLSsolver.computeresidual(res::RosenbrockB, x) = SVector(res.a * (1 - x[1]), res.b * (x[1] ^ 2 - x[2]))

using BenchmarkTools

