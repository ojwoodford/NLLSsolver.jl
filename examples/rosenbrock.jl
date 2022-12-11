import NLLSsolver

# Define the Rosenbrock cost function
struct Rosenbrock <: NLLSsolver.AbstractResidual
    a::Float64
    b::Float64
end
Rosenbrock() = Rosenbrock(1.0, 10.0)
NLLSsolver.nvars(::Rosenbrock) = 1 # Residual depends on 1 variable
NLLSsolver.varindices(res::Rosenbrock) = 1 # There's only one variable
function NLLSsolver.getvars(::Rosenbrock, vars::Vector)
    return (vars[1]::NLLSsolver.EuclideanVector{2, Float64},)
end
function NLLSsolver.computeresidual(res::Rosenbrock, x::NLLSsolver.EuclideanVector{2, Float64})
    return SVector(res.a - x[1], res.b * (x[1] ^ 2 - x[2]))
end
function NLLSsolver.computeresjac(::Val{varflags}, res::Rosenbrock, x::NLLSsolver.EuclideanVector{2, Float64}) where varflags
    @assert varflags == 1
    return SVector(res.a - x[1], res.b * (x[1] ^ 2 - x[2])),
           SMatrix{2}(-1., 2 * res.b * x[1], 0, res.b)
end

function optimizeRosenbrock()
    # Create the problem
    problem = NLLSsolver.NLLSProblem{NLLSsolver.EuclideanVector{2, Float64}}()
    NLLSsolver.addvariable!(problem, NLLSsolver.EuclideanVector(0., 0.))
    NLLSsolver.addresidual!(problem, Rosenbrock())
    
    # Optimize the cost
    result = NLLSsolver.optimize!(problem)

    # Plot the trajectory
    return result
end

result = optimizeRosenbrock();