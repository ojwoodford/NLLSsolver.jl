using LinearSolve
export NLLSProblem, addresidual, addvariable

mutable struct NLLSMutables
    linearsolve::LinearProblem
    callback # Function called at the end of every iteration
end
NLLSMutables() = NLLSMutables(LinearProblem(0, 0), (args...) -> false)
struct NLLSProblem{T<:AbstractFloat}
    # User provided
    residuals::IdDict{Any, Any}
    variables::Vector{Any}
    
    # Internal
    newvariables::Vector{T} # Updated copy of variables, for testing a step
    gradient::Vector{T} # Storage for the gradient
    hessian::Matrix{T} # Storage for the Hessian
    unfixed::BitArray{1} # Bit vector to store which variables are not fixed (same length as variables)
    condition::Vector{T}
    blockoffsets::Vector{UInt}
    mutables::NLLSMutables 

    # Debug outputs
    trajectory::Vector{Vector{T}}
    periteration::Vector{T}

    # Constructor
    function NLLSProblem{T}()
        return new(IdDict(), Vector{Any}(), Vector{T}(), Matrix{T}())
    end
end

function addresidual!(problem::NLLSProblem, residual)
    # Sanity checks
    @assert nvars(residual)>0 "Problem with nvars()"
    @assert reslen(residual)>0 "Problem with reslen()"
    @assert typeof(eltype(residual))===DataType "Problem with eltype()"
    @assert length(getvars(residual, problem.variables))>0 "Problem with getvars()"
    # Add to the problem
    push!(get!(problem.residuals, typeof(residual), Vector{typeof(residual)}()), residual)
end

function addvariable!(problem::NLLSProblem, variable)
    # Sanity checks
    @assert nvars(variable)>0 "Problem with nvars()"
    # Add the variable
    push!(problem.variables, variable)
    # Return the index
    return length(problem.variables)
end