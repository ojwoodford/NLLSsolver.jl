using LinearSolve
export NLLSProblem, addresidual!, addvariable!

mutable struct NLLSMutables
    linearsolve::LinearProblem
    callback # Function called at the end of every iteration, returning a boolean. If it returns true, terminate the optimization.
end
NLLSMutables() = NLLSMutables(LinearProblem(0, 0), (args...) -> false)
struct NLLSProblem{T}
    # User provided
    residuals::IdDict{DataType, Any}
    variables::Vector{Any}
    
    # Internal
    newvariables::Vector{Any} # Updated copy of variables, for testing a step
    gradient::Vector{T} # Storage for the gradient
    hessian::Matrix{T} # Storage for the Hessian
    unfixed::BitArray{1} # Bit vector to store which variables are not fixed (same length as variables)
    condition::Vector{T}
    blockoffsets::Vector{UInt}

    # Debug outputs
    trajectory::Vector{Vector{T}}
    periteration::Vector{T}

    # Mutables
    mutables::NLLSMutables 

    # Constructor
    function NLLSProblem{T}() where T
        return new(IdDict(), Vector{Any}(), Vector{Any}(), Vector{T}(), Matrix{T}(undef, 0, 0), BitArray{1}(), Vector{T}(), Vector{UInt}(), Vector{Vector{T}}(), Vector{T}(), NLLSMutables())
    end
end

function addresidual!(problem::NLLSProblem, residual)
    # Sanity checks
    N = nvars(residual)
    @assert N>0 "Problem with nvars()"
    @assert length(getvars(residual, problem.variables))==N "Problem with getvars()"
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