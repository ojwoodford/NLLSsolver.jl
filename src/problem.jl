using LinearSolve
export NLLSProblem, addresidual, addvariable
 
struct NLLSProblem{T}
    # User provided
    residuals::IdDict
    variables::Vector{Any}
    
    # Internal
    newvariables::Vector{T} # Updated copy of variables, for testing a step
    gradient::Vector{T} # Storage for the gradient
    hessian::Vector{T} # Storage for the Hessian
    unfixed::BitArray{1} # Bit vector to store which variables are not fixed (same length as variables)
    condition::Vector{T}
    blockoffsets::Vector{UInt}
    linearsolve::LinearSolve
    callback # Function called at the end of every iteration

    # Debug outputs
    trajectory::Vector{Vector{T}}
    periteration
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