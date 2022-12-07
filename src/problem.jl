using LinearSolve, SparseArrays
export NLLSProblem, addresidual!, addvariable!, fixvars!, unfixvars!, getschurvars

mutable struct NLLSMutables
    linearsolve::LinearProblem
    callback # Function called at the end of every iteration, returning a boolean. If it returns true, terminate the optimization.
    changed::Bool
end
NLLSMutables() = NLLSMutables(LinearProblem(0, 0), (args...) -> false, false)
struct NLLSProblem{T}
    # User provided
    residuals::IdDict{DataType, Any}
    variables::Vector{Any}
    
    # Internal
    newvariables::Vector{Any} # Updated copy of variables, for testing a step
    gradient::Vector{T} # Storage for the gradient
    hessian::Union{Matrix{T}, SparseMatrixCSC{T, UInt}} # Storage for a dense or sparse Hessian
    unfixed::BitVector # Bit vector to store which variables are not fixed (same length as variables)
    condition::Vector{T}
    blockoffsets::Vector{UInt}

    # Debug outputs
    trajectory::Vector{Vector{T}}
    periteration::Vector{T}

    # Mutables
    mutables::NLLSMutables 

    # Constructor
    function NLLSProblem{T}() where T
        return new(IdDict(), Vector{Any}(), Set{DataType}(), Vector{Any}(), Vector{T}(), Matrix{T}(undef, 0, 0), BitVector(), Vector{T}(), Vector{UInt}(), Vector{Vector{T}}(), Vector{T}(), NLLSMutables())
    end
end

function fixvars!(problem::NLLSProblem, indices)
    changed = any(problem.unfixed[indices])
    problem.unfixed[indices] .= false
    problem.mutables.changed = problem.mutables.changed || changed
    return changed
end

function unfixvars!(problem::NLLSProblem, indices)
    changed = !all(problem.unfixed[indices])
    problem.unfixed[indices] .= true
    problem.mutables.changed = problem.mutables.changed || changed
    return changed
end

function addresidual!(problem::NLLSProblem, residual)
    # Sanity checks
    N = nvars(residual)
    @assert N>0 "Problem with nvars()"
    @assert length(varindices(residual))==N "Problem with varindices()"
    @assert length(getvars(residual, problem.variables))==N "Problem with getvars()"
    # Set the used variables to be unfixed
    unfixvars!(problem, varindices(residual))
    # Add to the problem
    push!(get!(problem.residuals, typeof(residual), Vector{typeof(residual)}()), residual)
    return nothing
end

function addvariable!(problem::NLLSProblem, variable)
    # Sanity checks
    @assert nvars(variable)>0 "Problem with nvars()"
    # Add the variable
    push!(problem.variables, variable)
    # Set fixed to start with
    push!(problem.unfixed, false)
    # Return the index
    return length(problem.variables)
end

function updateinternals!(problem::NLLSProblem)
    # Compute the block offsets
    problem.blockoffsets = zeros(Uint, length(problem.variables))
    start = UInt(1)
    for index in eachindex(problem.blockoffsets)
        if problem.unfixed[index]
            problem.blockoffsets[index] = start
            start += nvars(problem.variables[index])::UInt
        end
    end
    start -= 1
    # Set the gradient and Hessian sizes
    problem.gradient = zeros(Float64, start)
    problem.hessian = zeros(Float64, start, start)
    # Mark that we have updated the internals
    problem.mutables.changed = false
    return start
end

# Get a bit vector showing which variables should be schur complemented out
function getschurvars(problem::NLLSProblem)
    return broadcast(v -> typeof(v) in val.schurvartypes, val.variables)
end

function update!(problem, step)
    # Copy the variables
    problem.newvariables = copy(problem.variables)
    # Update each variable
    for index in eachindex(problem.blockoffsets)
        if problem.blockoffsets[index] != 0
            problem.newvariables[index] = update(problem.newvariables[index], step, problem.blockoffsets[index])
        end
    end
    return nothing
end