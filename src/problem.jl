export NLLSProblem, addresidual!, addvariable!, fixvars!, unfixvars!

struct NLLSProblem{VarTypes}
    # User provided
    residuals::IdDict{DataType, Any}
    variables::Vector{VarTypes}
    unfixed::BitVector # Bit vector to store which variables are not fixed (same length as variables)

    # Constructor
    function NLLSProblem{VarTypes}() where VarTypes
        return new(IdDict{DataType, Any}(), Vector{VarTypes}(), BitVector())
    end
end

function fixvars!(problem::NLLSProblem, indices)
    problem.unfixed[indices] .= false
end

function unfixvars!(problem::NLLSProblem, indices)
    problem.unfixed[indices] .= true
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

