export NLLSProblem, subproblem, addresidual!, addvariable!, fixvars!, unfixvars!, numresiduals, lengthresiduals

ResidualStruct = Dict{DataType, Vector}

mutable struct NLLSProblem{VarTypes}
    # User provided
    residuals::ResidualStruct
    variables::Vector{VarTypes}
    varnext::Vector{VarTypes}
    varbest::Vector{VarTypes}
    unfixed::Union{UInt, BitVector} # Bit vector to store which variables are not fixed (same length as variables), or a single variable index

    # Constructor
    function NLLSProblem{VarTypes}(vars=Vector{VarTypes}(), unfixed=BitVector(), residuals=ResidualStruct(), varnext=Vector{VarTypes}(), varbest=Vector{VarTypes}()) where VarTypes
        @assert (typeof(unfixed) == UInt ? (0 < unfixed <= length(vars)) : length(unfixed) == length(vars))
        return new(residuals, vars, varnext, varbest, unfixed)
    end
end

function selectresiduals!(outres::ResidualStruct, inres::Vector{T}, unfixed::Integer) where T
    vec = inres[map(r -> any(i -> i == unfixed, varindices(r)), inres)]
    if !isempty(vec)
        outres[T] = vec
    end
end

function selectresiduals!(outres::ResidualStruct, inres::Vector{T}, unfixed::BitVector) where T
    vec = inres[map(r -> any(i -> unfixed[i], varindices(r)), inres)]
    if !isempty(vec)
        outres[T] = vec
    end
end

# Produce a subproblem containing only the relevant residuals
function subproblem(problem::NLLSProblem{T}, unfixed) where T
    # Copy residuals that have unfixed inputs
    residualstruct = ResidualStruct()
    for residuals in values(problem.residuals)
        selectresiduals!(residualstruct, residuals, unfixed)
    end
    # Create the new problem (note that variables are SHARED)
    return NLLSProblem{T}(problem.variables, unfixed, residualstruct, problem.varnext, problem.varbest)
end

function fixvars!(problem::NLLSProblem, indices)
    problem.unfixed[indices] .= false
end

function unfixvars!(problem::NLLSProblem, indices)
    problem.unfixed[indices] .= true
end

function addresidual!(problem::NLLSProblem, residual::T) where T
    # Sanity checks
    N = nvars(residual)
    @assert N>0 "Problem with nvars()"
    @assert length(varindices(residual))==N "Problem with varindices()"
    @assert length(getvars(residual, problem.variables))==N "Problem with getvars()"
    # Set the used variables to be unfixed
    unfixvars!(problem, varindices(residual))
    # Add to the problem
    push!(get!(problem.residuals, T, Vector{T}()), residual)
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

function numresiduals(residuals::ResidualStruct)
    num = 0
    @inbounds for vec in values(residuals)
        num += length(vec)
    end
    return num
end

function lengthresiduals(residuals::ResidualStruct)
    len = 0
    @inbounds for (key, vec) in residuals
        len += length(vec) * nres(key)
    end
    return len
end

