export NLLSProblem, subproblem, addresidual!, addvariable!, fixvars!, unfixvars!, numresiduals, lengthresiduals

ResidualStruct = Dict{DataType, Vector}

mutable struct NLLSProblem{VarTypes}
    # User provided
    residuals::ResidualStruct
    variables::Vector{VarTypes}
    varnext::Vector{VarTypes}
    varbest::Vector{VarTypes}

    # Constructor
    function NLLSProblem{VarTypes}(vars=Vector{VarTypes}(), residuals=ResidualStruct(), varnext=Vector{VarTypes}(), varbest=Vector{VarTypes}()) where VarTypes
        return new(residuals, vars, varnext, varbest)
    end
end

function selectresiduals!(outres::ResidualStruct, inres::Vector, unfixed::Integer)
    selectresiduals!(outres, inres, [i for (i, r) in enumerate(inres) if in(unfixed, varindices(r))])
end

function selectresiduals!(outres::ResidualStruct, inres::Vector, unfixed::BitVector)
    selectresiduals!(outres, inres, [i for (i, r) in enumerate(inres) if any(j -> unfixed[j], varindices(r))])
end

function selectresiduals!(outres::ResidualStruct, inres::Vector{T}, vec::Vector) where T
    if !isempty(vec)
        outres[T] = inres[vec]
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
    return NLLSProblem{T}(problem.variables, residualstruct, problem.varnext, problem.varbest)
end

function subproblem(problem::NLLSProblem{T}, resind::Vector) where T
    # Copy residuals that have unfixed inputs
    residualstruct = ResidualStruct()
    firstres = 0
    firstind = 0
    len = length(resind)
    for residuals in values(problem.residuals)
        lastres = firstres + length(residuals)
        lastind = firstind
        while lastind < len && resind[lastind+1] <= lastres
            lastind += 1
        end
        selectresiduals!(residualstruct, residuals, resind[firstind+1:lastind].-firstind)
        firstres = lastres
        firstind = lastind
    end
    # Create the new problem (note that variables are SHARED)
    return NLLSProblem{T}(problem.variables, residualstruct, problem.varnext, problem.varbest)
end

function addresidual!(problem::NLLSProblem, residual::T) where T
    # Sanity checks
    N = nvars(residual)
    @assert N>0 "Problem with nvars()"
    @assert length(varindices(residual))==N "Problem with varindices()"
    @assert length(getvars(residual, problem.variables))==N "Problem with getvars()"
    # Add to the problem
    push!(get!(problem.residuals, T, Vector{T}()), residual)
    return nothing
end

function addvariable!(problem::NLLSProblem, variable)
    # Sanity checks
    @assert nvars(variable)>0 "Problem with nvars()"
    # Add the variable
    push!(problem.variables, variable)
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
    @inbounds for vec in values(residuals)
        n = length(vec)
        if n != 0
            len += n * nres(vec[1])
        end
    end
    return len
end

