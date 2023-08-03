using Static, SparseArrays

const ResidualStruct = VectorRepo

mutable struct NLLSProblem{VarTypes, ResTypes}
    # User provided
    residuals::ResidualStruct{ResTypes}
    variables::Vector{VarTypes}

    # Used internally
    varnext::Vector{VarTypes}
    varbest::Vector{VarTypes}

    # Constructor
    function NLLSProblem{VarTypes, ResTypes}(vars=Vector{VarTypes}(), residuals=ResidualStruct{ResTypes}(), varnext=Vector{VarTypes}(), varbest=Vector{VarTypes}()) where {VarTypes, ResTypes}
        return new{VarTypes, ResTypes}(residuals, vars, varnext, varbest)
    end
end
NLLSProblem() = NLLSProblem{Any, Any}()
NLLSProblem(VT::Union{DataType, Union}) = NLLSProblem{VT, Any}()
NLLSProblem(VT::Union{DataType, Union}, RT::Union{DataType, Union}) = NLLSProblem{VT, RT}()
NLLSProblem(v::Vector{VT}) where VT = NLLSProblem{VT, Any}(v)
NLLSProblem(v::Vector{VT}, r::ResidualStruct{RT}) where {VT, RT} = NLLSProblem{VT, RT}(v, r)

function selectresiduals!(outres::ResidualStruct, inres::Vector, unfixed::Integer)
    selectresiduals!(outres, inres, [i for (i, r) in enumerate(inres) if in(unfixed, varindices(r))])
end

function selectresiduals!(outres::ResidualStruct, inres::Vector, unfixed::BitVector)
    selectresiduals!(outres, inres, [i for (i, r) in enumerate(inres) if any(j -> unfixed[j], varindices(r))])
end

function selectresiduals!(outres::ResidualStruct, inres::Vector{T}, vec::Vector) where T
    if !isempty(vec)
        outres.data[T] = inres[vec]
    end
end

# Produce a subproblem containing only the relevant residuals
function subproblem(problem::NLLSProblem{VT, RT}, unfixed) where {VT, RT}
    # Copy residuals that have unfixed inputs
    residualstruct = ResidualStruct{RT}()
    for residuals in values(problem.residuals)
        selectresiduals!(residualstruct, residuals, unfixed)
    end
    # Create the new problem (note that variables are SHARED)
    return NLLSProblem{VT, RT}(problem.variables, residualstruct, problem.varnext, problem.varbest)
end

function subproblem(problem::NLLSProblem{VT, RT}, resind::Vector) where {VT, RT}
    # Copy residuals that have unfixed inputs
    residualstruct = ResidualStruct{RT}()
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
    return NLLSProblem{VT, RT}(problem.variables, residualstruct, problem.varnext, problem.varbest)
end

function addresidual!(problem::NLLSProblem, residual::AbstractResidual)
    # Sanity checks
    N = ndeps(residual)
    @assert isa(N, StaticInt) && N>0 && N<=MAX_ARGS "Problem with ndeps()"
    M = nres(residual)
    @assert (isa(M, Integer) || isa(M, StaticInt)) && M>0 "Problem with nres()"
    @assert length(varindices(residual))==N "Problem with varindices()"
    @assert length(getvars(residual, problem.variables))==N "Problem with getvars()"
    # Add to the problem
    push!(problem.residuals, residual)
    return nothing
end

function addvariable!(problem::NLLSProblem, variable)
    # Sanity checks
    N = nvars(variable)
    @assert (isa(N, Integer) || isa(N, StaticInt)) && N>0 "Problem with nvars()"
    # Add the variable
    push!(problem.variables, variable)
    # Return the index
    return length(problem.variables)
end

reslen(vec::Vector) = length(vec)
# Support variable length residuals
resnum(vec::Vector) = length(vec) > 0 ? (dynamic(is_static(nres(vec[1]))) ? length(vec) * nres(vec[1]) : sum(nres, vec; init=0)) : 0
resdeps(vec::Vector) = length(vec) > 0 ? length(vec) * ndeps(vec[1]) : 0
@inline countresiduals(fun, residuals::ResidualStruct{Any}) = sum(fun, values(residuals); init=0)
@inline countresiduals(fun, residuals::ResidualStruct{T}) where T = countresiduals(fun, residuals, T)
@inline countresiduals(fun, residuals, T::Union) = countresiduals(fun, residuals, T.a) + countresiduals(fun, residuals, T.b)
@inline countresiduals(fun, residuals, T::DataType) = fun(get(residuals, T))

