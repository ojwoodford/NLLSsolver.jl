using Static, SparseArrays

const ResidualStruct = VectorRepo

mutable struct NLLSProblem{VarTypes, ResTypes}
    # User provided
    residuals::ResidualStruct{ResTypes}
    variables::Vector{VarTypes}

    # Used internally
    varnext::Vector{VarTypes}
    varbest::Vector{VarTypes}

    # Maps of dependencies
    varresmap::SparseMatrixCSC{Bool, Int}
    resvarmap::SparseMatrixCSC{Bool, Int}
    varvarmap::SparseMatrixCSC{Bool, Int}
    mapsvalid::Bool

    # Constructor
    function NLLSProblem{VarTypes, ResTypes}(vars=Vector{VarTypes}(), residuals=ResidualStruct{ResTypes}(), varnext=Vector{VarTypes}(), varbest=Vector{VarTypes}()) where {VarTypes, ResTypes}
        nvars = length(vars)
        nres = countresiduals(reslen, residuals)
        return new{VarTypes, ResTypes}(residuals, vars, varnext, varbest, spzeros(Bool, nvars, nres), spzeros(Bool, nres, nvars), spzeros(Bool, nvars, nvars), false)
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

function addresidual!(problem::NLLSProblem, residual)
    # Sanity checks
    N = ndeps(residual)
    @assert isa(N, StaticInt) && N>0 && N<=MAX_ARGS "Problem with ndeps()"
    M = nres(residual)
    @assert (isa(M, Integer) || isa(M, StaticInt)) && M>0 && M<=MAX_BLOCK_SZ "Problem with nres()"
    @assert length(varindices(residual))==N "Problem with varindices()"
    @assert length(getvars(residual, problem.variables))==N "Problem with getvars()"
    # Add to the problem
    push!(problem.residuals, residual)
    return nothing
end

function addvariable!(problem::NLLSProblem, variable)
    # Sanity checks
    N = nvars(variable)
    @assert (isa(N, Integer) || isa(N, StaticInt)) && N>0 && N<=MAX_BLOCK_SZ "Problem with nvars()"
    # Add the variable
    push!(problem.variables, variable)
    # Return the index
    return length(problem.variables)
end

reslen(vec::Vector) = length(vec)
resnum(vec::Vector) = length(vec) > 0 ? length(vec) * nres(vec[1]) : 0
resdeps(vec::Vector) = length(vec) > 0 ? length(vec) * ndeps(vec[1]) : 0
@inline countresiduals(fun, residuals::ResidualStruct{Any}) = sum(fun, values(residuals); init=0)
@inline countresiduals(fun, residuals::ResidualStruct{T}) where T = countresiduals(fun, residuals, T)
@inline countresiduals(fun, residuals, T::Union) = countresiduals(fun, residuals, T.a) + countresiduals(fun, residuals, T.b)
@inline countresiduals(fun, residuals, T::DataType) = fun(get(residuals, T))

function updatevarresmap!(varresmap::SparseMatrixCSC{Bool, Int}, residuals::Vector, colind::Int, rowind::Int)
    numres = length(residuals)
    if numres > 0
        ndeps_ = known(ndeps(residuals[1]))
        srange = SR(0, ndeps_-1)
        @inbounds for res in residuals
            varresmap.rowval[srange.+rowind] .= varindices(res)
            rowind += ndeps_
            colind += 1
            varresmap.colptr[colind] = rowind
        end
    end
    return colind, rowind
end

function updatevarresmap!(problem::NLLSProblem)
    # Pre-allocate all the necessary memory
    vrm = problem.varresmap
    res = problem.residuals
    resize!(vrm.rowval, countresiduals(resdeps, res))
    resize!(vrm.colptr, countresiduals(reslen, res)+1)
    prevlen = length(vrm.nzval)
    resize!(vrm.nzval, length(vrm.rowval))

    # Fill in the arrays
    vrm.nzval[prevlen+1:length(vrm.rowval)] .= true
    varresmap.colptr[1] = 1
    colind = 1
    rowind = 1
    @inbounds for r in values(res)
        colind, rowind = updatevarresmap!(vrm, r, colind, rowind)
    end
end

