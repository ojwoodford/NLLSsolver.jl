using Static, SparseArrays

const CostStruct = VectorRepo

mutable struct NLLSProblem{VarTypes, CostTypes}
    # User provided
    costs::CostStruct{CostTypes}
    variables::Vector{VarTypes}

    # Used internally
    varnext::Vector{VarTypes}
    varbest::Vector{VarTypes}

    # Constructor
    function NLLSProblem{VarTypes, CostTypes}(vars=Vector{VarTypes}(), costs=CostStruct{CostTypes}(), varnext=Vector{VarTypes}(), varbest=Vector{VarTypes}()) where {VarTypes, CostTypes}
        return new{VarTypes, CostTypes}(costs, vars, varnext, varbest)
    end
end
NLLSProblem() = NLLSProblem{Any, Any}()
NLLSProblem(VT::Union{DataType, Union}) = NLLSProblem{VT, Any}()
NLLSProblem(VT::Union{DataType, Union}, CT::Union{DataType, Union}) = NLLSProblem{VT, CT}()
NLLSProblem(v::Vector{VT}) where VT = NLLSProblem{VT, Any}(v)
NLLSProblem(v::Vector{VT}, r::CostStruct{CT}) where {VT, CT} = NLLSProblem{VT, CT}(v, r)

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::Integer)
    selectcosts!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if in(unfixed, varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::BitVector)
    selectcosts!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if any(j -> unfixed[j], varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector{T}, vec::Vector) where T
    if !isempty(vec)
        outcosts.data[T] = incosts[vec]
    end
end

# Produce a subproblem containing only the relevant costs
function subproblem(problem::NLLSProblem{VT, CT}, unfixed) where {VT, CT}
    # Copy costs that have unfixed inputs
    coststruct = CostStruct{CT}()
    for costs in values(problem.costs)
        selectcosts!(coststruct, costs, unfixed)
    end
    # Create the new problem (note that variables are SHARED)
    return NLLSProblem{VT, CT}(problem.variables, coststruct, problem.varnext, problem.varbest)
end

function subproblem(problem::NLLSProblem{VT, CT}, resind::Vector) where {VT, CT}
    # Copy costs that have unfixed inputs
    coststruct = CostStruct{CT}()
    firstres = 0
    firstind = 0
    len = length(resind)
    for costs in values(problem.costs)
        lastres = firstres + length(costs)
        lastind = firstind
        while lastind < len && resind[lastind+1] <= lastres
            lastind += 1
        end
        selectcosts!(coststruct, costs, resind[firstind+1:lastind].-firstind)
        firstres = lastres
        firstind = lastind
    end
    # Create the new problem (note that variables are SHARED)
    return NLLSProblem{VT, CT}(problem.variables, coststruct, problem.varnext, problem.varbest)
end

function addcost!(problem::NLLSProblem, cost::Cost) where Cost <: AbstractCost
    # Sanity checks
    N = ndeps(cost)
    @assert isa(N, StaticInt) && N>0 && N<=MAX_ARGS "Problem with ndeps()"
    @assert length(varindices(cost))==N "Problem with varindices()"
    vars = getvars(cost, problem.variables)
    @assert length(vars)==N "Problem with getvars()"
    @assert (!(Cost <: AbstractAdaptiveResidual) || isa(vars[1], AbstractAdaptiveRobustifier)) "Adaptive residual without adaptive robustifier"
    if Cost <: AbstractResidual
        M = nres(cost)
        @assert (isa(M, Integer) || isa(M, StaticInt)) && M>0 "Problem with nres()"
    end
    # Add to the problem
    push!(problem.costs, cost)
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
# Support variable length costs
resnum(vec::Vector) = length(vec) > 0 ? (dynamic(is_static(nres(vec[1]))) ? length(vec) * nres(vec[1]) : sum(nres, vec; init=0)) : 0
resdeps(vec::Vector) = length(vec) > 0 ? length(vec) * ndeps(vec[1]) : 0
@inline countcosts(fun, costs::CostStruct{Any}) = sum(fun, values(costs); init=0)
@inline countcosts(fun, costs::CostStruct{T}) where T = countcosts(fun, costs, T)
@inline countcosts(fun, costs, T::Union) = countcosts(fun, costs, T.a) + countcosts(fun, costs, T.b)
@inline countcosts(fun, costs, T::DataType) = fun(get(costs, T))

