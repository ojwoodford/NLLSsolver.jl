using Static, SparseArrays

const CostStruct = VectorRepo

mutable struct NLLSProblem{VarTypes, CostTypes}
    # User provided
    costs::CostStruct{CostTypes} # Set of cost and residual blocks that define the problem
    variables::Vector{VarTypes}  # Current state of variables in use by optimizer

    # Used internally
    varnext::Vector{VarTypes}    # Updated set of variables proposed by iterator
    varbest::Vector{VarTypes}    # Best set of variables found so far
    varcostmap::SparseMatrixCSC{Bool, Int} # Map of variables used by each cost block (i.e. each column represents a cost block, and dependent variables are set to 1)
    varcostmapvalid::Bool        # Flag indicating whether the above variable cost map is valid, or should be regenerated

    # Constructor
    function NLLSProblem{VarTypes, CostTypes}(vars=Vector{VarTypes}(), costs=CostStruct{CostTypes}(), varnext=Vector{VarTypes}(), varbest=Vector{VarTypes}(), varcostmap=spzeros(Bool, 0, 0), varcostmapvalid=false) where {VarTypes, CostTypes}
        return new{VarTypes, CostTypes}(costs, vars, varnext, varbest, varcostmap, varcostmapvalid)
    end
end
NLLSProblem() = NLLSProblem{Any, Any}()
NLLSProblem(VT::Union{DataType, Union}) = NLLSProblem{VT, Any}()
NLLSProblem(VT::Union{DataType, Union}, CT::Union{DataType, Union}) = NLLSProblem{VT, CT}()
NLLSProblem(v::Vector{VT}) where VT = NLLSProblem{VT, Any}(v)
NLLSProblem(v::Vector{VT}, r::CostStruct{CT}) where {VT, CT} = NLLSProblem{VT, CT}(v, r)

function Base.show(io::IO, x::NLLSProblem)
    print(io, "NLLSProblem with ", length(x.variables), " variable blocks of total dimension ", UInt(sum(nvars, x.variables)), 
           ",\n             and ", countcosts(costnum, x.costs), " residual blocks of total dimension ", countcosts(resnum, x.costs), ".")
end

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::Integer)
    selectcosts!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if in(unfixed, varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector, unfixed::BitVector)
    selectcosts!(outcosts, incosts, [i for (i, r) in enumerate(incosts) if any(j -> unfixed[j], varindices(r))])
end

function selectcosts!(outcosts::CostStruct, incosts::Vector{T}, vec::Vector) where T
    if !isempty(vec)
        outcosts.data[T] = @inbounds incosts[vec]
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

function selectcosts!(outcosts::CostStruct, incosts::Vector{T}, costindices, lastcost, index) where T
    tocostvec = get!(outcosts, T)::Vector{T}
    empty!(tocostvec)
    firstind = lastcost
    lastcost += length(incosts)
    len = length(costindices)
    while index <= len && @inbounds(costindices[index]) <= lastcost
        push!(tocostvec, @inbounds(incosts[costindices[index]-firstind]))
        index += 1
    end
    return lastcost, index
end

function selectcosts!(outcosts::CostStruct, incosts::CostStruct, costindices)
    lastcost = 0
    index = 1
    for costs in values(incosts)
        lastcost, index = selectcosts!(outcosts, costs, costindices, lastcost, index)
    end
end

function subproblem!(to::NLLSProblem{VT, CT}, from::NLLSProblem{VT, CT}, costindices::AbstractVector) where {VT, CT}
    # Copy costs that have unfixed inputs
    selectcosts!(to.costs, from.costs, costindices)
    # Return the sub-problem
    return to
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
    # Mark the var cost map as invalid
    problem.varcostmapvalid = false
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

function updatevarcostmap!(varcostmap::SparseMatrixCSC{Bool, Int}, costvec::Vector, colind::Int, rowind::Int)
    numres = length(costvec)
    if numres > 0
        @inbounds for cost in costvec
            ndeps_ = dynamic(ndeps(cost))
            if ndeps_ == 1
                varcostmap.rowval[rowind] = varindices(cost)[1]
            else
                varcostmap.rowval[SR(0, ndeps_-1).+rowind] .= sort(varindices(cost))
            end
            rowind += ndeps_
            colind += 1
            varcostmap.colptr[colind] = rowind
        end
    end
    return colind, rowind
end

function updatevarcostmap!(varcostmap::SparseMatrixCSC{Bool, Int}, costs::CostStruct)
    # Pre-allocate all the necessary memory
    resize!(varcostmap.rowval, countcosts(costdeps, costs))
    prevlen = length(varcostmap.nzval)
    resize!(varcostmap.nzval, length(varcostmap.rowval))

    # Fill in the arrays
    varcostmap.nzval[prevlen+1:length(varcostmap.rowval)] .= true
    varcostmap.colptr[1] = 1
    colind = 1
    rowind = 1
    @inbounds for costvec in values(costs)
        colind, rowind = updatevarcostmap!(varcostmap, costvec, colind, rowind)
    end
    return nothing
end

function updatevarcostmap!(problem::NLLSProblem)
    # Check the size is correct
    sz = (length(problem.variables), countcosts(costnum, problem.costs))
    if size(problem.varcostmap) != sz
        problem.varcostmap = spzeros(Bool, sz[1], sz[2])
    end
    retval = updatevarcostmap!(problem.varcostmap, problem.costs)
    problem.varcostmapvalid = true
    return retval
end

function getvarcostmap(problem::NLLSProblem)
    if !problem.varcostmapvalid
        updatevarcostmap!(problem)
    end
    return problem.varcostmap
end

function reordercostsforschur!(problem::NLLSProblem, schurvars)
    # Group residuals by Schur variable (i.e. variables to be factored out)
    # Get the cost index per Schur variable (0 if none)
    schurruns = Dict{DataType, Vector{Int}}()
    costvarmap = sparse(getvarcostmap(problem)')[:,schurvars]
    schurvarpercost = sum(costvarmap; dims=2)
    @assert all(x->(x<=1), schurvarpercost) "Each cost block can only depend on one schur variable at most"
    schurvarpercost[schurvarpercost .> 0] = costvarmap.rowval
    # Reorder each set of cost blocks such that blocks with Schur variables are grouped
    startind = 0
    for costs in values(problem.costs)
        if !isempty(costs)
            schurvarpercostsubset = view(schurvarpercost, startind+1:startind+length(costs))
            startind += length(costs)
            sortedorder = sortperm(schurvarpercostsubset)
            permute!(schurvarpercostsubset, sortedorder)
            schurruns[typeof(costs[1])] = runlengthencodesortedints(schurvarpercostsubset)
            permute!(costs, sortedorder)
        end
    end
    problem.varcostmapvalid = false
    return schurruns
end

costnum(vec::Vector)  = length(vec)
costdeps(vec::Vector) = length(vec) > 0 ? (dynamic(is_static(ndeps(vec[1]))) ? length(vec) * ndeps(vec[1]) : sum(ndeps, vec; init=0)) : 0 # Support variable number of dependencies
resnum(vec::Vector)   = length(vec) > 0 ? (dynamic(is_static( nres(vec[1]))) ? length(vec) *  nres(vec[1]) : sum( nres, vec; init=0)) : 0 # Support variable length costs
@inline countcosts(fun, costs::CostStruct{Any}) = sum(fun, values(costs); init=0)
@inline countcosts(fun, costs::CostStruct{T}) where T = countcosts(fun, costs, T)
@inline countcosts(fun, costs, T::Union) = countcosts(fun, costs, T.a) + countcosts(fun, costs, T.b)
@inline countcosts(fun, costs, T::DataType) = fun(get(costs, T))

