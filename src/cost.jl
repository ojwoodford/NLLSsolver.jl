using Static
import ForwardDiff

"""
    NLLSsolver.cost(problem::NLLSProblem)

Compute and return the scalar cost defined by `problem`.
"""
cost(problem::NLLSProblem, numthreads=StaticInt(Threads.nthreads())) = cost(problem.variables, problem.costs, numthreads)
cost(problem::NLLSProblem, numthreads::Integer) = cost(problem, static(numthreads))
cost(vars::AbstractVector, costs::CostStruct)::Float64 = cost(vars, costs, static(1))
cost(vars::AbstractVector, costs::CostStruct, ::StaticInt{1})::Float64 = cost_repo(vars, costs, static(1))
cost(vars::AbstractVector, costs::CostStruct, numthreads::Integer)::Float64 = cost(vars, costs, static(numthreads))
cost(vars::AbstractVector, costs::CostStruct, numthreads::StaticInt)::Float64 = cost_repo(vars, costs, numthreads)

cost_repo(vars, costs::CostStruct{T}, numthreads::StaticInt) where T = cost_repo_type(vars, costs, T, numthreads)
cost_repo_type(vars, costs, T::Union, numthreads) = cost_repo_type(vars, costs, T.a, numthreads) + cost_repo_type(vars, costs, T.b, numthreads)
cost_repo_type(vars, costs, T::DataType, numthreads) = cost_vec(vars, get(costs, T), numthreads)
cost_repo(vars, costs::CostStruct{Any}, numthreads::StaticInt) = sum(c -> cost_vec(vars, c, numthreads), values(costs.data))

cost_vec(vars, costlist::Vector, ::StaticInt{1})::Float64 = cost_range(vars, costlist, eachindex(costlist))

function cost_range(vars, costlist::AbstractVector, rng::AbstractUnitRange)::Float64
    tot = 0.0
    @inbounds for i in rng
        tot += computecost(vars, costlist[i])
    end
    return tot
end

@inline computecost(vars::AbstractVector, cost::AbstractCost)::Float64 = computecost(cost, getvars(cost, vars)...)

function gradhesshelper!(linsystem, cost::AbstractCost, vars, blockind, varflags::StaticInt)::Float64
    # Compute the residual
    c, g, H = computecostgradhess(varflags, cost, vars...)
    
    # Update the blocks in the problem
    updatesymlinearsystem!(linsystem, g, H, vars, varflags, blockind)

    # Return the cost
    return c
end
function gradhesshelper!(linsystem, cost::AbstractCost, vars, blockind, varflags::StaticInt{0})::Float64
    @warn "A cost for which all variables are fixed is being optimized. Remove such cost functions from the cost to avoid redundant computation, using the subproblem() method."
    return computecost(cost, vars...)
end

function gradhesshelper!(linsystem, cost::AbstractCost, vars, blockind, varflags::Integer)::Float64
    # Common case - all unfixed
    nvars = ndeps(cost)
    maxflags = static(2 ^ dynamic(nvars) - 1)
    if varflags == maxflags
        return gradhesshelper!(linsystem, cost, vars, blockind, maxflags)
    end

    if nvars <= 5
        # Value dispatch gradient computation based on the varflags
        return valuedispatch(static(0), maxflags-static(1), varflags, bindleadingargs(gradhesshelper!, linsystem, cost, vars, blockind))
    else
        # Fall back on dynamic dispatch
        return gradhesshelper!(linsystem, cost, vars, blockind, static(varflags))
    end
end

# Compute the variable flags indicating which variables are unfixed (i.e. to be optimized)
computevarflags(blockind) = mapreduce((ind, val) -> (val != 0) << ind, |, SR(0, length(blockind)-1), blockind)

function costgradhess!(linsystem, vars::AbstractVector, cost::AbstractCost)
    # Determine which variables are free
    blockind = getoffsets(cost, linsystem)
    varflags = computevarflags(blockind)
    # Complete the rest of the computation based on the varflags
    return gradhesshelper!(linsystem, cost, getvars(cost, vars), blockind, varflags)
end

function costgradhess!(linsystem, vars::AbstractVector, costs::CostStruct)::Float64
    zero!(linsystem)
    return sum(bindleadingargs(costgradhess!, linsystem, vars), costs; init=0.0)
end
