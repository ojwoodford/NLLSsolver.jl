using Static
import ForwardDiff

"""
    NLLSsolver.cost(problem::NLLSProblem)

Compute and return the scalar cost defined by `problem`.
"""
cost(problem::NLLSProblem, numthreads=StaticInt(Threads.nthreads())) = cost(problem.variables, problem.costs, numthreads)
cost(problem::NLLSProblem, numthreads::Integer) = cost(problem, static(numthreads))
cost(vars::AbstractVector, costs::CostStruct)::Float64 = sum(bindleadingargs(computecost, vars), costs; init=0.0)
cost(vars::AbstractVector, costs::CostStruct, ::StaticInt{1})::Float64 = cost(vars, costs)
cost(vars::AbstractVector, costs::CostStruct, numthreads::Integer)::Float64 = cost(vars, costs, static(numthreads))
computecost(vars::AbstractVector, cost::AbstractCost)::Float64 = computecost(cost, getvars(cost, vars)...)

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
# costgradhess!(linsystem, vars::Vector, costs::CostStruct, subsetfun)::Float64 = sumsubset(bindleadingargs(costgradhess!, linsystem, vars), subsetfun, costs; init=0.0)
