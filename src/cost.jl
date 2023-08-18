using Static
import ForwardDiff
import IfElse: ifelse

cost(problem::NLLSProblem) = cost(problem.variables, problem.costs)
cost(vars::Vector, costs::CostStruct)::Float64 = sum(Base.Fix1(computecost, vars), costs)
computecost(vars::Vector, cost::AbstractCost)::Float64 = computecost(cost, getvars(cost, vars)...)

function gradhesshelper!(linsystem, costblock::AbstractCost, vars, blockind, varflags)::Float64
    # Compute the residual
    c, g, H = computecostgradhess(varflags, costblock, vars...)
    
    # Update the blocks in the problem
    updatesymlinearsystem!(linsystem, g, H, vars, varflags, blockind)

    # Return the cost
    return c
end

# Compute the variable flags indicating which variables are unfixed (i.e. to be optimized)
computevarflags(blockind) = mapreduce((x, y) -> (x != 0) << (y - 1), |, blockind, SR(1, length(blockind)))

function costgradhess!(linsystem, vars::Vector, residual::AbstractCost)
    # Get the variables and associated data
    v = getvars(residual, vars)
    blockind = getoffsets(residual, linsystem)
    varflags = computevarflags(blockind)

    # Check that some variables are unfixed
    if varflags > 0
        # Common case - all unfixed
        maxflags = static(2 ^ dynamic(ndeps(residual)) - 1)
        if varflags == maxflags
            return gradhesshelper!(linsystem, residual, v, blockind, maxflags)
        end

        # Dispatch gradient computation based on the varflags, and return the cost
        if ndeps(residual) <= 5
            return valuedispatch(static(1), maxflags-static(1), varflags, fixallbutlast(gradhesshelper!, linsystem, residual, v, blockind))
        end
        return gradhesshelper!(linsystem, residual, v, blockind, static(varflags))
    end

    # No unfixed variables, so just return the cost
    return cost(residual, v)
end

costgradhess!(linsystem, vars::Vector, costs::CostStruct)::Float64 = sum(fixallbutlast(costgradhess!, linsystem, vars), costs)
