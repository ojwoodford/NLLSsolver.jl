using Static
import ForwardDiff
import IfElse: ifelse

cost(problem::NLLSProblem) = cost(problem.variables, problem.costs)
cost(vars::Vector, costs::CostStruct)::Float64 = sum(Base.Fix1(computecost, vars), costs)
computecost(vars::Vector, cost::AbstractCost)::Float64 = computecost(cost, getvars(cost, vars)...)

function computecost(residual::AbstractResidual, vars...)::Float64
    # Compute the residual
    r = computeresidual(residual, vars...)
    
    # Compute the robustified cost
    return 0.5 * robustify(robustkernel(residual), Float64(r' * r))[1]
end


function getoffsets(residual, linsystem::MultiVariateLS)
    return linsystem.blockindices[varindices(residual)]
end

function getoffsets(residual, linsystem::UniVariateLS)
    return convert.(UInt, SVector(varindices(residual)) .== linsystem.varindex)
end

function gradhesshelper!(linsystem, costblock::AbstractCost, vars, blockind, varflags)::Float64
    # Compute the residual
    c, g, H = computecostgradhess(varflags, costblock, vars...)
    
    # Update the blocks in the problem
    updatesymlinearsystem!(linsystem, g, H, vars, varflags, blockind)

    # Return the cost
    return c
end

function gradhesshelper!(linsystem, residual::AbstractResidual, vars, blockind, varflags)::Float64
    # Compute the residual
    res, jac = computeresjac(varflags, residual, vars...)

    # Compute the robustified cost and the IRLS weight
    c, w1, w2 = robustify(robustkernel(residual), res' * res)

    # If this residual has a weight...
    if w1 != 0    
        # Compute the unrobust gradient and Hessian
        g = jac' * res
        H = jac' * jac
        # Check for robust case
        if w1 != 1
            # IRLS reweighting of Hessian
            H *= w1
            if w2 < 0
                # Second order correction
                H += ((2 * w2) * g) * g'
            end
            # IRLS reweighting of gradient
            g *= w1
        end
        # Update the blocks in the problem
        updatesymlinearsystem!(linsystem, g, H, vars, varflags, blockind)
    end

    # Return the cost
    return 0.5 * c
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

function resjachelper!(linsystem, residual::AbstractResidual, vars, blockind, ind, varflags)::Float64
    # Compute the residual
    res, jac = computeresjac(varflags, residual, vars...)

    # Compute the robustified cost and the IRLS weight
    c, w1, = robustify(robustkernel(residual), res' * res)

    # If this residual has a weight...
    if w1 != 0    
        # Check for robust case
        if w1 != 1
            # IRLS reweighting
            w1 = sqrt(w1)
            res = res .* w1
            jac = jac .* w1
        end
        # Update the blocks in the problem
        updatelinearsystem!(linsystem, res, jac, ind, vars, varflags, blockind)
    end

    # Return the cost
    return 0.5 * c
end

function costresjac!(linsystem, vars::Vector, residual::AbstractResidual, ind)
    # Get the variables and associated data
    v = getvars(residual, vars)
    blockind = getoffsets(residual, linsystem)
    varflags = computevarflags(blockind)

    # Check that some variables are unfixed
    if varflags > 0
        # Common case - all unfixed
        maxflags = static(2 ^ dynamic(ndeps(residual)) - 1)
        if varflags == maxflags
            return resjachelper!(linsystem, residual, v, blockind, ind, maxflags)
        end

        # Dispatch gradient computation based on the varflags, and return the cost
        if ndeps(residual) <= 5
            return valuedispatch(static(1), maxflags-static(1), varflags, fixallbutlast(resjachelper!, linsystem, residual, v, blockind, ind))
        end
        return resjachelper!(linsystem, residual, v, blockind, ind, static(varflags))
    end

    # No unfixed variables, so just return the cost
    return cost(residual, v)
end

Base.length(::AbstractResidual) = 1

function costresjac!(linsystem, vars::Vector, residuals, ind=1)::Float64
    # Go over all residuals in the problem
    c = 0.
    @inbounds for res in values(residuals)
        c += costresjac!(linsystem, vars, res, ind)
        ind += length(res)
    end
    return c 
end
