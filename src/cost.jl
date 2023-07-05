using Static
import ForwardDiff
import IfElse: ifelse

cost(problem::NLLSProblem) = cost(problem.variables, problem.residuals)
cost(vars::Vector, residuals::ResidualStruct)::Float64 = sum(Base.Fix1(cost, vars), residuals)
cost(vars::Vector, residual::AbstractResidual)::Float64 = cost(residual, getvars(residual, vars))

function cost(residual::Residual, vars::Tuple)::Float64 where Residual <: AbstractResidual
    # Compute the residual
    r = computeresidual(residual, vars...)
    
    # Compute the robustified cost
    return robustify(robustkernel(residual), Float64(r' * r))[1]
end


function getoffsets(residual, linsystem::MultiVariateLS)
    return linsystem.blockindices[varindices(residual)]
end

function getoffsets(residual, linsystem::UniVariateLS)
    return convert.(UInt, SVector(varindices(residual)) .== linsystem.varindex)
end

function gradhesshelper!(linsystem, residual::Residual, vars, blockind, varflags)::Float64 where Residual <: AbstractResidual
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
    return c
end

# Compute the variable flags indicating which variables are unfixed (i.e. to be optimized)
computevarflags(blockind) = mapreduce((x, y) -> (x != 0) << (y - 1), |, blockind, SR(1, length(blockind)))

# Decision on whether to use static-sized autodiff - compile-time decision where possible
function usestatic(blockind, vars)
    nvs = map(nvars, vars)
    if sum(nv -> ifelse(is_static(nv), nv , static(MAX_STATIC_VAR+1)), nvs) <= static(MAX_STATIC_VAR)
        return static(true)
    end
    if all(map(nv -> ifelse(is_static(nv), static(nv > static(MAX_STATIC_VAR)), static(true)), nvs))
        return static(false)
    end
    return mapreduce((x, y) -> (x != 0) * ifelse(is_static(y), dynamic(y), MAX_STATIC_VAR+1), +, blockind, nvs) <= MAX_STATIC_VAR
end

function costgradhess!(linsystem, vars::Vector, residual::Residual) where Residual <: AbstractResidual
    # Get the variables and associated data
    v = getvars(residual, vars)
    blockind = getoffsets(residual, linsystem)
    varflags = computevarflags(blockind)

    # Check that some variables are unfixed
    if varflags > 0
        # Check if we can do statically sized stuff
        if dynamic(usestatic(blockind, v))
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
        
        # Dynamically sized version
        return gradhesshelper!(linsystem, residual, v, blockind, varflags)
    end

    # No unfixed variables, so just return the cost
    return cost(residual, v)
end

costgradhess!(linsystem, vars::Vector, residuals::ResidualStruct)::Float64 = sum(fixallbutlast(costgradhess!, linsystem, vars), residuals)

function resjachelper!(linsystem, residual::Residual, vars, blockind, ind, varflags)::Float64 where Residual <: AbstractResidual
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
    return c
end

function costresjac!(linsystem, vars::Vector, residual::Residual, ind) where Residual <: AbstractResidual
    # Get the variables and associated data
    v = getvars(residual, vars)
    blockind = getoffsets(residual, linsystem)
    varflags = computevarflags(blockind)

    # Check that some variables are unfixed
    if varflags > 0
        # Check if we can do statically sized stuff
        if dynamic(usestatic(blockind, v))
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
        
        # Dynamically sized version
        return resjachelper!(linsystem, residual, v, blockind, ind, varflags)
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
