using FLoops: @floop, @reduce
using Static
import ForwardDiff

cost(problem::NLLSProblem) = cost(problem.residuals, problem.variables)

function cost(residuals, vars::Vector)::Float64
    # Compute the cost of all residuals in the problem
    c = 0.
    @inbounds for res in values(residuals)
        c += cost(res, vars)
    end
    return c
end

function cost(residuals::Vector, vars::Vector)::Float64
    # Compute the total cost of all residuals in a container
    c = 0.
    @floop for res in residuals
        c_ = cost(res, vars)
        @reduce c += c_
    end
    return c
end

cost(residual::AbstractResidual, vars::Vector) = cost(residual, getvars(residual, vars))

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

function computevarlen(blockind, vars) 
    # Compute the total length of variables, plus the variable flags
    varflags = 0
    varlen = 0
    isstatic = true
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && blockind[i] != 0
            nv = nvars(vars[i])
            isstatic &= is_static(nv)
            varlen += Int(nvars(vars[i]))
            varflags |= 1 << (i - 1)
        end
    end
    return varlen, varflags, isstatic
end

function costgradhess!(linsystem, residual::Residual, vars::Vector) where Residual <: AbstractResidual
    # Get the variables and associated data
    v = getvars(residual, vars)
    blockind = getoffsets(residual, linsystem)
    varlen, varflags, isstatic = computevarlen(blockind, v)

    # Check that there are some unfixed variables
    if varflags > 0
        if isstatic && varlen <= 64
            # We can do statically sized stuff from now on
            # Common case - all unfixed
            maxflags = static(2 ^ known(ndeps(residual)) - 1)
            if varflags == maxflags
                return gradhesshelper!(linsystem, residual, v, blockind, maxflags)
            end

            # Dispatch gradient computation based on the varflags, and return the cost
            if ndeps(residual) <= 5
                return valuedispatch(static(1), maxflags-static(1), v, fixallbutlast(gradhesshelper!, linsystem, residual, v, blockind))
            end
            return gradhesshelper!(linsystem, residual, v, blockind, static(varflags))
        end

        # Dynamically sized version
        return gradhesshelper!(linsystem, residual, v, blockind, varflags)
    end
    
    # No unfixed variables, so just return the cost
    return cost(residual, v)
end

function costgradhess!(linsystem, residuals, vars::Vector)::Float64
    # Go over all resdiduals in the problem
    c = 0.
    for res in values(residuals)
        c += costgradhess!(linsystem, res, vars)
    end
    return c 
end

function resjachelper!(linsystem, residual::Residual, vars, blockind, ind, varflags)::Float64 where Residual <: AbstractResidual
    # Compute the residual
    res, jac = computeresjac(varflags, residual, vars...)

    # Compute the robustified cost and the IRLS weight
    c, w1, unused = robustify(robustkernel(residual), res' * res)

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

function costresjac!(linsystem, residual::Residual, vars::Vector, ind) where Residual <: AbstractResidual
    # Get the variables and associated data
    v = getvars(residual, vars)
    blockind = getoffsets(residual, linsystem)
    varlen, varflags, isstatic = computevarlen(blockind, v)

    # Check that there are some unfixed variables
    if varflags > 0
        if isstatic && varlen <= 64
            # We can do statically sized stuff from now on
            # Common case - all unfixed
            maxflags = static(2 ^ known(ndeps(residual)) - 1)
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

function costresjac!(linsystem, residuals, vars::Vector, ind=1)::Float64
    # Go over all resdiduals in the problem
    c = 0.
    for res in values(residuals)
        c += costresjac!(linsystem, res, vars, ind)
        ind += length(res)
    end
    return c 
end
