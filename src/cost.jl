using FLoops: @floop, @reduce
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

function cost(residual::Residual, vars::Vector)::Float64 where Residual <: AbstractResidual
    # Get the variables required to compute the residual
    v = getvars(residual, vars)

    # Compute the residual
    r = computeresidual(residual, v...)
    
    # Compute the robustified cost
    return robustify(robustkernel(residual), Float64(r' * r))[1]
end

function getoffsets(residual, linsystem::MultiVariateLS)
    return linsystem.blockindices[varindices(residual)]
end

function getoffsets(residual, linsystem::UniVariateLS)
    return convert.(UInt, SVector(varindices(residual)) .== linsystem.varindex)
end

function gradhesshelper!(linsystem, residual::Residual, vars::Vector, blockind, ::Val{varflags})::Float64 where {varflags, Residual <: AbstractResidual}
    # Get the variables
    v = getvars(residual, vars)

    # Compute the residual
    res, jac = computeresjac(Val(varflags), residual, v...)

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
        updatesymlinearsystem!(linsystem, g, H, v, Val(varflags), blockind)
    end

    # Return the cost
    return c
end

function costgradhess!(linsystem, residual::Residual, vars::Vector) where Residual <: AbstractResidual
    # Get the bitset for the input variables, as an integer
    blockind = getoffsets(residual, linsystem)
    varflags = Int(foldl((x, y) -> (x << 1) + (y != 0), reverse(blockind), init=UInt(0)))

    # If there are no variables, just return the cost
    if varflags == 0
        return cost(residual, vars)
    end

    # Dispatch gradient computation based on the varflags, and return the cost
    if ndeps(residual) <= 5
        return valuedispatch(Val(1), Val((2^ndeps(residual))-1), varflags, fixallbutlast(gradhesshelper!, linsystem, residual, vars, blockind))
    end
    return gradhesshelper!(linsystem, residual, vars, blockind, Val(varflags))
end

function costgradhess!(linsystem, residuals, vars::Vector)::Float64
    # Go over all resdiduals in the problem
    c = 0.
    for res in values(residuals)
        c += costgradhess!(linsystem, res, vars)
    end
    return c 
end

function resjachelper!(linsystem, residual::Residual, vars::Vector, blockind, ind, ::Val{varflags})::Float64 where {varflags, Residual <: AbstractResidual}
    # Get the variables
    v = getvars(residual, vars)

    # Compute the residual
    res, jac = computeresjac(Val(varflags), residual, v...)

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
        updatelinearsystem!(linsystem, res, jac, ind, v, Val(varflags), blockind)
    end

    # Return the cost
    return c
end

function costresjac!(linsystem, residual::Residual, vars::Vector, ind) where Residual <: AbstractResidual
    # Get the bitset for the input variables, as an integer
    blockind = getoffsets(residual, linsystem)
    varflags = Int(foldl((x, y) -> (x << 1) + (y != 0), reverse(blockind), init=UInt(0)))

    # If there are no variables, just return the cost
    if varflags == 0
        return cost(residual, vars)
    end

    # Dispatch gradient computation based on the varflags, and return the cost
    if ndeps(residual) <= 5
        return valuedispatch(Val(1), Val((2^ndeps(residual))-1), varflags, fixallbutlast(resjachelper!, linsystem, residual, vars, blockind, ind))
    end
    return resjachelper!(linsystem, residual, vars, blockind, ind, Val(varflags))
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
