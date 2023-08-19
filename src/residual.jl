using Static

function computecost(residual::AbstractResidual, vars...)::Float64
    # Compute the residual
    r = computeresidual(residual, vars...)
    
    # Compute the robustified cost
    return 0.5 * robustify(robustkernel(residual), Float64(sqnorm(r)))
end

function computecostgradhess(varflags, residual::AbstractResidual, vars...)
    # Compute the residual and Jacobian
    res, jac = computeresjac(varflags, residual, vars...)
 
    # Compute the unrobust gradient and Hessian
    g = jac' * res
    H = jac' * jac

    # Compute the robustified cost and the IRLS weight
    c, dc, d2c = robustifydcost(robustkernel(residual), sqnorm(res))

    # Check for robust case
    if dc != 1
        # IRLS reweighting of Hessian
        H *= dc
        if d2c < 0
            # Second order correction
            H += ((2 * d2c) * g) * g'
        end
        # IRLS reweighting of gradient
        g *= dc
    end

    # Return the cost and derivatives
    return 0.5 * Float64(c), g, H
end

function computecost(residual::AbstractAdaptiveResidual, kernel, vars...)::Float64
    # Compute the residual
    r = computeresidual(residual, vars...)
    
    # Compute the robustified cost
    return 0.5 * robustify(kernel, Float64(sqnorm(r)))
end

function computecostgradhess(varflags, residual::AbstractAdaptiveResidual, kernel, vars...)
    kernelvarind = SR(1, nvars(kernel))

    # Check for case that only kernel is optimized
    varflagsres = varflags >> static(1)
    if varflagsres == 0
        # Only the kernel is optimized
        res = computeresidual(residual, vars...)
        cost, dc, d2c = robustifydkernel(kernel, sqnorm(res))
        return 0.5 * Float64(cost), dc[kernelvarind], d2c[kernelvarind, kernelvarind]
    end

    # Compute the residual and Jacobian
    res, jac = computeresjac(varflagsres, residual, vars...)
 
    # Compute the unrobust cost, gradient and Hessian
    cost = sqnorm(res)
    g = jac' * res
    H = jac' * jac

    if varflags & 1 == 0
        # The kernel is not optimized. Differentiate w.r.t. the cost only
        cost, dc, d2c = robustifydcost(kernel, cost)

        # IRLS reweighting of Hessian
        H *= dc
        if d2c < 0
            # Second order correction
            H += ((2 * d2c) * g) * g'
        end

        # IRLS reweighting of gradient
        g *= dc

        # Return the cost
        return 0.5 * Float64(cost), g, H
    end

    # Differentiate w.r.t. the cost and kernel parameters
    cost, dc, d2c = robustifydkernel(kernel, cost)

    # Compute the d^2/dkernel.dvariables block
    dkdv = g * view(d2c, kernelvarind, nvars(kernel)+1)'

    # IRLS reweighting of Hessian
    H *= dc[end]
    if d2c[end] < 0
        # Second order correction
        H += ((2 * d2c[end]) * g) * g'
    end
    
    # IRLS reweighting of gradient
    g *= dc[end]

    # Add on the kernel derivative blocks
    g = vcat(view(dc, kernelvarind), g)
    H = hcat(vcat(view(d2c, kernelvarind, kernelvarind), dkdv), vcat(dkdv', H))
    
    # Return the cost and derivatives
    return 0.5 * Float64(cost), g, H
end

resjachelper!(linsystem, residual::AbstractResidual, vars, blockind, ind, varflags) = resjachelper!(linsystem, residual, vars, blockind, ind, varflags, robustkernel(residual))
resjachelper!(linsystem, residual::AbstractAdaptiveResidual, vars, blockind, ind, varflags) = resjachelper!(linsystem, residual, vars[2:end], blockind, ind, varflags >> static(1), vars[1])

function resjachelper!(linsystem, residual::AbstractResidual, vars, blockind, ind, varflags, kernel)::Float64
    # Compute the residual
    res, jac = computeresjac(varflags, residual, vars...)

    # Compute the robustified cost and the IRLS weight
    c, w1, = robustifydcost(kernel, sqnorm(res))

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
