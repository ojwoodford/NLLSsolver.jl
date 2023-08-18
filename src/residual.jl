function computecost(residual::AbstractResidual, vars...)::Float64
    # Compute the residual
    r = computeresidual(residual, vars...)
    
    # Compute the robustified cost
    return 0.5 * robustify(robustkernel(residual), Float64(r' * r))
end

function computecostgradhess(varflags, residual::AbstractResidual, vars...)
    # Compute the residual and Jacobian
    res, jac = computeresjac(varflags, residual, vars...)
 
    # Compute the unrobust gradient and Hessian
    g = jac' * res
    H = jac' * jac

    # Compute the robustified cost and the IRLS weight
    c, dc, d2c = robustifydcost(robustkernel(residual), res' * res)

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
    return 0.5 * robustify(kernel, Float64(r' * r))
end

function computecostgradhess(varflags, residual::AbstractAdaptiveResidual, kernel, vars...)
    kernelvarind = SR(1, nvars(kernel))

    # Check for case that only kernel is optimized
    varflagsres = varflags >> static(1)
    if varflagsres == 0
        # Only the kernel is optimized
        res = computeresidual(residual, vars...)
        cost, dc, d2c = robustifydkernel(kernel, res' * res)
        return 0.5 * Float64(cost), dc[kernelvarind], d2c[kernelvarind, kernelvarind]
    end

    # Compute the residual and Jacobian
    res, jac = computeresjac(varflagsres, residual, vars...)
 
    # Compute the unrobust cost, gradient and Hessian
    cost = res' * res
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
