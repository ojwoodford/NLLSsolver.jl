using Static

computecost(residual::AbstractResidual, vars...) = computerescost(residual, robustkernel(residual), vars)
computecostgradhess(varflags, residual::AbstractResidual, vars...) = computerescostgradhess(varflags << static(1), residual, robustkernel(residual), vars)
computecost(residual::AbstractAdaptiveResidual, kernel, vars...) = computerescost(residual, kernel, vars)
computecostgradhess(varflags, residual::AbstractAdaptiveResidual, kernel, vars...) = computerescostgradhess(varflags, residual, kernel, vars)

function computerescost(residual, kernel, vars)::Float64
    # Compute the residual
    r = computeresidual(residual, vars...)
    
    # Compute the robustified cost
    return 0.5 * robustify(kernel, Float64(sqnorm(r)))
end

function computerescostgradhess(varflags, residual, kernel, vars)
    # Check for case that only kernel is optimized
    varflagsres = varflags >> static(1)
    if varflagsres == 0
        # Only the kernel is optimized
        res = computeresidual(residual, vars...)
        cost, dc, d2c = robustifydkernel(kernel, sqnorm(res))
        kernelvarind = SR(1, nvars(kernel))
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
        if dc != 1
            H *= dc
        end
        # Second order correction
        if d2c != 0
            H += ((2 * d2c) * g) * g'
        end
        # IRLS reweighting of gradient
        if dc != 1
            g *= dc
        end

        # Return the cost
        return 0.5 * Float64(cost), g, H
    end

    # Differentiate w.r.t. the cost and kernel parameters
    cost, dc, d2c = robustifydkernel(kernel, cost)

    # Compute the d^2/dkernel.dvariables block
    kernelvarind = SR(1, nvars(kernel))
    dkdv = g * view(d2c, kernelvarind, nvars(kernel)+1)'

    # IRLS reweighting of Hessian
    if dc[end] != 1
        H *= dc[end]
    end
    # Second order correction
    if d2c[end] != 0
        H += ((2 * d2c[end]) * g) * g'
    end
    # IRLS reweighting of gradient
    if dc[end] != 1
        g *= dc[end]
    end

    # Add on the kernel derivative blocks
    g = vcat(view(dc, kernelvarind), g)
    H = hcat(vcat(view(d2c, kernelvarind, kernelvarind), dkdv), vcat(dkdv', H))
    
    # Return the cost and derivatives
    return 0.5 * Float64(cost), g, H
end
