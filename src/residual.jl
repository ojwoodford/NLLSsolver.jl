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
        return @inbounds (0.5 * Float64(cost), dc[kernelvarind], d2c[kernelvarind, kernelvarind])
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
    else
        # Differentiate w.r.t. the cost and kernel parameters
        cost, dc_, d2c_ = robustifydkernel(kernel, cost)
        @inbounds dc = dc_[end]
        @inbounds d2c = d2c_[end,end]
    
        # Compute the d^2/dkernel.dvariables block
        kernelvarind = SR(1, nvars(kernel))
        @inbounds dkdv = g * view(d2c_, kernelvarind, nvars(kernel)+1)'
    end

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

    if varflags & 1 == 1
        # Add on the kernel derivative blocks
        @inbounds g = vcat(view(dc_, kernelvarind), g)
        @inbounds H = hcat(vcat(view(d2c_, kernelvarind, kernelvarind), dkdv), vcat(dkdv', H))
    end
    
    # Return the cost and derivatives
    return 0.5 * Float64(cost), g, H
end
