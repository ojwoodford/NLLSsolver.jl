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
    g = jac * res

    if varflags & 1 == 0
        # The kernel is not optimized. Differentiate w.r.t. the cost only
        cost, dc, d2c = robustifydcost(kernel, cost)
    else
        # Differentiate w.r.t. the cost and kernel parameters
        cost, dc_, d2c_ = robustifydkernel(kernel, cost)
        dc = dc_[end]
        d2c = d2c_[end,end]
    
        # Compute the d^2/dkernel.dvariables block
        kernelvarind = SR(1, nvars(kernel))
        @fastmath dkdv = @inbounds(view(d2c_, kernelvarind, nvars(kernel)+1)) * g'
    end

    # IRLS reweighting
    if dc != 1
        # Second order correction
        if d2c == 0
            H = A_mul_B(jac, jac' * dc)
        else
            H = A_mul_B(hcat(jac, g), hcat(jac * dc, g * (2 * d2c))')
        end
        g *= dc
    else
        H = A_mul_B(jac, jac')
    end

    if varflags & 1 == 1
        # Add on the kernel derivative blocks
        g = vcat(view(dc_, kernelvarind), g)
        H = hcat(vcat(view(d2c_, kernelvarind, kernelvarind), dkdv'), vcat(dkdv, H))
    end
    
    # Return the cost and derivatives
    return 0.5 * Float64(cost), g, H
end
