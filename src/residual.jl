function computecost(residual::AbstractResidual, vars...)::Float64
    # Compute the residual
    r = computeresidual(residual, vars...)
    
    # Compute the robustified cost
    return 0.5 * robustify(robustkernel(residual), Float64(r' * r))[1]
end

function computecostgradhess(varflags, residual::AbstractResidual, vars...)
    # Compute the residual
    res, jac = computeresjac(varflags, residual, vars...)
 
    # Compute the unrobust gradient and Hessian
    g = jac' * res
    H = jac' * jac

    # Compute the robustified cost and the IRLS weight
    c, w1, w2 = robustify(robustkernel(residual), res' * res)

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

    # Return the cost
    return 0.5 * c, g, H
end
