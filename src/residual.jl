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

    # Compute the residual and unrobustified cost
    res, jac = computeresjac(varflagsres, residual, vars...)
    cost_ = sqnorm(res)

    # Robustify
    cost, dc, d2c = robustifydcost(kernel, cost_)

    # Compute the gradient and Hessian
    g = jac * res
    H = compute_hessian(jac, g, dc, d2c)

    if varflags & 1 == 0
        # IRLS reweighting of gradient
        g *= dc
    else
        # Differentiate w.r.t. the cost and kernel parameters
        cost_, dc, d2c = robustifydkernel(kernel, cost_)
    
        # Compute the d^2/dkernel.dvariables block
        kernelvarind = SR(1, nvars(kernel))
        @inbounds dkdv = g * view(d2c, kernelvarind, nvars(kernel)+1)'

        # Add on the kernel derivative blocks
        @inbounds g = vcat(view(dc, kernelvarind), g * dc[end])
        @inbounds H = hcat(vcat(view(d2c, kernelvarind, kernelvarind), dkdv), vcat(dkdv', H))
    end
    
    # Return the cost and derivatives
    return 0.5 * Float64(cost), g, H
end

@inline function compute_hessian(jac::StaticArray, g, dc, d2c)
    N = Size(jac)[1]
    H = MMatrix{N, N, eltype(jac), N*N}(undef)
    if dc != 1
        # IRLS reweighting
        if d2c == 0
            @turbo for n in axes(jac, 1), m in axes(jac, 1)
                Cmn = zero(eltype(C))
                for k in axes(jac, 2)
                    Cmn += jac[m,k] * jac[n,k] * dc
                end
                H[m,n] = Cmn
            end
        else
            d2c *= 2
            # Second order correction
            @turbo for n in axes(jac, 1), m in axes(jac, 1)
                Cmn = zero(eltype(C))
                for k in axes(jac, 2)
                    Cmn += jac[m,k] * jac[n,k] * dc
                end
                Cmn += g[m] * g[n] * d2c
                H[m,n] = Cmn
            end
        end
    else
        @turbo for n in axes(jac, 1), m in axes(jac, 1)
            Cmn = zero(eltype(C))
            for k in axes(jac, 2)
                Cmn += jac[m,k] * jac[n,k]
            end
            H[m,n] = Cmn
        end
    end
    return SMatrix(H)
end

@inline function compute_hessian(jac, g, dc, d2c)
    N = size(jac, 1)
    H = Matrix{eltype(jac)}(undef, (N, N))
    if dc != 1
        # IRLS reweighting
        if d2c == 0
            @turbo for n in axes(jac, 1), m in axes(jac, 1)
                Cmn = zero(eltype(C))
                for k in axes(jac, 1)
                    Cmn += jac[m,k] * jac[n,k] * dc
                end
                H[m,n] = Cmn
            end
        else
            d2c *= 2
            # Second order correction
            @turbo for n in axes(jac, 1), m in axes(jac, 1)
                Cmn = zero(eltype(C))
                for k in axes(jac, 2)
                    Cmn += jac[m,k] * jac[n,k] * dc
                end
                Cmn += g[m] * g[n] * d2c
                H[m,n] = Cmn
            end
        end
    else
        @turbo for n in axes(jac, 1), m in axes(jac, 1)
            Cmn = zero(eltype(C))
            for k in axes(jac, 2)
                Cmn += jac[m,k] * jac[n,k]
            end
            H[m,n] = Cmn
        end
    end
    return H
end
