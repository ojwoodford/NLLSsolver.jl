using Static

# Simple residuals modelling measurement noise errors
struct SimpleError2{N, T, V1, V2} <: AbstractResidual
    measurement::SVector{N, T} # Measurement
    varind::SVector{2, Int} # Variable indices
end
SimpleError2{V1, V2}(meas::SVector{N, T}, vi1, vi2) where {N, T, V1, V2} = SimpleError2{N, T, V1, V2}(meas, SVector{2, Int}(vi1, vi2))
ndeps(::SimpleError2) = static(2) # Residual depends on 2 variables
nres(::SimpleError2{N, T, V1, V2}) where {N, T, V1, V2} = static(N) # Residual vector has length N
varindices(res::SimpleError2) = res.varind
getvars(res::SimpleError2{N, T, V1, V2}, vars::Vector) where {N, T, V1, V2} = vars[@inbounds(res.varind[1])]::V1, vars[@inbounds(res.varind[2])]::V2
computeresidual(res::SimpleError2, arg1, arg2) = generatemeasurement(arg1, arg2) - res.measurement
Base.eltype(::SimpleError2{N, T, V1, V2}) where {N, T, V1, V2} = T

struct SimpleError3{N, T, V1, V2, V3} <: AbstractResidual
    measurement::SVector{N, T} # Measurement
    varind::SVector{3, Int} # Variable indices
end
SimpleError3{V1, V2, V3}(meas::SVector{N, T}, vi1, vi2, vi3) where {N, T, V1, V2, V3} = SimpleError3{N, T, V1, V2, V3}(meas, SVector{3, Int}(vi1, vi2, vi3))
ndeps(::SimpleError3) = static(3) # Residual depends on 3 variables
nres(::SimpleError3{N, T, V1, V2, V3}) where {N, T, V1, V2, V3} = static(N) # Residual vector has length N
varindices(res::SimpleError3) = res.varind
getvars(res::SimpleError3{N, T, V1, V2, V3}, vars::Vector) where {N, T, V1, V2, V3} = vars[@inbounds(res.varind[1])]::V1, vars[@inbounds(res.varind[2])]::V2, vars[@inbounds(res.varind[3])]::V3
computeresidual(res::SimpleError3, arg1, arg2, arg3) = generatemeasurement(arg1, arg2, arg3) - res.measurement
Base.eltype(::SimpleError3{N, T, V1, V2, V3}) where {N, T, V1, V2, V3} = T

struct SimpleError4{N, T, V1, V2, V3, V4} <: AbstractResidual
    measurement::SVector{N, T} # Measurement
    varind::SVector{4, Int} # Variable indices
end
SimpleError4{V1, V2, V3, V4}(meas::SVector{N, T}, vi1, vi2, vi3, vi4) where {N, T, V1, V2, V3, V4} = SimpleError4{N, T, V1, V2, V3, V4}(meas, SVector{4, Int}(vi1, vi2, vi3, vi4))
ndeps(::SimpleError4) = static(4) # Residual depends on 4 variables
nres(::SimpleError4{N, T, V1, V2, V3, V4}) where {N, T, V1, V2, V3, V4} = static(N) # Residual vector has length N
varindices(res::SimpleError4) = res.varind
getvars(res::SimpleError4{N, T, V1, V2, V3, V4}, vars::Vector) where {N, T, V1, V2, V3, V4} = vars[@inbounds(res.varind[1])]::V1, vars[@inbounds(res.varind[2])]::V2, vars[@inbounds(res.varind[3])]::V3, vars[@inbounds(res.varind[4])]::V4
computeresidual(res::SimpleError4, arg1, arg2, arg3, arg4) = generatemeasurement(arg1, arg2, arg3, arg4) - res.measurement
Base.eltype(::SimpleError4{N, T, V1, V2, V3, V4}) where {N, T, V1, V2, V3, V4} = T

# Dummy definition
generatemeasurement() = nothing

# Functions for computing costs and gradients of residuals
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
