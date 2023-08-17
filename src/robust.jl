using Static
# import Plots.plot

# Robustification
struct NoRobust <: AbstractRobustifier
end

robustkernel(::AbstractResidual) = NoRobust()
@inline robustify(::NoRobust, cost) = cost
@inline robustifyd(::NoRobust, cost) = cost, one(cost), zero(cost)

robustifyd(kernel::AbstractRobustifier, cost) = autorobustifyd(kernel, cost)
robustifydd(kernel::AbstractAdaptiveRobustifier, cost) = autorobustifydd(kernel, cost)

struct Scaled{T<:Real,Robustifier<:AbstractRobustifier} <: AbstractRobustifier
    robust::Robustifier
    height::T
end
robustify(kernel::Scaled, cost) = robustify(kernel.robust, cost) * kernel.height
robustifyd(kernel::Scaled{Real, NoRobust}, cost) = cost * kernel.height, kernel.height, zero(cost)
function robustifyd(kernel::Scaled, cost)
    c, d1, d2 = robustifyd(kernel.robust, cost)
    return c * kernel.height, d1 * kernel.height, d2 * kernel.height
end


struct HuberKernel{T<:Real, B} <: AbstractRobustifier
    width::T
    width_squared::T
    secondorder::B
end
HuberKernel(w) = HuberKernel(w, w*w, static(false))
Huber2oKernel(w) = HuberKernel(w, w*w, static(true))

robustify(kernel::HuberKernel, cost) = cost < kernel.width_squared ? cost : sqrt(cost) * (kernel.width * 2) - kernel.width_squared
function robustifyd(kernel::HuberKernel, cost)
    if cost < kernel.width_squared
        return cost, one(cost), zero(cost)
    end
    sqrtcost = sqrt(cost)
    return sqrtcost * (kernel.width * 2) - kernel.width_squared, kernel.width / sqrtcost, dynamic(kernel.secondorder) ? (-0.5 * kernel.width) / (cost * sqrtcost) : zero(cost)
end


struct GemanMcclureKernel{T<:Real} <: AbstractRobustifier
    width_squared::T
    function GemanMcclureKernel{T}(w::T) where T
        return new(w * w)
    end
end
GemanMcclureKernel(w::T) where T = GemanMcclureKernel{T}(w)

robustify(kernel::GemanMcclureKernel, cost) = cost * kernel.width_squared / (cost + kernel.width_squared)
function robustifyd(kernel::GemanMcclureKernel, cost)
    r = 1.0 / (cost + kernel.width_squared)
    w = kernel.width_squared * r
    w2 = w * w
    return cost * w, w2, -2 * w2 * r
end
