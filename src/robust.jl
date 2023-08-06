using Static
# import Plots.plot

# Robustification
struct NoRobust <: AbstractRobustifier
end

function robustify(::NoRobust, cost::T) where T
    return cost, T(1), T(0)
end

function robustkernel(::AbstractResidual)
    return NoRobust()
end


struct Scaled{T<:Real,Robustifier<:AbstractRobustifier} <: AbstractRobustifier
    robust::Robustifier
    height::T
end
robustify(kernel::Scaled{Real, NoRobust}, cost) = cost * kernel.height, kernel.height, zero(cost)

function robustify(kernel::Scaled, cost)
    c, d1, d2 = robustify(kernel.robust, cost)
    return c * kernel.height, d1 * kernel.height, d2 * kernel.height
end


struct HuberKernel{T<:Real, B} <: AbstractRobustifier
    width::T
    width_squared::T
    secondorder::B
end
HuberKernel(w) = HuberKernel(w, w*w, static(false))
Huber2oKernel(w) = HuberKernel(w, w*w, static(true))

function robustify(kernel::HuberKernel, cost)
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

function robustify(kernel::GemanMcclureKernel{T}, cost) where T
    w = kernel.width_squared / (cost + kernel.width_squared)
    return cost * w, w * w, T(0)
end

struct AdaptiveKernel <: AbstractRobustifier
end
struct AdaptiveKernelPartitionNegLog{T} <: AbstractCost
    varind::Int
end
ndeps(::AdaptiveKernelPartitionNegLog) = static(1) # The residual depends on one variable
varindices(cost::AdaptiveKernelPartitionNegLog) = cost.varind
computecost(cost::AdaptiveKernelPartitionNegLog, kernel::AdaptiveKernel) = 0.1
computecostgradhess(varflags, cost::AdaptiveKernelPartitionNegLog, kernel::AdaptiveKernel) = (0.1, 1.0, nothing)
getvars(cost::AdaptiveKernelPartitionNegLog, vars::Vector) = (vars[cost.varind]::AdaptiveKernel,)
Base.eltype(::AdaptiveKernelPartitionNegLog{T}) where T = T


# function displaykernel(kernel, maxval=1)
#     x = range(0, maxval, 1000)
#     cost = x .^ 2
#     weight = similar(cost)
#     for (ind, c) in enumerate(cost)
#         cost[ind], weight[ind], = robustify(kernel, c)
#     end
#     plot(x, [cost, weight])
# end
