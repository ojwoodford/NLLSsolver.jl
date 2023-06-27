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
robustify(kernel::Scaled{T, NoRobust}, cost) where T = cost * kernel.height, kernel.height, T(0)

function robustify(kernel::Scaled, cost)
    c, d1, d2 = robustify(kernel.robust, cost)
    return c * kernel.height, d1 * kernel.height, d2 * kernel.height
end


struct HuberKernel{T<:Real} <: AbstractRobustifier
    width::T
    width_squared::T
end
HuberKernel(w) = HuberKernel(w, w*w)

function robustify(kernel::HuberKernel{T}, cost) where T
    if cost < kernel.width_squared
        return cost, T(1), T(0)
    end
    sqrtcost = sqrt(cost)
    return sqrtcost * (kernel.width * 2) - kernel.width_squared, kernel.width / sqrtcost, T(0)
end


struct Huber2oKernel{T<:Real} <: AbstractRobustifier
    width::T
    width_squared::T
end
Huber2oKernel(w) = Huber2oKernel(w, w*w)

function robustify(kernel::Huber2oKernel{T}, cost) where T
    if cost < kernel.width_squared
        return cost, T(1), T(0)
    end
    sqrtcost = sqrt(cost)
    return sqrtcost * (kernel.width * 2) - kernel.width_squared, kernel.width / sqrtcost, (-0.5 * kernel.width) / (cost * sqrtcost)
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


# function displaykernel(kernel, maxval=1)
#     x = range(0, maxval, 1000)
#     cost = x .^ 2
#     weight = similar(cost)
#     for (ind, c) in enumerate(cost)
#         cost[ind], weight[ind], unused = robustify(kernel, c)
#     end
#     plot(x, [cost, weight])
# end
