export Robustifier, NoRobust, HuberKernel, GemanMcclureKernel
export robustify, robustkernel
# import Plots.plot

# Robustification
abstract type Robustifier end
struct NoRobust <: Robustifier
end

function robustify(::NoRobust, cost)
    return cost, 1, 0
end

function robustkernel(::AbstractResidual)
    return NoRobust()
end

struct ScaledNoRobust{T<:Real} <: Robustifier
    height::T
    height_sqrt::T
end

ScaledNoRobust(h) = ScaledNoRobust(h, sqrt(h))

function robustify(kernel::ScaledNoRobust, cost)
    return cost * kernel.height, kernel.height_sqrt, 0
end

struct HuberKernel{T<:Real} <: Robustifier
    width::T
    width_squared::T
    height::T
    height_sqrt::T
end

HuberKernel(w, h) = HuberKernel(w, w*w, h, sqrt(h))

function robustify(kernel::HuberKernel, cost)
    if cost < kernel.width_squared
        return cost * kernel.height, kernel.height_sqrt, 0
    end
    sqrtcost = sqrt(cost)
    return (sqrtcost * (kernel.width * 2) - kernel.width_squared) * kernel.height, kernel.width * kernel.height_sqrt / sqrtcost, 0
end


struct GemanMcclureKernel{T<:Real} <: Robustifier
    width_squared::T
    height::T
    height_sqrt::T
end

GemanMcclureKernel(w, h) = GemanMcclureKernel(w*w, h/(w*w), sqrt(h)/w)

function robustify(kernel::GemanMcclureKernel, cost)
    w = kernel.width_squared / (cost + kernel.width_squared)
    return cost * w * kernel.height, w * w * kernel.height_sqrt, 0
end


# function displaykernel(kernel, maxval=1)
#     x = LinRange(0, maxval, 1000)
#     cost = x .^ 2
#     weight = similar(cost)
#     for ind in eachindex(cost)
#         cost[ind], weight[ind], unused = robustify(kernel, cost[ind])
#     end
#     plot(x, [cost, weight])
# end
