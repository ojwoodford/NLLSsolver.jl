using Static, StaticArrays

struct ContaminatedGaussian{T<:Real} <: AbstractAdaptiveRobustifier
    invsigma1::ZeroToInfScalar{T}
    invsigma2::ZeroToInfScalar{T}
    w::ZeroToOneScalar{T}
    s1sq::T
    s2sq::T
    halfs2sqminuss1sq::T
    halfs2sq::T
end
function ContaminatedGaussian(s1::ZeroToInfScalar{T}, s2::ZeroToInfScalar{T}, w::ZeroToOneScalar{T}) where T <: Real
    if T <: AbstractFloat # Don't bother for autodiff duals
        s1, s2 = ifelse(s1.val >= s2.val, (s1, s2), (s2, s1)) # Ensure the the first Gaussian is the narrowest (so largest inverse sigma)
    end
    s1sq = s1.val * s1.val
    s2sq = s2.val * s2.val
    ContaminatedGaussian{T}(s1, s2, w, s1sq, s2sq, 0.5 * (s2sq - s1sq), 0.5 * s2sq)
end
ContaminatedGaussian(s1::T, s2::T, w::T) where T <: Real = ContaminatedGaussian(ZeroToInfScalar{T}(1.0/s1), ZeroToInfScalar{T}(1.0/s2), ZeroToOneScalar{T}(w))
nvars(::ContaminatedGaussian) = static(3)
update(var::ContaminatedGaussian, updatevec, start=1) = ContaminatedGaussian(update(var.invsigma1, updatevec, start), update(var.invsigma2, updatevec, start+1), update(var.w, updatevec, start+2))
params(var::ContaminatedGaussian) = SVector(1.0 / var.invsigma1.val, 1.0 / var.invsigma2.val, var.w.val)

robustify(kernel::ContaminatedGaussian, cost) = cost * kernel.halfs2sq - log(kernel.w.val * kernel.invsigma1.val * exp(cost * kernel.halfs2sqminuss1sq) + (1 - kernel.w.val) * kernel.invsigma2.val)
function robustifydcost(kernel::ContaminatedGaussian, cost)
    c = cost * kernel.halfs2sq
    s = kernel.w.val * kernel.invsigma1.val * exp(cost * kernel.halfs2sqminuss1sq)
    t = (1 - kernel.w.val) * kernel.invsigma2.val
    den = 1 / (s + t)
    s *= kernel.halfs2sqminuss1sq
    return c + log(den), kernel.halfs2sq - s * den, -s * kernel.halfs2sqminuss1sq * t * den * den
end

function optimize(kernel::ContaminatedGaussian{T}, squarederrors::Vector{T}, maxiters=10)::ContaminatedGaussian{T} where T
    # Optimize the parameters of the kernel using Expectation-Maximization algorithm
    totalsquarederror = sum(squarederrors)
    oldparams = params(kernel)
    for iter = 1:maxiters
        wratio = ((1 - kernel.w.val) * kernel.invsigma2.val) / (kernel.invsigma1.val * kernel.w.val)
        halfs1sqminuss2sq = -kernel.halfs2sqminuss1sq
        sigma1 = T(0)
        totalweight = T(0)
        for (ind, err) in enumerate(squarederrors)
            # Expectation step - compute latent variables as a likelihood ratio
            w = 1 / (1 + wratio * exp(halfs1sqminuss2sq * err))
            # Compute running totals for the Maximization step
            sigma1 += w * err
            totalweight += w
        end
        # Complete the Maximization step (computing the optimal distribution parameters) by normalizing by the total weights
        newparams = SVector(sqrt(sigma1 / totalweight), sqrt((totalsquarederror - sigma1) / (length(squarederrors) - totalweight)), totalweight / length(squarederrors))
        kernel = ContaminatedGaussian(newparams...)
        if isapprox(oldparams, newparams; rtol=1.e-6)
            break
        end
        oldparams = newparams
    end
    return kernel
end
