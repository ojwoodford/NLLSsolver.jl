
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

function robustify(kernel::ContaminatedGaussian, cost)
    c = cost * kernel.halfs2sq
    return c - log(kernel.w.val * kernel.invsigma1.val * exp(cost * kernel.halfs2sqminuss1sq) + (1 - kernel.w.val) * kernel.invsigma2.val)
end
function robustifydcost(kernel::ContaminatedGaussian, cost)
    c = cost * kernel.halfs2sq
    s = kernel.w.val * kernel.invsigma1.val * exp(cost * kernel.halfs2sqminuss1sq)
    t = (1 - kernel.w.val) * kernel.invsigma2.val
    den = 1 / (s + t)
    s *= kernel.halfs2sqminuss1sq
    return c + log(den), kernel.halfs2sq - s * den, -s * kernel.halfs2sqminuss1sq * t * den * den
end
function robustifydkernel(kernel::ContaminatedGaussian, cost)
    c = cost * kernel.halfs2sq
    ex = exp(cost * kernel.halfs2sqminuss1sq)
    wex = kernel.w.val * ex
    swex = wex * kernel.invsigma1.val
    oneminusw = 1 - kernel.w.val
    tw = oneminusw * kernel.invsigma2.val
    den = 1 / (swex + tw)
    s = swex * kernel.halfs2sqminuss1sq
    return c + log(den), # Value
           SVector{4}(wex * (kernel.s1sq * cost - 1) * den, oneminusw * (2 * c - 1) * den, (kernel.invsigma2.val - kernel.invsigma1.val * ex) * den, kernel.halfs2sq - s * den), # Gradient
           zeros(SMatrix{4, 4})
end