using StaticArrays
export EuclideanVector, ZeroToInfScalar, ZeroToOneScalar

# Scalar
nvars(@objtype(Number)) = 1
function update(var::Number, updatevec, start=1)
    return var + updatevec[start]
end

# Standard Euclidean vector of length N
const EuclideanVector{N, T} = SVector{N, T}
nvars(@objtype(EuclideanVector{N, T})) where {N, T} = N
function update(var::EuclideanVector{N, T}, updatevec, start=1)  where {N, T}
    return EuclideanVector(var + updatevec[SR(0, N-1) .+ start])
end

# A scalar in the range zero to +Inf
struct ZeroToInfScalar{T}
    val::T
end
nvars(@objtype(ZeroToInfScalar)) = 1
function update(var::ZeroToInfScalar, updatevec, start=1)
    return ZeroToInfScalar((var.val > 0 ? var.val : floatmin(var.val)) * exp(updatevec[start]))
end

# A scalar in the range zero to one
struct ZeroToOneScalar{T}
    val::T
end
nvars(@objtype(ZeroToOneScalar)) = 1
function update(var::ZeroToOneScalar, updatevec, start=1)
    val = (var.val > 0 ? var.val : floatmin(var.val)) * exp(updatevec[start])
    return ZeroToOneScalar(val < Inf ? val / (1 + (val - var.val)) : convert(typeof(val), 1))
end