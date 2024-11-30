using StaticArrays

# Scalar
nvars(::Number) = static(1)
update(var::Number, updatevec, start=1) = var + updatevec[start]

# Standard fixed-length Euclidean vector of length N
const EuclideanVector{N, T} = SVector{N, T}
nvars(::EuclideanVector{N, T}) where {N, T} = static(N)
update(var::EuclideanVector{N, T}, updatevec, start=1)  where {N, T} = EuclideanVector(var + view(updatevec, SR(0, N-1) .+ start))

# Variable-length Euclidean vector
const DynamicVector{T} = Vector{T}
nvars(v::DynamicVector) = length(v)
update(var::DynamicVector, updatevec, start=1) = DynamicVector(var + view(updatevec, start:start-1+length(var)))

# A scalar in the range zero to +Inf
struct ZeroToInfScalar{T}
    val::T
end
nvars(::ZeroToInfScalar) = static(1)
update(var::ZeroToInfScalar, updatevec, start=1) = ZeroToInfScalar(ifelse(var.val > 0, var.val, floatmin(var.val)) * exp(updatevec[start]))

# A scalar in the range zero to one
struct ZeroToOneScalar{T}
    val::T
end
nvars(::ZeroToOneScalar) = static(1)
function update(var::ZeroToOneScalar, updatevec, start=1)
    val = ifelse(var.val > 0, var.val, floatmin(var.val)) * exp(updatevec[start])
    return ZeroToOneScalar(ifelse(val < Inf, val / (1 + (val - var.val)), one(val)))
end