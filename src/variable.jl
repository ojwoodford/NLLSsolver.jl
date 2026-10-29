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

"""
    NLLSsolver.iseliminable(::Type{V}) -> StaticBool

Whether variables of type `V` may be eliminated from the normal equations by
the Schur complement, before the remaining (retained) variables are solved
for. The default is `static(false)`.

Return `static(true)` for a type with many instances, each of a small, static
size (`nvars` a `StaticInt`), that no cost couples to another eliminable
variable — the 3D points of a bundle adjustment are the standard case:

```julia
NLLSsolver.iseliminable(::Type{<:MyPoint}) = NLLSsolver.static(true)
```

When at least one unfixed variable is eliminable and at least one is not,
`optimize!` uses the Schur-complement linear system,
`MultiVariateLSschur`: each iteration it factorizes every eliminable
variable's diagonal block, solves the dense reduced system over the retained
variables, and back-substitutes the eliminated ones. A cost that depends on
two unfixed eliminable variables is refused with an `ArgumentError`.
"""
iseliminable(::Type) = static(false)
iseliminable(var) = iseliminable(typeof(var))
