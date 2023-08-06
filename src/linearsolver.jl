using LDLFactorizations, LinearSolve

@inline function symmetricsolve(A::SparseMatrixCSC, b::AbstractVector, options)
    return ldl(A) \ b
end

@inline function symmetricsolve(A::AbstractMatrix, b::AbstractVector, options)
    return Symmetric(A) \ b
end

@inline function linearsolve(A::AbstractMatrix, b::AbstractVector, options)
    return solve!(init(LinearProblem(A, b))).u
end
