using LDLFactorizations, LinearSolve

function symmetricsolve(A::SparseMatrixCSC, b::AbstractVector, options)
    return ldl(A) \ b
end

function symmetricsolve(A::AbstractMatrix, b::AbstractVector, options)
    return Symmetric(A) \ b
end

function linearsolve(A::AbstractMatrix, b::AbstractVector, options)
    return solve!(init(LinearProblem(A, b))).u
end
