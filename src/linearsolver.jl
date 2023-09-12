using LDLFactorizations, LinearAlgebra, LinearSolve, StaticArrays

symmetricsolve(A::SparseMatrixCSC, b::AbstractVector, options) = ldl(A) \ b

function symmetricsolve(A::StaticMatrix, b::AbstractVector, options)
    cholfac = StaticArrays._cholesky(Size(A), A, false)
    if issuccess(cholfac)
        return cholfac \ b
    end
    return qr(A) \ b
end

function symmetricsolve(A::AbstractMatrix, b::AbstractVector, options)
    cholfac = cholesky(A; check=false)
    if issuccess(cholfac)
        return cholfac \ b
    end
    return qr(A) \ b
end
