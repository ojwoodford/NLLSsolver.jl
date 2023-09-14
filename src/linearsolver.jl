using LDLFactorizations, LinearAlgebra, LinearSolve, StaticArrays

# Fix for when ldiv! isn't supported
import LinearAlgebra.ldiv!
ldiv!(x, A, b) = x .= ldiv(A, b)

symmetricsolve!(x::AbstractVector, A::SparseMatrixCSC, b::AbstractVector, options) = ldiv!(x, ldl(A), b)

function symmetricsolve!(x::AbstractVector, A::StaticMatrix, b::AbstractVector, options)
    cholfac = StaticArrays._cholesky(Size(A), A, false)
    if issuccess(cholfac)
        return ldiv!(x, cholfac, b)
    end
    return ldiv!(x, qr(A), b)
end

function symmetricsolve!(x::AbstractVector, A::AbstractMatrix, b::AbstractVector, options)
    cholfac = cholesky(A; check=false)
    if issuccess(cholfac)
        return ldiv!(x, cholfac, b)
    end
    return ldiv!(x, qr(A), b)
end
