using LDLFactorizations, LinearAlgebra, StaticArrays

# Fix for when ldiv! isn't supported
import LinearAlgebra.ldiv!
ldiv!(x, A, b) = x .= ldiv(A, b)

function try_cholesky!(x::StaticVector, A::StaticMatrix, b::StaticVector)
    # Convert to immutable type, to avoid allocations
    A_ = SMatrix(A)
    b_ = SVector(b)
    # Handle this issue with StaticArrays cholesky: https://github.com/JuliaArrays/StaticArrays.jl/issues/1194
    cholfac = StaticArrays._cholesky(Size(A_), A_, false)
    if issuccess(cholfac)
        return x .= cholfac \ b_
    end
    # Handle this issue with StaticArrays qr: https://github.com/JuliaArrays/StaticArrays.jl/issues/1192
    return x .= A_ \ b_
end

function try_cholesky!(x, A, b)
    cholfac = cholesky(A; check=false)
    if issuccess(cholfac)
        return ldiv!(x, cholfac, b)
    end
    return ldiv!(x, qr(A), b)
end

solve!(ls, options) = solve!(ls.x, ls.A, ls.b)
solve!(ls::MultiVariateLSsparse, options) = ldiv!(ls.x, ldl_factorize!(gethessian(ls), ls.ldlfac), ls.b)
solve!(ls::MultiVariateLSdense, options) = solve!(ls.x, ls.A.data, ls.b)
solve!(x, A, b) = try_cholesky!(x, A, b)
solve!(x, A::SparseMatrixCSC, b) = ldiv!(x, ldl(A), b)

# Symmetric matrix inversion
invsym(A::AbstractMatrix) = inv(bunchkaufman(A))
invsym(A::StaticMatrix) = Size(A)[1] <= 14 ? inv(A) : inv(bunchkaufman(A))
