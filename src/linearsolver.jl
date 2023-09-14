using LDLFactorizations, LinearAlgebra, LinearSolve, StaticArrays

# Fix for when ldiv! isn't supported
import LinearAlgebra.ldiv!
ldiv!(x, A, b) = x .= ldiv(A, b)

function try_cholesky!(x::StaticVector, A::StaticMatrix, b::StaticVector)
    # Handle this issue with StaticArrays cholesky: https://github.com/JuliaArrays/StaticArrays.jl/issues/1194
    cholfac = StaticArrays._cholesky(Size(A), A, false)
    if issuccess(cholfac)
        return ldiv!(x, cholfac, b)
    end
    # Handle this issue with StaticArrays qr: https://github.com/JuliaArrays/StaticArrays.jl/issues/1192
    return x .= A \ b
end

function try_cholesky!(x, A, b)
    cholfac = cholesky(A; check=false)
    if issuccess(cholfac)
        return ldiv!(x, cholfac, b)
    end
    return ldiv!(x, qr(A), b)
end

solve!(ls, options) = solve!(ls.x, ls.A, ls.b)
solve!(ls::MultiVariateLSsparse, options) = solve!(ls.x, ls.hessian, ls.b)
solve!(ls::MultiVariateLSdense, options) = solve!(ls.x, ls.A.data, ls.b)
solve!(x, A, b) = try_cholesky!(x, A, b)
solve!(x, A::SparseMatrixCSC, b) = ldiv!(x, ldl(A), b)
