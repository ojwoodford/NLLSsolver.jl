using NLLSsolver, SparseArrays, Test

@testset "linearsolve.jl" begin
    # Construct a linear problem with known solution
    A = NLLSsolver.proj2orthonormal(randn(5, 5))
    A = A * Diagonal(rand(5)) * A'
    x = randn(5)
    b = A * x

    # Test the solvers
    y = zeros(length(b))
    @test isapprox(NLLSsolver.symmetricsolve!(y, A, b, "test"), x)
    y = zeros(length(b))
    @test isapprox(NLLSsolver.symmetricsolve!(y, sparse(A), b, nothing), x)
end