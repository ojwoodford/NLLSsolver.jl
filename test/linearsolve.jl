using NLLSsolver, SparseArrays, Test

@testset "linearsolve.jl" begin
    # Construct a linear problem with known solution
    A = NLLSsolver.proj2orthonormal(randn(5, 5))
    A = A * Diagonal(rand(5)) * A'
    x = randn(5)
    b = A * x

    # Test the solvers
    @test isapprox(NLLSsolver.symmetricsolve(A, b, "test"), x)
    @test isapprox(NLLSsolver.symmetricsolve(sparse(A), b, nothing), x)
    @test isapprox(NLLSsolver.linearsolve(A, b, 1), x)
end