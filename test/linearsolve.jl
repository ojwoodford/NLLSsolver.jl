using NLLSsolver, SparseArrays, StaticArrays, LinearAlgebra, Test

@testset "linearsolve.jl" begin
    # Construct a symmetric positive definite linear problem with known solution
    A = randn(5, 5)
    A = A' * A
    x = randn(5)
    b = A * x

    # Test the solvers
    y = zeros(length(b))
    @test isapprox(NLLSsolver.solve!(y, A, b), x)
    y = zeros(length(b))
    @test isapprox(NLLSsolver.solve!(MVector{5}(y), MMatrix{5, 5}(A),  MVector{5}(b)), x)
    y = zeros(length(b))
    @test isapprox(NLLSsolver.solve!(y, sparse(A), b), x)

    # Construct a non-symmetric linear problem with known solution
    A = randn(5, 5)
    x = randn(5)
    b = A * x

    # Test the solvers
    y = zeros(length(b))
    @test isapprox(NLLSsolver.solve!(y, A, b), x)
    y = zeros(length(b))
    @test isapprox(NLLSsolver.solve!(MVector{5}(y), MMatrix{5, 5}(A),  MVector{5}(b)), x)
    y = zeros(length(b))
    @test !isapprox(NLLSsolver.solve!(y, sparse(A), b), x) # Currently expected to fail

    # Construct a symmetric negative-definite linear problem with known solution
    A = randn(5, 5)
    b = 2 * rand(5)
    A = A' * A .- b' * b
    @test !isposdef(A)
    x = randn(5)
    b = A * x

    # Test the solvers
    y = zeros(length(b))
    @test isapprox(NLLSsolver.solve!(y, A, b), x)
    y = zeros(length(b))
    @test isapprox(NLLSsolver.solve!(MVector{5}(y), MMatrix{5, 5}(A),  MVector{5}(b)), x)
    y = zeros(length(b))
    @test isapprox(NLLSsolver.solve!(y, sparse(A), b), x)
end