using NLLSsolver, LinearAlgebra, Test

@testset "geometry.jl" begin
    # Test utility functions
    @test NLLSsolver.rodrigues(0., 0., 0.) == I
    @test isapprox(NLLSsolver.rodrigues(Float64(pi), 0., 0.), Diagonal(SVector(1., -1., -1.)))
    O = NLLSsolver.proj2orthonormal(randn(5, 5))
    @test isapprox(O' * O, I)
end