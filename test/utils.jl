using NLLSsolver, Test, SparseArrays, StaticArrays, LinearAlgebra

test_fast_bAb(A, b) = @test NLLSsolver.fast_bAb(A, b) ≈ dot(b, A, b)

@testset "utils.jl" begin
    @test NLLSsolver.runlengthencodesortedints([0, 0, 1, 1, 3]) == [1, 3, 5, 5, 6]
    @test NLLSsolver.runlengthencodesortedints([3]) == [1, 1, 1, 1, 2]
    @test NLLSsolver.runlengthencodesortedints([0]) == [1, 2]

    x = [1, 2, 3]
    @test cumsum!(x) == [1, 3, 6]
    @test x == [1, 3, 6]

    test_fast_bAb(randn(20, 20), randn(20))
    test_fast_bAb(sprandn(100, 100, 0.02), randn(100))
    test_fast_bAb(randn(SMatrix{10, 10, Float64, 100}), randn(SVector{10, Float64}))
end
