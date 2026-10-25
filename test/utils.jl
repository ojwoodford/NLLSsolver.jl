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

    @test NLLSsolver.timestring(500) == "500ns"
    @test NLLSsolver.timestring(1500) == "1.5μs"
    @test NLLSsolver.timestring(1500000) == "1.5ms"
    @test NLLSsolver.timestring(1500000000) == "1.5s"
    @test NLLSsolver.timestring(90000000000) == "1.5 minutes"
    @test NLLSsolver.timestring(7200000000000) == "2 hours"
    @test NLLSsolver.timestring(720000000000000) == "200 hours"

    @test NLLSsolver.bytesstring(500) == "500 bytes"
    @test NLLSsolver.bytesstring(2048) == "2KB"
    @test NLLSsolver.bytesstring(1048576 * 5) == "5MB"
    @test NLLSsolver.bytesstring(1073741824 * 3) == "3GB"
    @test NLLSsolver.bytesstring(1099511627776 * 4) == "4TB"
    @test NLLSsolver.bytesstring(1099511627776 * 4000) == "4000TB"

    s1 = NLLSsolver.Stats()
    x = zeros(1000)
    s2 = NLLSsolver.Stats()
    @test s2.bytes_allocd >= s1.bytes_allocd
    @test s2.time_ns >= s1.time_ns
    @test s2.num_allocs >= s1.num_allocs
end
