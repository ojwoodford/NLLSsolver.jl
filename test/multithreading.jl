using Test
import NLLSsolver

@testset "multithreading.jl" begin
    v = randn(10000)
    fun(x) = x * x - x
    s1 = sum(fun, v)
    s2 = NLLSsolver.multithreadedsum(fun, v)
    @test isapprox(s1, s2; rtol=1.e-12)
end
