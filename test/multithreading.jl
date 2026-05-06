using Test
import NLLSsolver

@testset "multithreading.jl" begin
    # Cost sums
    fun(x) = x * x * x
    v = rand(1000000)
    s1 = sum(fun, v)
    s2 = NLLSsolver.multithreadedsum(fun, v)
    @test isapprox(s1, s2; rtol=1.e-13)
    # Subset sums with Vector{Bool}
    subset = rand(Bool, length(v))
    s1 = sum(fun(v[i]) for i in eachindex(v) if subset[i])
    s2 = NLLSsolver.sumsubsetloop(fun, subset, v)
    @test isapprox(s1, s2; rtol=1.e-13)
    # Subset sums with BitVector
    subset = BitVector(subset)
    s2 = NLLSsolver.sumsubsetloop(fun, subset, v)
    @test isapprox(s1, s2; rtol=1.e-13)
    # Subset sums with Vector{Int}
    subset = findall(subset)
    s2 = NLLSsolver.sumsubsetloop(fun, subset, v)
    @test isapprox(s1, s2; rtol=1.e-13)
end
