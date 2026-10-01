using Test, Static, NLLSsolver

myfun(x) = x * x * x
NLLSsolver.computecost(::StaticInt, ::Vector{Int}, x::Float64) = myfun(x)
NLLSsolver.computecost(::Vector{Int}, x::Float64) = myfun(x)

@testset "multithreading.jl" begin
    # Cost sums
    v = rand(Int(1e6))
    s1 = sum(myfun, v)
    s2 = sum(NLLSsolver.bindleadingargs(computecost, static(1), Vector{Int}()), v)
    @test isapprox(s1, s2; rtol=1.e-13)
    s3 = sum(NLLSsolver.bindleadingargs(computecost, static(Threads.nthreads()), Vector{Int}()), v)
    @test isapprox(s1, s3; rtol=1.e-13)
    # # Subset sums with Vector{Bool}
    # subset = rand(Bool, length(v))
    # s1 = sum(fun(v[i]) for i in eachindex(v) if subset[i])
    # s2 = NLLSsolver.sumsubsetloop(fun, subset, v)
    # @test isapprox(s1, s2; rtol=1.e-13)
    # # Subset sums with BitVector
    # subset = BitVector(subset)
    # s2 = NLLSsolver.sumsubsetloop(fun, subset, v)
    # @test isapprox(s1, s2; rtol=1.e-13)
    # # Subset sums with Vector{Int}
    # subset = findall(subset)
    # s2 = NLLSsolver.sumsubsetloop(fun, subset, v)
    # @test isapprox(s1, s2; rtol=1.e-13)
end
