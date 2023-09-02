using NLLSsolver, Test

@testset "utils.jl" begin
    @test NLLSsolver.runlengthencodesortedints([0, 0, 1, 1, 3]) == [1, 3, 5, 5, 6]
    @test NLLSsolver.runlengthencodesortedints([3]) == [1, 1, 1, 1, 2]
    @test NLLSsolver.runlengthencodesortedints([0]) == [1, 2]
end