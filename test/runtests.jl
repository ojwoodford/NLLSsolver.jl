using Test

@testset "NLLSsolver.jl" begin
    include("functional.jl")
    include("optimizeba.jl")
    include("OptimizeCache.jl")
end
