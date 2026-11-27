using Test

@testset "NLLSsolver.jl" begin
    include("exports.jl")
    include("functional.jl")
    include("passthrough.jl")
    include("optimizeba.jl")
    include("OptimizeCache.jl")
    include("aqua.jl")
end
