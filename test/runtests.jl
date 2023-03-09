using Test

@testset "NLLSsolver.jl" begin
    include("camera.jl")
    include("BlockSparseMatrix.jl")
    include("marginalize.jl")
    include("functional.jl")
end
