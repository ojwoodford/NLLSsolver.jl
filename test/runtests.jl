using Test

@testset "NLLSsolver.jl" begin
    include("camera.jl")
    include("BlockSparseMatrix.jl")
    include("functional.jl")
    include("marginalize.jl")
end
