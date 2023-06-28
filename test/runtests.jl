using Test

@testset "NLLSsolver.jl" begin
    include("camera.jl")
    include("geometry.jl")
    include("BlockSparseMatrix.jl")
    include("VectorRepo.jl")
    include("linearsolve.jl")
    include("functional.jl")
    include("marginalize.jl")
end
