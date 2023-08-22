using Test

@testset "NLLSsolver.jl" begin
    include("camera.jl")
    include("geometry.jl")
    include("BlockSparseMatrix.jl")
    include("VectorRepo.jl")
    include("robust.jl")
    include("linearsolve.jl")
    include("functional.jl")
    include("dynamicvars.jl")
    include("nonsquaredcost.jl")
    include("adaptivecost.jl")
    include("marginalize.jl")
end
