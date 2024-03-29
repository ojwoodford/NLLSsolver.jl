using Test

@testset "NLLSsolver.jl" begin
    include("BlockSparseMatrix.jl")
    include("VectorRepo.jl")
    include("robust.jl")
    include("linearsolve.jl")
    include("functional.jl")
    include("optimizeba.jl")
    include("dynamicvars.jl")
    include("nonsquaredcost.jl")
    include("adaptivecost.jl")
    include("utils.jl")
end
