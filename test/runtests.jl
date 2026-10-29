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
    include("multithreading.jl")
    include("utils.jl")
    include("nantrial.jl")
    include("copyvars.jl")
    include("OptimizeCache.jl")
    include("trialreuse.jl")
end
