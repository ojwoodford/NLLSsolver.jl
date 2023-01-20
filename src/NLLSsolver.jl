module NLLSsolver
# Abstract types
export AbstractResidual
abstract type AbstractResidual end

# Helper functions
include("utils.jl")
include("unroll.jl")

# Problem definition
include("problem.jl")

# Variables
include("variable.jl")
include("camera.jl")
include("geometry.jl")

# Optimization
include("BlockSparseMatrix.jl")
include("linearsystem.jl")
include("robust.jl")
include("cost.jl")
include("optimize.jl")
end
