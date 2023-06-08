module NLLSsolver
# Abstract types
export AbstractResidual
abstract type AbstractResidual end

# Constants
const MAX_ARGS = 10 # Maximum number of variables a residual can depend on

# Helper functions
include("utils.jl")
include("unroll.jl")

# Problem definition
include("problem.jl")

# Variables
include("variable.jl")
include("visualgeometry/camera.jl")
include("visualgeometry/geometry.jl")

# Types
include("BlockSparseMatrix.jl")
include("linearsystem.jl")
include("structs.jl")

# Optimization
include("marginalize.jl")
include("robust.jl")
include("autodiff.jl")
include("cost.jl")
include("linearsolver.jl")
include("iterators.jl")
include("optimize.jl")
end
