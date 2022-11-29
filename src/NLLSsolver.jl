module NLLSsolver
# Abstract types
export AbstractResidual
abstract type AbstractResidual end

# Helper functions
include("utils.jl")

# Problem definition
include("problem.jl")

# Variables
include("variable.jl")
include("camera.jl")
include("geometry.jl")

# Optimization
include("cost.jl")
include("optimize.jl")
include("robust.jl")
end
