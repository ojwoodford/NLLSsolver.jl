module NLLSsolver
# Export the API
# Types
export AbstractCost, AbstractResidual, AbstractRobustifier # Abstract problem definition types
export NLLSProblem, NLLSOptions, NLLSResult, NLLSIterator # Concrete problem & solver types
export AbstractRobustifier, NoRobust, HuberKernel, GemanMcclureKernel # Robustifiers
export EuclideanVector, ZeroToInfScalar, ZeroToOneScalar # Concrete general variables
export Rotation3DR, Rotation3DL, Point3D, Pose3D, EffPose3D, UnitPose3D # Concrete 3D geometry variable types
export SimpleCamera, NoDistortionCamera, ExtendedUnifiedCamera, BarrelDistortion, EULensDistortion # Concrete camera sensor & lens variable types
# Functions
export addresidual!, addvariable!, subproblem, nvars, nres # Construct a problem
export update, nvars # Variable interface
export nres, ndeps, varindices, getvars, computeresidual # Residual interface 
export cost, computeresjac # Compute the objective
export optimize!  # Optimize the objective
export rodrigues, project, epipolarerror, proj2orthonormal # Multi-view geometry helper functions
export ideal2image, image2ideal, pixel2image, image2pixel, ideal2distorted, distorted2ideal, convertlens

# Exported abstract types
abstract type AbstractCost end
abstract type AbstractResidual <: AbstractCost end
abstract type AbstractRobustifier end

# Constants
const MAX_ARGS = 10 # Maximum number of variables a residual can depend on
const MAX_BLOCK_SZ = 32 # Maximum DoF of a variable and also maximum length of a residual allowed for various operations, such as marginalization
const MAX_STATIC_VAR = 64 # Maximum static variable size for static sized autodiff to be used

# Helper functions
include("utils.jl")
include("unroll.jl")

# Problem definition
include("VectorRepo.jl")
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
