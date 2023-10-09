using NLLSsolver, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra

# Simple reprojection residual
struct ProjError <: AbstractResidual
    measurement::SVector{2, Float64} # Landmark image coordinates (x, y)
    varind::SVector{2, Int} # Variable indices for the residual (image, landmark)
end
ProjError(meas, cam, lndm) = ProjError(SVector{2}(meas[1], meas[2]), SVector{2, Int}(cam, lndm))
NLLSsolver.ndeps(::ProjError) = static(2) # Residual depends on 2 variables
NLLSsolver.nres(::ProjError) = static(2) # Residual vector has length 2
NLLSsolver.varindices(res::ProjError) = res.varind
NLLSsolver.getvars(res::ProjError, vars::Vector) = vars[res.varind[1]]::EffPose3D{Float64}, vars[res.varind[2]]::Point3D{Float64}
NLLSsolver.computeresidual(res::ProjError, pose::EffPose3D, X::Point3D) = project(pose * X) - res.measurement
Base.eltype(::ProjError) = Float64

function create_ba_problem(ncameras, nlandmarks, propvisible)
    problem = NLLSProblem(Union{EffPose3D{Float64}, Point3D{Float64}}, ProjError)

    # Generate the cameras on a unit sphere, pointing to the origin
    for i = 1:ncameras
        camcenter = normalize(randn(SVector{3, Float64}))
        addvariable!(problem, EffPose3D(Rotation3DL(UnitVec3D(-camcenter).v.m), Point3D(camcenter)))
    end
    
    # Generate the landmarks in a unit cube centered on the origin
    for i = 1:nlandmarks
        addvariable!(problem, Point3D{Float64}(rand(SVector{3, Float64}) .- 0.5))
    end

    # Generate the measurements
    visibility = sprand(nlandmarks, ncameras, propvisible)
    for camind = 1:ncameras
        camera = problem.variables[camind]::EffPose3D{Float64}
        for landmark = view(visibility.rowval, visibility.colptr[camind]:visibility.colptr[camind+1]-1)
            landmarkind = landmark + ncameras
            addcost!(problem, ProjError(project(camera * problem.variables[landmarkind]::Point3D{Float64}), camind, landmarkind))
        end
    end

    # Return the NLLSProblem
    return problem
end

function perturb_ba_problem(problem, pointnoise, posenoise)
    for ind in 1:lastindex(problem.variables)
        if isa(problem.variables[ind], Point3D)
            problem.variables[ind]::Point3D{Float64} = update(problem.variables[ind]::Point3D{Float64}, randn(SVector{3, Float64}) * pointnoise)
        else
            problem.variables[ind]::EffPose3D{Float64} = update(problem.variables[ind]::EffPose3D{Float64}, randn(SVector{6, Float64}) * posenoise)
        end
    end
    return problem
end

@testset "optimizeba.jl" begin
    # Generate some test data for a dense problem
    Random.seed!(20)
    problem = create_ba_problem(10, 100, 0.3)

    # Test reordering the costs
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    costbefore = cost(problem)
    NLLSsolver.reordercostsforschur!(problem, isa.(problem.variables, Point3D{Float64}))
    @test cost(problem) ≈ costbefore

    # Optimze just the landmarks
    result = NLLSsolver.optimizesingles!(problem, NLLSOptions(), Point3D{Float64})
    @test result.bestcost < 1.e-15

    # Optimize problem
    problem = perturb_ba_problem(problem, 0.001, 0.001)
    result = optimize!(problem)
    @test result.bestcost < 1.e-15

    # Generate & optimize a sparse problem
    problem = create_ba_problem(10, 400, 0.3)
    problem = perturb_ba_problem(problem, 1.e-7, 1.e-7)
    result = optimize!(problem)
    @test result.bestcost < result.startcost * 1e-10
end
