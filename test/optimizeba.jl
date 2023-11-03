using NLLSsolver, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra

# Simple affine projection transform
NLLSsolver.generatemeasurement(pose::EuclideanVector{6, T}, X::EuclideanVector{3, U}) where {T, U} = SVector(dot(@inbounds(view(pose, NLLSsolver.SR(1, 3))), X), dot(@inbounds(view(pose, NLLSsolver.SR(4, 6))), X))

function create_ba_problem(ncameras, nlandmarks, propvisible)
    problem = NLLSProblem(Union{EuclideanVector{6, Float64}, EuclideanVector{3, Float64}}, SimpleError2{2, Float64, EuclideanVector{6, Float64}, EuclideanVector{3, Float64}})

    # Generate the cameras on a unit sphere, pointing to the origin
    camoffset = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for i = 1:ncameras
        addvariable!(problem, randn(EuclideanVector{6, Float64}) .+ camoffset)
    end
    
    # Generate the landmarks in a unit cube centered on the origin
    lmoffset = SVector(-0.5, -0.5, 10.0)
    for i = 1:nlandmarks
        addvariable!(problem, rand(EuclideanVector{3, Float64}) .+ lmoffset)
    end

    # Generate the measurements
    visibility = abs.(repeat(vec(1:ncameras), outer=(1, nlandmarks)) .- LinRange(2, ncameras-1, nlandmarks)')
    visibility = visibility .<= sort(vec(visibility))[Int(ceil(length(visibility)*propvisible))]
    for camind = 1:ncameras
        camera = problem.variables[camind]::EuclideanVector{6, Float64}
        for (landmark, tf) in enumerate(view(visibility, camind, :)')
            if tf
                landmarkind = landmark + ncameras
                addcost!(problem, SimpleError2{EuclideanVector{6, Float64}, EuclideanVector{3, Float64}}(generatemeasurement(camera, problem.variables[landmarkind]::EuclideanVector{3, Float64}), camind, landmarkind))
            end
        end
    end

    # Return the NLLSProblem
    return problem
end

function perturb_ba_problem(problem, pointnoise, posenoise)
    for ind in 1:lastindex(problem.variables)
        if isa(problem.variables[ind], EuclideanVector{3, Float64})
            problem.variables[ind]::EuclideanVector{3, Float64} += randn(SVector{3, Float64}) * pointnoise
        else
            problem.variables[ind]::EuclideanVector{6, Float64} += randn(SVector{6, Float64}) * posenoise
        end
    end
    return problem
end

@testset "optimizeba.jl" begin
    # Generate some test data for a dense problem
    Random.seed!(1)
    problem = create_ba_problem(3, 5, 1.0)

    # Test reordering the costs
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    costbefore = cost(problem)
    NLLSsolver.reordercostsforschur!(problem, isa.(problem.variables, EuclideanVector{3, Float64}))
    @test cost(problem) ≈ costbefore

    # Optimze just the landmarks
    NLLSsolver.optimizesingles!(problem, NLLSOptions(), EuclideanVector{3, Float64})
    @test NLLSsolver.cost(problem) < 1.e-15

    # Optimize problem
    problem = perturb_ba_problem(problem, 0.001, 0.001)
    result = optimize!(problem)
    @test NLLSsolver.cost(problem) == result.bestcost
    @test result.bestcost < 1.e-15

    # Generate & optimize a sparse problem
    problem = create_ba_problem(10, 50, 0.3)
    problem = perturb_ba_problem(problem, 0.001, 0.001)
    result = optimize!(problem)
    @test NLLSsolver.cost(problem) == result.bestcost
    @test result.bestcost < 1.e-15
end
