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

function compute_cost(problem)
    allocd = @allocated cost = NLLSsolver.cost(problem, static(1))
    return (cost, allocd)
end

@testset "optimizeba.jl" begin
    # Generate some test data for a dense problem
    Random.seed!(1)
    problem = create_ba_problem(3, 5, 1.0)
    options = NLLSOptions(numthreads = static(1))

    # Test reordering the costs
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    costbefore = cost(problem)
    landmarks = isa.(problem.variables, EuclideanVector{3, Float64})
    NLLSsolver.reordercostsforschur!(problem, landmarks)
    @test cost(problem) ≈ costbefore

    # Optimze just the landmarks
    prob_m = deepcopy(problem)
    result = NLLSsolver.optimizesingles!(problem, options, findall(landmarks))
    result_m = NLLSsolver.optimizesingles!(prob_m, NLLSOptions(numthreads=static(4)), findall(landmarks))
    costafter = NLLSsolver.cost(problem, static(1))
    costafter_m = NLLSsolver.cost(prob_m, static(1))
    @test costbefore ≈ result.startcost
    @test isapprox(costafter, result.bestcost; atol=1.e-22)
    @test costafter < 1.e-15
    @test isapprox(costafter_m, result_m.bestcost; atol=1.e-22)
    @test costafter_m < 1.e-15

    # Optimze just the landmarks, specified by type
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    costbefore, allocd = compute_cost(problem)
    @test allocd == 0
    result = NLLSsolver.optimizesingles!(problem, options, EuclideanVector{3, Float64})
    display(result)
    costafter = NLLSsolver.cost(problem)
    @test costbefore ≈ result.startcost
    @test isapprox(costafter, result.bestcost; atol=1.e-22)
    @test costafter < 1.e-15

    # Optimize problem
    problem = perturb_ba_problem(problem, 0.001, 0.001)
    result = optimize!(problem, options)
    @test isapprox(NLLSsolver.cost(problem), result.bestcost; atol=1.e-22)
    @test result.bestcost < 1.e-15
    problem = perturb_ba_problem(problem, 0.001, 0.001)
    result = optimize!(problem, options)
    @test result.optimizeallocated <= 30960

    # Generate & optimize a sparse problem
    problem = create_ba_problem(10, 50, 0.3)
    problem = perturb_ba_problem(problem, 0.001, 0.001)
    result = optimize!(problem, options)
    @test isapprox(NLLSsolver.cost(problem), result.bestcost; atol=1.e-22)
    @test result.bestcost < 1.e-15
    problem = perturb_ba_problem(problem, 0.001, 0.001)
    result = optimize!(problem, options)
    @test result.optimizeallocated == 0

    # The same with D'D damping: dense, single-variable (static) and sparse linear systems
    dtdoptions = NLLSOptions(numthreads = static(1), iterator = :levenbergmarquardt_dtd)
    problem = perturb_ba_problem(create_ba_problem(3, 5, 1.0), 0.001, 0.001)
    result = optimize!(problem, dtdoptions)
    @test isapprox(NLLSsolver.cost(problem), result.bestcost; atol=1.e-22)
    @test result.bestcost < 1.e-15
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    problem_m = deepcopy(problem)
    dtdoptions_m = NLLSOptions(numthreads = static(4), iterator = :levenbergmarquardt_dtd)
    result = NLLSsolver.optimizesingles!(problem, dtdoptions, EuclideanVector{3, Float64})
    result_m = NLLSsolver.optimizesingles!(problem_m, dtdoptions_m, EuclideanVector{3, Float64})
    @test NLLSsolver.cost(problem) < 1.e-15
    @test NLLSsolver.cost(problem_m) < 1.e-15
    @test result.bestcost ≈ result_m.bestcost
    problem = perturb_ba_problem(create_ba_problem(10, 50, 0.3), 0.001, 0.001)
    result = optimize!(problem, dtdoptions)
    @test isapprox(NLLSsolver.cost(problem), result.bestcost; atol=1.e-22)
    @test result.bestcost < 1.e-15
    @test result.optimizeallocated == 0
end
