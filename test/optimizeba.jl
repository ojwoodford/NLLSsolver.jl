using NLLSsolver, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra
import NLLSsolver.OptimizationVariableInterface

struct BALandmark{T}
    p::SVector{3, T}
end
OptimizationVariableInterface.nvars(::Type{<:BALandmark}) = static(3)
OptimizationVariableInterface.nvars(::BALandmark) = static(3)
OptimizationVariableInterface.update(v::BALandmark, step, offset) = BALandmark(v.p + SVector(step[offset], step[offset+1], step[offset+2]))
OptimizationVariableInterface.iseliminable(::Type{<:BALandmark}) = static(true)

# Simple affine projection transform
NLLSsolver.generatemeasurement(pose::EuclideanVector{6, T}, X::EuclideanVector{3, U}) where {T, U} = SVector(dot(@inbounds(view(pose, NLLSsolver.SR(1, 3))), X), dot(@inbounds(view(pose, NLLSsolver.SR(4, 6))), X))
NLLSsolver.generatemeasurement(pose::EuclideanVector{6, T}, X::BALandmark{U}) where {T, U} = SVector(dot(@inbounds(view(pose, NLLSsolver.SR(1, 3))), X.p), dot(@inbounds(view(pose, NLLSsolver.SR(4, 6))), X.p))

function create_schur_ba_problem(ncameras, nlandmarks, propvisible)
    problem = NLLSProblem(Union{EuclideanVector{6, Float64}, BALandmark{Float64}}, SimpleError2{2, Float64, EuclideanVector{6, Float64}, BALandmark{Float64}})
    camoffset = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for i = 1:ncameras
        addvariable!(problem, randn(EuclideanVector{6, Float64}) .+ camoffset)
    end
    lmoffset = SVector(-0.5, -0.5, 10.0)
    for i = 1:nlandmarks
        addvariable!(problem, BALandmark(rand(SVector{3, Float64}) .+ lmoffset))
    end
    visibility = abs.(repeat(vec(1:ncameras), outer=(1, nlandmarks)) .- LinRange(2, ncameras-1, nlandmarks)')
    visibility = visibility .<= sort(vec(visibility))[Int(ceil(length(visibility)*propvisible))]
    for camind = 1:ncameras
        camera = problem.variables[camind]::EuclideanVector{6, Float64}
        for (landmark, tf) in enumerate(view(visibility, camind, :)')
            if tf
                landmarkind = landmark + ncameras
                addcost!(problem, SimpleError2{EuclideanVector{6, Float64}, BALandmark{Float64}}(generatemeasurement(camera, problem.variables[landmarkind]::BALandmark{Float64}), camind, landmarkind))
            end
        end
    end
    return problem
end

function perturb_schur_ba_problem(problem, pointnoise, posenoise)
    for ind in 1:lastindex(problem.variables)
        if isa(problem.variables[ind], BALandmark{Float64})
            p = problem.variables[ind]::BALandmark{Float64}
            problem.variables[ind] = BALandmark(p.p + randn(SVector{3, Float64}) * pointnoise)
        else
            problem.variables[ind]::EuclideanVector{6, Float64} += randn(SVector{6, Float64}) * posenoise
        end
    end
    return problem
end

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
    options = NLLSOptions()

    # Test reordering the costs
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    costbefore = cost(problem)
    landmarks = isa.(problem.variables, EuclideanVector{3, Float64})
    NLLSsolver.reordercostsforschur!(problem, landmarks)
    @test cost(problem) ≈ costbefore

    # Optimze just the landmarks
    result = NLLSsolver.optimizesingles!(problem, options, findall(landmarks))
    costafter = NLLSsolver.cost(problem, static(1))
    @test costbefore ≈ result.startcost
    @test isapprox(costafter, result.bestcost; atol=1.e-22)
    @test costafter < 1.e-15

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
    dtdoptions = NLLSOptions(iterator = :levenbergmarquardt_dtd)
    problem = perturb_ba_problem(create_ba_problem(3, 5, 1.0), 0.001, 0.001)
    result = optimize!(problem, dtdoptions)
    @test isapprox(NLLSsolver.cost(problem), result.bestcost; atol=1.e-22)
    @test result.bestcost < 1.e-15
    problem = perturb_ba_problem(problem, 0.003, 0.0)
    result = NLLSsolver.optimizesingles!(problem, dtdoptions, EuclideanVector{3, Float64})
    @test NLLSsolver.cost(problem) < 1.e-15
    problem = perturb_ba_problem(create_ba_problem(10, 50, 0.3), 0.001, 0.001)
    result = optimize!(problem, dtdoptions)
    @test isapprox(NLLSsolver.cost(problem), result.bestcost; atol=1.e-22)
    @test result.bestcost < 1.e-15
    @test result.optimizeallocated == 0

    # Schur-Complement Solvers (Dense & Sparse)
    @testset "Schur solvers" begin
        # 1. Dense Schur
        schur_dense_opts = NLLSOptions(linsolver=:dense_schur)
        prob_dense = perturb_schur_ba_problem(create_schur_ba_problem(3, 8, 1.0), 0.001, 0.001)
        res_dense = optimize!(prob_dense, schur_dense_opts)
        @test isapprox(NLLSsolver.cost(prob_dense), res_dense.bestcost; rtol=1e-8, atol=1e-15)
        @test res_dense.bestcost < 1e-9

        # 2. Sparse Schur
        schur_sparse_opts = NLLSOptions(linsolver=:sparse_schur)
        prob_sparse = perturb_schur_ba_problem(create_schur_ba_problem(8, 20, 0.4), 0.001, 0.001)
        res_sparse = optimize!(prob_sparse, schur_sparse_opts)
        @test isapprox(NLLSsolver.cost(prob_sparse), res_sparse.bestcost; rtol=1e-8, atol=1e-15)
        @test res_sparse.bestcost < 1e-9

        # 3. Auto Schur selection via eliminable trait
        prob_auto = perturb_schur_ba_problem(create_schur_ba_problem(4, 10, 0.8), 0.001, 0.001)
        res_auto = optimize!(prob_auto, NLLSOptions())
        @test isapprox(NLLSsolver.cost(prob_auto), res_auto.bestcost; rtol=1e-8, atol=1e-15)
        @test res_auto.bestcost < 1e-9

        # 4. Dense Schur with D'D damping
        schur_dtd_opts = NLLSOptions(iterator=:levenbergmarquardt_dtd, linsolver=:dense_schur)
        prob_dtd = perturb_schur_ba_problem(create_schur_ba_problem(3, 8, 1.0), 0.001, 0.001)
        res_dtd = optimize!(prob_dtd, schur_dtd_opts)
        @test isapprox(NLLSsolver.cost(prob_dtd), res_dtd.bestcost; rtol=1e-8, atol=1e-15)
        @test res_dtd.bestcost < 1e-6

        # 5. Sparse Schur with D'D damping
        sparse_dtd_opts = NLLSOptions(iterator=:levenbergmarquardt_dtd, linsolver=:sparse_schur)
        prob_sdtd = perturb_schur_ba_problem(create_schur_ba_problem(6, 15, 0.5), 0.001, 0.001)
        res_sdtd = optimize!(prob_sdtd, sparse_dtd_opts)
        @test isapprox(NLLSsolver.cost(prob_sdtd), res_sdtd.bestcost; rtol=1e-8, atol=1e-15)
        @test res_sdtd.bestcost < 1e-6
    end
end
