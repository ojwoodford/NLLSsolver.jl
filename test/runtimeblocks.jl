using NLLSsolver, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra

# Affine projection, for a pose and point of static or run-time size
NLLSsolver.generatemeasurement(pose::AbstractVector, X::AbstractVector) = SVector(dot(view(pose, 1:3), X), dot(view(pose, 4:6), X))

# The same bundle adjustment problem, with statically or dynamically sized variables
function create_runtime_ba_problem(P, X, ncameras, nlandmarks)
    problem = NLLSProblem(Union{P, X}, SimpleError2{2, Float64, P, X})
    for i = 1:ncameras
        addvariable!(problem, P(randn(6) .+ [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    end
    for i = 1:nlandmarks
        addvariable!(problem, X(rand(3) .+ [-0.5, -0.5, 10.0]))
    end
    for camind = 1:ncameras, landmark = 1:nlandmarks
        meas = NLLSsolver.generatemeasurement(problem.variables[camind]::P, problem.variables[landmark+ncameras]::X) + 0.01 * randn(SVector{2, Float64})
        addcost!(problem, SimpleError2{P, X}(meas, camind, landmark + ncameras))
    end
    return problem
end

function create_linear_system(problem, sparse::Bool)
    unfixed = trues(length(problem.variables))
    blocksizes, blockindices = NLLSsolver.block_sizes_indices(problem.variables, unfixed, length(problem.variables))
    if sparse
        bsm = NLLSsolver.BlockSparseMatrix{Float64}(NLLSsolver.block_sparsity(problem, unfixed), blocksizes, blocksizes)
        return NLLSsolver.MultiVariateLSsparse(bsm, blockindices)
    end
    return NLLSsolver.MultiVariateLSdense(blocksizes, blockindices)
end

function linearize(problem, sparse::Bool)
    linsystem = create_linear_system(problem, sparse)
    c = NLLSsolver.costgradhess!(linsystem, problem.variables, problem.costs)
    return c, copy(NLLSsolver.getgrad(linsystem)), Matrix(NLLSsolver.gethessian(linsystem))
end

# Accumulate one cost's gradient and Hessian into the linear system
function assemble!(linsystem, g, H, vars, blockind)
    NLLSsolver.updatesymlinearsystem!(linsystem, g, H, vars, static(3), blockind)
    return nothing
end
assemblyallocations(linsystem, g, H, vars, blockind) = @allocated assemble!(linsystem, g, H, vars, blockind)

@testset "runtimeblocks.jl" begin
    Random.seed!(1)
    problem_s = create_runtime_ba_problem(EuclideanVector{6, Float64}, EuclideanVector{3, Float64}, 5, 20)
    Random.seed!(1)
    problem_d = create_runtime_ba_problem(Vector{Float64}, Vector{Float64}, 5, 20)
    costtype = (SimpleError2{2, Float64, EuclideanVector{6, Float64}, EuclideanVector{3, Float64}}, SimpleError2{2, Float64, Vector{Float64}, Vector{Float64}})

    for sparse in (true, false)
        # Statically and dynamically sized variables give the same linearization
        c_s, g_s, H_s = linearize(problem_s, sparse)
        c_d, g_d, H_d = linearize(problem_d, sparse)
        @test c_d ≈ c_s
        @test g_d ≈ g_s
        @test H_d ≈ H_s

        # Block accessors are type stable for run-time sizes
        A = NLLSsolver.getA(create_linear_system(problem_d, sparse))
        @test isconcretetype(Base.promote_op(NLLSsolver.block, typeof(A), Int, Int, Int, Int))
        @test isconcretetype(Base.promote_op(NLLSsolver.block, typeof(A), Int, Int, StaticInt{6}, Int))
        @test isconcretetype(Base.promote_op(NLLSsolver.block, typeof(A), Int, Int, StaticInt{6}, StaticInt{3}))

        # Assembly of run-time sized blocks constructs no types, so does not allocate
        for (problem, ct, g, H) in ((problem_s, costtype[1], @SVector(randn(9)), @SMatrix(randn(9, 9))), (problem_d, costtype[2], randn(9), randn(9, 9)))
            linsystem = create_linear_system(problem, sparse)
            cost = get(problem.costs, ct)[1]
            vars = NLLSsolver.getvars(cost, problem.variables)
            blockind = NLLSsolver.getoffsets(cost, linsystem)
            assemblyallocations(linsystem, g, H, vars, blockind)
            @test assemblyallocations(linsystem, g, H, vars, blockind) == 0
        end
    end

    # The optimization converges to the same cost
    result_s = NLLSsolver.optimize!(deepcopy(problem_s), NLLSOptions(numthreads=static(1)))
    result_d = NLLSsolver.optimize!(deepcopy(problem_d), NLLSOptions(numthreads=static(1)))
    @test result_d.bestcost ≈ result_s.bestcost
end
