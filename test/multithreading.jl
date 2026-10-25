using Test, Static, StaticArrays, SparseArrays, LinearAlgebra, NLLSsolver

myfun(x) = x * x * x
NLLSsolver.computecost(::StaticInt, ::AbstractVector, x::Float64) = myfun(x)
NLLSsolver.computecost(::AbstractVector, x::Float64) = myfun(x)

struct MTTestScalarRes <: NLLSsolver.AbstractResidual
    val::Float64
    ind::Int
end
NLLSsolver.ndeps(::MTTestScalarRes) = static(1)
NLLSsolver.nres(::MTTestScalarRes) = 1
NLLSsolver.varindices(res::MTTestScalarRes) = res.ind
NLLSsolver.getvars(res::MTTestScalarRes, vars::AbstractVector) = (vars[res.ind]::Float64,)
NLLSsolver.computeresidual(res::MTTestScalarRes, x) = x - res.val
Base.eltype(::MTTestScalarRes) = Float64

struct MTTestPairRes <: NLLSsolver.AbstractResidual
    val::Float64
    i::Int
    j::Int
end
NLLSsolver.ndeps(::MTTestPairRes) = static(2)
NLLSsolver.nres(::MTTestPairRes) = static(1)
NLLSsolver.varindices(res::MTTestPairRes) = SVector(res.i, res.j)
NLLSsolver.getvars(res::MTTestPairRes, vars::AbstractVector) = (vars[res.i]::Float64, vars[res.j]::Float64)
NLLSsolver.computeresidual(res::MTTestPairRes, x, y) = SVector(x - y - res.val)
Base.eltype(::MTTestPairRes) = Float64

struct MTTestDynamicRes <: NLLSsolver.AbstractResidual
    y::Float64
    X::Vector{Float64}
end
NLLSsolver.ndeps(::MTTestDynamicRes) = static(1)
NLLSsolver.nres(::MTTestDynamicRes) = static(1)
NLLSsolver.varindices(::MTTestDynamicRes) = 1
NLLSsolver.getvars(::MTTestDynamicRes, vars::AbstractVector) = (vars[1]::NLLSsolver.DynamicVector{Float64},)
NLLSsolver.computeresidual(res::MTTestDynamicRes, w) = dot(res.X, w) - res.y
Base.eltype(::MTTestDynamicRes) = Float64

@testset "multithreading.jl" begin
    # Cost sums
    v = rand(Int(1e6))
    s1 = sum(myfun, v)
    s2 = sum(NLLSsolver.bindleadingargs(computecost, static(1), Vector{Int}()), v)
    @test isapprox(s1, s2; rtol=1.e-13)
    s3 = sum(NLLSsolver.bindleadingargs(computecost, static(Threads.nthreads()), Vector{Int}()), v)
    @test isapprox(s1, s3; rtol=1.e-13)

    # Multi-threaded costgradhess! vs single-threaded
    # 1. UniVariateLSstatic
    prob_static = NLLSProblem(Float64)
    addvariable!(prob_static, 2.0)
    for k in 1:120
        addcost!(prob_static, MTTestScalarRes(k * 0.1, 1))
    end
    ls_static_single = NLLSsolver.LinearSystem{NLLSsolver.UniVariateLSstatic_x{1}, NLLSsolver.UniVariateLSstatic_ls{1, 1}}((1,), ())
    ls_static_shared = NLLSsolver.sharedlinearsystem(ls_static_single, static(4))
    c_s = NLLSsolver.costgradhess!(ls_static_single, prob_static.variables, prob_static.costs)
    c_m = NLLSsolver.costgradhess!(ls_static_shared, prob_static.variables, prob_static.costs)
    @test c_s ≈ c_m
    @test NLLSsolver.getgrad(ls_static_single) ≈ NLLSsolver.getgrad(ls_static_shared)
    @test NLLSsolver.gethessian(ls_static_single) ≈ NLLSsolver.gethessian(ls_static_shared)
    c_cost_s = NLLSsolver.cost(prob_static.variables, prob_static.costs, static(1))
    c_cost_m = NLLSsolver.cost(prob_static.variables, prob_static.costs, static(4))
    @test c_cost_s ≈ c_cost_m
    @test c_cost_s ≈ c_s

    # Verify ephemeral stack wrapper construction has 0 allocations
    ls_ephemeral = NLLSsolver.LinearSystem(ls_static_shared, 1)
    @test isa(ls_ephemeral, NLLSsolver.LinearSystem)
    allocs = @allocated NLLSsolver.LinearSystem(ls_static_shared, 1)
    @test allocs == 0

    # 2. UniVariateLSdynamic
    dim = 20
    prob_dyn = NLLSProblem(NLLSsolver.DynamicVector{Float64})
    addvariable!(prob_dyn, zeros(dim))
    for k in 1:120
        addcost!(prob_dyn, MTTestDynamicRes(1.0, randn(dim)))
    end
    ls_dyn_single = NLLSsolver.LinearSystem{NLLSsolver.UniVariateLSdynamic_x, NLLSsolver.UniVariateLSdynamic_ls}((1, dim), (dim,))
    ls_dyn_shared = NLLSsolver.sharedlinearsystem(ls_dyn_single, static(4))
    c_ds = NLLSsolver.costgradhess!(ls_dyn_single, prob_dyn.variables, prob_dyn.costs)
    c_dm = NLLSsolver.costgradhess!(ls_dyn_shared, prob_dyn.variables, prob_dyn.costs)
    @test c_ds ≈ c_dm
    @test NLLSsolver.getgrad(ls_dyn_single) ≈ NLLSsolver.getgrad(ls_dyn_shared)
    @test NLLSsolver.gethessian(ls_dyn_single) ≈ NLLSsolver.gethessian(ls_dyn_shared)

    # 3. MultiVariateLSdense
    prob_dense = NLLSProblem(Float64)
    addvariable!(prob_dense, 1.0)
    addvariable!(prob_dense, 2.0)
    for k in 1:120
        addcost!(prob_dense, MTTestPairRes(0.5, 1, 2))
        addcost!(prob_dense, MTTestScalarRes(1.0, 1))
    end
    blocksizes, blockindices = NLLSsolver.block_sizes_indices(prob_dense.variables, trues(2), 2)
    ls_dense_single = NLLSsolver.MultiVariateLSdense(blocksizes, blockindices)
    ls_dense_shared = NLLSsolver.sharedlinearsystem(ls_dense_single, static(4))
    c_dens_s = NLLSsolver.costgradhess!(ls_dense_single, prob_dense.variables, prob_dense.costs)
    c_dens_m = NLLSsolver.costgradhess!(ls_dense_shared, prob_dense.variables, prob_dense.costs)
    @test c_dens_s ≈ c_dens_m
    @test NLLSsolver.getgrad(ls_dense_single) ≈ NLLSsolver.getgrad(ls_dense_shared)
    @test NLLSsolver.gethessian(ls_dense_single) ≈ NLLSsolver.gethessian(ls_dense_shared)

    # 4. MultiVariateLSsparse
    prob_sparse = NLLSProblem(Float64)
    nvars = 40
    for i in 1:nvars
        addvariable!(prob_sparse, Float64(i))
    end
    for i in 1:nvars-1
        addcost!(prob_sparse, MTTestPairRes(1.0, i, i+1))
    end
    b_sizes, b_indices = NLLSsolver.block_sizes_indices(prob_sparse.variables, trues(nvars), nvars)
    sparsity = NLLSsolver.block_sparsity(prob_sparse, trues(nvars))
    bsm = NLLSsolver.BlockSparseMatrix{Float64}(sparsity, b_sizes, b_sizes)
    ls_sparse_single = NLLSsolver.MultiVariateLSsparse(bsm, b_indices, true)
    ls_sparse_shared = NLLSsolver.sharedlinearsystem(ls_sparse_single, static(4))
    c_sp_s = NLLSsolver.costgradhess!(ls_sparse_single, prob_sparse.variables, prob_sparse.costs)
    c_sp_m = NLLSsolver.costgradhess!(ls_sparse_shared, prob_sparse.variables, prob_sparse.costs)
    @test c_sp_s ≈ c_sp_m
    @test NLLSsolver.getgrad(ls_sparse_single) ≈ NLLSsolver.getgrad(ls_sparse_shared)
    @test NLLSsolver.gethessian(ls_sparse_single) ≈ NLLSsolver.gethessian(ls_sparse_shared)

    # Full optimize! end-to-end tests
    opt_single = NLLSOptions(; numthreads=static(1))
    opt_multi = NLLSOptions(; numthreads=static(4))

    # Static univariate optimization equivalence
    res_static_s = optimize!(deepcopy(prob_static), opt_single)
    res_static_m = optimize!(deepcopy(prob_static), opt_multi)
    @test res_static_s.bestcost ≈ res_static_m.bestcost

    # Dynamic univariate optimization equivalence
    res_dyn_s = optimize!(deepcopy(prob_dyn), opt_single)
    res_dyn_m = optimize!(deepcopy(prob_dyn), opt_multi)
    @test res_dyn_s.bestcost ≈ res_dyn_m.bestcost

    # Dense multivariate optimization equivalence
    res_dense_s = optimize!(deepcopy(prob_dense), opt_single)
    res_dense_m = optimize!(deepcopy(prob_dense), opt_multi)
    @test res_dense_s.bestcost ≈ res_dense_m.bestcost

    # Sparse multivariate optimization equivalence
    res_sp_s = optimize!(deepcopy(prob_sparse), opt_single)
    res_sp_m = optimize!(deepcopy(prob_sparse), opt_multi)
    @test res_sp_s.bestcost ≈ res_sp_m.bestcost

    # Threshold-based decision test (< 1e4 stays single-threaded, >= 1e4 creates LinearSystemShared)
    @test isa(NLLSsolver.sharedlinearsystem(ls_static_single, prob_static, opt_multi), NLLSsolver.LinearSystem)
    prob_large = NLLSProblem(Float64)
    addvariable!(prob_large, 2.0)
    for k in 1:10000
        addcost!(prob_large, MTTestScalarRes(k * 0.001, 1))
    end
    ls_large_single = NLLSsolver.LinearSystem{NLLSsolver.UniVariateLSstatic_x{1}, NLLSsolver.UniVariateLSstatic_ls{1, 1}}((1,), ())
    @test isa(NLLSsolver.sharedlinearsystem(ls_large_single, prob_large, opt_multi), NLLSsolver.LinearSystemShared)
    res_large_s = optimize!(deepcopy(prob_large), opt_single)
    res_large_m = optimize!(deepcopy(prob_large), opt_multi)
    @test res_large_s.bestcost ≈ res_large_m.bestcost
end
