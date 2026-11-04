using NLLSsolver, Test, Static, StaticArrays, SparseArrays, Random, LinearAlgebra

# Points in two flavours: eliminable (Schur) and plain (dense/sparse), otherwise identical
struct SchurTestPoint{N, T}
    x::SVector{N, T}
end
struct PlainTestPoint{N, T}
    x::SVector{N, T}
end
const TestPoints = Union{SchurTestPoint, PlainTestPoint}
NLLSsolver.nvars(::SchurTestPoint{N}) where N = static(N)
NLLSsolver.nvars(::PlainTestPoint{N}) where N = static(N)
NLLSsolver.update(p::SchurTestPoint{N}, δ, start=1) where N = SchurTestPoint(p.x + view(δ, NLLSsolver.SR(0, N-1) .+ start))
NLLSsolver.update(p::PlainTestPoint{N}, δ, start=1) where N = PlainTestPoint(p.x + view(δ, NLLSsolver.SR(0, N-1) .+ start))
NLLSsolver.iseliminable(::Type{<:SchurTestPoint}) = static(true)

# An affine camera (6 parameters) observing a 3D point, or a 2D point through a 2D affinity
schurtestproject(cam, X::SVector{3}) = SVector(dot(view(cam, NLLSsolver.SR(1, 3)), X), dot(view(cam, NLLSsolver.SR(4, 6)), X))
schurtestproject(cam, X::SVector{2}) = SVector(cam[1] * X[1] + cam[2] * X[2] + cam[3], cam[4] * X[1] + cam[5] * X[2] + cam[6])

struct SchurTestProjection{P} <: NLLSsolver.AbstractResidual
    measurement::SVector{2, Float64}
    varind::SVector{2, Int}
end
NLLSsolver.ndeps(::SchurTestProjection) = static(2)
NLLSsolver.nres(::SchurTestProjection) = static(2)
NLLSsolver.varindices(res::SchurTestProjection) = res.varind
NLLSsolver.getvars(res::SchurTestProjection{P}, vars::AbstractVector) where P = (vars[res.varind[1]]::EuclideanVector{6, Float64}, vars[res.varind[2]]::P)
NLLSsolver.computeresidual(res::SchurTestProjection, cam, X) = schurtestproject(cam, X.x) - res.measurement
Base.eltype(::SchurTestProjection) = Float64

# A prior coupling two cameras
struct SchurTestCameraPair <: NLLSsolver.AbstractResidual
    measurement::SVector{6, Float64}
    varind::SVector{2, Int}
end
NLLSsolver.ndeps(::SchurTestCameraPair) = static(2)
NLLSsolver.nres(::SchurTestCameraPair) = static(6)
NLLSsolver.varindices(res::SchurTestCameraPair) = res.varind
NLLSsolver.getvars(res::SchurTestCameraPair, vars::AbstractVector) = (vars[res.varind[1]]::EuclideanVector{6, Float64}, vars[res.varind[2]]::EuclideanVector{6, Float64})
NLLSsolver.computeresidual(res::SchurTestCameraPair, c1, c2) = c1 - c2 - res.measurement
Base.eltype(::SchurTestCameraPair) = Float64

# A cost coupling two points (refused by the Schur system)
struct SchurTestPointPair <: NLLSsolver.AbstractResidual
    varind::SVector{2, Int}
end
NLLSsolver.ndeps(::SchurTestPointPair) = static(2)
NLLSsolver.nres(::SchurTestPointPair) = static(3)
NLLSsolver.varindices(res::SchurTestPointPair) = res.varind
NLLSsolver.getvars(res::SchurTestPointPair, vars::AbstractVector) = (vars[res.varind[1]]::SchurTestPoint{3, Float64}, vars[res.varind[2]]::SchurTestPoint{3, Float64})
NLLSsolver.computeresidual(res::SchurTestPointPair, a, b) = a.x - b.x
Base.eltype(::SchurTestPointPair) = Float64

# A bundle-shaped problem: ncams affine cameras, n3 3D and n2 2D points, each point seen by every camera,
# measurements with noise, the start perturbed. PT is the point flavour.
function schurtestproblem(PT, ncams, n3, n2; noise=0.001, perturb=0.03, camerapairs=false, rng=MersenneTwister(1))
    VT = n2 == 0 ? Union{EuclideanVector{6, Float64}, PT{3, Float64}} : Union{EuclideanVector{6, Float64}, PT{3, Float64}, PT{2, Float64}}
    CT = n2 == 0 ? SchurTestProjection{PT{3, Float64}} : Union{SchurTestProjection{PT{3, Float64}}, SchurTestProjection{PT{2, Float64}}}
    problem = camerapairs ? NLLSProblem(Any, Any) : NLLSProblem(VT, CT)
    cams = [randn(rng, EuclideanVector{6, Float64}) for _ in 1:ncams]
    for c in cams
        addvariable!(problem, c)
    end
    for (n, N) in ((n3, 3), (n2, 2))
        for _ in 1:n
            X = randn(rng, SVector{N, Float64})
            ind = addvariable!(problem, PT(X + perturb * randn(rng, SVector{N, Float64})))
            for (c, cam) in enumerate(cams)
                addcost!(problem, SchurTestProjection{PT{N, Float64}}(schurtestproject(cam, X) + noise * randn(rng, SVector{2, Float64}), SVector(c, ind)))
            end
        end
    end
    if camerapairs
        for c in 4:ncams # Between unfixed cameras (the first two are fixed)
            addcost!(problem, SchurTestCameraPair(cams[c-1] - cams[c] + noise * randn(rng, SVector{6, Float64}), SVector(c - 1, c)))
        end
    end
    for c in 1:ncams
        problem.variables[c] = cams[c] + perturb * randn(rng, SVector{6, Float64})
    end
    return problem
end

# The Schur system's dense Hessian and gradient, permuted to the order of the variables
function schurtodense(ls, problem, unfixed)
    A = Matrix(NLLSsolver.getA(ls))
    b = NLLSsolver.getb(ls)
    perm = Int[]
    for (i, v) in enumerate(problem.variables)
        unfixed[i] || continue
        o = Int(ls.x.boffsets[ls.x.blockindices[i]])
        append!(perm, o:o+Int(dynamic(nvars(v)))-1)
    end
    return A[perm, perm], b[perm], perm
end

@testset "schur.jl" begin
    @test NLLSsolver.iseliminable(EuclideanVector{3, Float64}) === static(false)
    @test NLLSsolver.iseliminable(SchurTestPoint(SVector(1.0, 2.0, 3.0))) === static(true)

    @testset "Schur solve equals dense solve: ncams=$ncams, n3=$n3, n2=$n2" for (ncams, n3, n2) in ((1, 5, 0), (3, 10, 4), (5, 40, 7), (8, 200, 30))
        # Two more cameras, fixed (which fixes the affine gauge)
        problem = schurtestproblem(SchurTestPoint, ncams + 2, n3, n2; camerapairs=true)
        unfixed = trues(length(problem.variables))
        unfixed[1:2] .= false
        @test NLLSsolver.useschur(problem.variables, unfixed)
        ls = NLLSsolver.MultiVariateLSschur(problem, unfixed)
        cost = NLLSsolver.costgradhess!(ls, problem.variables, problem.costs)
        A = NLLSsolver.getA(ls)
        b = copy(NLLSsolver.getb(ls))

        # The assembled Hessian and gradient equal the dense linear system's
        dense = NLLSsolver.MultiVariateLSdense(NLLSsolver.block_sizes_indices(problem.variables, unfixed, sum(unfixed))...)
        @test NLLSsolver.costgradhess!(dense, problem.variables, problem.costs) ≈ cost
        Hd = Symmetric(NLLSsolver.gethessian(dense), :L)
        Hs, bs, perm = schurtodense(ls, problem, unfixed)
        @test Hs ≈ Hd rtol=1e-14
        @test bs ≈ NLLSsolver.getb(dense) rtol=1e-14
        @test size(A) == size(Hs)

        # Undamped, then damped with each LM iterator's damping
        x = randn(MersenneTwister(2), length(b))
        @test NLLSsolver.fast_bAb(A, x) ≈ dot(x, Matrix(A) * x) rtol=1e-12
        for damping in (nothing, 1.0, :marquardt)
            if damping isa Real
                NLLSsolver.dampenhessian!(A, NLLSsolver.LevenbergData(damping), damping)
            elseif damping === :marquardt
                lmd = NLLSsolver.MarquardtData(0.0, zeros(length(b)))
                NLLSsolver.updatescaling!(lmd, A)
                lmd.lambda = 0.5
                @test lmd.scaling ≈ max.(abs.(diag(Matrix(A))), 1e-8 * maximum(abs, diag(Matrix(A))))
                NLLSsolver.dampenhessian!(A, lmd, lmd.lambda)
            end
            M = Matrix(A)
            xdense = cholesky(Symmetric(M)) \ b
            xschur = copy(NLLSsolver.solve!(ls))
            @test norm(xschur - xdense) <= 1e-12 * norm(xdense)
            @test norm(M * xschur - b) <= 1e-12 * norm(b) * cond(M)
            # The reduced system is the Schur complement of the retained block
            nc = ls.x.structure.nc
            Sref = M[1:nc, 1:nc] - M[1:nc, nc+1:end] * (M[nc+1:end, nc+1:end] \ M[nc+1:end, 1:nc])
            @test NLLSsolver.schurcomplement!(ls) ≈ Sref rtol=1e-10
        end

        # The solve allocates nothing
        NLLSsolver.solve!(ls)
        @test (@allocated NLLSsolver.solve!(ls)) == 0

        # The shared (multi-threaded) system reduces to the same Hessian
        shared = NLLSsolver.sharedlinearsystem(NLLSsolver.MultiVariateLSschur(problem, unfixed), static(2))
        @test NLLSsolver.costgradhess!(shared, problem.variables, problem.costs) ≈ cost
        @test Matrix(NLLSsolver.getA(shared)) ≈ Hs[invperm(perm), invperm(perm)]
    end

    @testset "Every iterator converges to the dense/sparse optimum: $iterator" for iterator in (:newton, :levenbergmarquardt, :levenbergmarquardt_dtd, :dogleg)
        # callbackchangesstate selects the Levenberg-Marquardt iterators without (true) and with (false) trial-point reuse
        for (n3, n2) in ((6, 0), (60, 9)), changesstate in (false, true)
            # Four cameras, the first two fixed (which fixes the affine gauge)
            schur = schurtestproblem(SchurTestPoint, 4, n3, n2)
            plain = schurtestproblem(PlainTestPoint, 4, n3, n2)
            unfixed = trues(length(schur.variables))
            unfixed[1:2] .= false
            options = NLLSOptions(iterator=iterator, numthreads=static(1), callbackchangesstate=changesstate)
            rs = optimize!(schur, options, unfixed)
            rp = optimize!(plain, options, unfixed)
            @test rs.startcost ≈ rp.startcost
            @test rs.bestcost < 0.5 * rs.startcost
            @test rs.bestcost ≈ rp.bestcost rtol=1e-9
            for (vs, vp) in zip(schur.variables, plain.variables)
                xs = vs isa TestPoints ? vs.x : vs
                xp = vp isa TestPoints ? vp.x : vp
                @test xs ≈ xp rtol=1e-6
            end
            # The Schur system was the one used
            @test NLLSsolver.useschur(schur.variables, unfixed)
            @test !NLLSsolver.useschur(plain.variables, unfixed)
        end
    end

    @testset "Optimization allocates nothing after setup" begin
        problem = schurtestproblem(SchurTestPoint, 4, 30, 0)
        unfixed = trues(length(problem.variables))
        unfixed[1:2] .= false
        options = NLLSOptions(iterator=:levenbergmarquardt_dtd, numthreads=static(1))
        optimize!(deepcopy(problem), options, unfixed)
        result = optimize!(problem, options, unfixed)
        @test result.bestcost < result.startcost
        @test result.optimizeallocated == 0
    end

    @testset "A cost coupling two eliminable variables is refused" begin
        problem = schurtestproblem(SchurTestPoint, 2, 4, 0; camerapairs=true)
        addcost!(problem, SchurTestPointPair(SVector(3, 4)))
        @test_throws ArgumentError optimize!(problem, NLLSOptions(numthreads=static(1)))
        # ... but not when one of the two is fixed
        unfixed = trues(length(problem.variables))
        unfixed[4] = false
        @test optimize!(problem, NLLSOptions(numthreads=static(1)), unfixed).bestcost >= 0
    end
end
