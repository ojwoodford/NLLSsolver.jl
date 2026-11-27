using Test, StaticArrays, Static, NLLSsolver, Random, LinearAlgebra
import SecondOrderOptimization

# A small bundle-adjustment problem with eliminable landmarks, as in optimizeba.jl, used to check that every
# feature of the new optimizer reaches it through NLLSsolver's positional optimize! plus keywords, and gives
# the same result as the same call made directly on NLLSModel + SecondOrderOptimization.
struct BALandmark{T}
    p::SVector{3, T}
end
NLLSsolver.OptimizationVariableInterface.nvars(::Type{<:BALandmark}) = static(3)
NLLSsolver.OptimizationVariableInterface.nvars(::BALandmark) = static(3)
NLLSsolver.OptimizationVariableInterface.update(v::BALandmark, step, offset) = BALandmark(v.p + SVector(step[offset], step[offset+1], step[offset+2]))
NLLSsolver.OptimizationVariableInterface.iseliminable(::Type{<:BALandmark}) = static(true)
NLLSsolver.generatemeasurement(pose::EuclideanVector{6, T}, X::BALandmark{U}) where {T, U} = SVector(dot(view(pose, SVector(1,2,3)), X.p), dot(view(pose, SVector(4,5,6)), X.p))

function makeba(seed=1)
    Random.seed!(seed)
    ncameras, nlandmarks, propvisible = 4, 8, 1.0
    problem = NLLSProblem(Union{EuclideanVector{6, Float64}, BALandmark{Float64}}, SimpleError2{2, Float64, EuclideanVector{6, Float64}, BALandmark{Float64}})
    camoffset = SVector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    for i = 1:ncameras; addvariable!(problem, randn(EuclideanVector{6, Float64}) .+ camoffset); end
    lmoffset = SVector(-0.5, -0.5, 10.0)
    for i = 1:nlandmarks; addvariable!(problem, BALandmark(rand(SVector{3, Float64}) .+ lmoffset)); end
    for camind = 1:ncameras
        camera = problem.variables[camind]::EuclideanVector{6, Float64}
        for landmark = 1:nlandmarks
            landmarkind = landmark + ncameras
            addcost!(problem, SimpleError2{EuclideanVector{6, Float64}, BALandmark{Float64}}(generatemeasurement(camera, problem.variables[landmarkind]::BALandmark{Float64}), camind, landmarkind))
        end
    end
    # perturb
    for ind in 1:lastindex(problem.variables)
        if problem.variables[ind] isa BALandmark{Float64}
            problem.variables[ind] = BALandmark(problem.variables[ind].p + randn(SVector{3, Float64}) * 0.001)
        else
            problem.variables[ind] = problem.variables[ind]::EuclideanVector{6, Float64} + randn(SVector{6, Float64}) * 0.001
        end
    end
    return problem
end

# Compare a wrapper call (positional form + keywords) against the same call made directly on the new packages.
function samecall(wrap, direct; rtol=1e-10)
    p1 = makeba(); r1 = wrap(p1)
    p2 = makeba(); r2 = direct(p2)
    @test isapprox(r1.bestcost, r2.bestcost; rtol, atol=1e-18)
    @test isapprox(Float64.(reduce(vcat, [v isa BALandmark ? v.p : v for v in p1.variables])),
                   Float64.(reduce(vcat, [v isa BALandmark ? v.p : v for v in p2.variables])); rtol, atol=1e-12)
    return r1
end

@testset "passthrough.jl" begin
    # reduction = :eliminate, storage = :dense and :sparse
    for storage in (:dense, :sparse)
        samecall(p -> optimize!(p, NLLSOptions(), nothing; reduction=:eliminate, storage=storage),
                 p -> SecondOrderOptimization.optimize!(p, NLLSOptions(); reduction=:eliminate, storage=storage))
    end
    # formulation = :jacobian
    samecall(p -> optimize!(p, NLLSOptions(), nothing; formulation=:jacobian, reduction=:none),
             p -> SecondOrderOptimization.optimize!(p, NLLSOptions(); formulation=:jacobian, reduction=:none))
    # precision = Float32 (lower accuracy, so a looser tolerance)
    samecall(p -> optimize!(p, NLLSOptions(), nothing; precision=Float32, reduction=:none),
             p -> SecondOrderOptimization.optimize!(p, NLLSOptions(); precision=Float32, reduction=:none); rtol=1e-3)
    # numthreads = 1
    samecall(p -> optimize!(p, NLLSOptions(), nothing; numthreads=1, reduction=:none),
             p -> SecondOrderOptimization.optimize!(p, NLLSOptions(); numthreads=1, reduction=:none))
    # NLLSOptions(linsolver = :ldl)
    samecall(p -> optimize!(p, NLLSOptions(linsolver=:ldl), nothing; reduction=:none),
             p -> SecondOrderOptimization.optimize!(p, NLLSOptions(linsolver=:ldl); reduction=:none))
    # A recommendation, made and run through NLLSsolver
    samecall(p -> optimize!(p, recommendoptimizer(p)),
             p -> SecondOrderOptimization.optimize!(p, recommendoptimizer(p)))

    # Giving unfixed both positionally and by keyword is an error
    @test_throws ArgumentError optimize!(makeba(), NLLSOptions(), 2; unfixed=3)
end
