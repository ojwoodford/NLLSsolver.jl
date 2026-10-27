using Test, StaticArrays, Static, NLLSsolver

# A chain of 3D points, each tied to the next by a measured offset and the first to the origin: 20 points,
# 60 unknowns (a sparse linear system), or 2 (a dense one)
struct ChainLink <: AbstractResidual
    offset::SVector{3, Float64}
    varind::SVector{2, Int}
end
NLLSsolver.ndeps(::ChainLink) = static(2)
NLLSsolver.nres(::ChainLink) = static(3)
NLLSsolver.varindices(r::ChainLink) = r.varind
NLLSsolver.getvars(r::ChainLink, vars::Vector) = (vars[r.varind[1]]::SVector{3, Float64}, vars[r.varind[2]]::SVector{3, Float64})
NLLSsolver.computeresidual(r::ChainLink, a, b) = b - a - r.offset + 0.1 * sin.(a)
Base.eltype(::ChainLink) = Float64
struct ChainAnchor <: AbstractResidual
end
NLLSsolver.ndeps(::ChainAnchor) = static(1)
NLLSsolver.nres(::ChainAnchor) = static(3)
NLLSsolver.varindices(::ChainAnchor) = SVector(1)
NLLSsolver.getvars(::ChainAnchor, vars::Vector) = (vars[1]::SVector{3, Float64},)
NLLSsolver.computeresidual(::ChainAnchor, a) = a
Base.eltype(::ChainAnchor) = Float64

function chainproblem(n, seed=1)
    problem = NLLSProblem(SVector{3, Float64}, Union{ChainLink, ChainAnchor})
    for i in 1:n
        addvariable!(problem, SVector(0.1 * i, sin(i + seed), cos(i * seed)))
    end
    addcost!(problem, ChainAnchor())
    for i in 1:n-1
        addcost!(problem, ChainLink(SVector(1.0, 0.5 * sin(i * seed), 0.2), SVector(i, i + 1)))
    end
    return problem
end
startvalues(n, seed=1) = [SVector(0.1 * i, sin(i + seed), cos(i * seed)) for i in 1:n]
# For a single unfixed variable, only its start changes (fixed variables keep their values between calls)
startvalues(n, seed, unfixed::Integer) = (v = startvalues(n); v[unfixed] = startvalues(n, seed)[unfixed]; v)
startvalues(n, seed, unfixed) = startvalues(n, seed)

@testset "optimizecache.jl" begin
    for (n, unfixed, iterator) in ((20, nothing, :levenbergmarquardt), (20, nothing, :levenbergmarquardt_dtd), (20, nothing, :dogleg),
                                   (20, nothing, :newton), (2, nothing, :levenbergmarquardt), (20, 7, :levenbergmarquardt_dtd))
        options = NLLSOptions(iterator=iterator)
        # (With one unfixed variable, the subproblem of the costs that depend on it)
        makeproblem() = unfixed isa Integer ? subproblem(chainproblem(n), unfixed) : chainproblem(n)
        problem = makeproblem()
        cache = OptimizeCache()
        @test isempty(cache)
        for seed in 1:3
            # Without the cache, on a fresh problem
            reference = makeproblem()
            reference.variables .= startvalues(n, seed, unfixed)
            refresult = optimize!(reference, options, unfixed)
            # With the cache, the same problem reset to the start
            problem.variables .= startvalues(n, seed, unfixed)
            result = optimize!(problem, options, unfixed, nullcallback, cache)
            @test !isempty(cache)
            @test problem.variables == reference.variables
            @test result.bestcost == refresult.bestcost
            @test result.startcost == refresult.startcost
            @test result.niterations == refresult.niterations
            @test result.termination == refresult.termination
            @test result.costcomputations == refresult.costcomputations
            @test result.gradientcomputations == refresult.gradientcomputations
        end
    end

    # The stored data are reused, and constructed anew when the call or the problem's structure changes
    options = NLLSOptions(iterator=:levenbergmarquardt_dtd)
    problem = chainproblem(20)
    cache = OptimizeCache()
    unfixed = trues(20)
    firstresult = optimize!(problem, options, unfixed, nullcallback, cache)
    data = cache.data[1]
    problem.variables .= startvalues(20, 2)
    result = optimize!(problem, options, unfixed, nullcallback, cache)
    @test cache.data[1] === data
    @test result.initallocations < 10 < firstresult.initallocations # Nothing is constructed for the linear system
    unfixed[3] = false
    optimize!(problem, options, unfixed, nullcallback, cache)
    @test cache.data[1] !== data
    data = cache.data[1]
    optimize!(problem, NLLSOptions(iterator=:levenbergmarquardt), unfixed, nullcallback, cache)
    @test cache.data[1] !== data
    data = cache.data[1]
    addcost!(problem, ChainLink(SVector(2.0, 0.0, 0.0), SVector(1, 3)))
    result = optimize!(problem, NLLSOptions(iterator=:levenbergmarquardt), unfixed, nullcallback, cache)
    @test cache.data[1] !== data
    reference = deepcopy(problem)
    @test optimize!(reference, NLLSOptions(iterator=:levenbergmarquardt), unfixed).bestcost == optimize!(problem, NLLSOptions(iterator=:levenbergmarquardt), unfixed, nullcallback, cache).bestcost
    addvariable!(problem, SVector(0.0, 0.0, 0.0))
    optimize!(problem, options, nothing, nullcallback, cache)
    @test length(cache.data[1].linsystem.x.blockindices) == 21
    empty!(cache)
    @test isempty(cache)
end

# A cache built for its problem (`OptimizeCache(problem, options, unfixed)`) is concretely typed: the same
# results as an untyped cache and as no cache, and a repeated optimization on a one-variable problem with a
# static variable allocates nothing for the cache (an untyped one allocates a few small objects per call)
struct Offset3 <: AbstractResidual
    y::SVector{3, Float64}
end
NLLSsolver.ndeps(::Offset3) = static(1)
NLLSsolver.nres(::Offset3) = static(3)
NLLSsolver.varindices(::Offset3) = SVector(1)
NLLSsolver.getvars(::Offset3, vars::Vector) = (vars[1]::SVector{3, Float64},)
NLLSsolver.computeresidual(r::Offset3, a) = a + 0.1 * sin.(a) - r.y
Base.eltype(::Offset3) = Float64
function offsetproblem()
    problem = NLLSProblem(SVector{3, Float64}, Offset3)
    addvariable!(problem, SVector(0.0, 0.0, 0.0))
    for k in 1:5
        addcost!(problem, Offset3(SVector(1.0 + k / 10, -0.5, 2.0 - k / 7)))
    end
    return problem
end
typedrun!(problem, options, cache) = (problem.variables[1] = SVector(0.0, 0.0, 0.0); optimize!(problem, options, 1, nullcallback, cache))

@testset "optimizecache.jl: a typed cache" begin
    options = NLLSOptions(iterator=:levenbergmarquardt_dtd, numthreads=static(1))
    for (makeproblem, unfixed) in ((offsetproblem, 1), (() -> chainproblem(20), nothing))
        reference = makeproblem()
        refresult = optimize!(reference, options, unfixed)
        problem = makeproblem()
        cache = OptimizeCache(problem, options, unfixed)
        @test !isempty(cache) && isconcretetype(typeof(cache).parameters[2])
        result = optimize!(problem, options, unfixed, nullcallback, cache)
        @test problem.variables == reference.variables && result.bestcost == refresult.bestcost
        @test result.niterations == refresult.niterations
        # a structure change rebuilds the data, as for an untyped cache (the multi-variable problem: adding a
        # variable changes its unfixed set)
        if unfixed === nothing
            data = cache.data[1]
            addvariable!(problem, problem.variables[1])
            optimize!(problem, options, unfixed, nullcallback, cache)
            @test cache.data[1] !== data && length(cache.data[1].linsystem.x.blockindices) == 21
        end
    end
    problem = offsetproblem()
    cache = OptimizeCache(problem, options, 1)
    untyped = OptimizeCache()
    typedrun!(problem, options, cache); typedrun!(problem, options, cache)
    typedrun!(problem, options, untyped); typedrun!(problem, options, untyped)
    @test (@allocations typedrun!(problem, options, cache)) < (@allocations typedrun!(problem, options, untyped))
    @test (@allocations typedrun!(problem, options, cache)) == 0
end
