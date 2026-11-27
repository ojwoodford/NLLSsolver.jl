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

@testset "OptimizeCache.jl" begin
    # The cached call gives the same result as an uncached one (D5: OptimizeCache returns a ModelWorkspace,
    # which reuses the model's setup as well as the optimizer's state, so a repeated call is allocation-free).
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

    # The setup (`cache.setup`) is reused while the model's structure and the keywords are unchanged, and
    # rebuilt when the unfixed set, a variable count or a sparse pattern changes. A ModelWorkspace keys on the
    # model, its variable vector, the structure and the keywords — not the options — so a cost's data can
    # change between calls and the setup is still reused.
    options = NLLSOptions(iterator=:levenbergmarquardt_dtd)
    problem = chainproblem(20)
    cache = OptimizeCache()
    unfixed = trues(20)
    optimize!(problem, options, unfixed, nullcallback, cache)
    setup = cache.setup
    problem.variables .= startvalues(20, 2)
    optimize!(problem, options, unfixed, nullcallback, cache)
    @test cache.setup === setup   # same structure and keywords: reused
    unfixed[3] = false
    optimize!(problem, options, unfixed, nullcallback, cache)
    @test cache.setup !== setup   # unfixed set changed: rebuilt
    setup = cache.setup
    addcost!(problem, ChainLink(SVector(2.0, 0.0, 0.0), SVector(1, 3)))
    reference = deepcopy(problem)
    result = optimize!(problem, options, unfixed, nullcallback, cache)
    @test cache.setup !== setup   # a cost was added (structure changed): rebuilt
    @test result.bestcost == optimize!(reference, options, unfixed).bestcost
    addvariable!(problem, SVector(0.0, 0.0, 0.0))
    setup = cache.setup
    optimize!(problem, options, nothing, nullcallback, cache)
    @test cache.setup !== setup   # a variable was added: rebuilt
    # The cache is populated before the optimization loop runs
    optimize!(problem, options, nothing, (cost, unusedargs...) -> (@test !isempty(cache); (cost, 0)), cache)
    empty!(cache)
    @test isempty(cache)
end

# A cache built for its problem (`OptimizeCache(problem, options, unfixed)`) is concretely typed: the same
# results as an untyped cache and as no cache, and a repeated call reuses the model's linearization setup as
# well as the optimizer's state (D5), so it allocates next to nothing. The Offset3 problem's variable is
# static (a `Vector` variable would allocate in its own `update`).
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
# Single-threaded, so the measurement sees no per-call task overhead (as NLLSModel's own allocation test is):
# multithreading a single static variable allocates its tasks each call, which is orthogonal to the cache.
typedrun!(problem, options, cache) = (problem.variables[1] = SVector(0.0, 0.0, 0.0); optimize!(problem, options, 1, nullcallback, cache; numthreads=static(1)))

@testset "OptimizeCache.jl: a typed cache" begin
    options = NLLSOptions(iterator=:levenbergmarquardt_dtd)
    @test isbitstype(typeof(options))
    @test !isempty(OptimizeCache(offsetproblem()))
    for (makeproblem, unfixed) in ((offsetproblem, 1), (() -> chainproblem(20), nothing))
        reference = makeproblem()
        refresult = optimize!(reference, options, unfixed)
        problem = makeproblem()
        cache = OptimizeCache(problem, options, unfixed)
        @test !isempty(cache) && isconcretetype(typeof(cache).parameters[2])
        result = optimize!(problem, options, unfixed, nullcallback, cache)
        @test problem.variables == reference.variables && result.bestcost == refresult.bestcost
        @test result.niterations == refresult.niterations
    end
    # A repeated typed-cache call reuses the setup rather than rebuilding it (D5): it allocates ~0 B on release
    # and ~864 B on 1.10 (the deprecated positional wrapper's keyword forwarding), well below a rebuilt call
    # (≥ 2256 B). Measured in bytes, which is steadier across Julia versions than the object count.
    problem = offsetproblem()
    cache = OptimizeCache(problem, options, 1; numthreads=static(1))
    typedrun!(problem, options, cache); typedrun!(problem, options, cache)
    @test (@allocated typedrun!(problem, options, cache)) < 1600
end
