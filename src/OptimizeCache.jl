# OptimizeCache: storage for an optimization's internal data (its linear system and iterator state), for reuse
# by later optimize! calls on the same problem. See the OptimizeCache docstring.

"""
    NLLSsolver.OptimizeCache()

Storage for the internal data of an optimization (its linear system and iterator state), for reuse by later
calls of `optimize!` on the same problem:
```julia
cache = NLLSsolver.OptimizeCache()
for data in datasets
    # ... update the problem's variables and the costs' measurements, in place ...
    optimize!(problem, options, unfixed, callback, cache)
end
```
The first call constructs the internal data, as `optimize!` without a cache does, and stores it; later calls
reuse it while the problem's structure is unchanged, so they skip constructing the linear system (for a sparse
or Schur-complement system, its sparsity analysis) and allocate nothing for it. Results are identical to those
of `optimize!` without a cache.

The data is reconstructed when the call differs in its problem, unfixed variables, iterator or number of
threads, or when variables or costs have been added to the problem (with `addvariable!` or `addcost!`). After
changing the problem in any other way that alters its structure (e.g. editing `problem.costs` directly, or
changing the size of a dynamically sized variable), call `empty!(cache)`.
"""
mutable struct OptimizeCache{K, D}
    key::Union{Nothing, K}
    data::Union{Nothing, D}
end
OptimizeCache() = OptimizeCache{Any, Any}(nothing, nothing)

"""
    NLLSsolver.OptimizeCache(problem::NLLSProblem, options::NLLSOptions, unfixed=nothing)

An [`OptimizeCache`](@ref) built for `problem`, `options` and `unfixed`: the internal data are constructed now
(nothing is optimized), and the cache is concretely typed by them, so a later
`optimize!(problem, options, unfixed, callback, cache)` is type-stable and allocates nothing for the cache
(an `OptimizeCache()` holds its data untyped: each reuse dispatches dynamically and allocates a few small
objects). When the problem's structure changes, the data are rebuilt as for an untyped cache; if the rebuilt
data have another type, that call optimizes without storing them.
"""
function OptimizeCache(problem::NLLSProblem, options::NLLSOptions, unfixed=nothing)
    unfixed = convertunfixed(unfixed, problem)
    data = optimizewith(BuildOnly(), problem, options, unfixed, Stats(), nullcallback)
    key = cachekey(problem, options, unfixed)
    return OptimizeCache{typeof(key), typeof(data)}(key, data)
end
# Construct the internal data, without optimizing
struct BuildOnly end
(::BuildOnly)(problem, options, data, iteratedata, callback) = (data, iteratedata)

Base.empty!(cache::OptimizeCache) = (cache.key = nothing; cache.data = nothing; cache)
Base.isempty(cache::OptimizeCache) = isnothing(cache.data)

# What the internal data depend on: the problem, its unfixed variables and their sizes, the number of variables
# and costs, and the iterator and number of threads (the options' type)
function cachekey(problem::NLLSProblem, options::NLLSOptions, unfixed::Integer)
    return (problem, typeof(options), Int(unfixed), Int(dynamic(nvars(problem.variables[unfixed]))))
end
function cachekey(problem::NLLSProblem, options::NLLSOptions, unfixed::AbstractVector)
    # The costs' structure is also tracked through the validity of the variable cost map: addcost! and
    # addvariable! invalidate it
    getvarcostmap(problem)
    sizes = [unfixed_ ? Int(dynamic(nvars(var))) : 0 for (var, unfixed_) in zip(problem.variables, unfixed)]
    return (problem, typeof(options), BitVector(unfixed), sizes, countcosts(costnum, problem.costs))
end
function cachevalid(cache::OptimizeCache, problem::NLLSProblem, options::NLLSOptions, unfixed::Integer)
    key = cache.key
    return !isnothing(key) && key[1] === problem && key[2] === typeof(options) && key[3] == unfixed && key[4] == dynamic(nvars(problem.variables[unfixed]))
end
function cachevalid(cache::OptimizeCache, problem::NLLSProblem, options::NLLSOptions, unfixed::AbstractVector)
    key = cache.key
    (!isnothing(key) && key[1] === problem && key[2] === typeof(options) && problem.varcostmapvalid) || return false
    (key[3]::BitVector == unfixed && key[5]::Int == countcosts(costnum, problem.costs)) || return false
    return samesizes(key[4]::Vector{Int}, problem.variables, unfixed)
end
function samesizes(sizes::Vector{Int}, variables::Vector, unfixed)
    for (i, var) in enumerate(variables)
        if @inbounds(unfixed[i]) && @inbounds(sizes[i]) != dynamic(nvars(var))
            return false
        end
    end
    return true
end

# Store the internal data in the cache, then optimize
struct StoreInCache{C <: OptimizeCache}
    cache::C
end
function (store::StoreInCache)(problem, options, data, iteratedata, callback)
    storedata!(store.cache, (data, iteratedata))
    return optimizeinternal!(problem, options, data, iteratedata, callback)
end
# A typed cache keeps only data of its type
storedata!(cache::OptimizeCache{K, D}, d) where {K, D} = (d isa D && (cache.data = d); cache)
storekey!(cache::OptimizeCache{K, D}, key) where {K, D} = (!isnothing(cache.data) && key isa K && (cache.key = key); cache)

