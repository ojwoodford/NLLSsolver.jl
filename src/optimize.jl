using SecondOrderOptimization, NLLSModel, OptimizationVariableInterface
using Static, StaticArrays, SparseArrays
import NLLSModel: Model, convertunfixed, SingleVarCostFunc, SingleVarCostGradHess, updateunfixed, selectcosts!, getvarcostmap, CostStruct, countcosts, costnum

# Callback wrapper to expose NLLSModel variables and state to callbacks
struct NLLSCallbackWrapper{CB, P, U, UI, O}
    cb::CB
    problem::P
    unfixed::U
    unfixed_inds::UI
    options::O
end
NLLSCallbackWrapper(cb, problem, unfixed=nothing) = NLLSCallbackWrapper(cb, problem, unfixed, nothing, OptimizationOptions())

@inline function (w::NLLSCallbackWrapper)(cost, x_trial, data, iteratedata)
    if w.unfixed isa Integer
        @inbounds w.problem.variables[w.unfixed] = data.variables
        vnext = copy(w.problem.variables)
        @inbounds vnext[w.unfixed] = x_trial
        res = task_local_storage(:nlls_varnext, vnext) do
            task_local_storage(:nlls_varbest, data.varbest) do
                task_local_storage(:nlls_unfixed, w.unfixed) do
                    w.cb(cost, w.problem, data, iteratedata)
                end
            end
        end
        if dynamic(w.options.callbackchangesstate)
            @inbounds for i in eachindex(w.problem.variables, vnext)
                if i != w.unfixed
                    w.problem.variables[i] = vnext[i]
                end
            end
            @inbounds data.varnext = vnext[w.unfixed]
        end
        return res
    elseif w.unfixed_inds isa Vector{Int}
        @inbounds for (k, idx) in enumerate(w.unfixed_inds)
            w.problem.variables[idx] = data.variables[k]
        end
        vnext = copy(w.problem.variables)
        @inbounds for (k, idx) in enumerate(w.unfixed_inds)
            vnext[idx] = x_trial[k]
        end
        vbest = copy(w.problem.variables)
        @inbounds for (k, idx) in enumerate(w.unfixed_inds)
            vbest[idx] = data.varbest[k]
        end
        res = task_local_storage(:nlls_varnext, vnext) do
            task_local_storage(:nlls_varbest, vbest) do
                task_local_storage(:nlls_unfixed, w.unfixed) do
                    w.cb(cost, w.problem, data, iteratedata)
                end
            end
        end
        if dynamic(w.options.callbackchangesstate)
            @inbounds for i in eachindex(w.problem.variables, vnext)
                w.problem.variables[i] = vnext[i]
            end
            @inbounds for (k, idx) in enumerate(w.unfixed_inds)
                x_trial[k] = vnext[idx]
            end
        end
        return res
    else
        w.problem.variables = data.variables
        return task_local_storage(:nlls_varnext, x_trial) do
            task_local_storage(:nlls_varbest, data.varbest) do
                task_local_storage(:nlls_unfixed, w.unfixed) do
                    w.cb(cost, w.problem, data, iteratedata)
                end
            end
        end
    end
end
@inline (w::NLLSCallbackWrapper{typeof(nullcallback), P, U, UI, O})(cost, x_trial, data, iteratedata) where {P, U, UI, O} = (cost, 0)

@inline function update_problem_variables!(problem::Model, unfixed::Integer, ::Any, x_opt)
    @inbounds problem.variables[unfixed] = x_opt
    return problem
end
@inline function update_problem_variables!(problem::Model, ::AbstractVector, ::Nothing, x_opt)
    problem.variables = x_opt
    return problem
end
@inline function update_problem_variables!(problem::Model, ::AbstractVector, unfixed_inds::Vector{Int}, x_opt)
    @inbounds for (k, idx) in enumerate(unfixed_inds)
        problem.variables[idx] = x_opt[k]
    end
    return problem
end

get_unfixed_inds(fdf!::SingleVarCostGradHess) = nothing
get_unfixed_inds(fdf!) = fdf!.unfixed_inds

mutable struct NLLSInternalData{LS, F, FDF, HType, D}
    linsystem::LS
    f::F
    fdf!::FDF
    H::HType
    so_data::D
end

# Public optimize! entry points
@inline optimize!(problem::Model, options::OptimizationOptions=OptimizationOptions(), unfixed=nothing, callback=nullcallback; numthreads=StaticInt(Threads.nthreads())) =
    optimize_dispatch!(problem, options, convertunfixed(unfixed, problem), callback, nothing, SecondOrderOptimization.Stats(), static(numthreads))

@inline optimize!(problem::Model, options::OptimizationOptions, unfixed, callback::CB, workspace::WS; numthreads=StaticInt(Threads.nthreads())) where {CB, WS<:Union{Nothing, OptimizationWorkspace}} =
    optimize_dispatch!(problem, options, convertunfixed(unfixed, problem), callback, workspace, SecondOrderOptimization.Stats(), static(numthreads))

@inline function optimize_cached_dispatch!(problem::Model, options::OptimizationOptions, unfixed, callback, workspace::OptimizationWorkspace, startstats::SecondOrderOptimization.Stats)
    return run_cached_nlls!(something(workspace.data), problem, options, unfixed, callback, startstats)
end

@inline function run_cached_nlls!(cached::Tuple{D, ID}, problem::Model, options::OptimizationOptions, unfixed, callback::CB, startstats::SecondOrderOptimization.Stats) where {D, ID, CB}
    data, so_iteratedata = cached
    unfixed_inds = get_unfixed_inds(data.fdf!)
    if unfixed isa Integer
        x0 = problem.variables[unfixed]
    elseif isnothing(unfixed_inds)
        x0 = problem.variables
    else
        x0 = problem.variables[unfixed_inds]
    end
    cb = callback === nullcallback ? nullcallback : NLLSCallbackWrapper(callback, problem, unfixed, unfixed_inds, options)
    x_opt, res = SecondOrderOptimization.run_cached_optimize!((data.so_data, so_iteratedata), x0, data.f, data.fdf!, options, cb, startstats)
    update_problem_variables!(problem, unfixed, unfixed_inds, x_opt)
    return res
end

@inline function run_nlls_optimvars!(problem::Model, options::OptimizationOptions, unfixed, callback::CB, workspace::WS, startstats::SecondOrderOptimization.Stats, numthreads::StaticInt=StaticInt(Threads.nthreads())) where {CB, WS}
    f, fdf!, x0, H = optimvars(problem, unfixed, numthreads)
    unfixed_inds = get_unfixed_inds(fdf!)
    cb = callback === nullcallback ? nullcallback : NLLSCallbackWrapper(callback, problem, unfixed, unfixed_inds, options)
    if isnothing(workspace)
        x_opt, res = optimize!(f, fdf!, x0, options; H=H, callback=cb, startstats=startstats)
        update_problem_variables!(problem, unfixed, unfixed_inds, x_opt)
        return res
    else
        so_ls = SecondOrderOptimization.makelinearsystem(x0, H, options.linsolver)
        so_data = OptimizationInternal(so_ls, x0, startstats)
        so_iteratedata = SecondOrderOptimization.construct(options.iterator, so_ls, options.callbackchangesstate)
        data = NLLSInternalData(fdf!.linsystem, f, fdf!, H, so_data)
        key = nlls_cachekey(problem, options, unfixed)
        SecondOrderOptimization.storeworkspace!(workspace, key, (data, so_iteratedata))
        x_opt, res = SecondOrderOptimization.run_optimize!(so_data, so_iteratedata, f, fdf!, options, cb, x0)
        update_problem_variables!(problem, unfixed, unfixed_inds, x_opt)
        return res
    end
end

# Single variable optimization
@inline function optimize_dispatch!(problem::Model, options::OptimizationOptions, unfixed::Integer, callback::CB, workspace::WS, startstats::SecondOrderOptimization.Stats=SecondOrderOptimization.Stats(), numthreads::StaticInt=StaticInt(Threads.nthreads())) where {CB, WS}
    if !isnothing(workspace) && !isnothing(workspace.key) && nlls_keyvalid(workspace.key, problem, options, unfixed)
        return optimize_cached_dispatch!(problem, options, unfixed, callback, workspace, startstats)
    end
    return run_nlls_optimvars!(problem, options, unfixed, callback, workspace, startstats, numthreads)
end

# Multi-variable optimization
@inline function optimize_dispatch!(problem::Model, options::OptimizationOptions, unfixed::AbstractVector, callback::CB, workspace::WS, startstats::SecondOrderOptimization.Stats=SecondOrderOptimization.Stats(), numthreads::StaticInt=StaticInt(Threads.nthreads())) where {CB, WS}
    @assert length(problem.variables) > 0
    nblocks = sum(unfixed)
    if nblocks == 1
        ind = findfirst(unfixed)
        return optimize_dispatch!(problem, options, ind, callback, workspace, startstats, numthreads)
    end
    if !isnothing(workspace) && !isnothing(workspace.key) && nlls_keyvalid(workspace.key, problem, options, unfixed)
        return optimize_cached_dispatch!(problem, options, unfixed, callback, workspace, startstats)
    end
    return run_nlls_optimvars!(problem, options, unfixed, callback, workspace, startstats, numthreads)
end

# Cache key and workspace management
function nlls_cachekey(problem::Model, options::OptimizationOptions, unfixed::Integer)
    return (problem, typeof(options), Int(unfixed), dynamic(nvars(problem.variables[unfixed])), countcosts(costnum, problem.costs))
end

function nlls_cachekey(problem::Model, options::OptimizationOptions, unfixed::AbstractVector)
    getvarcostmap(problem)
    sizes = [unfixed_ ? Int(dynamic(nvars(var))) : 0 for (var, unfixed_) in zip(problem.variables, unfixed)]
    return (problem, typeof(options), BitVector(unfixed), sizes, countcosts(costnum, problem.costs))
end

function nlls_keyvalid(key::Tuple{Model, DataType, Int, Any, Int}, problem::Model, options::OptimizationOptions, unfixed::Integer)
    return key[1] === problem && key[2] === typeof(options) && key[3] == unfixed && key[4] === dynamic(nvars(problem.variables[unfixed])) && key[5] == countcosts(costnum, problem.costs)
end

function nlls_keyvalid(key::Tuple{Model, DataType, BitVector, Vector{Int}, Int}, problem::Model, options::OptimizationOptions, unfixed::AbstractVector)
    (key[1] === problem && key[2] === typeof(options) && problem.varcostmapvalid) || return false
    (key[3] == unfixed && key[5] == countcosts(costnum, problem.costs)) || return false
    length(key[4]) == length(problem.variables) || return false
    for (i, var) in enumerate(problem.variables)
        if @inbounds(unfixed[i]) && @inbounds(key[4][i]) != dynamic(nvars(var))
            return false
        end
    end
    return true
end
nlls_keyvalid(key, problem, options, unfixed) = false

# OptimizeCache constructor
const OptimizeCache = OptimizationWorkspace

function OptimizeCache(problem::Model, options::OptimizationOptions=OptimizationOptions(), unfixed=nothing; numthreads=StaticInt(Threads.nthreads()))
    startstats = SecondOrderOptimization.Stats()
    unfixed_ = convertunfixed(unfixed, problem)
    if unfixed_ isa AbstractVector && sum(unfixed_) == 1
        unfixed_ = findfirst(unfixed_)
    end
    f, fdf!, x0, H = optimvars(problem, unfixed_, static(numthreads))
    so_ls = SecondOrderOptimization.makelinearsystem(x0, H, options.linsolver)
    so_data = OptimizationInternal(so_ls, x0, startstats)
    so_iteratedata = SecondOrderOptimization.construct(options.iterator, so_ls, options.callbackchangesstate)
    data = NLLSInternalData(fdf!.linsystem, f, fdf!, H, so_data)
    key = nlls_cachekey(problem, options, unfixed_)
    return OptimizationWorkspace{typeof(key), Tuple{typeof(data), typeof(so_iteratedata)}}(key, (data, so_iteratedata))
end

# Optimizing variables one at a time (e.g. alternating minimization in BA)
function optimizesingles!(problem::Model{VT, CT}, options::OptimizationOptions, type::DataType; numthreads=StaticInt(Threads.nthreads()))::OptimizationResult where {VT, CT}
    indices = findall(v -> isa(v, type), problem.variables)
    return optimizesingles!(problem, options, indices, !isempty(indices) && dynamic(is_static(nvars(problem.variables[indices[1]])) ); numthreads=numthreads)
end

function optimizesingles!(problem::Model{VT, CT}, options::OptimizationOptions, indices, sortedbysize=false; numthreads=StaticInt(Threads.nthreads()))::OptimizationResult where {VT, CT}
    startstats = SecondOrderOptimization.Stats()
    costindices = sparse(getvarcostmap(problem)')
    allcosts = problem.costs
    problem.costs = CostStruct{CT}()
    if !sortedbysize
        indices = sort!(indices, lt=(i, j) -> nvars(problem.variables[i]) <= nvars(problem.variables[j]))
    end
    result = OptimizationResult(0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, getiterator(options), 0, 0, 0, 0)
    first = 1
    while first <= length(indices)
        varsz = dynamic(nvars(problem.variables[indices[first]]))::Int
        last = first + 1
        while last <= length(indices) && dynamic(nvars(problem.variables[indices[last]]))::Int == varsz
            last += 1
        end
        result = setupsinglevarls(problem, options, indices[first], allcosts, costindices, @inbounds(view(indices, first:last-1)), result, first == 1 ? startstats : SecondOrderOptimization.Stats(); numthreads=numthreads)
        first = last
    end
    problem.costs = allcosts
    return result
end

function setupsinglevarls(problem::Model, options::OptimizationOptions, unfixed::Integer, allcosts::CostStruct, costindices, varindices, result, startstats::SecondOrderOptimization.Stats=SecondOrderOptimization.Stats(); numthreads=StaticInt(Threads.nthreads()))
    nthr = static(numthreads)
    _, fdf0!, x0_init, H = optimvars(problem, unfixed, nthr)
    ls = fdf0!.linsystem
    so_ls = SecondOrderOptimization.makelinearsystem(x0_init, H, options.linsolver)
    data = OptimizationInternal(so_ls, x0_init, startstats)
    iteratedata = SecondOrderOptimization.construct(options.iterator, so_ls, options.callbackchangesstate)
    iternum = 0
    termination = 0
    startcost = 0.0
    bestcost = 0.0
    data.init = SecondOrderOptimization.Stats()
    for ind in varindices
        ls_ind = updateunfixed(ls, ind)
        ls = ls_ind
        selectcosts!(problem.costs, allcosts, @inbounds(view(costindices.rowval, costindices.colptr[ind]:costindices.colptr[ind+1]-1)))
        x0 = problem.variables[ind]
        if x0 === data.varnext
            data.variables, data.varnext = data.varnext, data.variables
        else
            data.variables = x0
        end
        data.varnext = syncvars!(data.varnext, x0)
        if data.varbestvalid
            data.varbest = syncvars!(data.varbest, x0)
        else
            data.varbest = x0
        end
        data.iternum = 0
        SecondOrderOptimization.reset!(iteratedata)
        f = SingleVarCostFunc(problem, ind, nthr)
        fdf! = SingleVarCostGradHess(ls_ind, problem, ind)
        termination |= SecondOrderOptimization.optimizeloop!(data, iteratedata, f, fdf!, options, nullcallback)
        problem.variables[ind] = data.variables
        startcost += data.startcost
        bestcost += data.bestcost
        iternum += data.iternum
    end
    data.iternum = iternum
    data.startcost = startcost
    data.bestcost = bestcost
    data.optimize = SecondOrderOptimization.Stats()
    result += OptimizationResult(data, termination, getiterator(options))
    return result
end
