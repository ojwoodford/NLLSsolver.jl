using Polyester

@inline function chunk_indices(len::Integer, nchunks::Integer, chunkid::Integer)
    chunkid > nchunks && return 1:0
    q, r = divrem(len, nchunks)
    start = (chunkid - 1) * q + min(chunkid - 1, r) + 1
    stop = start + q - 1 + (chunkid <= r ? 1 : 0)
    return start:stop
end

@inline function ithchunk(N, taskid, collection)
    return view(collection, chunk_indices(length(collection), N, taskid))
end

function cost_vec(vars, costlist::Vector, ::StaticInt{N})::Float64 where {N}
    len = length(costlist)
    if len < 10000
        return cost_vec(vars, costlist, static(1))
    end
    tot = 0.0
    @batch reduction=((+, tot),) for taskid in 1:N
        tot += cost_range(vars, costlist, chunk_indices(len, N, taskid))
    end
    return tot
end

costgradhess_chunk_repo!(ls, vars, costs::CostStruct{T}, N, taskid) where T = costgradhess_chunk_repo_type!(ls, vars, costs, T, N, taskid)
costgradhess_chunk_repo_type!(ls, vars, costs, T::Union, N, taskid) = costgradhess_chunk_repo_type!(ls, vars, costs, T.a, N, taskid) + costgradhess_chunk_repo_type!(ls, vars, costs, T.b, N, taskid)
function costgradhess_chunk_repo_type!(ls, vars, costs, T::DataType, N, taskid)
    costlist = get(costs, T)
    rng = chunk_indices(length(costlist), N, taskid)
    return costgradhess_chunk_vec!(ls, vars, costlist, rng)
end
costgradhess_chunk_repo!(ls, vars, costs::CostStruct{Any}, N, taskid) = sum(c -> begin
    rng = chunk_indices(length(c), N, taskid)
    costgradhess_chunk_vec!(ls, vars, c, rng)
end, values(costs.data))

function costgradhess_chunk_vec!(ls, vars, costlist::Vector, rng::UnitRange{Int})::Float64
    tot = 0.0
    sub = @inbounds view(costlist, rng)
    @inbounds for c in sub
        tot += costgradhess!(ls, vars, c)
    end
    return tot
end

function costgradhess!(linsystem::LinearSystemShared{F, V, N}, vars::AbstractVector, costs::CostStruct)::Float64 where {F, V, N}
    if N == 1
        ls = LinearSystem(linsystem, 1)
        zero!(ls)
        linsystem.costs[1] = costgradhess!(ls, vars, costs)
        return linsystem.costs[1]
    end

    # Phase 1: Parallel gradient & cost chunk evaluation across threads
    @batch for taskid in 1:N
        ls = LinearSystem(linsystem, taskid)
        zero!(ls)
        linsystem.costs[taskid] = costgradhess_chunk_repo!(ls, vars, costs, N, taskid)
    end

    # Phase 2: Parallel tree reduction across the N buffers
    n = N
    while n > 1
        n_ = (n + 1) >> 1
        m = n - n_
        @batch for taskid in 1:m
            add!(linsystem.ls[taskid], linsystem.ls[taskid + n_])
            linsystem.costs[taskid] += linsystem.costs[taskid + n_]
        end
        n = n_
    end

    return linsystem.costs[1]
end

@inline function optimizesinglesinternal!(problem::NLLSProblem{VT, CT}, options::NLLSOptions{S, N}, data::NLLSInternal{LST}, iteratedata, callback::F, allcosts::CostStruct, costindices, varinds, result) where {VT, CT, S, N, LST<:UniVariateLS, F}
    if length(varinds) <= 1 || N == 1
        data.init = Stats()
        termination = optimizesingles_worker!(problem, singlethreaded(options), data, iteratedata, callback, allcosts, costindices, varinds)
        data.optimize = Stats()
        result += NLLSResult(data, termination, getiterator(options))
        return result
    end

    nactive = min(N, length(varinds))
    singleoptions = singlethreaded(options)

    if length(problem.variables) != length(problem.varbest)
        problem.varbest = deepcopy(problem.variables)
    end

    costs_vec = [CostStruct{CT}() for _ in 1:nactive]
    probs = [i == 1 ? problem : NLLSProblem{VT, CT}(problem.variables, costs_vec[i], problem.varnext, problem.varbest) for i in 1:nactive]
    data_vec = [i == 1 ? data : NLLSInternal(copy_structure(data.linsystem), data.start) for i in 1:nactive]
    iterdata_vec = [i == 1 ? iteratedata : construct(options.iterator, probs[i], data_vec[i].linsystem) for i in 1:nactive]
    terminations = zeros(Int, nactive)

    data.init = Stats()

    @batch for taskid in 1:nactive
        chunk = ithchunk(nactive, taskid, varinds)
        terminations[taskid] = optimizesingles_worker!(probs[taskid], singleoptions, data_vec[taskid], iterdata_vec[taskid], callback, allcosts, costindices, chunk)
    end

    stop_stats = Stats()
    data.optimize = stop_stats
    for taskid in 2:nactive
        data_vec[taskid].start = stop_stats
        data_vec[taskid].init = stop_stats
        data_vec[taskid].optimize = stop_stats
    end

    for taskid in 1:nactive
        result += NLLSResult(data_vec[taskid], terminations[taskid], getiterator(options))
    end
    return result
end
