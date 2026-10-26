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
