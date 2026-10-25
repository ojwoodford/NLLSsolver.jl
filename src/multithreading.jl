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

function cost(vars::AbstractVector, costs::CostStruct, ::StaticInt{N})::Float64 where {N}
    if N == 1 || countcosts(costnum, costs) < 10000
        return cost(vars, costs)
    end
    costsvec = MVector{N, Float64}(undef)
    @batch for taskid in 1:N
        costsvec[taskid] = sumsubset(bindleadingargs(computecost, vars), costs, bindleadingargs(ithchunk, N, taskid); init=0.0)
    end
    return sum(costsvec)
end

function costgradhess!(linsystem::LinearSystemShared{F, V, N}, vars::AbstractVector, costs::CostStruct)::Float64 where {F, V, N}
    if N == 1
        ls = LinearSystem(linsystem, 1)
        zero!(ls)
        linsystem.costs[1] = sumsubset(bindleadingargs(costgradhess!, ls, vars), costs, bindleadingargs(ithchunk, 1, 1); init=0.0)
        return linsystem.costs[1]
    end

    # Phase 1: Parallel gradient & cost chunk evaluation across threads
    @batch for taskid in 1:N
        ls = LinearSystem(linsystem, taskid)
        zero!(ls)
        linsystem.costs[taskid] = sumsubset(bindleadingargs(costgradhess!, ls, vars), costs, bindleadingargs(ithchunk, N, taskid); init=0.0)
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
