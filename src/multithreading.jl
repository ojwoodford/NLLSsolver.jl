using OhMyThreads

function scheduleunsticky(t::Task)
    t.sticky = false
    schedule(t)
    return nothing
end

# Overload sum to use multithreading for large vectors
function Base.sum(fun::Bind{typeof(computecost), Tuple{StaticInt{N}, Vector{VT}}}, vec::AbstractVector; kw...)::Float64 where {N, VT}
    if N == 1 || length(vec) < Int(1e5)
        # Single threaded version
        return sum(Base.Fix1(computecost, fun.args[2]), vec; kw...)
    end
    # Multithreaded version
    return tmapreduce(fun, +, vec; scheduler=StaticScheduler(ntasks=N), kw...)
end

function ithchunk(N, taskid, collection)
    c = index_chunks(collection; n=N)
    return view(collection, taskid <= length(c) ? c[taskid] : 1:0)
end

function costgradhesschunk!(linsystem::LinearSystemShared{F, V, N}, vars::Vector, costs::CostStruct, taskid, tasks::Vector{Task})::Nothing where {F, V, N}
    ls = LinearSystem(linsystem, taskid)
    zero!(ls)
    linsystem.costs[taskid] = sumsubset(bindleadingargs(costgradhess!, ls, vars), costs, bindleadingargs(ithchunk, N, taskid); init=0.0)

    # Parallel tree reduction over the local buffers
    n = N
    while n > 1
        n_ = (n + 1) >> 1
        i = taskid + n_
        if i > n
            break
        end
        n = n_
        # Wait for task i to finish
        wait(tasks[i-1])
        # Reduce data
        add!(linsystem.ls[taskid], linsystem.ls[i])
        linsystem.costs[taskid] += linsystem.costs[i]
    end
    return nothing
end

function costgradhess!(linsystem::LinearSystemShared{F, V, N}, vars::Vector, costs::CostStruct)::Float64 where {F, V, N}
    if N == 1
        ls = LinearSystem(linsystem, 1)
        zero!(ls)
        linsystem.costs[1] = sumsubset(bindleadingargs(costgradhess!, ls, vars), costs, bindleadingargs(ithchunk, 1, 1); init=0.0)
        return linsystem.costs[1]
    end
    tasks = sizehint!(Vector{Task}(), N-1)
    for taskid in 2:N
        push!(tasks, @task costgradhesschunk!(linsystem, vars, costs, taskid, tasks))
    end
    foreach(scheduleunsticky, tasks)
    costgradhesschunk!(linsystem, vars, costs, 1, tasks)
    return linsystem.costs[1]
end
