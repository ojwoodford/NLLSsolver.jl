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

function costgradhesschunk!(linsystem::LinearSystemShared{F, V, N}, vars::Vector, costs::CostStruct, taskid, tasks::Vector{Task})::Nothing where {F, V, N}
    # Construct the task local values
    subsetfun = (v) -> chunks(v, N)[taskid]
    ls = LinearSystem(linsystem, taskid)
    zero!(ls) # Zero the local data buffer
    # Sum over the subsets
    ls.ls.cost = sumsubset(bindleadingargs(costgradhess!, linsystem, vars), costs, subsetfun; init=0.0)
    # Reduce over the local buffers
    n = N
    while n > 1
        n_ = (n + 1) / 2
        i = taskid + n_
        if i > n
            break
        end
        n = n_
        # Wait for task i to finish
        wait(tasks[i-1])
        # Reduce data
        ls.ls.A += linsystem.ls[i].A
        ls.ls.b += linsystem.ls[i].b
        ls.ls.cost += linsystem.ls[i].cost
    end
    return nothing
end

# Overload sum to use multithreading for large vectors
function costgradhess!(linsystem::LinearSystemShared{F, V, N}, vars::Vector, costs::CostStruct)::Float64 where {F, V, N}
    tasks = sizehint!(Vector{Task}(), N-1)
    for taskid in 2:N
        push!(tasks, @task costgradhesschunk!(linsystem, vars, cost, taskid, tasks))
    end
    foreach(scheduleunsticky, tasks)
    costgradhesschunk!(linsystem, vars, cost, 1, tasks)
    return linsystem.ls[1].cost
end
