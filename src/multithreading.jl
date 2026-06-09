using OhMyThreads

# Overload sum to use multithreading for large vectors
function Base.sum(fun::Bind{typeof(computecost), Tuple{StaticInt{N}, Vector{T}}}, vec::AbstractVector; init=0.0)::Float64 where {N, T}
    if N == 1 || length(vec) < Int(1e6)
        # Single threaded version
        return sum(Base.Fix1(computecost, fun.args[2]), vec; init=init)
    end
    # Multithreaded version
    return tmapreduce(fun, +, vec; init=init, scheduler=StaticScheduler(ntasks=N))
end
