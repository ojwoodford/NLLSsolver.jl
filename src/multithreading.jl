using OhMyThreads

# Overload sum to use multithreading for large vectors
function Base.sum(fun::Bind{typeof(computecost), Tuple{StaticInt{N}, Vector{T}}}, vec::AbstractVector; kw...)::Float64 where {N, T}
    if N == 1 || length(vec) < Int(1e5)
        # Single threaded version
        return sum(Base.Fix1(computecost, fun.args[2]), vec; kw...)
    end
    # Multithreaded version
    return tmapreduce(fun, +, vec; scheduler=StaticScheduler(ntasks=N), kw...)
end
