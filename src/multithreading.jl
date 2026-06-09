using OhMyThreads

# Overload sum to use multithreading for large vectors
function Base.sum(fun::Bind{typeof(computecost), A}, vec::AbstractVector; init=0.0) where A
    return tmapreduce(fun, +, vec; scheduler=:static, init=init)
end
