using OhMyThreads

# Overload sum to use multithreading for large vectors
function Base.sum(fun::Bind{typeof(computecost), A}, vec::Vector; init=init) where A
    return tmapreduce(fun, +, vec; scheduler=:static, init=init)
end
