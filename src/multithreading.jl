using OhMyThreads

# @inline multithreadedsum(fun, vector; init=0.0) = sum(fun, vector; init=init)::Float64
@inline multithreadedsum(fun, vector; init=0.0) = tmapreduce(fun, +, vector; scheduler=:static, init=init)::Float64
