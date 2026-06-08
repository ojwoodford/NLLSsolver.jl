using OhMyThreads

# Single threaded fallback
multithreadedsum(fun, vec, init) = sum(fun, vec; init=init)

multithreadedsum(fun::Bind{typeof(computecost), A}, vec, init) where A = tmapreduce(fun, +, vec; scheduler=:static, init=init)
