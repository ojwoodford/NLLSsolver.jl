using OhMyThreads

multithreadedsum(fun, v::Vector; init=0.0) = tmapreduce(fun, +, v; scheduler=:dynamic, init=init)
