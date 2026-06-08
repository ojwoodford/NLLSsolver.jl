using OhMyThreads

multithreadedsum(fun, vec, init) = sum(fun, vec; init=init)

