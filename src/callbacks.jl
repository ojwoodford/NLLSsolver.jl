import Printf.@printf

# Default: do nothing
"""
    NLLSsolver.nullcallback(cost, problem, data, iteratedata)

A callback that does nothing and returns the tuple `(cost, 0)`.

# Example

Optimizing an NLLSsolver problem as follows:
```julia
    NLLSsolver.optimize!(problem, options, unfixed, nullcallback)
```
will incur no callback overhead.

!!! note
    This is the default callback used by `optimize!`.
"""
nullcallback(cost, unusedargs...) = (cost, 0)

# Print out per-iteration results
"""
    NLLSsolver.printoutcallback(cost, problem, data, iteratedata)

A callback that prints out cost, change in cost and step size each optimization iteration.

# Example

Optimizing an NLLSsolver problem as follows:
```julia
    NLLSsolver.optimize!(problem, options, unfixed, printoutcallback)
```
will cause per iteration information to print out in the terminal.

!!! note
    This callback is useful for debugging, but will incur a performance penalty.
"""
function printoutcallback(cost, problem, data, trailingargs...)
    prevcost = data.bestcost
    if data.iternum == 1
        prevcost = data.startcost
        # First iteration, so print out column headers and the zeroth iteration (i.e. start) values
        println("iter      cost      cost_change    |step|")
        @printf("% 4d % 8e  % 4.3e   % 3.2e\n", 0, prevcost, 0, 0)
    end
    @printf("% 4d % 8e  % 4.3e   % 3.2e\n", data.iternum, cost, prevcost-cost, norm(data.linsystem.x))
    return cost, 0
end
function printoutcallback(cost, data, trradius::Float64)
    prevcost = data.bestcost
    if data.iternum == 1
        prevcost = data.startcost
        # First iteration, so print out column headers and the zeroth iteration (i.e. start) values
        println("iter      cost      cost_change    |step|    tr_radius")
        @printf("% 4d % 8e  % 4.3e   % 3.2e   % 2.1e\n", 0, prevcost, 0, 0, trradius)
    end
    @printf("% 4d % 8e  % 4.3e   % 3.2e   % 2.1e\n", data.iternum, cost, prevcost-cost, norm(data.linsystem.x), trradius)
    return cost, 0
end

# Store per-iteration costs
function storecostscallback(costs::Vector{Float64}, cost, unusedargs...)
    push!(costs, cost)
    return (cost, 0)
end

"""
    NLLSsolver.CostTrajectory

A type with the fields
```julia
    costs::Vector{Float64}
    times_ns::Vector{UInt64}
    trajectory::Vector{Vector{Float64}}
```
that can be used to store per iteration optimization information.

---
```julia
    CostTrajectory()
```
Construct an empty `CostTrajectory` struct, ready for use with [`storecostscallback`](@ref).
"""
struct CostTrajectory
    costs::Vector{Float64}
    times_ns::Vector{UInt64}
    trajectory::Vector{Vector{Float64}}
    
    function CostTrajectory()
        return new(sizehint!(Vector{Float64}(), 50), sizehint!(Vector{UInt64}(), 50), sizehint!(Vector{Vector{Float64}}(), 50))
    end
end
function Base.empty!(ct::CostTrajectory)
    empty!(ct.costs)
    empty!(ct.times_ns)
    empty!(ct.trajectory)
    return ct
end

# Store per-iteration costs and trajectory
function storecostscallback(store::CostTrajectory, cost, problem, data, unusedargs...)
    push!(store.costs, cost)
    push!(store.times_ns, Base.time_ns() - data.starttime)
    push!(store.trajectory, Vector{Float64}(data.linsystem.x))
    return (cost, 0)
end

# Callback function generator
"""
    NLLSsolver.storecostscallback(store)

Generate a callback that stores per iteration information in `store` for later inspection.

# Example

Optimizing an NLLSsolver problem as follows:
```julia
    costs = Vector{Float64}()
    NLLSsolver.optimize!(problem, options, unfixed, storecostscallback(costs))
```
will cause the cost at the end of each iteration to be stored in the vector `costs`, while:
```julia
    costtrajectory = CostTrajectory()
    NLLSsolver.optimize!(problem, options, unfixed, storecostscallback(costtrajectory))
```
will cause the cost, step and optimization time at the end of each iteration to be stored in
a [`CostTrajectory`](@ref) structure.

!!! note
    This method is useful for debugging, but will incur a performance penalty.
"""
storecostscallback(store) = (args...) -> storecostscallback(store, args...)
