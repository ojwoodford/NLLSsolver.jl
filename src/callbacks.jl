import Printf.@printf

# Default: do nothing
nullcallback(cost, unusedargs...) = (cost, 0)

# Print out per-iteration results
function printoutcallback(cost, problem, data, trailingargs...)
    if data.iternum == 1
        # First iteration, so print out column headers and the zeroth iteration (i.e. start) values
        println("iter      cost      cost_change    |step|")
        @printf("% 4d % 8e  % 4.3e   % 3.2e\n", 0, data.bestcost, 0, 0)
    end
    @printf("% 4d % 8e  % 4.3e   % 3.2e\n", data.iternum, cost, data.bestcost-cost, norm(data.linsystem.x))
    return cost, 0
end
function printoutcallback(cost, data, trradius::Float64)
    if data.iternum == 1
        # First iteration, so print out column headers and the zeroth iteration (i.e. start) values
        println("iter      cost      cost_change    |step|    tr_radius")
        @printf("% 4d % 8e  % 4.3e   % 3.2e   % 2.1e\n", 0, data.bestcost, 0, 0, trradius)
    end
    @printf("% 4d % 8e  % 4.3e   % 3.2e   % 2.1e\n", data.iternum, cost, data.bestcost-cost, norm(data.linsystem.x), trradius)
    return cost, 0
end

# Store per-iteration costs
function storecostscallback(costs::Vector{Float64}, cost, unusedargs...)
    push!(costs, cost)
    return (cost, 0)
end

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
storecostscallback(store) = (args...) -> storecostscallback(store, args...)
