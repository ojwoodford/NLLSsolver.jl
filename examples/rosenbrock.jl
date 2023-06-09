using StaticArrays, GLMakie, LinearAlgebra
import NLLSsolver

# Define the Rosenbrock cost function
struct Rosenbrock <: NLLSsolver.AbstractResidual
    a::Float64
    b::Float64
end
Rosenbrock() = Rosenbrock(1.0, 10.0)
Base.eltype(::Rosenbrock) = Float64
NLLSsolver.nvars(NLLSsolver.@objtype(Rosenbrock)) = 1 # Residual depends on 1 variable
NLLSsolver.nres(NLLSsolver.@objtype(Rosenbrock)) = 2 # Residual has length 2
NLLSsolver.varindices(::Rosenbrock) = SVector(1) # There's only one variable
function NLLSsolver.getvars(::Rosenbrock, vars::Vector)
    return (vars[1]::NLLSsolver.EuclideanVector{2, Float64},)
end
function NLLSsolver.computeresidual(res::Rosenbrock, x::NLLSsolver.EuclideanVector{2, Float64})
    return SVector(res.a - x[1], res.b * (x[1] ^ 2 - x[2]))
end
function NLLSsolver.computeresjac(::Val{varflags}, res::Rosenbrock, x::NLLSsolver.EuclideanVector{2, Float64}) where varflags
    @assert varflags == 1
    return SVector(res.a - x[1], res.b * (x[1] ^ 2 - x[2])),
           SMatrix{2}(-1., 2 * res.b * x[1], 0, -res.b)
end

function computeCostGrid(func, X, Y)
    grid = Matrix{Float64}(undef, length(Y), length(X))
    for (b, y) in enumerate(Y)
        for (a, x) in enumerate(X)
            grid[a,b] = func(SVector(x, y))
        end
    end
    return grid
end

struct RosenbrockResult
    costs::Observable{Vector{Float64}}
    trajectory::Observable{Vector{Point2f}}
end
RosenbrockResult() = RosenbrockResult(Observable(Vector{Float64}()), Observable(Vector{Point2f}()))

function runoptimizers!(results, problem, start, iterators)
    for (ind, iter) in enumerate(iterators)
        # Set the start
        problem.variables[1] = NLLSsolver.EuclideanVector(start[1], start[2])

        # Optimize the cost
        options = NLLSsolver.NLLSOptions(reldcost=1.e-6, iterator=iter, storetrajectory=true, storecosts=true)
        result = NLLSsolver.optimize!(problem, options)

        # Construct the trajectory
        resize!(results[ind].trajectory.val, length(result.trajectory)+1)
        @inbounds results[ind].trajectory.val[1] = Point2f(start[1], start[2])
        for (i, step) in enumerate(result.trajectory)
            @inbounds results[ind].trajectory.val[i+1] = results[ind].trajectory.val[i] + Point2f(step[1], step[2])
        end

        # Set the costs
        resize!(results[ind].costs.val, length(result.costs)+1)
        @inbounds results[ind].costs.val[1] = result.startcost
        @inbounds results[ind].costs.val[2:end] .= max.(result.costs, 1.e-38)
    end
    for res in results
        notify(res.costs)
        notify(res.trajectory)
    end
end

function optimizeRosenbrock(start=[-0.5, 2.5], iterators=[NLLSsolver.gaussnewton, NLLSsolver.levenbergmarquardt, NLLSsolver.dogleg])
    # Create the problem
    problem = NLLSsolver.NLLSProblem{NLLSsolver.EuclideanVector{2, Float64}}()
    NLLSsolver.addvariable!(problem, NLLSsolver.EuclideanVector(0., 0.))
    residual = Rosenbrock()
    NLLSsolver.addresidual!(problem, residual)

    # Compute the results 
    results = [RosenbrockResult() for i in range(1, length(iterators))]
    runoptimizers!(results, problem, start, iterators)

    # Compute costs over a grid 
    X = range(-1.5, 3., 1000)
    Y = range(-1.5, 3., 1000)
    grid = computeCostGrid(x -> log1p(norm(NLLSsolver.computeresidual(residual, x))), X, Y)

    # Create the plot
    GLMakie.activate!(inline=false)
    fig = Figure()
    ax1 = Axis(fig[1, 1]; limits=(X[1], X[end], Y[1], Y[end]), title="Trajectories")
    ax2 = Axis(fig[1, 2]; title="Costs", xlabel="Iteration", yscale=log10)
    heatmap!(ax1, X, Y, grid)
    contour!(ax1, X, Y, grid, linewidth=2, color=:white, levels=10)

    # Plot the trajectory and costs
    colors = [:black, :red, :navy]
    for (ind, iter) in enumerate(iterators)
        color = colors[mod(ind-1, length(colors)) + 1]
        scatterlines!(ax1, results[ind].trajectory, color=color, linewidth=2.5)
        scatterlines!(ax2, results[ind].costs, color=color, linewidth=1.5, label=String(iter))
    end
    axislegend(ax2)

    # Allow start points to be selected
    on(events(ax1).mousebutton) do event
        if event.button == Mouse.left && event.action == Mouse.press
            runoptimizers!(results, problem, mouseposition(ax1.scene), iterators)
        end
    end
    return fig
end

optimizeRosenbrock()