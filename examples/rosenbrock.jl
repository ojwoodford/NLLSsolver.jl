# This example:
#   1. Constructs and optimizes the 2D Rosenbrock cost function, using a number of different solvers.
#   2. Visualizes the cost space and the trajectories taken by the different optimizers through that space.
#   3. Different optimization start points can be iteractively selected within the state space using the mouse.

using StaticArrays, GLMakie, LinearAlgebra, Static
import NLLSsolver

# Define the Rosenbrock cost function
struct Rosenbrock <: NLLSsolver.AbstractResidual
    a::Float64
    b::Float64
end
Rosenbrock() = Rosenbrock(1.0, 10.0)
Base.eltype(::Rosenbrock) = Float64
NLLSsolver.ndeps(::Rosenbrock) = static(1) # Residual depends on 1 variable
NLLSsolver.nres(::Rosenbrock) = static(2) # Residual has length 2
NLLSsolver.varindices(::Rosenbrock) = SVector(1) # There's only one variable
NLLSsolver.getvars(::Rosenbrock, vars::Vector) = (vars[1]::NLLSsolver.EuclideanVector{2, Float64},)
NLLSsolver.computeresidual(res::Rosenbrock, x) = SVector(res.a - x[1], res.b * (x[1] ^ 2 - x[2]))

function constructrosenbrockprob()
    # Create the problem
    problem = NLLSsolver.NLLSProblem(NLLSsolver.EuclideanVector{2, Float64}, Rosenbrock)
    NLLSsolver.addvariable!(problem, NLLSsolver.EuclideanVector(0., 0.))
    NLLSsolver.addcost!(problem, Rosenbrock())
    return problem
end

function computeCostGrid(costs, X, Y)
    grid = Matrix{Float64}(undef, length(Y), length(X))
    vars = [SVector(0.0, 0.0)]
    for (b, y) in enumerate(Y)
        for (a, x) in enumerate(X)
            vars[1] = SVector(x, y)
            grid[a,b] = NLLSsolver.cost(vars, costs)
        end
    end
    return grid
end

struct OptimResult
    costs::Observable{Vector{Float64}}
    trajectory::Observable{Vector{Point2f}}
end
OptimResult() = OptimResult(Observable(Vector{Float64}()), Observable(Vector{Point2f}()))

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

function optimize2DProblem(problem, start, xrange, yrange; iterators=[NLLSsolver.newton, NLLSsolver.levenbergmarquardt, NLLSsolver.dogleg, NLLSsolver.gradientdescent])
    # Compute the results 
    results = [OptimResult() for i in range(1, length(iterators))]
    runoptimizers!(results, problem, start, iterators)

    # Compute costs over a grid
    grid = computeCostGrid(problem.costs, xrange, yrange)
    minval = minimum(grid)
    grid = map(x -> log1p(x - minval), grid)

    # Create the plot
    GLMakie.activate!(inline=false)
    fig = Figure()
    ax1 = Axis(fig[1, 1]; limits=(xrange[1], xrange[end], yrange[1], yrange[end]), title="Trajectories")
    ax2 = Axis(fig[1, 2]; title="Costs", xlabel="Iteration", yscale=log10)
    heatmap!(ax1, xrange, yrange, grid)
    contour!(ax1, xrange, yrange, grid, linewidth=2, color=:white, levels=10)

    # Plot the trajectory and costs
    colors = [:black, :red, :navy, :green]
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

optimize2DProblem(constructrosenbrockprob(), [-0.5, 2.5], range(-1.5, 3.0, 1000), range(-1.5, 3.0, 1000))