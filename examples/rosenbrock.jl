using StaticArrays, GLMakie
import NLLSsolver

# Define the Rosenbrock cost function
struct Rosenbrock <: NLLSsolver.AbstractResidual
    a::Float64
    b::Float64
end
Rosenbrock() = Rosenbrock(1.0, 10.0)
NLLSsolver.nvars(::Rosenbrock) = 1 # Residual depends on 1 variable
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

function constructtrajectory(start, trajectory)
    x = start[1]
    y = start[2]
    X = [x]
    Y = [y]
    for step in trajectory
        x += step[1]
        y += step[2]
        push!(X, x)
        push!(Y, y)
    end
    return X, Y
end

function optimizeRosenbrock(start=[-0.5, 2.5], iterators=[NLLSsolver.gaussnewton, NLLSsolver.levenbergmarquardt, NLLSsolver.dogleg])
    # Compute costs over a grid 
    residual = Rosenbrock()
    X = range(-1.5, 3., 1000)
    Y = range(-1.5, 3., 1000)
    grid = computeCostGrid(x -> log1p(norm(NLLSsolver.computeresidual(residual, x))), X, Y)

    # Create the plot
    fig = Figure()
    ax1 = Axis(fig[1, 1]; limits=(X[1], X[end], Y[1], Y[end]), title="Trajectories")
    ax2 = Axis(fig[1, 2]; title="Costs", xlabel="Iteration", yscale=log10)
    heatmap!(ax1, X, Y, grid)
    contour!(ax1, X, Y, grid, linewidth=2, color=:white, levels=10)

    # Create the problem
    problem = NLLSsolver.NLLSProblem{NLLSsolver.EuclideanVector{2, Float64}}()
    NLLSsolver.addvariable!(problem, NLLSsolver.EuclideanVector(0., 0.))
    NLLSsolver.addresidual!(problem, residual)

    colors = [:black, :red, :navy]
    for (ind, iter) in enumerate(iterators)
        # Set the start
        problem.variables[1] = NLLSsolver.EuclideanVector(start[1], start[2])

        # Optimize the cost
        options = NLLSsolver.NLLSOptions(dcost=1.e-6, iterator=iter, storetrajectory=true, storecosts=true)
        result = NLLSsolver.optimize!(problem, options)

        # Construct the trajectory
        X, Y = constructtrajectory(start, result.trajectory)

        # Plot the trajectory and costs
        color = colors[mod(ind-1, length(colors)) + 1]
        scatterlines!(ax1, X, Y, color=color, linewidth=2.5)
        scatterlines!(ax2, max.(pushfirst!(result.costs, result.startcost), 1.e-38), color=color, linewidth=1.5, label=String(iter))
    end
    axislegend(ax2)
    return fig
end

optimizeRosenbrock()