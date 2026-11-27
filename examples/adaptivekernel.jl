# This example:
#   1. Constructs and optimizes a simple data fitting problem (in generatedata()) using a ContaminatedGaussian
#      adaptive robustifer to fit to a set of points coming from two gaussians with the same mean.
#   2. Visualizes the fit of the ContaminatedGaussian robust kernel to the data (in plotdata()), with a slider to
#      select any state throughout the optimization.
#
# The kernel's cost is plotted with the kernel's own `robustify` (the exact cost the optimizer minimises), so
# the curve matches the optimized cost. NLLSModel has no public accessor for a ContaminatedGaussian's
# parameters (its `params` is internal), so the example never reads them; see the NLLSModel fixes file.

using NLLSsolver, Static, StaticArrays, GLMakie

struct OffsetResidual <: NLLSsolver.AbstractAdaptiveResidual
    data::Float64
end
NLLSsolver.ndeps(::OffsetResidual) = static(2) # Residual depends on 2 variables (the kernel and the offset)
NLLSsolver.nres(::OffsetResidual) = static(1) # Residual has length 1
NLLSsolver.varindices(::OffsetResidual) = SVector(1, 2)
NLLSsolver.getvars(::OffsetResidual, vars::Vector) = (vars[1]::NLLSsolver.ContaminatedGaussian{Float64, 1}, vars[2]::Float64)
# A length-1 residual and its 1x1 Jacobian w.r.t. the offset (the kernel's derivatives are handled by the
# adaptive machinery). NLLSModel wants a residual vector and a Jacobian matrix, not bare scalars, even for
# length 1 (NLLSsolver 5 accepted scalars here).
NLLSsolver.computeresidual(res::OffsetResidual, mean) = SVector(mean - res.data)
NLLSsolver.computeresjac(varflags, res::OffsetResidual, mean) = SVector(mean - res.data), SMatrix{1, 1}(1.0)
Base.eltype(::OffsetResidual) = Float64

function generatedata(numinliers=100, numoutliers=200, inliersigma=1.0, outliersigma=10.0, offset=1.0, startsigma1=0.5, startsigma2=5.0, startinlierratio=0.6, startoffset=0.0)
    # Generate the test data
    points = offset .+ vcat(randn(numinliers) * inliersigma, randn(numoutliers) * outliersigma)

    # Create the problem. The kernel is a ContaminatedGaussian of residual length 1 (static(1)).
    problem = NLLSsolver.NLLSProblem(Union{NLLSsolver.ContaminatedGaussian{Float64, 1}, Float64}, OffsetResidual)
    NLLSsolver.addvariable!(problem, ContaminatedGaussian(startsigma1, startsigma2, startinlierratio, static(1)))
    NLLSsolver.addvariable!(problem, startoffset)
    for p in points
        NLLSsolver.addcost!(problem, OffsetResidual(p))
    end

    # Optimize the cost
    costtraj = NLLSsolver.CostTrajectory()
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=:levenbergmarquardt), nothing, NLLSsolver.storecostscallback(costtraj))

    # Replay the optimization to get the kernel and offset at every step. The step vector is the update to
    # [kernel parameters (3); offset (1)]; CostTrajectory stores each step in `steps` (was `trajectory` in
    # NLLSsolver 5).
    cg = ContaminatedGaussian(startsigma1, startsigma2, startinlierratio, static(1))
    offset = startoffset
    cgstates = Vector{Tuple{typeof(cg), Float64}}(undef, length(costtraj.steps)+1)
    for (ind, updatevec) in enumerate(costtraj.steps)
        cgstates[ind] = (cg, offset)
        cg = update(cg, updatevec)
        offset += updatevec[end]
    end
    cgstates[end] = (cg, offset)

    # Generate the line sample points
    min, max = extrema(points)
    padding = 0.05 * (max - min)
    x = vcat(Vector(range(min-padding, max+padding, 1000)), points)
    ind = sortperm(x)
    x = x[ind]
    points = ind .> 1000

    return x, points, cgstates
end

# The cost the optimizer minimises at a point `xi`, for the kernel and offset of one optimization state: half
# the robustified squared residual (the adaptive kernel's cost is exactly -2 log p, so it can be negative).
costfunc(x, state) = [0.5 * NLLSsolver.robustify(state[1], (xi - state[2])^2) for xi in x]

function plotdata(x, points, cgstates)
    # Construct the figure
    GLMakie.activate!(inline=false)
    fig = Figure()
    ax = Axis(fig[1, 1])
    # Use a slider to select the point along the optimization
    slider = Slider(fig[2, 1], range=1:length(cgstates), startvalue=length(cgstates))
    y = lift(slider.value) do ind
        costfunc(x, cgstates[ind])
    end
    # Plot the data
    lines!(ax, x, y; color=:blue)
    colors = ones(UInt, length(x))
    colors[points] .= 2
    scatter!(ax, x, y; color=colors, colormap=[:transparent, :black])
    return fig
end

plotdata(generatedata()...)
