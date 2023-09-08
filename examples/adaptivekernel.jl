# This example:
#   1. Constructs and optimizes a simple data fitting problem (in generatedata()) using a ContaminatedGaussian 
#      adaptive robustifer to fit to a set of points coming from two gaussians with the same mean.
#   2. Visualizes the fit of the ContaminatedGaussian robust kernel to the data (in plotdata()), with a slider to
#      select any state throughout the optimization

using NLLSsolver, Static, StaticArrays, GLMakie

struct OffsetResidual <: NLLSsolver.AbstractAdaptiveResidual
    data::Float64
end
NLLSsolver.ndeps(::OffsetResidual) = static(2) # Residual depends on 2 variables
NLLSsolver.nres(::OffsetResidual) = static(1) # Residual has length 1
NLLSsolver.varindices(::OffsetResidual) = SVector(1, 2)
NLLSsolver.getvars(::OffsetResidual, vars::Vector) = (vars[1]::NLLSsolver.ContaminatedGaussian{Float64}, vars[2]::Float64)
NLLSsolver.computeresidual(res::OffsetResidual, mean) = mean - res.data
NLLSsolver.computeresjac(varflags, res::OffsetResidual, mean) = mean - res.data, one(mean)
Base.eltype(::OffsetResidual) = Float64

function generatedata(numinliers=100, numoutliers=200, inliersigma=1.0, outliersigma=10.0, offset=1.0, startsigma1=0.5, startsigma2=5.0, startinlierratio=0.6, startoffset=0.0)
    # Generate the test data
    points = offset .+ vcat(randn(numinliers) * inliersigma, randn(numoutliers) * outliersigma)
    
    # Create the problem
    problem = NLLSsolver.NLLSProblem(Union{NLLSsolver.ContaminatedGaussian{Float64}, Float64}, OffsetResidual)
    NLLSsolver.addvariable!(problem, ContaminatedGaussian(startsigma1, startsigma2, startinlierratio))
    NLLSsolver.addvariable!(problem, startoffset)
    for p in points
        NLLSsolver.addcost!(problem, OffsetResidual(p))
    end

    # Optimize the cost
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.levenbergmarquardt, storetrajectory=true))

    # Generate the parameters
    cg = ContaminatedGaussian(startsigma1, startsigma2, startinlierratio)
    offset = startoffset
    cgparams = Vector{SVector{4, Float64}}(undef, length(result.trajectory)+1)
    for (ind, updatevec) in enumerate(result.trajectory)
        cgparams[ind] = vcat(NLLSsolver.params(cg), offset)
        cg = update(cg, updatevec)
        offset += updatevec[end]
    end
    cgparams[end] = vcat(NLLSsolver.params(cg), offset)

    # Generate the line sample points
    min, max = extrema(points)
    padding = 0.05 * (max - min)
    x = vcat(Vector(range(min-padding, max+padding, 1000)), points)
    ind = sortperm(x)
    x = x[ind]
    points = ind .> 1000

    return x, points, cgparams
end

function costfunc(x, cgparams)
    y = (x .- cgparams[end]) .^ 2
    y = y / (2 * cgparams[2] ^ 2) - log.(exp.(y * (0.5 / cgparams[2] ^ 2 - 0.5 / cgparams[1] ^ 2)) * (cgparams[3] / cgparams[1]) .+ (1 - cgparams[3]) / cgparams[2])
    return 0.5 * y
end

function plotdata(x, points, cgparams)
    # Construct the figure
    GLMakie.activate!(inline=false)
    fig = Figure()
    ax = Axis(fig[1, 1])
    # Use a slider to select the point along the optimization
    slider = Slider(fig[2, 1], range=1:length(cgparams), startvalue=length(cgparams))
    y = lift(slider.value) do ind
        costfunc(x, cgparams[ind])
    end
    # Plot the data
    lines!(ax, x, y; color=:blue)
    colors = ones(UInt, length(x))
    colors[points] .= 2
    scatter!(ax, x, y; color=colors, colormap=[:transparent, :black])
    return fig
end

plotdata(generatedata()...)

