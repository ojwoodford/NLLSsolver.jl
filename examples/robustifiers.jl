using NLLSsolver, GLMakie

function displaykernels(maxval, kernels...)
    # Compute the values for plotting
    x = range(-maxval, maxval, 1000)
    cost = x .^ 2
    v = [Vector{Float64}(undef, 1000) for i in 1:length(kernels)]
    dv = [Vector{Float64}(undef, 1000) for i in 1:length(kernels)]
    d2v = [Vector{Float64}(undef, 1000) for i in 1:length(kernels)]
    for (k, kernel) in enumerate(kernels)
        for (ind, c) in enumerate(cost) 
            v[k][ind], dv[k][ind], d2v[k][ind] = robustifydcost(kernel, c)
        end
    end

    # Construct the plot
    fig = Figure()
    ax1 = Axis(fig[1, 1]; title="Costs")
    ax2 = Axis(fig[1, 2]; title="First derivative")
    ax3 = Axis(fig[1, 3]; title="Second derivative")
    for k = 1:length(kernels)
        lines!(ax1, x, v[k])
        lines!(ax2, x, dv[k])
        lines!(ax3, x, d2v[k])
    end
    return fig
end

displaykernels(10.0, NoRobust(), Huber2oKernel(1.0), GemanMcclureKernel(1.0))
