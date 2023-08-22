using NLLSsolver, Test, StaticArrays

# Generic function for testing a robustifier
function testrobustifier(kernel, costs, expected)
    for (i, cost) in enumerate(costs)
        # Check the value is correct
        @test robustify(kernel, cost) ≈ expected[i]
        # Check the derivatives are correct
        @test SVector(robustifydcost(kernel, cost)) ≈ SVector(NLLSsolver.autorobustifydcost(kernel, cost))
        if isa(kernel, NLLSsolver.AbstractAdaptiveRobustifier)
            # Check the derivatives w.r.t. the kernel parameters
            c, dc, d2c = robustifydkernel(kernel, cost)
            c_, dc_, d2c_ = NLLSsolver.autorobustifydkernel(kernel, cost)
            @test c ≈ c_
            # @test dc ≈ dc_
            # @test d2c ≈ d2c_
        end
    end
end

@testset "robust.jl" begin
    costs = SVector(0.0, 0.1, 0.3, 0.7, 1.3, 2.0, 5.0) .^ 2

    # No robustifier
    testrobustifier(NoRobust(), costs, costs)

    # Scaled robustifier
    testrobustifier(Scaled(NoRobust(), 2.0), costs, 2 * costs)

    # Huber
    sigma = 0.7
    out = [ifelse(c <= sigma ^ 2, c, 2 * sigma * sqrt(c) - sigma ^ 2) for c in costs]
    testrobustifier(Huber2oKernel(sigma), costs, out)

    # Scaled Huber
    testrobustifier(Scaled(Huber2oKernel(sigma), 3.0), costs, 3 * out)

    # Geman-McClure
    sigma = 0.6
    out = costs * sigma ^ 2 ./ (costs .+ sigma ^ 2)
    testrobustifier(GemanMcclureKernel(sigma), costs, out)

    # Contaminated Gaussian (Adaptive kernel)
    s1 = 0.6
    s2 = 9.0
    w = 0.7
    out = -log.((w / s1) * exp.(costs / (-2 * s1^2)) .+ ((1 - w) / s2) * exp.(costs / (-2 * s2^2)))
    testrobustifier(ContaminatedGaussian(s1, s2, w), costs, out)
end
