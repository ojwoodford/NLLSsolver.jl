using NLLSsolver, Test, StaticArrays

# Generic function for testing a robustifier
function testrobustifier(kernel, costs, expected)
    for (i, cost) in enumerate(costs)
        # Check the value is correct
        @test robustify(kernel, cost) == expected[i]
        # Check the derivatives are correct
        @test all(robustifyd(kernel, cost) .≈ NLLSsolver.autorobustifyd(kernel, cost))
    end
end

@testset "robust.jl" begin
    costs = SVector(0.0, 0.1, 0.3, 0.7, 1.3, 2.0, 5.0) .^ 2

    # No robustifier
    testrobustifier(NoRobust(), costs, costs)

    # Scaled robustifier
    testrobustifier(Scaled(NoRobust(), 2.0), costs, 2 * costs)

    # Huber
    out = vcat(SVector(0.0, 0.1, 0.3) .^ 2, SVector(0.7, 1.3, 2.0, 5.0) .- 0.25)
    testrobustifier(Huber2oKernel(0.5), costs, out)

    # Scaled Huber
    testrobustifier(Scaled(Huber2oKernel(0.5), 3.0), costs, 3 * out)

    # Geman-McClure
    out = costs * 0.25 ./ (costs .+ 0.25)
    testrobustifier(GemanMcclureKernel(0.5), costs, out)
end
