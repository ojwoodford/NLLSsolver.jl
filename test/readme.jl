using Test, NLLSsolver

@testset "readme.jl" begin
    readme = read(joinpath(pkgdir(NLLSsolver), "README.md"), String)
    blocks = [m.captures[1] for m in eachmatch(r"```julia\n(.*?)```"s, readme)]
    # The translation guide's worked example: the "before" block (NLLSsolver 5 API) and the "after" block
    # (NLLSModel + SecondOrderOptimization directly).
    @test length(blocks) == 2
    costs = Float64[]
    for (i, blk) in enumerate(blocks)
        mod = Module(Symbol(:ReadmeBlock, i))
        push!(costs, Float64(Base.include_string(mod, blk)))
    end
    @test costs[1] < 1e-6                              # the NLLSsolver 5 API converges to the minimum
    @test isapprox(costs[1], costs[2]; atol=1e-10)     # both routes reach the same cost

    # "Behaviour that changed", each checked by running:
    @test NLLSOptions().reldcost == 1e-12              # was 1e-15 in NLLSsolver 5
    @test NLLSOptions().maxstall == 3                  # maxstall/stalltol replace maxfails
    @test isdefined(NLLSsolver, :TERMINATION_STALLED)
    @test isdefined(NLLSsolver, :isconverged)
    @test !isdefined(NLLSsolver, :updatevarcostmap!)   # removed
    @test_throws MethodError ContaminatedGaussian(0.5, 5.0, 0.6)  # needs the residual-length argument
end
