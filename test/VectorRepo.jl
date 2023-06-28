using NLLSsolver, Test

function fillrepo(vr, floats, ints)
    # Fill the repo with the data
    Nf = length(floats)
    Ni = length(ints)
    append!(vr, floats) 
    append!(vr, ints)
    push!(vr, 0.0)
    push!(vr, 0)
    get!(vr, Char)

    # Test that the lengths and values are as expected
    floats = get(vr, Float64)
    @test length(floats) == Nf+1 && floats[end] == 0
    ints = get(vr, Int)
    @test length(ints) == Ni+1 && ints[end] == 0
end

@testset "VectorRepo.jl" begin
    # Generate random data
    floats = rand(10) * 100
    ints = convert(Vector{Int}, ceil.(floats))
    total = sum(floats) + sum(ints)

    # Construct repos and test the sum reduction
    # Any container
    vr1 = NLLSsolver.VectorRepo()
    @test sum(i->2i, vr1) == 0.0
    fillrepo(vr1, floats, ints)
    @test isapprox(sum(i->π*i, vr1), total * π)

    # Union container
    vr2 = NLLSsolver.VectorRepo{Union{Float64, Int, Char}}()
    @test sum(i->2i, vr2) == 0.0
    fillrepo(vr2, floats, ints)
    @test isapprox(sum(i->π*i, vr2), total * π)
end
