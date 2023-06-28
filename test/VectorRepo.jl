using NLLSsolver, Test

function fillrepo(vr, floats, ints)
    # Fill the repo with the data
    Nf = length(floats)
    Ni = length(ints)
    append!(vr, floats) 
    append!(vr, ints)
    push!(vr, 0.0)
    push!(vr, 0)

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

    # Construct repos and test the interface
    vr1 = NLLSsolver.VectorRepo()
    fillrepo(vr1, floats, ints)
    vr2 = NLLSsolver.VectorRepo{Union{Float64, Int}}()
    fillrepo(vr2, floats, ints)

    # Test the sum reduction
    @test isapprox(sum(i->2i, vr1)*2, sum(i->4.0*i, vr2))
end
