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
    total = sum(floats) + sum(ints)
    halftotal = sum(floats[1:5]) + sum(ints[1:5])
    rangefun(v::Vector{T}) where T = T <: Char ? T[] : view(v, 1:5)
    indicesfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, [1, 2, 3, 4, 5])
    bitvecfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, 1:11 .<= 5)
    boolvecfun(v::Vector{T}) where T = T <: Char ? T[] : view(v, map(x->x<=5, 1:11))

    # Construct repos and test the sum reduction
    # Any container
    vr1 = NLLSsolver.VectorRepo()
    @test sum(i->2i, vr1; init=0.0) == 0.0
    fillrepo(vr1, floats, ints)
    @test sum(i->π*i, vr1; init=0.0) ≈ total * π
    vec = values(vr1)
    @test length(vec) == 2 && any(Base.Fix2(isa, Vector{Float64}), vec) && any(Base.Fix2(isa, Vector{Int}), vec)

    # Test subset reductions
    @test NLLSsolver.sumsubset(Float64, vr1, rangefun) == halftotal
    @test NLLSsolver.sumsubset(Float64, vr1, indicesfun) == halftotal
    @test NLLSsolver.sumsubset(Float64, vr1, bitvecfun) == halftotal
    @test NLLSsolver.sumsubset(Float64, vr1, boolvecfun) ≈ halftotal

    # Union container
    vr2 = NLLSsolver.VectorRepo{Union{Float64, Int, Char}}()
    keytup = keys(vr2)
    @test isa(keytup, Tuple) && length(keytup) == 3
    @test any(keytup .== Float64) && any(keytup .== Int) && any(keytup .== Char)
    @test sum(i->2i, vr2; init=0.0) == 0.0
    fillrepo(vr2, floats, ints)
    @test sum(i->π*i, vr2; init=0.0) ≈ total * π
    valuetup = values(vr2)
    @test isa(valuetup, Tuple) && length(valuetup) == 3
    @test any(Base.Fix2(isa, Vector{Float64}), valuetup) && any(Base.Fix2(isa, Vector{Int}), valuetup) && any(Base.Fix2(isa, Vector{Char}), valuetup)

    # Test subset reductions
    @test NLLSsolver.sumsubset(Float64, vr2, rangefun) == halftotal
    @test NLLSsolver.sumsubset(Float64, vr2, indicesfun) == halftotal
    @test NLLSsolver.sumsubset(Float64, vr2, bitvecfun) == halftotal
    @test NLLSsolver.sumsubset(Float64, vr2, boolvecfun) ≈ halftotal
end
