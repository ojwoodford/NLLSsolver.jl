using Test, StaticArrays, Static, NLLSsolver

# Points (3D) on lines through the origin (2D direction angles), with a union variable type: each point is
# tied to its line, and every point to a noisy measurement of itself
struct OnLine <: AbstractResidual
    varind::SVector{2, Int}
end
NLLSsolver.ndeps(::OnLine) = static(2)
NLLSsolver.nres(::OnLine) = static(3)
NLLSsolver.varindices(r::OnLine) = r.varind
NLLSsolver.getvars(r::OnLine, vars::Vector) = (vars[r.varind[1]]::SVector{3, Float64}, vars[r.varind[2]]::SVector{2, Float64})
function NLLSsolver.computeresidual(::OnLine, X, a)
    d = SVector(cos(a[1]) * cos(a[2]), sin(a[1]) * cos(a[2]), sin(a[2]))
    return X - (d' * X) * d
end
Base.eltype(::OnLine) = Float64
struct Measured <: AbstractResidual
    X::SVector{3, Float64}
    varind::Int
end
NLLSsolver.ndeps(::Measured) = static(1)
NLLSsolver.nres(::Measured) = static(3)
NLLSsolver.varindices(r::Measured) = SVector(r.varind)
NLLSsolver.getvars(r::Measured, vars::Vector) = (vars[r.varind]::SVector{3, Float64},)
NLLSsolver.computeresidual(r::Measured, X) = X - r.X
Base.eltype(::Measured) = Float64

function linesproblem(n)
    problem = NLLSProblem(Union{SVector{3, Float64}, SVector{2, Float64}}, Union{OnLine, Measured})
    for k in 1:n
        line = addvariable!(problem, SVector(0.1 * k, 0.05 * k))
        for j in 1:5
            X = SVector(cos(k) + j, sin(k) - j, 0.3 * j * k)
            point = addvariable!(problem, X + SVector(0.5, -0.5, 0.2))
            addcost!(problem, Measured(X, point))
            addcost!(problem, OnLine(SVector(point, line)))
        end
    end
    return problem
end

@testset "copyvars.jl" begin
    # Variables of isbits types (and unions of them) are copied without per-variable allocations
    @test NLLSsolver.isbitsvartype(SVector{3, Float64})
    @test NLLSsolver.isbitsvartype(Union{SVector{3, Float64}, SVector{2, Float64}, Float64})
    @test !NLLSsolver.isbitsvartype(Vector{Float64})
    @test !NLLSsolver.isbitsvartype(Union{SVector{3, Float64}, Vector{Float64}})
    @test !NLLSsolver.isbitsvartype(Any)
    vars = linesproblem(100).variables
    varscopy = NLLSsolver.copyvars(vars)
    @test varscopy == vars && varscopy !== vars
    NLLSsolver.copyvars(vars)
    @test (@allocations NLLSsolver.copyvars(vars)) <= 4 # (deepcopy: one per variable and more)
    dynvars = [[1.0, 2.0], [3.0]]
    dyncopy = NLLSsolver.copyvars(dynvars)
    @test dyncopy == dynvars && dyncopy[1] !== dynvars[1]

    # A first optimization allocates less than one allocation per variable (it copied the variables to
    # varnext, and on the first cost increase to varbest, with deepcopy), and the result is unchanged
    options = NLLSOptions(iterator=:levenbergmarquardt_dtd)
    optimize!(linesproblem(4), options)
    problem = linesproblem(200)
    startcost = cost(problem)
    result = optimize!(problem, options)
    @test result.bestcost < startcost
    @test cost(problem) == result.bestcost
    @test result.initallocations + result.optimizeallocations < length(problem.variables)
end
