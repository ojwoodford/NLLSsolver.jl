using NLLSsolver, Test, Static, StaticArrays, ForwardDiff, LinearAlgebra

struct TrivialResidual <: AbstractResidual end
NLLSsolver.ndeps(::TrivialResidual) = static(1)
NLLSsolver.nres(::TrivialResidual) = static(2)
NLLSsolver.varindices(::TrivialResidual) = 1
NLLSsolver.getvars(::TrivialResidual, vars::Vector) = (vars[1], )
NLLSsolver.computeresidual(::TrivialResidual, x) = SVector(x, x*x)
Base.eltype(::TrivialResidual) = Float64

function vec2skew(v, ind, N)
    M = MMatrix{N, N, eltype(v), N*N}(undef)
    for i = 1:N
        M[i,i] = 1.0
        for j = i+1:N
            M[i,j] = v[ind]
            M[j,i] = -v[ind]
            ind += 1
        end
    end
    return SMatrix(M)
end

function proj2rot(M)
    s = svd(M);
    M = s.U * s.V'
    if det(M) < 0.0
        M = s.U * diagm(convert(typeof(s.S), vcat(ones(length(s.S)-1), -1.0))) * s.V';
    end
    return M
end

struct Rotation{N, N2, T}
    R::SMatrix{N, N, T, N2}
end
Rotation(R::SMatrix{N, N, T, N2}) where {N, N2, T} = Rotation{N, N2, T}(R)
NLLSsolver.nvars(::Rotation{N, N2, T}) where {N, N2, T} = StaticInt((N*(N-1))/2)
proj2rot(M::SMatrix{N, N, T, N2}) where {N, T<:ForwardDiff.Dual, N2} = M
NLLSsolver.update(var::Rotation{N, N2, T}, updatevec, start=1) where {N, N2, T} = Rotation(var.R * proj2rot(vec2skew(updatevec, start, N)))

struct Procrustes{N} <: AbstractResidual
    A::SVector{N, Float64}
    B::SVector{N, Float64}
end
Procrustes(N) = Procrustes{N}(@SVector(randn(N)), @SVector(randn(N)))
NLLSsolver.ndeps(::Procrustes) = static(1)
NLLSsolver.nres(::Procrustes{N}) where N = static(N)
NLLSsolver.varindices(::Procrustes) = 1
NLLSsolver.getvars(res::Procrustes, vars::Vector) = (vars[1], )
NLLSsolver.computeresidual(res::Procrustes, var::Rotation) = res.A - var.R * res.B
Base.eltype(::Procrustes) = Float64

function compare_results(res, x)
    rs, Js = NLLSsolver.computeresjacstatic(static(1), res, (x,))
    rd, Jd = NLLSsolver.computeresjacdynamic(static(1), res, (x,))
    @test rs == rd
    @test Js == Jd
end

@testset "autodiff.jl" begin
    # Check that the outputs of computeresjacstatic and computeresjacdynamic are the same
    compare_results(TrivialResidual(), 1.0)
    compare_results(Procrustes(6), Rotation(proj2rot(@SMatrix(randn(6, 6)))))
end
