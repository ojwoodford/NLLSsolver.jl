using NLLSsolver, LinearAlgebra, Test

@testset "geometry.jl" begin
    # Test utility functions
    @test NLLSsolver.rodrigues(0., 0., 0.) == I
    @test isapprox(NLLSsolver.rodrigues(Float64(pi), 0., 0.), Diagonal(SVector(1., -1., -1.)))
    O = NLLSsolver.proj2orthonormal(randn(5, 5))
    @test isapprox(O' * O, I)

    # Test points
    updatevec = zeros(3)
    point = NLLSsolver.update(NLLSsolver.Point3D(normalize(SVector(0.9, 1.1, -1.0))), updatevec)
    pointu = NLLSsolver.update(NLLSsolver.UnitVec3D(NLLSsolver.getvec(point)), updatevec)
    @test project(point) == SVector(-0.9, -1.1)
    @test project(pointu) == SVector(-0.9, -1.1)
    @test point - pointu == Point3D()

    # Test rotations
    updatevec = randn(3)
    rotr = update(Rotation3DR(), updatevec)
    rotl = update(Rotation3DL(), -updatevec)
    @test isapprox(NLLSsolver.getvec((rotl * rotr) * point), NLLSsolver.getvec(point))
    @test isapprox(NLLSsolver.getvec((NLLSsolver.inverse(rotl) * NLLSsolver.inverse(rotr)) * point), NLLSsolver.getvec(point))

    # Test poses
    updatevec = zeros(6)
    pose = NLLSsolver.update(NLLSsolver.Pose3D(rotr, point), updatevec)
    poseu = NLLSsolver.update(NLLSsolver.UnitPose3D(rotr, pointu), updatevec)
    @test isapprox(NLLSsolver.getvec(pose * (NLLSsolver.inverse(poseu) * point)), NLLSsolver.getvec(point))
    @test isapprox(NLLSsolver.getvec(poseu * (NLLSsolver.inverse(pose) * point)), NLLSsolver.getvec(point))
    posee = NLLSsolver.update(NLLSsolver.EffPose3D(pose), updatevec)
    @test isapprox(NLLSsolver.getvec(NLLSsolver.inverse(pose) * (posee * point)), NLLSsolver.getvec(point))
    @test isapprox(NLLSsolver.getvec(pose * (NLLSsolver.inverse(posee) * point)), NLLSsolver.getvec(point))
end