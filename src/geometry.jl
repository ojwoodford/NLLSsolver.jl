export rodrigues, project, epipolarerror, proj2orthonormal
export Rotation3DR, Rotation3DL, Point3D, Pose3D, EffPose3D, UnitPose3D
using StaticArrays, LinearAlgebra
import nllssolver.AbstractVariable, nllssolver.SR

function rodrigues(x::T, y::T, z::T) where T<:Number
    if x == 0 && y == 0 && z == 0
        # Short cut for derivatives at identity
        return SMatrix{3, 3, T}(T(1), z, -y, -z, T(1), x, y, -x, T(1))
    end
    theta2 = x * x + y * y + z * z
    cosf = T(0.5)
    sinc = T(1)
    if theta2 > T(2.23e-16)
        theta = sqrt(theta2)
        sinc, cosf = sincos(theta)
        cosf -= 1
        sinc /= theta
        cosf /= -theta2
    end
    a = x * y * cosf
    b = sinc * z
    c = x * z * cosf
    d = sinc * y
    e = y * z * cosf
    f = sinc * x
    return SMatrix{3, 3, T}((x * x - theta2) * cosf + 1, a + b, c - d,
                            a - b, (y * y - theta2) * cosf + 1, e + f,
                            c + d, e - f, (z * z - theta2) * cosf + 1)
end

function proj2orthonormal(M)
    s = svd(M);
    return s.U * s.V';
end

function epipolarerror(RX, T, x, W=Nothing)
    Tp = proj(T)
    xe = x - Tp
    RXp = proj(RX) - Tp
    RXp = SVector(RXp[2], -RXp[1])
    if W != Nothing
        RXp = W' \ RXp
        xe = W * xe
    end
    return dot(xe, normalize(RXp))
end

abstract type AbstractPoint3D <: AbstractVariable end
struct Point3D{T<:Real} <: AbstractPoint3D
    v::SVector{3, T}
end
Point3D(x, y, z) = Point3D(SVector{3}(x, y, z))
Point3D() = Point3D(SVector{3}(0., 0., 0.))
nvars(::Point3D) = 3
function update(var::Point3D, updatevec, start=1)
    return Point3D(var.v + updatevec[SR(0, 2) .+ start])
end

function project(x::Point3D)
    return SVector(x.v[1], x.v[2]) ./ x.v[3]
end


abstract type AbstractRotation3D end
struct Rotation3DR{T<:Real} <: AbstractRotation3D
    m::SMatrix{3, 3, T, 9}
end
Rotation3DR(x, y, z) = Rotation3DR(rodrigues(x, y, z))
Rotation3DR() = Rotation3DR(SMatrix{3, 3, Float64}(1., 0., 0., 0., 1., 0., 0., 0., 1.))
nvars(::Rotation3DR) = 3
function update(var::Rotation3DR, updatevec, start=1)
    return var * rodrigues(updatevec[start], updatevec[start+1], updatevec[start+2])
end
transform(rota::Rotation3DR, rotb::Rotation3DR) = Rotation3DR(rota.m * rotb.m)
transform(rot::Rotation3DR, point::Point3D) = Point3D(rot.m * point.v)

struct Rotation3DL{T<:Real} <: AbstractRotation3D
    m::SMatrix{3, 3, T, 9}
end
Rotation3DL(x, y, z) = Rotation3DL(rodrigues(x, y, z))
Rotation3DL() = Rotation3DL(SMatrix{3, 3, Float64}(1., 0., 0., 0., 1., 0., 0., 0., 1.))
nvars(::Rotation3DL) = 3
function update(var::Rotation3DL, updatevec, start=1)
    return Rotation3DL(updatevec[start], updatevec[start+1], updatevec[start+2]) * var
end
transform(rota::Rotation3DL, rotb::Rotation3DL) = Rotation3DL(rota.m * rotb.m)
transform(rot::Rotation3DL, point::Point3D) = Point3D(rot.m * point.v)


abstract type AbstractPose3D end
struct Pose3D{T<:Real} <: AbstractPose3D
    rot::Rotation3DR{T}
    trans::Point3D{T}
end
Pose3D(rx, ry, rz, tx, ty, tz) = Pose3D(Rotation3DR(rx, ry, rz), Point3D(tx, ty, tz))
Pose3D() = Pose3D(Rotation3DR(), Point3D())
nvars(::Pose3D) = 6
function update(var::Pose3D, updatevec, start=1)
    return Pose3D(update(var.rot, updatevec, start), update(var.trans, updatevec, start+3))
end
function inverse(var::Pose3D)
    return Pose3D(var.rot', var.rot' * -var.trans)
end
transform(pose::Pose3D, point::Point3D) = Point3D(pose.rot.m * point.v + pose.trans.v)

struct EffPose3D{T<:Real} <: AbstractPose3D
    rot::Rotation3DL{T}
    camcenter::Point3D{T}
end
EffPose3D(rx, ry, rz, cx, cy, cz) = EffPose3D(Rotation3DL(rx, ry, rz), Point3D(cx, cy, cz))
EffPose3D() = Pose3D(Rotation3DL(), Point3D())
nvars(::EffPose3D) = 6
function update(var::EffPose3D, updatevec, start=1)
    return EffPose3D(update(var.rot, updatevec, start), update(var.camcenter, updatevec, start+3))
end
function inverse(var::EffPose3D)
    return EffPose3D(var.rot', var.rot * -var.camcenter)
end
transform(pose::EffPose3D, point::Point3D) = Point3D(pose.rot.m * (point.v - pose.camcenter.v))


struct UnitPose3D{T<:Real} <: AbstractVariable
    rot::Rotation3DL{T}
    trans::Rotation3DL{T}
end
UnitPose3D() = Pose3D(Rotation3DL(), Rotation3DL())
UnitPose3D((rx, ry, rz, tx, ty, tz)) = Pose3D(Rotation3DL(rx, ry, rz), Rotation3DL()) # Normalize translation and initialize y & z axes
nvars(::UnitPose3D) = 5
function update(var::UnitPose3D, updatevec, start=1)
    return UnitPose3D(update(var.rot, updatevec, start), update(var.trans, SVector(0, updatevec[start+3], updatevec[start+4])))
end
function inverse(var::UnitPose3D)
    return Pose3D(var.rot', var.rot' * -var.trans.m[:,1])
end
transform(pose::UnitPose3D, point::Point3D) = Point3D(pose.rot.m * point.v + pose.trans.m[:,1])

# Overload multiplication with transformation
Base.:*(rota::AbstractRotation3D, rotb::AbstractRotation3D) = transform(rota, rotb)
Base.:*(rot::AbstractRotation3D, point::AbstractPoint3D) = transform(rot, point)
Base.:*(posea::AbstractPose3D, poseb::AbstractPose3D) = transform(posea, poseb)
Base.:*(pose::AbstractPose3D, point::AbstractPoint3D) = transform(pose, point)
