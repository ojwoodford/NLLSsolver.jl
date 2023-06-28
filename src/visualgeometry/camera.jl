using StaticArrays, LinearAlgebra, Static


function image2pixel(halfimsz, x)
    return x .* halfimsz[1] .+ halfimsz
end

function pixel2image(halfimsz, x)
    return (x .- halfimsz) ./ halfimsz[1]
end

function pixel2image(halfimsz, x, W)
    return ((x .- halfimsz) ./ halfimsz[1], W .* halfimsz[1])
end

function ideal2image(camera, x)
    return x .* focallength(camera) .+ cameracenter(camera)
end

function image2ideal(camera, x)
    return (x .- cameracenter(camera)) ./ focallength(camera)
end

function image2ideal(camera, x, W)
    return (x .- cameracenter(camera)) ./ focallength(camera), W .* focallength(camera)' 
end

struct SimpleCamera{T}
    f::ZeroToInfScalar{T}
    function SimpleCamera{T}(v::T) where T
        return new{T}(ZeroToInfScalar{T}(v))
    end
end
SimpleCamera(v::T) where T = SimpleCamera{T}(v::T)
nvars(::SimpleCamera) = static(1)
update(var::SimpleCamera, updatevec, start=1) = SimpleCamera(update(var.f, updatevec, start).val)
@inline cameracenter(::SimpleCamera) = 0
@inline focallength(camera::SimpleCamera) = camera.f.val


struct NoDistortionCamera{T}
    fx::ZeroToInfScalar{T}
    fy::ZeroToInfScalar{T}
    c::EuclideanVector{2, T}
end
NoDistortionCamera(fx::T, fy::T, cx::T, cy::T) where T = NoDistortionCamera(ZeroToInfScalar(fx), ZeroToInfScalar(fy), EuclideanVector(cx, cy))
NoDistortionCamera(f, c) = NoDistortionCamera(f[1], f[2], c[1], c[2])
nvars(::NoDistortionCamera) = static(4)
update(var::NoDistortionCamera, updatevec, start=1) = NoDistortionCamera(update(var.fx, updatevec, start), update(var.fy, updatevec, start+1), update(var.c, updatevec, start+2))
@inline cameracenter(camera::NoDistortionCamera) = camera.c
@inline focallength(camera::NoDistortionCamera{T}) where T = SVector{2, T}(camera.fx.val, camera.fy.val)


struct EULensDistortion{T}
    alpha::ZeroToOneScalar{T}
    beta::ZeroToInfScalar{T}
end
EULensDistortion(alpha::T, beta::T) where T = EULensDistortion{T}(ZeroToOneScalar{T}(alpha), ZeroToInfScalar{T}(beta))
nvars(::EULensDistortion) = static(2)
update(var::EULensDistortion, updatevec, start=1) = EULensDistortion(update(var.alpha, updatevec, start), update(var.beta, updatevec, start+1))
Base.eltype(::EULensDistortion{T}) where T = T

function ideal2distorted(lens::EULensDistortion, x)
    z = 1 / (1 + lens.alpha.val * (sqrt(lens.beta.val * (x' * x) + 1) - 1))
    return x * z
end

function distorted2ideal(lens::EULensDistortion, x)
    z = lens.beta.val * (x' * x)
    z = (1 + lens.alpha.val * (sqrt(1 + z * (1 - 2 * lens.alpha.val)) - 1)) / (1 - z * (lens.alpha.val ^ 2))
    return x * z
end

function distorted2ideal(lens::EULensDistortion, x, W)
    t = 1 - 2 * lens.alpha.val
    n = x' * x
    u = lens.beta.val * n
    v = 1 / (1 - u * (lens.alpha.val ^ 2))
    w = sqrt(t * u + 1)
    z = (1 + lens.alpha.val * (w - 1)) * v
    zi = 1 / z
    dzdn = zi * 2 * lens.alpha.val * lens.beta.val * v * (z * lens.alpha.val + t / (2 * w))
    return x * z, (W - (W * (x * x')) * (dzdn / (1 + n * dzdn))) * zi
end

struct ExtendedUnifiedCamera{T<:Number}
    sensor::NoDistortionCamera{T}
    lens::EULensDistortion{T}
end
ExtendedUnifiedCamera(f, c, a, b) = ExtendedUnifiedCamera(NoDistortionCamera(f[1], f[2], c[1], c[2]), EULensDistortion(a, b))
nvars(::ExtendedUnifiedCamera) = static(6)
update(var::ExtendedUnifiedCamera, updatevec, start=1) = ExtendedUnifiedCamera(update(var.sensor, updatevec, start), update(var.lens, updatevec, start+4))

function ideal2image(camera::ExtendedUnifiedCamera, x)
    return ideal2image(camera.sensor, ideal2distorted(camera.lens, x))
end

function image2ideal(camera::ExtendedUnifiedCamera, x)
    return distorted2ideal(camera.lens, image2ideal(camera.sensor, x))
end

function image2ideal(camera::ExtendedUnifiedCamera, x, W)
    y, W_ = image2ideal(camera.sensor, x, W)
    return distorted2ideal(camera.lens, y, W_)
end

struct BarrelDistortion{T}
    k1::T
    k2::T
end
nvars(::BarrelDistortion) = static(2)
update(var::BarrelDistortion, updatevec, start=1) = BarrelDistortion(var.k1 + updatevec[start], var.k2 + updatevec[start+1])
Base.eltype(::BarrelDistortion{T}) where T = T

function ideal2distorted(lens::BarrelDistortion, x)
    z = x' * x
    z = z * (lens.k1 + z * lens.k2) + 1
    return x * z
end

struct LensDistortResidual{T} <: AbstractResidual
    rlinear::T
    rdistort::T
end
ndeps(::LensDistortResidual) = static(1) # The residual depends on one variable
nres(::LensDistortResidual) = 1 # The residual vector has length one
varindices(::LensDistortResidual) = SVector(1)
computeresidual(residual::LensDistortResidual, lens::EULensDistortion) = SVector(residual.rdistort - ideal2distorted(lens, residual.rlinear))
getvars(::LensDistortResidual{T}, vars::Vector{LT}) where {T, LT} = (vars[1]::LT,)
Base.eltype(::LensDistortResidual{T}) where T = T

function convertlens(tolens, fromlens, halfimsz)
    # Create an optimization problem to convert the lens distortion
    @assert eltype(tolens)<:AbstractFloat
    problem = NLLSsolver.NLLSProblem(typeof(tolens), LensDistortResidual{eltype(tolens)})
    NLLSsolver.addvariable!(problem, tolens)
    for x in LinRange(0., convert(Float64, halfimsz), 100)
        NLLSsolver.addresidual!(problem, LensDistortResidual(x, ideal2distorted(fromlens, x)))
    end
    # Optimize
    NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.dogleg))
    # Return the lens
    return problem.variables[1]
end
