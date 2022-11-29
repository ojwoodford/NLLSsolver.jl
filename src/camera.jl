export SimpleCamera, NoDistortionCamera, ExtendedUnifiedCamera, BALCamera
export ideal2image, image2ideal, pixel2image, image2pixel, update, nvars
using StaticArrays, LinearAlgebra


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


const SimpleCamera{T} = PositiveScalar{T}
@inline cameracenter(::SimpleCamera) = 0
@inline focallength(camera::SimpleCamera) = camera.val


struct NoDistortionCamera{T}
    fx::ZeroToInfScalar{T}
    fy::ZeroToInfScalar{T}
    c::EuclideanVector{2, T}
end
nvars(var::NoDistortionCamera) = 4
function update(var::NoDistortionCamera, updatevec, start=1)
    return NoDistortionCamera(update(var.fx, updatevec, start), update(var.fy, updatevec, start+1), update(var.c, updatevec, start+2))
end
@inline cameracenter(camera::NoDistortionCamera) = camera.c
@inline focallength(camera::NoDistortionCamera) = SVector(camera.fx.val, camera.fy.val)


struct EULensDistortion{T}
    alpha::ZeroToOneScalar{T}
    beta::ZeroToInfScalar{T}
end
nvars(::EULensDistortion) = 2
function update(var::EULensDistortion, updatevec, start=1)
    return EULensDistortion(update(var.alpha, updatevec, start), update(var.beta, updatevec, start+1))
end

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
    u = lens.beta * n
    v = 1 - u * (lens.alpha.val ^ 2)
    w = sqrt(t * u + 1)
    z = (1 + lens.alpha.val * (w - 1)) / v
    z_ = (lens.alpha.val * lens.beta.val) * ((t * v / w) + (2 * lens.alpha.val) * (lens.alpha.val * (w - 1) + 1)) / v
    return x * z, W * (I / z - (x * x') * (z_ / (z * (z + n * z_))))
end

struct ExtendedUnifiedCamera{T<:Number}
    sensor::NoDistortionCamera{T}
    lens::EULensDistortion{T}
end
ExtendedUnifiedCamera(f, c, a, b) = ExtendedUnifiedCamera(NoDistortionCamera(f[1], f[2], c), EULensDistortion(a, b))
nvars(::ExtendedUnifiedCamera) = 6
function update(var::ExtendedUnifiedCamera, updatevec, start=1)
    return ExtendedUnifiedCamera(update(var.sensor, updatevec, start),
                                 update(var.lens, updatevec, start+4))
end

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
nvars(::BarrelDistortion) = 2
function update(var::BarrelDistortion, updatevec, start=1)
    return BarrelDistortion(var.k1 + updatevec[start], var.k2 + updatevec[start+1])
end

function ideal2distorted(lens::BarrelDistortion, x)
    z = x' * x
    z = z * (lens.k1 + z * lens.k2) + 1
    return x * z
end

struct BALCamera{T}
    sensor::SimpleCamera{T}
    lens::BarrelDistortion{T}
end
BALCamera(f, k1, k2) = BALCamera(SimpleCamera(f), BarrelDistortion(k1, k2))
nvars(var::BALCamera) = 3
function update(var::BALCamera, updatevec, start=1)
    return BALCamera(update(var.sensor, updatevec, start),
                     update(var.lens, updatevec, start+1))
end

function ideal2image(camera::BALCamera, x)
    return ideal2image(camera.sensor, ideal2distorted(camera.lens, x))
end

struct LensDistortResidual{T} <: AbstractResidual
    rlinear::T
    rdistort::T
end
function computeresidual(residual::LensDistortResidual, lens::EULensDistortion)
    return residual.rdistort - ideal2distorted(lens, residual.rlinear)
end
function varindices(::LensDistortResidual)
    return 1
end

function makeeucamera(coeffs)
    # Compute the sensor values
    #halfimsz = 

    # Create an optimization problem to convert the lens distortion
    #residuals = 

    

end