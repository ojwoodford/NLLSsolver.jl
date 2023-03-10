export SimpleCamera, NoDistortionCamera, ExtendedUnifiedCamera, BarrelDistortion
export ideal2image, image2ideal, pixel2image, image2pixel, update, nvars, nres, varindices, getvars, computeresidual, barrel2eulens
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

struct SimpleCamera{T}
    f::ZeroToInfScalar{T}
    function SimpleCamera{T}(v::T) where T
        return new{T}(ZeroToInfScalar{T}(v))
    end
end
SimpleCamera(v::T) where T = SimpleCamera{T}(v::T)
nvars(@objtype(SimpleCamera)) = 1
function update(var::SimpleCamera, updatevec, start=1)
    return SimpleCamera(update(var.f, updatevec, start).val)
end
@inline cameracenter(::SimpleCamera) = 0
@inline focallength(camera::SimpleCamera) = camera.f.val


struct NoDistortionCamera{T}
    fx::ZeroToInfScalar{T}
    fy::ZeroToInfScalar{T}
    c::EuclideanVector{2, T}
end
NoDistortionCamera(fx::T, fy::T, cx::T, cy::T) where T = NoDistortionCamera(ZeroToInfScalar(fx), ZeroToInfScalar(fy), EuclideanVector(cx, cy))
NoDistortionCamera(f, c) = NoDistortionCamera(f[1], f[2], c[1], c[2])
nvars(@objtype(NoDistortionCamera)) = 4
function update(var::NoDistortionCamera, updatevec, start=1)
    return NoDistortionCamera(update(var.fx, updatevec, start), update(var.fy, updatevec, start+1), update(var.c, updatevec, start+2))
end
@inline cameracenter(camera::NoDistortionCamera) = camera.c
@inline focallength(camera::NoDistortionCamera) = SVector(camera.fx.val, camera.fy.val)


struct EULensDistortion{T}
    alpha::ZeroToOneScalar{T}
    beta::ZeroToInfScalar{T}
end
EULensDistortion(alpha::T, beta::T) where T = EULensDistortion(ZeroToOneScalar(alpha), ZeroToInfScalar(beta))
nvars(@objtype(EULensDistortion)) = 2
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
    u = lens.beta.val * n
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
ExtendedUnifiedCamera(f, c, a, b) = ExtendedUnifiedCamera(NoDistortionCamera(f[1], f[2], c[1], c[2]), EULensDistortion(a, b))
nvars(@objtype(ExtendedUnifiedCamera)) = 6
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
nvars(@objtype(BarrelDistortion)) = 2
function update(var::BarrelDistortion, updatevec, start=1)
    return BarrelDistortion(var.k1 + updatevec[start], var.k2 + updatevec[start+1])
end

function ideal2distorted(lens::BarrelDistortion, x)
    z = x' * x
    z = z * (lens.k1 + z * lens.k2) + 1
    return x * z
end

struct LensDistortResidual{T} <: AbstractResidual
    rlinear::T
    rdistort::T
end
nvars(@objtype(LensDistortResidual)) = 1 # The residual depends on one variable
nres(@objtype(LensDistortResidual)) = 1 # The residual vector has length one
varindices(::LensDistortResidual) = [1]
function computeresidual(residual::LensDistortResidual, lens::EULensDistortion)
    return SVector(residual.rdistort - ideal2distorted(lens, residual.rlinear))
end
function getvars(::LensDistortResidual{T}, vars::Vector) where T
    return (vars[1]::EULensDistortion{T},)
end

function barrel2eulens(k1, k2, halfimsz)
    # Create an optimization problem to convert the lens distortion
    problem = NLLSsolver.NLLSProblem{EULensDistortion}()
    NLLSsolver.addvariable!(problem, EULensDistortion(0.01, 1000.))
    for x in range(0, halfimsz, 100)
        x2 = x ^ 2
        NLLSsolver.addresidual!(problem, LensDistortResidual(x, x * (1 + x2 * (k1 + x2 * k2))))
    end
    # Optimize
    NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(dcost=1.e-6, iterator=NLLSsolver.dogleg, storetrajectory=true))
    # Return the lens
    return problem.variables[1]
end