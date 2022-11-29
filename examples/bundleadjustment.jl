using VisualGeometryDatasets, NLLSsolver
export optimizeBALproblem
export getvars, computeresidual, robustkernel, nvars, transform, makeBALproblem

struct BALImage{T}
    pose::EffPose3D{T}
    camera::BALCamera{T}
end
nvars(::BALImage) = 9
function update(var::BALImage, updatevec, start=1)
    return BALImage(update(var.pose, updatevec, start),
                    update(var.camera, updatevec, start+6))
end
function transform(im::BALImage, X::Point3D)
    return ideal2image(im.camera, -project(im.pose * X))
end
function makeBALImage(rx::T, ry::T, rz::T, tx::T, ty::T, tz::T, f::T, k1::T, k2::T) where T<:Real
    R = Rotation3DL(rx, ry, rz)
    return BALImage{T}(EffPose3D(R, Point3D(R.m' * -SVector(tx, ty, tz))), BALCamera(f, k1, k2))
end
makeBALImage(v) = makeBALImage(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9])

struct BALResidual{T} <: AbstractResidual
    measurement::SVector{2, T}
    varind::SVector{2, Int}
end
BALResidual(m, v) = BALResidual(SVector{2}(m[1], m[2]), SVector{2, Int}(v[1], v[2]))
nvars(::BALResidual) = 2 # Residual depends on 2 variables
reslen(::BALResidual) = 2 # The residual is a vector of length 2
function eltype(::BALResidual{T}) where T
    return T
end
function getvars(res::BALResidual{T}, vars::Vector) where T
    return vars[res.varind[1]]::BALImage{T}, vars[res.varind[2]]::Point3D{T}
end
function computeresidual(res::BALResidual, im::BALImage, X::Point3D)
    return transform(im, X) - res.measurement
end
const balrobustifier = HuberKernel(2., 4., 1., 1.)
function robustkernel(::BALResidual)
    return balrobustifier
end

function makeBALproblem(name)
    # Load the data
    data = loadbaldataset(name)

    # Create the problem
    problem = NLLSProblem{Float64}()

    # Add the cameras
    for cam in data.cameras
        addvariable!(problem, makeBALImage(cam))
    end
    numcameras = length(data.cameras)
    # Add the landmarks
    for lm in data.landmarks
        addvariable!(problem, Point3D(lm[1], lm[2], lm[3]))
    end

    # Add the residuals
    for meas in data.measurements
        addresidual!(problem::NLLSProblem, BALResidual(SVector(meas.x, meas.y), SVector(meas.camera, meas.landmark + numcameras)))
    end

    # Return the optimization problem
    return problem
end

function optimizeBALproblem(name="problem-16-22106")
    # Create the problem
    problem = makeBALproblem(name)
    # Compute the current RMS error

    # Optimize the cost

    # Compute the new RMS error

    # Print out the timings

    # Plot the costs

end