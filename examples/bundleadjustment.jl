using VisualGeometryDatasets, StaticArrays, Static
import NLLSsolver
export optimizeBALproblem

# Description of BAL image, and function to transform a landmark from world coordinates to pixel coordinates
struct BALImage{T}
    pose::NLLSsolver.EffPose3D{T}   # Extrinsic parameters (rotation & translation)
    sensor::NLLSsolver.SimpleCamera{T} # Intrinsic sensor parameters (just a single focal length)
    lens::NLLSsolver.BarrelDistortion{T} # Intrinsic lens parameters (k1 & k2 barrel distortion)
end
NLLSsolver.nvars(::BALImage) = static(9) # 9 DoF in total for the image (6 extrinsic, 3 intrinsic)
function NLLSsolver.update(var::BALImage, updatevec, start=1)
    return BALImage(NLLSsolver.update(var.pose, updatevec, start),
                    NLLSsolver.update(var.sensor, updatevec, start+6),
                    NLLSsolver.update(var.lens, updatevec, start+7))
end
function BALImage(rx::T, ry::T, rz::T, tx::T, ty::T, tz::T, f::T, k1::T, k2::T) where T<:Real
    return BALImage{T}(NLLSsolver.EffPose3D(NLLSsolver.Pose3D(rx, ry, rz, tx, ty, tz)), 
                       NLLSsolver.SimpleCamera(f), 
                       NLLSsolver.BarrelDistortion(k1, k2))
end
BALImage(v) = BALImage(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9])
transform(im::BALImage, X::NLLSsolver.Point3D) = NLLSsolver.ideal2image(im.sensor, NLLSsolver.ideal2distorted(im.lens, -NLLSsolver.project(im.pose * X)))

# Residual that defines the reprojection error of a BAL measurement
struct BALResidual{T} <: NLLSsolver.AbstractResidual
    measurement::SVector{2, T} # Landmark image coordinates (x, y)
    varind::SVector{2, Int} # Variable indices for the residual (image, landmark)
end
BALResidual(m, v) = BALResidual(SVector{2}(m[1], m[2]), SVector{2, Int}(v[1], v[2]))
NLLSsolver.ndeps(::BALResidual) = static(2) # Residual depends on 2 variables
NLLSsolver.nres(::BALResidual) = static(2) # Residual vector has length 2
NLLSsolver.varindices(res::BALResidual) = res.varind
NLLSsolver.getvars(res::BALResidual{T}, vars::Vector) where T = vars[res.varind[1]]::BALImage{T}, vars[res.varind[2]]::NLLSsolver.Point3D{T}
NLLSsolver.computeresidual(res::BALResidual, im::BALImage, X::NLLSsolver.Point3D) = transform(im, X) - res.measurement
const balrobustifier = NLLSsolver.HuberKernel(2.)
NLLSsolver.robustkernel(::BALResidual) = balrobustifier
Base.eltype(::BALResidual{T}) where T = T

function NLLSsolver.computeresjac(::StaticInt{3}, residual::BALResidual, im, point)
    # Exploit the parameterization to make the jacobian computation more efficient
    res, jac = NLLSsolver.computeresjac(static(1), residual, im, point)
    return res, hcat(jac, -view(jac, :, NLLSsolver.SR(4, 6)))
end

# Function to create a NLLSsolver problem from a BAL dataset
function makeBALproblem(data)
    # Create the problem
    problem = NLLSsolver.NLLSProblem(Union{BALImage{Float64}, NLLSsolver.Point3D{Float64}}, BALResidual{Float64})

    # Add the cameras
    for cam in data.cameras
        NLLSsolver.addvariable!(problem, BALImage(cam))
    end
    numcameras = length(data.cameras)
    # Add the landmarks
    for lm in data.landmarks
        NLLSsolver.addvariable!(problem, NLLSsolver.Point3D(lm[1], lm[2], lm[3]))
    end

    # Add the residuals
    for meas in data.measurements
        NLLSsolver.addresidual!(problem, BALResidual(SVector(meas.x, meas.y), SVector(meas.camera, meas.landmark + numcameras)))
    end

    # Return the optimization problem
    return problem
end

# Function to optimize a BAL problem
function optimizeBALproblem(name)
    # Create the problem
    data = loadbaldataset(name)
    show(data)
    problem = makeBALproblem(data)
    # Compute the mean cost per measurement
    println("   Start mean cost per measurement: ", NLLSsolver.cost(problem)/length(data.measurements))
    # Optimize the cost
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.levenbergmarquardt))
    # Compute the new mean cost per measurement
    println("   End mean cost per measurement: ", result.bestcost/length(data.measurements))
    # Print out the solver summary
    show(result)
end

optimizeBALproblem("problem-16-22106")
