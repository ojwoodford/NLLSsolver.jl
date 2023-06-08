using VisualGeometryDatasets, StaticArrays
import NLLSsolver
export optimizeBALproblem

# Description of BAL image, and function to transform a landmark from world coordinates to pixel coordinates
struct BALImage{T}
    pose::NLLSsolver.EffPose3D{T}
    sensor::NLLSsolver.SimpleCamera{T}
    lens::NLLSsolver.BarrelDistortion{T}
end
NLLSsolver.nvars(NLLSsolver.@objtype(BALImage{T})) where T = 9
function NLLSsolver.update(var::BALImage, updatevec, start=1)
    return BALImage(NLLSsolver.update(var.pose, updatevec, start),
                    NLLSsolver.update(var.sensor, updatevec, start+6),
                    NLLSsolver.update(var.lens, updatevec, start+7))
end
function BALImage(rx::T, ry::T, rz::T, tx::T, ty::T, tz::T, f::T, k1::T, k2::T) where T<:Real
    R = NLLSsolver.Rotation3DL(rx, ry, rz)
    return BALImage{T}(NLLSsolver.EffPose3D(R, NLLSsolver.Point3D(R.m' * -SVector(tx, ty, tz))), 
                       NLLSsolver.SimpleCamera(f), 
                       NLLSsolver.BarrelDistortion(k1, k2))
end
BALImage(v) = BALImage(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9])
function transform(im::BALImage, X::NLLSsolver.Point3D)
    return NLLSsolver.ideal2image(im.sensor, NLLSsolver.ideal2distorted(im.lens, -NLLSsolver.project(im.pose * X)))
end

# Residual that defines the reprojection error of a BAL measurement
struct BALResidual{T} <: NLLSsolver.AbstractResidual
    measurement::SVector{2, T}
    varind::SVector{2, Int}
end
BALResidual(m, v) = BALResidual(SVector{2}(m[1], m[2]), SVector{2, Int}(v[1], v[2]))
NLLSsolver.nvars(NLLSsolver.@objtype(BALResidual)) = 2 # Residual depends on 2 variables
NLLSsolver.nres(NLLSsolver.@objtype(BALResidual)) = 2 # Residual vector has length 2
NLLSsolver.varindices(res::BALResidual) = res.varind
NLLSsolver.getvars(res::BALResidual{T}, vars::Vector) where T = vars[res.varind[1]]::BALImage{T}, vars[res.varind[2]]::NLLSsolver.Point3D{T}
NLLSsolver.computeresidual(res::BALResidual, im::BALImage, X::NLLSsolver.Point3D) = transform(im, X) - res.measurement
const balrobustifier = NLLSsolver.HuberKernel(2., 4., 1.)
NLLSsolver.robustkernel(::BALResidual) = balrobustifier
Base.eltype(::BALResidual{T}) where T = T

function NLLSsolver.computeresjac(::Val{3}, residual::BALResidual{T}, vars...) where T
    # Exploit the parameterization to make the jacobian computation more efficient
    res, jac = NLLSsolver.extract_resjac(NLLSsolver.computeresidual(residual, NLLSsolver.dualvars(vars, Val(1), T)...))
    return res, hcat(jac, -view(jac, :, NLLSsolver.SR(4, 6)))
end

# Function to create a NLLSsolver problem from a BAL dataset
function makeBALproblem(data)
    # Create the problem
    problem = NLLSsolver.NLLSProblem{Union{BALImage{Float64}, NLLSsolver.Point3D{Float64}}}()

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
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.levenbergmarquardt, storecosts=true))
    # Compute the new mean cost per measurement
    println("   End mean cost per measurement: ", minimum(result.costs)/length(data.measurements))
    # Print out the solver summary
    show(result)
end

optimizeBALproblem("problem-16-22106")
