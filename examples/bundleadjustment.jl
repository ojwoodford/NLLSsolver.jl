using VisualGeometryDatasets, StaticArrays, BenchmarkTools
import NLLSsolver, GLMakie
export optimizeBALproblem

# Description of BAL image, and function to transform a landmark from world coordinates to pixel coordinates
struct BALImage{T}
    pose::NLLSsolver.EffPose3D{T}
    sensor::NLLSsolver.SimpleCamera{T}
    lens::NLLSsolver.BarrelDistortion{T}
end
NLLSsolver.nvars(::BALImage) = 9
function NLLSsolver.update(var::BALImage, updatevec, start=1)
    return BALImage(update(var.pose, updatevec, start),
                    update(var.sensor, updatevec, start+6),
                    update(var.lens, updatevec, start+7))
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
NLLSsolver.nvars(::BALResidual) = 2 # Residual depends on 2 variables
NLLSsolver.varindices(res::BALResidual) = res.varind
function NLLSsolver.getvars(res::BALResidual{T}, vars::Vector) where T
    return vars[res.varind[1]]::BALImage{T}, vars[res.varind[2]]::NLLSsolver.Point3D{T}
end
function NLLSsolver.computeresidual(res::BALResidual, im::BALImage, X::NLLSsolver.Point3D)
    return transform(im, X) - res.measurement
end
const balrobustifier = NLLSsolver.HuberKernel(2., 4., 1., 1.)
function NLLSsolver.robustkernel(::BALResidual)
    return balrobustifier
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

function filterBALlandmarks(data, landmarks)
    # Find the residuals associated with the landmarks
    deleteat!(data.measurements, findall(broadcast(m -> m.landmark ∉ landmarks, data.measurements)))
    # Delete the unused cameras and landmarks
    cameras = trues(length(data.cameras))
    landmarks = trues(length(data.landmarks))
    for measurement in data.measurements
        cameras[measurement.camera] = false
        landmarks[measurement.landmark] = false
    end
    deleteat!(data.cameras, findall(cameras))
    deleteat!(data.landmarks, findall(landmarks))
    # Remap the indices
    cameras = cumsum(.!cameras)
    landmarks = cumsum(.!landmarks)
    for ind in eachindex(data.measurements)
        data.measurements[ind] = VisualGeometryDatasets.BALMeasurement(
                                    data.measurements[ind].x, data.measurements[ind].y,
                                    cameras[data.measurements[ind].camera], landmarks[data.measurements[ind].landmark])
    end
    return data
end


# Function to optimize a BAL problem
function optimizeBALproblem(name="problem-16-22106")
    # Create the problem
    data = filterBALlandmarks(loadbaldataset(name), 1)
    show(data)
    problem = makeBALproblem(data)
    NLLSsolver.fixvars!(problem, range(1, length(problem.variables)-1))
    # Compute the mean cost per measurement
    println("   Mean cost per measurement: ", NLLSsolver.cost(problem)/length(data.measurements))
    # Optimize the cost
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator="Dogleg"))
    # Compute the new mean cost per measurement
    println("   Mean cost per measurement: ", minimum(result.costs)/length(data.measurements))
    # Print out the timings

    # Plot the costs
    fig = Figure()
    ax = Axis(fig[1, 1])
    GLMakie.lines!(ax, result.costs)
    fig
end

val = optimizeBALproblem()