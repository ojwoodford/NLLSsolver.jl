using VisualGeometryDatasets, StaticArrays, Static, LinearAlgebra
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

    # Add the camera variable blocks
    for cam in data.cameras
        NLLSsolver.addvariable!(problem, BALImage(cam))
    end
    numcameras = length(data.cameras)
    # Add the landmark variable blocks
    for lm in data.landmarks
        NLLSsolver.addvariable!(problem, NLLSsolver.Point3D(lm[1], lm[2], lm[3]))
    end

    # Add the measurement cost blocks
    for meas in data.measurements
        NLLSsolver.addcost!(problem, BALResidual(SVector(meas.x, meas.y), SVector(meas.camera, meas.landmark + numcameras)))
    end

    # Return the optimization problem
    return problem
end

# Compute the Area Under Curve for reprojection errors, truncated at a given threshold
function computeauc(problem, threshold=10.0, residuals=problem.costs.data[BALResidual{Float64}])
    # Compute all the errors
    invthreshold = 1.0 / threshold
    errors = Vector{Float64}(undef, length(residuals)+1)
    ind = 1
    errors[ind] = 0.0
    for res in residuals
        ind += 1
        errors[ind] = norm(NLLSsolver.computeresidual(res, NLLSsolver.getvars(res, problem.variables)...)) * invthreshold
    end

    # Sort the errors and truncate, compute recall
    sort!(errors)
    if errors[end] > 1.0
        cutoff = findfirst(x -> x > 1.0, errors)
        recall = (cutoff - 1.0) / length(residuals)
        recallfinal = (1.0 - errors[cutoff-1]) / ((errors[cutoff] - errors[cutoff-1]) * length(residuals))
        errors[cutoff] = 1.0
        resize!(errors, cutoff)
        recall = vcat(range(0.0, recall, cutoff-1), recall + recallfinal)
    else
        recall = vcat(range(0.0, 1.0, length(errors)), 1.0)
        errors = vcat(errors, 1.0)
    end

    # Compute the AUC
    return 0.5 * sum(diff(errors) .* (recall[1:end-1] .+ recall[2:end]))
end

# Function to optimize a BAL problem
function optimizeBALproblem(name)
    # Create the problem
    t = @elapsed begin
        data = loadbaldataset(name)
        problem = makeBALproblem(data)
    end
    show(data)
    println("Data loading and problem construction took ", t, " seconds.")
    # Compute the starting AUC
    println("   Start AUC: ", computeauc(problem, 2.0))
    # Optimize the cost
    result = NLLSsolver.optimize!(problem, NLLSsolver.NLLSOptions(iterator=NLLSsolver.levenbergmarquardt, reldcost=1.0e-6))
    # Compute the final AUC
    println("   Final AUC: ", computeauc(problem, 2.0))
    # Print out the solver summary
    display(result)
end

optimizeBALproblem("problem-16-22106")
