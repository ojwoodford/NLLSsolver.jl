using SparseArrays

negate!(x) = @.(x = -x)

# Iterators assume that the linear problem has been constructed

# Newton optimization (undamped-Hessian form)
struct NewtonData
end
NewtonData(::NLLSProblem, ::NLLSInternal) = NewtonData()
reset!(nd::NewtonData, ::NLLSProblem, ::NLLSInternal) = nd

function iterate!(::NewtonData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    # Compute the step
    gethessian(data.linsystem)
    data.timesolver += @elapsed_ns negate!(solve!(data.linsystem, options))
    data.linearsolvers += 1
    # Update the new variables
    update!(problem.varnext, problem.variables, data.linsystem)
    # Return the cost
    data.timecost += @elapsed_ns cost_ = cost(problem.varnext, problem.costs)
    data.costcomputations += 1
    return cost_
end

# Dogleg optimization
mutable struct DoglegData{T}
    trustradius::Float64
    cauchy::T

    function DoglegData(::NLLSProblem, data::NLLSInternal)
        return new{typeof(data.linsystem.x)}(0.0, similar(data.linsystem.x))
    end
end
function reset!(dd::DoglegData{T}, ::NLLSProblem, data::NLLSInternal) where T<:Vector
    dd.trustradius = 0.0
    resize!(dd.cauchy, length(data.linsystem.x))
    return
end
function reset!(dd::DoglegData{T}, ::NLLSProblem, data::NLLSInternal) where T<:StaticVector
    dd.trustradius = 0.0
    return
end

function iterate!(doglegdata::DoglegData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    hessian, gradient = gethessgrad(data.linsystem)
    data.timesolver += @elapsed_ns begin
        # Compute the Cauchy step
        gnorm2 = gradient' * gradient
        a = gnorm2 / (fast_bAb(hessian, gradient) + floatmin(eltype(gradient)))
        doglegdata.cauchy .= -a * gradient
        alpha2 = a * a * gnorm2
        alpha = sqrt(alpha2)
        if doglegdata.trustradius == 0
            # Make first step the Cauchy point
            doglegdata.trustradius = alpha
        end
        if alpha < doglegdata.trustradius
            # Compute the Newton step
            negate!(solve!(data.linsystem, options))
            beta = norm(data.linsystem.x)
            data.linearsolvers += 1
        end
    end
    cost_ = data.bestcost
    while true
        # Determine the step
        if !(alpha < doglegdata.trustradius)
            # Along first leg
            data.linsystem.x .= (doglegdata.trustradius / alpha) * doglegdata.cauchy
            linear_approx = doglegdata.trustradius * (2 * alpha - doglegdata.trustradius) / (2 * a)
        else
            # Along second leg
            if beta <= doglegdata.trustradius
                # Do the full Newton step
                linear_approx = cost_
            else
                # Find the point along the Cauchy -> Newton line on the trust
                # region circumference
                data.linsystem.x .-= doglegdata.cauchy
                sq_leg = data.linsystem.x' * data.linsystem.x
                c = doglegdata.cauchy' * data.linsystem.x
                trsq = doglegdata.trustradius * doglegdata.trustradius - alpha2
                step = sqrt(c * c + sq_leg * trsq)
                if c <= 0
                    step = (-c + step) / sq_leg
                else
                    step = trsq / (c + step)
                end
                data.linsystem.x .*= step
                data.linsystem.x .+= doglegdata.cauchy
                linear_approx = 0.5 * (a * (1 - step) ^ 2 * gnorm2) + step * (2 - step) * cost_
            end
        end
        # Update the new variables
        update!(problem.varnext, problem.variables, data.linsystem)
        # Compute the cost
        data.timecost += @elapsed_ns cost_ = cost(problem.varnext, problem.costs)
        data.costcomputations += 1
        # Update trust region radius
        mu = (data.bestcost - cost_) / linear_approx
        if mu > 0.375
            doglegdata.trustradius = max(doglegdata.trustradius, 3 * norm(data.linsystem.x))
        elseif mu < 0.125
            doglegdata.trustradius *= 0.5
        end
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.linsystem.x) < options.dstep)
            # Return the cost
            return cost_
        end
    end
end

printoutcallback(cost, problem, data, iteratedata::DoglegData) = printoutcallback(cost, data, iteratedata.trustradius)

# Levenberg-Marquardt optimization
mutable struct LevMarData
    lambda::Float64

    function LevMarData(::NLLSProblem, ::NLLSInternal)
        return new(0.0)
    end
end
reset!(lmd::LevMarData, ::NLLSProblem, ::NLLSInternal) = lmd.lambda = 0.0

function initlambda(hessian)
    m = zero(eltype(hessian))
    for i in indices(hessian, 1)
        @inbounds m = max(m, abs(hessian[i,i]))
    end
    return m * 1e-6
end

function iterate!(levmardata::LevMarData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    @assert levmardata.lambda >= 0.
    hessian, gradient = gethessgrad(data.linsystem)
    if levmardata.lambda == 0
        levmardata.lambda = initlambda(hessian)
    end
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        uniformscaling!(hessian, levmardata.lambda - lastlambda)
        lastlambda = levmardata.lambda
        # Solve the linear system
        data.timesolver += @elapsed_ns negate!(solve!(data.linsystem, options))
        data.linearsolvers += 1
        # Update the new variables
        update!(problem.varnext, problem.variables, data.linsystem)
        # Compute the cost
        data.timecost += @elapsed_ns cost_ = cost(problem.varnext, problem.costs)
        data.costcomputations += 1
        # Check for exit
        if !(cost_ > data.bestcost) || maximum(abs, data.linsystem.x) < options.dstep
            # Success (or convergence) - update lambda
            uniformscaling!(hessian, -lastlambda)
            stepquality = (cost_ - data.bestcost) / (0.5 * fast_bAb(hessian, data.linsystem.x) + dot(gradient, data.linsystem.x))
            levmardata.lambda *= stepquality < 0.983 ? 1 - (2 * stepquality - 1) ^ 3 : 0.1
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        levmardata.lambda *= mu
        mu *= 2.
    end
end

# Gradient descent optimization
mutable struct GradientDescentData
    stepsize::Float64

    function GradientDescentData(::NLLSProblem, ::NLLSInternal)
        return new(1.0)
    end
end
reset!(gdd::GradientDescentData, ::NLLSProblem, ::NLLSInternal) = gdd.stepsize = 1.0

function iterate!(gddata::GradientDescentData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    gradient = getgrad(data.linsystem)
    # Evaluate the current step size
    data.linsystem.x .= -gradient * gddata.stepsize
    update!(problem.varnext, problem.variables, data.linsystem)
    data.timecost += @elapsed_ns costc = cost(problem.varnext, problem.costs)
    data.costcomputations += 1
    # Iterate until we find a lower cost
    while costc > data.bestcost
        # Compute the expected cost
        coststep = data.linsystem.x' * gradient
        costdiff = data.bestcost + coststep - costc
        # Compute the optimal step size assuming quadratic fit
        gddata.stepsize *= 0.5 * coststep / costdiff
        # Evaluate the new step size
        data.linsystem.x .= -gradient * gddata.stepsize
        update!(problem.varnext, problem.variables, data.linsystem)
        data.timecost += @elapsed_ns costc = cost(problem.varnext, problem.costs)
        data.costcomputations += 1
    end
    gddata.stepsize *= 2
    return costc
end

printoutcallback(cost, problem, data, iteratedata::GradientDescentData) = printoutcallback(cost, data, iteratedata.stepsize)

mutable struct VarProData{CT}
    lambda::Float64
    firstschurvar::Int
    singlesoptions::NLLSOptions
    marginalizedls::MultiVariateLS
    costs::CostStruct{CT}
    costindices::SparseMatrixCSC{Bool, Int}
    schurindices::Vector{Int}

    function VarProData(problem::NLLSProblem{VT, CT}, linsystem::MultiVariateLSsparse, firstschurvar::Int, singlesoptions::NLLSOptions) where {VT, CT}
        indices = firstschurvar:length(problem.variables)
        reordercostsforschur!(problem, indices)
        costindices = sparse(getvarcostmap(problem)')
        indices = sortperm([dynamic(nvars(problem.variables[i])) for i in indices]) .+ (firstschurvar - 1)
        return new{CT}(0.0, firstschurvar, singlesoptions, constructcrop(linsystem, firstschurvar), CostStruct{CT}(), costindices, indices)
    end
end
reset!(vpd::VarProData, ::NLLSProblem, ::NLLSInternal) = vpd.lambda = 0.0

function iterate!(varprodata::VarProData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    @assert varprodata.lambda >= 0.
    # Store the current schur variables
    @inbounds problem.varbest[varprodata.firstschurvar:end] .= view(problem.variables, varprodata.firstschurvar, lastindex(problem.variables))
    # Compute the reduced system
    initcrop!(varprodata.marginalizedls, data.linsystem, varprodata.firstschurvar)
    marginalize!(varprodata.marginalizedls, data.linsystem, varprodata.firstschurvar)
    hessian = gethessian(varprodata.marginalizedls)
    if varprodata.lambda == 0
        varprodata.lambda = initlambda(hessian)
    end
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        uniformscaling!(hessian, varprodata.lambda - lastlambda)
        lastlambda = varprodata.lambda
        # Solve the linear system
        data.timesolver += @elapsed_ns negate!(solve!(varprodata.marginalizedls, options))
        data.linearsolvers += 1
        # Update the reduced variables
        update!(problem.varnext, problem.variables, varprodata.marginalizedls, varprodata.marginalizedls.x)
        # Optimize the other variables
        problem.varnext, problem.variables = problem.variables, problem.varnext
        varprodata.costs, problem.costs = problem.costs, varprodata.costs
        first = 1
        while first <= length(varprodata.schurindices)
            # Optimize all variables of the same size
            first = setupiterator(optimizesinglesinternal!, problem, varprodata.singlesoptions, NLLSInternal(UInt(1), nvars(problem.variables[varprodata.schurindices[first]]), UInt64(0)), varprodata.costs, varprodata.costindices, varprodata.schurindices, first)
        end
        varprodata.costs, problem.costs = problem.costs, varprodata.costs
        problem.varnext, problem.variables = problem.variables, problem.varnext
        # Compute the cost
        data.timecost += @elapsed_ns cost_ = cost(problem.varnext, problem.costs)
        data.costcomputations += 1
        # Check for exit
        stepmax = maximum(abs, varprodata.marginalizedls.x)
        if !(cost_ > data.bestcost) || stepmax < options.dstep
            # Success (or convergence) - update lambda
            varprodata.lambda *= 0.333
            data.linsystem.x[1] = stepmax
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        varprodata.lambda *= mu
        mu *= 2.
        # Reset the schur variables
        @inbounds problem.variables[varprodata.firstschurvar:end] .= view(problem.varbest, varprodata.firstschurvar, lastindex(problem.varbest))
    end
end

printoutcallback(cost, problem, data, iteratedata::Union{LevMarData, VarProData}) = printoutcallback(cost, data, 1.0/iteratedata.lambda)

