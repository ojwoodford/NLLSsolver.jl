using SparseArrays

negate!(x) = @.(x = -x)

function Base.String(iterator::NLLSIterator) 
    if iterator == newton
        return "Newton"
    elseif iterator == levenbergmarquardt
        return "Levenberg-Marquardt"
    elseif iterator == levenbergmarquardt_dtd
        return "Levenberg-Marquardt (D'D scaling)"
    elseif iterator == dogleg
        return "Dogleg"
    elseif iterator == gradientdescent
        return "Gradient descent"
    else
        return "Unknown iterator"
    end
end

# Newton optimization (undamped-Hessian form)
struct NewtonData
end
NewtonData(::NLLSProblem, ::NLLSInternal) = NewtonData()
construct(::StaticSymbol{:newton}, problem, data) = NewtonData(problem, data)
reset!(nd::NewtonData, ::NLLSProblem, ::NLLSInternal) = nothing

@inline function iterate!(::NewtonData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    # Compute the step
    gethessian(data.linsystem)
    data.solves += @elapsed_ns negate!(solve!(data.linsystem))
    # Update the new variables
    update!(problem.varnext, problem.variables, data.linsystem)
    # Return -Inf to signal the cost wasn't computed
    return -Inf
end

# Dogleg optimization
mutable struct DoglegData{T}
    trustradius::Float64
    cauchy::T

    function DoglegData(::NLLSProblem, data::NLLSInternal)
        return new{typeof(getx(data.linsystem))}(0.0, similar(getx(data.linsystem)))
    end
end
construct(::StaticSymbol{:dogleg}, problem, data) = DoglegData(problem, data)
reset!(dd::DoglegData, ::NLLSProblem, data::NLLSInternal) = dd.trustradius = 0.0

@inline function iterate!(doglegdata::DoglegData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    hessian, gradient = gethessgrad(data.linsystem)
    data.solves += @elapsed_ns begin
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
            negate!(solve!(data.linsystem))
            beta = norm(getx(data.linsystem))
        end
    end
    cost_ = data.bestcost
    while true
        # Determine the step
        if !(alpha < doglegdata.trustradius)
            # Along first leg
            getx(data.linsystem) .= (doglegdata.trustradius / alpha) * doglegdata.cauchy
            linear_approx = doglegdata.trustradius * (2 * alpha - doglegdata.trustradius) / (2 * a)
        else
            # Along second leg
            if beta <= doglegdata.trustradius
                # Do the full Newton step
                linear_approx = cost_
            else
                # Find the point along the Cauchy -> Newton line on the trust
                # region circumference
                getx(data.linsystem) .-= doglegdata.cauchy
                sq_leg = getx(data.linsystem)' * getx(data.linsystem)
                c = doglegdata.cauchy' * getx(data.linsystem)
                trsq = doglegdata.trustradius * doglegdata.trustradius - alpha2
                step = sqrt(c * c + sq_leg * trsq)
                if c <= 0
                    step = (-c + step) / sq_leg
                else
                    step = trsq / (c + step)
                end
                getx(data.linsystem) .*= step
                getx(data.linsystem) .+= doglegdata.cauchy
                linear_approx = 0.5 * (a * (1 - step) ^ 2 * gnorm2) + step * (2 - step) * cost_
            end
        end
        # Update the new variables
        update!(problem.varnext, problem.variables, data.linsystem)
        # Compute the cost
        data.costs += @elapsed_ns cost_ = cost(problem.varnext, problem.costs, options.numthreads)
        # Update trust region radius
        mu = (data.bestcost - cost_) / linear_approx
        if mu > 0.375
            doglegdata.trustradius = max(doglegdata.trustradius, 3 * norm(getx(data.linsystem)))
        elseif mu < 0.125
            doglegdata.trustradius *= 0.5
        end
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, getx(data.linsystem))::Float64 < options.dstep)
            # Return the cost
            return cost_
        end
    end
end

printoutcallback(cost, problem, data, iteratedata::DoglegData) = printoutcallback(cost, data, iteratedata.trustradius)

# Levenberg-Marquardt optimization
# :levenbergmarquardt damps the Hessian with lambda * I.
# :levenbergmarquardt_dtd damps it with lambda * D'D (Marquardt scaling), where D'D is the running maximum of diag(J'J)
# over the iterations so far, floored at 1e-8 times its largest entry (Transtrum & Sethna, arXiv:1201.5885).
mutable struct LevMarData
    lambda::Float64
    dtd::Bool
    scaling::Vector{Float64} # D'D (dtd only)
    hdiag::Vector{Float64}   # Undamped diagonal of the current Hessian (dtd only)

    function LevMarData(::NLLSProblem, ::NLLSInternal, dtd::Bool=false)
        return new(0.0, dtd, Float64[], Float64[])
    end
end
construct(::StaticSymbol{:levenbergmarquardt}, problem, data) = LevMarData(problem, data)
construct(::StaticSymbol{:levenbergmarquardt_dtd}, problem, data) = LevMarData(problem, data, true)
reset!(lmd::LevMarData, ::NLLSProblem, ::NLLSInternal) = (empty!(lmd.scaling); lmd.lambda = 0.0)

# Store the undamped diagonal of the Hessian, and update the running maximum D'D
function updatescaling!(lmd::LevMarData, hessian)
    n = size(hessian, 1)
    if length(lmd.scaling) != n
        lmd.scaling = zeros(n)
        lmd.hdiag = zeros(n)
    end
    m = 0.0
    for i in 1:n
        @inbounds h = hessian[i,i]
        @inbounds lmd.hdiag[i] = h
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], abs(h))
        m = max(m, d)
    end
    m *= 1e-8
    for i in 1:n
        @inbounds lmd.scaling[i] = max(lmd.scaling[i], m)
    end
end

# Set the Hessian diagonal to hdiag + lambda * scaling (lambda = 0 restores the undamped Hessian exactly)
function dampdiagonal!(hessian, lmd::LevMarData, lambda)
    for i in 1:size(hessian, 1)
        @inbounds hessian[i,i] = lmd.hdiag[i] + lambda * lmd.scaling[i]
    end
end

function initlambda(hessian)
    m = zero(eltype(hessian))
    for i in indices(hessian, 1)
        @inbounds m = max(m, abs(hessian[i,i]))
    end
    return m * 1e-6
end

@inline function iterate!(levmardata::LevMarData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    @assert levmardata.lambda >= 0.
    hessian, gradient = gethessgrad(data.linsystem)
    if levmardata.dtd
        updatescaling!(levmardata, hessian)
        if levmardata.lambda == 0
            levmardata.lambda = 1e-6
        end
    elseif levmardata.lambda == 0
        levmardata.lambda = initlambda(hessian)
    end
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        if levmardata.dtd
            dampdiagonal!(hessian, levmardata, levmardata.lambda)
        else
            uniformscaling!(hessian, levmardata.lambda - lastlambda)
        end
        lastlambda = levmardata.lambda
        # Solve the linear system
        data.solves += @elapsed_ns negate!(solve!(data.linsystem))
        # Update the new variables
        update!(problem.varnext, problem.variables, data.linsystem)
        # Compute the cost
        data.costs += @elapsed_ns cost_ = cost(problem.varnext, problem.costs, options.numthreads)
        # Check for exit
        if !(cost_ > data.bestcost) || maximum(abs, getx(data.linsystem))::Float64 < options.dstep
            # Success (or convergence) - update lambda
            if levmardata.dtd
                dampdiagonal!(hessian, levmardata, 0.0)
            else
                uniformscaling!(hessian, -lastlambda)
            end
            stepquality = (cost_ - data.bestcost) / (0.5 * fast_bAb(hessian, getx(data.linsystem)) + dot(gradient, getx(data.linsystem)))
            levmardata.lambda *= stepquality < 0.983 ? 1 - (2 * stepquality - 1) ^ 3 : 0.1
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        levmardata.lambda *= mu
        mu *= 2.
    end
end

printoutcallback(cost, problem, data, iteratedata::LevMarData) = printoutcallback(cost, data, 1.0/iteratedata.lambda)

# Gradient descent optimization
mutable struct GradientDescentData
    stepsize::Float64

    function GradientDescentData(::NLLSProblem, ::NLLSInternal)
        return new(1.0)
    end
end
construct(::StaticSymbol{:gradientdescent}, problem, data) = GradientDescentData(problem, data)
reset!(gdd::GradientDescentData, ::NLLSProblem, ::NLLSInternal) = gdd.stepsize = 1.0

@inline function iterate!(gddata::GradientDescentData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    gradient = getgrad(data.linsystem)
    # Evaluate the current step size
    getx(data.linsystem) .= -gradient * gddata.stepsize
    update!(problem.varnext, problem.variables, data.linsystem)
    data.costs += @elapsed_ns costc = cost(problem.varnext, problem.costs, options.numthreads)
    # Iterate until we find a lower cost
    while costc > data.bestcost
        # Compute the expected cost
        coststep = getx(data.linsystem)' * gradient
        costdiff = data.bestcost + coststep - costc
        # Compute the optimal step size assuming quadratic fit
        gddata.stepsize *= 0.5 * coststep / costdiff
        # Evaluate the new step size
        getx(data.linsystem) .= -gradient * gddata.stepsize
        update!(problem.varnext, problem.variables, data.linsystem)
        data.costs += @elapsed_ns costc = cost(problem.varnext, problem.costs, options.numthreads)
    end
    gddata.stepsize *= 2
    return costc
end

printoutcallback(cost, problem, data, iteratedata::GradientDescentData) = printoutcallback(cost, data, iteratedata.stepsize)
