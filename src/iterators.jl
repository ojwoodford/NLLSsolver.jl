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
        return "gradient descent"
    else
        return "unknown iterator"
    end
end

# Newton optimization (undamped-Hessian form)
struct NewtonData
end
construct(::StaticSymbol{:newton}, problem, ls) = NewtonData()
reset!(nd::NewtonData) = nothing

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

    function DoglegData(::NLLSProblem, ls)
        return new{typeof(getx(ls))}(0.0, similar(getx(ls)))
    end
end
construct(::StaticSymbol{:dogleg}, problem, data) = DoglegData(problem, data)
reset!(dd::DoglegData) = dd.trustradius = 0.0

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
            beta = fast_norm(getx(data.linsystem))
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
            doglegdata.trustradius = max(doglegdata.trustradius, 3 * fast_norm(getx(data.linsystem)))
        elseif !(mu >= 0.125) # a NaN cost (mu NaN) shrinks the region like a poor step
            doglegdata.trustradius *= 0.5
        end
        # Check for exit
        if cost_ <= data.bestcost || (maximum(abs, getx(data.linsystem))::Float64 < options.dstep)
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
abstract type LevMarData end

# Both hold a second linear system, trial, into which the first trial step of an iteration computes the cost,
# gradient and Hessian, rather than the cost alone. When the step is accepted, the optimizer swaps it in
# (swaptrial!) rather than recomputing every residual at the new point.
mutable struct LevenbergData{L} <: LevMarData
    lambda::Float64
    trial::L
    trialready::Bool # Whether trial holds the linear system at the point last returned by iterate!
end
construct(::StaticSymbol{:levenbergmarquardt}, problem, ls) = LevenbergData(0.0, copy_structure(ls), false)
reset!(lmd::LevenbergData) = (lmd.trialready = false; lmd.lambda = 0.0)

function updatescaling!(lmd::LevenbergData, hessian)
    if lmd.lambda == 0
        m = zero(eltype(hessian))
        for i in axes(hessian, 1)
            @inbounds m = max(m, abs(hessian[i,i]))
        end
        lmd.lambda = m * 1e-6
    end
end

dampenhessian!(hessian, ::LevenbergData, dlambda) = uniformscaling!(hessian, dlambda)

mutable struct MarquardtData{T, L} <: LevMarData
    lambda::Float64
    scaling::T
    trial::L
    trialready::Bool
end
construct(::StaticSymbol{:levenbergmarquardt_dtd}, problem, ls) = MarquardtData(0.0, zero(getx(ls)), copy_structure(ls), false)
reset!(lmd::MarquardtData) = (fill!(lmd.scaling, 0.0); lmd.trialready = false; lmd.lambda = 0.0)

# Update the running maximum D'D of the Hessian diagonal
function updatescaling!(lmd::MarquardtData, hessian)
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    m = zero(eltype(hessian))
    for i in axes(hessian, 1)
        @inbounds d = lmd.scaling[i] = max(lmd.scaling[i], abs(hessian[i,i]))
        m = max(m, d)
    end
    lmd.scaling .= max.(lmd.scaling, m * 1e-8)
end

function dampenhessian!(hessian, lmd::MarquardtData, dlambda)
    for i in axes(hessian, 1)
        @inbounds hessian[i,i] += dlambda * lmd.scaling[i]
    end
end

# x' D x, for the damping matrix D
dampingnorm(::LevenbergData, x) = fast_dot(x, x)
function dampingnorm(lmd::MarquardtData, x)
    total = 0.0
    @inbounds for i in eachindex(x)
        total += lmd.scaling[i] * x[i] * x[i]
    end
    return total
end

# The trial linear system, if iterate! left it holding the linear system at the point it returned
hastrial(iteratedata) = false
hastrial(lmd::LevMarData) = lmd.trialready

# Make the trial linear system the current one, keeping the current one's blocks for the next trial
function swaptrial!(lmd::LevMarData, linsystem)
    trial = sharex(lmd.trial, linsystem)
    lmd.trial = linsystem
    lmd.trialready = false
    return trial
end

@inline function iterate!(levmardata::LevMarData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    @assert levmardata.lambda >= 0.
    hessian, gradient = gethessgrad(data.linsystem)
    updatescaling!(levmardata, hessian)
    lastlambda = 0.
    mu = 2.
    levmardata.trialready = false
    firsttrial = true
    while true
        # Dampen the Hessian
        dampenhessian!(hessian, levmardata, levmardata.lambda - lastlambda)
        lastlambda = levmardata.lambda
        # Solve the linear system
        data.solves += @elapsed_ns negate!(solve!(data.linsystem))
        # Update the new variables
        update!(problem.varnext, problem.variables, data.linsystem)
        # Compute the cost. The first trial of an iteration is usually accepted, so compute the gradient and
        # Hessian there too, unless the model's predicted decrease, -g'x - x'Hx/2 = (lambda x'Dx - g'x)/2
        # given (H + lambda D) x = -g, is too small for the optimization to continue
        step = getx(data.linsystem)
        if firsttrial && 0.5 * (levmardata.lambda * dampingnorm(levmardata, step) - fast_dot(gradient, step)) >= data.bestcost * options.reldcost
            levmardata.trial = sharex(levmardata.trial, data.linsystem)
            data.gradients += @elapsed_ns cost_ = costgradhess!(levmardata.trial, problem.varnext, problem.costs)
            levmardata.trialready = true
        else
            data.costs += @elapsed_ns cost_ = cost(problem.varnext, problem.costs, options.numthreads)
        end
        firsttrial = false
        # Check for exit
        # Accept only a cost that is not greater than the best: a NaN cost (a step past the domain of the
        # residual) is a failed step, like a higher one
        if cost_ <= data.bestcost || maximum(abs, getx(data.linsystem))::Float64 < options.dstep
            # Success (or convergence) - update lambda
            dampenhessian!(hessian, levmardata, -lastlambda)
            stepquality = (cost_ - data.bestcost) / (0.5 * fast_bAb(hessian, getx(data.linsystem)) + fast_dot(gradient, getx(data.linsystem)))
            levmardata.lambda *= stepquality < 0.983 ? 1 - (2 * stepquality - 1) ^ 3 : 0.1
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        levmardata.trialready = false
        levmardata.lambda *= mu
        mu *= 2.
    end
end

printoutcallback(cost, problem, data, iteratedata::LevMarData) = printoutcallback(cost, data, 1.0/iteratedata.lambda)

# Gradient descent optimization
mutable struct GradientDescentData
    stepsize::Float64
end
construct(::StaticSymbol{:gradientdescent}, problem, data) = GradientDescentData(1.0)
reset!(gdd::GradientDescentData) = gdd.stepsize = 1.0

@inline function iterate!(gddata::GradientDescentData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    gradient = getgrad(data.linsystem)
    # Evaluate the current step size
    getx(data.linsystem) .= -gradient * gddata.stepsize
    update!(problem.varnext, problem.variables, data.linsystem)
    data.costs += @elapsed_ns costc = cost(problem.varnext, problem.costs, options.numthreads)
    # Iterate until we find a lower cost
    while !(costc <= data.bestcost)
        if isfinite(costc)
            # Compute the expected cost
            coststep = getx(data.linsystem)' * gradient
            costdiff = data.bestcost + coststep - costc
            # Compute the optimal step size assuming quadratic fit
            gddata.stepsize *= 0.5 * coststep / costdiff
        else
            # A non-finite cost (a step past the domain of the residual): halve the step
            gddata.stepsize *= 0.5
        end
        # Evaluate the new step size
        getx(data.linsystem) .= -gradient * gddata.stepsize
        update!(problem.varnext, problem.variables, data.linsystem)
        data.costs += @elapsed_ns costc = cost(problem.varnext, problem.costs, options.numthreads)
    end
    gddata.stepsize *= 2
    return costc
end

printoutcallback(cost, problem, data, iteratedata::GradientDescentData) = printoutcallback(cost, data, iteratedata.stepsize)
