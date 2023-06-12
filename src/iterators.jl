function update!(to::Vector, from::Vector, linsystem::MultiVariateLS, step)
    # Update each variable
    @inbounds for (i, j) in enumerate(linsystem.blockindices)
        if j != 0
            to[i] = update(from[i], step, linsystem.soloffsets[j])
        end
    end
end

function update!(to::Vector, from::Vector, linsystem::UniVariateLS, step)
    # Update one variable
    to[linsystem.varindex] = update(from[linsystem.varindex], step)
end

# Iterators assume that the linear problem has been constructed

# Newton optimization
struct NewtonData
end

function iterate!(::NewtonData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    # Compute the step
    data.timesolver += @elapsed begin
        jacobian, residual = getjacres(data.linsystem)
        data.step .= -linearsolve(jacobian, residual, options)
    end
    data.linearsolvers += 1
    # Update the new variables
    update!(problem.varnext, problem.variables, data.linsystem, data.step)
    # Return the cost
    data.timecost += @elapsed cost_ = cost(problem.residuals, problem.varnext)
    data.costcomputations += 1
    return cost_
end

# Dogleg optimization
mutable struct DoglegData
    trustradius::Float64

    function DoglegData()
        return new(0.0)
    end
end

function iterate!(doglegdata::DoglegData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    hessian, gradient = gethessgrad(data.linsystem)
    data.timesolver += @elapsed begin
        # Compute the Cauchy step
        gnorm2 = gradient' * gradient
        a = gnorm2 / ((gradient' * hessian) * gradient + floatmin(eltype(gradient)))
        cauchy = -a * gradient
        alpha2 = a * a * gnorm2
        alpha = sqrt(alpha2)
        if doglegdata.trustradius == 0
            # Make first step the Cauchy point
            doglegdata.trustradius = alpha
        end
        if alpha < doglegdata.trustradius
            # Compute the Newton step
            data.step .= -symmetricsolve(hessian, gradient, options)
            beta = norm(data.step)
            data.linearsolvers += 1
        end
    end
    cost_ = data.bestcost
    while true
        # Determine the step
        if !(alpha < doglegdata.trustradius)
            # Along first leg
            data.step .= (doglegdata.trustradius / alpha) * cauchy
            linear_approx = doglegdata.trustradius * (2 * alpha - doglegdata.trustradius) / (2 * a)
        else
            # Along second leg
            if beta <= doglegdata.trustradius
                # Do the full Newton step
                linear_approx = cost_
            else
                # Find the point along the Cauchy -> Newton line on the trust
                # region circumference
                data.step .-= cauchy
                sq_leg = data.step' * data.step
                c = cauchy' * data.step
                trsq = doglegdata.trustradius * doglegdata.trustradius - alpha2
                step = sqrt(c * c + sq_leg * trsq)
                if c <= 0
                    step = (-c + step) / sq_leg
                else
                    step = trsq / (c + step)
                end
                data.step .*= step
                data.step .+= cauchy
                linear_approx = 0.5 * (a * (1 - step) ^ 2 * gnorm2) + step * (2 - step) * cost_
            end
        end
        # Update the new variables
        update!(problem.varnext, problem.variables, data.linsystem, data.step)
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, problem.varnext)
        data.costcomputations += 1
        # Update trust region radius
        mu = (data.bestcost - cost_) / linear_approx
        if mu > 0.75
            doglegdata.trustradius = max(doglegdata.trustradius, 3 * norm(data.step))
        elseif mu < 0.25
            doglegdata.trustradius *= 0.5
        end
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Return the cost
            return cost_
        end
    end
end

# Levenberg-Marquardt optimization
mutable struct LevMarData
    lambda::Float64

    function LevMarData()
        return new(1.0)
    end
end

function iterate!(levmardata::LevMarData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    @assert levmardata.lambda >= 0.
    hessian, gradient = gethessgrad(data.linsystem)
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        uniformscaling!(hessian, levmardata.lambda - lastlambda)
        lastlambda = levmardata.lambda
        # Solve the linear system
        data.timesolver += @elapsed data.step .= -symmetricsolve(hessian, gradient, options)
        data.linearsolvers += 1
        # Update the new variables
        update!(problem.varnext, problem.variables, data.linsystem, data.step)
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, problem.varnext)
        data.costcomputations += 1
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Success (or convergence) - update lambda
            uniformscaling!(hessian, -lastlambda)
            step_quality = (cost_ - data.bestcost) / (((data.step' * hessian) * 0.5 + gradient') * data.step)
            levmardata.lambda *= step_quality < 1.966 ? 1 - (step_quality - 1) ^ 3 : 0.1
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        levmardata.lambda *= mu;
        mu *= 2.;
    end
end



# Levenberg-Marquardt optimization with Schur complement
mutable struct LevMarSchurData
    lambda::Float64

    function LevMarSchurData()
        return new(1.0)
    end
end

function iterate!(levmardata::LevMarSchurData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        uniformscaling!(data.linsystem.a, levmardata.lambda - lastlambda)
        lastlambda = levmardata.lambda
        # Solve the linear system
        data.timesolver += @elapsed begin
            # Compute the reduced system
            # Solve the reduced system
            data.step .= -symmetricsolve(hessian, gradient, options)
            # Back substitute to find the other variables

        end
        data.linearsolvers += 1
        # Update the new variables
        update!(problem.varnext, problem.variables, data.linsystem, data.step)
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, problem.varnext)
        data.costcomputations += 1
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Success (or convergence) - update lambda
            uniformscaling!(hessian, -lastlambda)
            step_quality = (cost_ - data.bestcost) / (((data.step' * hessian) * 0.5 + gradient') * data.step)
            levmardata.lambda *= step_quality < 1.966 ? 1 - (step_quality - 1) ^ 3 : 0.1
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        levmardata.lambda *= mu;
        mu *= 2.;
    end
end


# Levenberg-Marquardt optimization
mutable struct VarProData
    lambda::Float64

    function VarProData()
        return new(1.0)
    end
end

function iterate!(varprodata::VarProData, data, problem::NLLSProblem, options::NLLSOptions)::Float64
    @assert varprodata.lambda >= 0.
    # Compute the reduced system
    hessian, gradient = gethessgrad(data.linsystem)
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        uniformscaling!(hessian, varprodata.lambda - lastlambda)
        lastlambda = varprodata.lambda
        # Solve the linear system
        data.timesolver += @elapsed data.step .= -symmetricsolve(hessian, gradient, options)
        data.linearsolvers += 1
        # Update the reduced variables
        update!(problem.varnext, problem.variables, data.linsystem, data.step)
        # Optimize the other variables
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, problem.varnext)
        data.costcomputations += 1
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Success (or convergence) - update lambda
            varprodata.lambda *= 0.333
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        varprodata.lambda *= mu;
        mu *= 2.;
    end
end
