using SparseArrays, LinearSolve
import Printf.@printf
export optimize!, NLLSOptions, NLLSResult, NLLSIterator

@enum NLLSIterator gaussnewton levenbergmarquardt dogleg

struct NLLSOptions
    dcost::Float64
    dstep::Float64
    maxfails::Int
    maxiters::Int
    iterator::NLLSIterator
    linearsolver
    callback
    storetrajectory::Bool
end
function NLLSOptions(; maxiters=100, dcost=1.e-6, dstep=1.e-6, maxfails=3, iterator=gaussnewton, callback=(args...)->false, storetrajectory=false, linearsolver=nothing)
    NLLSOptions(dcost, dstep, maxfails, maxiters, iterator, linearsolver, callback, storetrajectory)
end

mutable struct NLLSInternal{VarTypes}
    variables::Vector{VarTypes}
    gradient::Vector{Float64}
    hessian::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int}}
    step::Vector{Float64}
    blockoffsets::Vector{UInt}
    bestcost::Float64
    lambda::Float64
    timecost::Float64
    timegradient::Float64
    timesolver::Float64
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int

    function NLLSInternal{VarTypes}(problem::NLLSProblem) where VarTypes
        # Compute the block offsets
        offsets = zeros(UInt, length(problem.variables))
        start = UInt(1)
        for index in eachindex(offsets)
            if problem.unfixed[index]
                @inbounds offsets[index] = start
                start += convert(UInt, nvars(problem.variables[index]))
            end
        end
        start -= 1
        # Initialize everything
        return new(Vector{VarTypes}(undef, length(offsets)), zeros(Float64, start), zeros(Float64, start, start), Vector{Float64}(undef, start), offsets, 0., 0., 0., 0., 0., 0, 0, 0)
    end
end

struct NLLSResult
    costs::Vector{Float64}
    trajectory::Vector{Vector{Float64}}
    timetotal::Float64
    timecost::Float64
    timegradient::Float64
    timesolver::Float64
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int
end

function Base.show(io::IO, x::NLLSResult)
    bestcost = minimum(x.costs)
    otherstuff = x.timetotal - x.timecost - x.timegradient - x.timesolver
    @printf(io, "NLLSsolver optimization took %f seconds and %d iterations to reduce the cost from %f to %f (a %.2f%% reduction), using:
   %d cost computations in %f seconds (%.2f%% of total time),
   %d gradient computations in %f seconds (%.2f%% of total time),
   %d linear solver computations in %f seconds (%.2f%% of total time),
   %f seconds for other stuff (%.2f%% of total time).\n", 
            x.timetotal, length(x.costs)-1, x.costs[1], bestcost, 100*(1-bestcost/x.costs[1]), 
            x.costcomputations, x.timecost, 100*x.timecost/x.timetotal,
            x.gradientcomputations, x.timegradient, 100*x.timegradient/x.timetotal,
            x.linearsolvers, x.timesolver, 100*x.timesolver/x.timetotal,
            otherstuff, 100*otherstuff/x.timetotal)
end

function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions=NLLSOptions())::NLLSResult where VarTypes
    t = @elapsed begin
        # Set up all the internal data structures
        data = NLLSInternal{VarTypes}(problem)
        bestvariables = problem.variables
        # Initialize the linear problem
        data.timegradient += @elapsed data.bestcost = costgradhess!(data.gradient, data.hessian, problem.residuals, problem.variables, data.blockoffsets)
        data.gradientcomputations += 1
        # Initialize the results
        costs = [data.bestcost]
        trajectory = Vector{Vector{Float64}}()
        fails = 0
        # Initialize the iterator
        if options.iterator == gaussnewton
            # Gauss-Newton or Newton
            iterator = newton_iteration!
        elseif options.iterator == levenbergmarquardt
            # Levenberg-Marquardt
            data.lambda = 1.
            iterator = levenberg_iteration!
        elseif options.iterator == dogleg
            # Dogleg
            data.lambda = 0.
            iterator = dogleg_iteration!
        end
        # Do the iterations
        iter = 0
        while true
            iter += 1
            # Call the per iteration solver
            cost = iterator(data, problem, options)
            push!(costs, cost)
            # Store the best result
            dcost = data.bestcost - cost
            if dcost > 0
                data.bestcost = cost
                bestvariables = data.variables
                fails = 0
            else
                dcost = cost
                fails += 1
            end
            if options.storetrajectory
                # Store the variable trajectory (as update vectors)
                push!(trajectory, data.step)
            end
            # Check for convergence
            if options.callback(problem, data, cost) || !(dcost >= data.bestcost * options.dcost) || (maximum(abs, data.step) < options.dstep) || (fails > options.maxfails) || iter >= options.maxiters
                break
            end
            # Update the variables
            problem.variables .= data.variables
            # Construct the linear problem
            data.timegradient += @elapsed begin
                fill!(data.gradient, 0)
                fill!(data.hessian, 0)
                costgradhess!(data.gradient, data.hessian, problem.residuals, problem.variables, data.blockoffsets)
            end
            data.gradientcomputations += 1
        end
        # Update the problem variables
        problem.variables .= bestvariables
    end
    # Return the result
    return NLLSResult(costs, trajectory, t, data.timecost, data.timegradient, data.timesolver, data.costcomputations, data.gradientcomputations, data.linearsolvers)
end

function update!(variables, offsets, step)
    # Update each variable
    for index in eachindex(offsets)
        if offsets[index] != 0
            variables[index] = update(variables[index], step, offsets[index])
        end
    end
end

# Iterators assume that the linear problem has been constructed
function newton_iteration!(data::NLLSInternal, problem::NLLSProblem, options::NLLSOptions)::Float64
    # Compute the step
    data.timesolver += @elapsed data.step = -solve(LinearProblem(data.hessian, data.gradient), options.linearsolver).u
    data.linearsolvers += 1
    # Update the new variables
    data.variables = copy(problem.variables)
    update!(data.variables, data.blockoffsets, data.step)
    # Return the cost
    data.timecost += @elapsed cost_ = cost(problem.residuals, data.variables)
    data.costcomputations += 1
    return cost_
end

function dogleg_iteration!(data::NLLSInternal, problem::NLLSProblem, options::NLLSOptions)::Float64
    data.timesolver += @elapsed begin
        # Compute the Cauchy step
        gnorm2 = data.gradient' * data.gradient
        a = gnorm2 / (data.gradient' * data.hessian * data.gradient + floatmin(eltype(data.gradient)))
        cauchy = -a * data.gradient
        alpha2 = a * a * gnorm2
        alpha = sqrt(alpha2)
        if data.lambda == 0
            # Make first step the Cauchy point
            data.lambda = alpha
        end
        if alpha < data.lambda
            # Compute the Newton step
            newton = -solve(LinearProblem(data.hessian, data.gradient), options.linearsolver).u
            beta = norm(newton)
            data.linearsolvers += 1
        end
    end
    cost_ = data.bestcost
    while true
        # Determine the step
        if !(alpha < data.lambda)
            # Along first leg
            data.step = (data.lambda / alpha) * cauchy
            linear_approx = data.lambda * (2 * alpha - data.lambda) / (2 * a)
        else
            # Along second leg
            if beta <= data.lambda
                # Do the full Newton step
                data.step = newton
                linear_approx = cost_
            else
                # Find the point along the Cauchy -> Newton line on the trust
                # region circumference
                leg = newton - cauchy
                sq_leg = leg' * leg
                c = cauchy' * leg
                step = sqrt(c * c + sq_leg * (data.lambda * data.lambda - alpha2))
                if c <= 0
                    step = (-c + step) / sq_leg
                else
                    step = (data.lambda * data.lambda - alpha2) / (c + step)
                end
                data.step = cauchy + step * leg
                linear_approx = 0.5 * (a * (1 - step) ^ 2 * gnorm2) + step * (2 - step) * cost_
            end
        end
        # Update the new variables
        data.variables = copy(problem.variables)
        update!(data.variables, data.blockoffsets, data.step)
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, data.variables)
        data.costcomputations += 1
        # Update lambda
        mu = (data.bestcost - cost_) / linear_approx
        if mu > 0.75
            data.lambda = max(data.lambda, 3 * norm(data.step))
        elseif mu < 0.25
            data.lambda *= 0.5
        end
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Return the cost
            return cost_
        end
    end
end

function levenberg_iteration!(data::NLLSInternal, problem::NLLSProblem, options::NLLSOptions)::Float64
    @assert data.lambda >= 0.
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        data.hessian += (data.lambda - lastlambda) * I
        lastlambda = data.lambda
        # Solve the linear system
        data.timesolver += @elapsed data.step = -solve(LinearProblem(data.hessian, data.gradient), options.linearsolver).u
        data.linearsolvers += 1
        # Update the new variables
        data.variables = copy(problem.variables)
        update!(data.variables, data.blockoffsets, data.step)
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, data.variables)
        data.costcomputations += 1
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Success (or convergence) - update lambda
            step_quality = (cost_ - data.bestcost) / (data.step' * (data.gradient + data.hessian * data.step * 0.5))
            data.lambda *= max(0.333, 1 - (step_quality - 1) ^ 3)
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        data.lambda *= mu;
        mu *= 2.;
    end
end
