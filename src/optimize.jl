
# Uni-variate optimization (single unfixed variable)
function optimize!(problem::NLLSProblem, options::NLLSOptions, unfixed::Integer, type::DataType=typeof(problem.variables[unfixed]), t::UInt64=Base.time_ns())::NLLSResult
    # Copy the variables
    if length(problem.variables) != length(problem.varnext)
        problem.varnext = copy(problem.variables)
    end
    # One unfixed variable
    varlen = UInt(nvars(problem.variables[unfixed]))
    return optimizeinternal!(problem, options, NLLSInternalSingleVar(UInt(unfixed), varlen, varlen), t)
end

# Multi-variate optimization
optimize!(problem::NLLSProblem, options::NLLSOptions, unfixed::Type) = optimize!(problem, options, typeof.(variables).==unfixed)
function optimize!(problem::NLLSProblem, options::NLLSOptions=NLLSOptions(), unfixed::AbstractVector=trues(length(problem.variables)))::NLLSResult
    t = Base.time_ns()
    @assert length(problem.variables) > 0
    # Compute the number of free variables (nblocks)
    nblocks = sum(unfixed)
    if nblocks == 1
        # One unfixed variable
        return optimize!(problem, options, findfirst(unfixed), typeof(problem.variables[unfixed]), t)
    end
    # Copy the variables
    if length(problem.variables) != length(problem.varnext)
        problem.varnext = copy(problem.variables)
    end
    # Multiple variables. Use a block sparse matrix
    return optimizeinternal!(problem, options, NLLSInternalMultiVar(makesymmvls(problem, unfixed, nblocks)), t)
end

# Optimize one variable at a time
function optimizesingles!(problem::NLLSProblem, options::NLLSOptions, unfixed::DataType)::NLLSResult
    t = Base.time_ns()
    # Compute initial cost
    timecost = @elapsed startcost = cost(problem)
    # Initialize stats
    timeinit = 0.0
    timegradient = 0.0
    timesolver = 0.0
    iternum = 0
    costcomputations = 2 # Count final cost computation here
    gradientcomputations = 0
    linearsolvers = 0
    # Optimize each variable of the given type, in sequence
    for (ind, var) in enumerate(problem.variables)
        # Skip variables of a different type
        if !isa(var, unfixed)
            continue
        end
        # Construct the subset of residuals that depend
        timeinit += @elapsed subprob = subproblem(problem, ind)
        # Optimize the subproblem
        result = optimize!(subprob, options, ind, unfixed)
        # Accumulate stats
        timeinit += result.timeinit
        timecost += result.timecost
        timegradient += result.timegradient
        timesolver += result.timesolver
        iternum += result.niterations
        costcomputations += result.costcomputations
        gradientcomputations += result.gradientcomputations
        linearsolvers += result.linearsolvers
    end
    # Compute final cost
    timecost += @elapsed endcost = cost(problem)
    return NLLSResult(startcost, endcost, (Base.time_ns() - t)*1.e-9, timeinit, timecost, timegradient, timesolver, 0, iternum, costcomputations, gradientcomputations, linearsolvers, Vector{Float64}(), Vector{Vector{Float64}}())
end

function optimizeinternal!(problem::NLLSProblem, options::NLLSOptions, data, starttimens::UInt64)::NLLSResult
    # Call the optimizer with the required iterator struct
    if options.iterator == newton || options.iterator == gaussnewton
        # Newton's method, using Gauss' approximation to the Hessian (optimizing Hessian form)
        newtondata = NewtonData()
        return optimizeinternal!(problem, options, data, newtondata, (Base.time_ns() - starttimens) * 1.e-9)
    end
    if options.iterator == levenbergmarquardt
        # Levenberg-Marquardt
        levmardata = LevMarData()
        return optimizeinternal!(problem, options, data, levmardata, (Base.time_ns() - starttimens) * 1.e-9)
    end
    if options.iterator == dogleg
        # Dogleg
        doglegdata = DoglegData()
        return optimizeinternal!(problem, options, data, doglegdata, (Base.time_ns() - starttimens) * 1.e-9)
    end
    if options.iterator == gradientdescent
        # Gradient descent
        gddata = GradientDescentData(1.0)
        return optimizeinternal!(problem, options, data, gddata, (Base.time_ns() - starttimens) * 1.e-9)
    end
    error("Iterator not recognized")
end

function optimizeinternal!(problem::NLLSProblem, options::NLLSOptions, data, iteratedata, timeinit::Float64)::NLLSResult
    t = @elapsed begin
        # Initialize the linear problem
        data.timegradient += @elapsed data.bestcost = costgradhess!(data.linsystem, problem.variables, problem.costs)
        data.gradientcomputations += 1
        # Initialize the results
        startcost = data.bestcost
        costs = Vector{Float64}()
        trajectory = Vector{Vector{Float64}}()
        # Do the iterations
        fails = 0
        cost = data.bestcost
        converged = 0
        while true
            data.iternum += 1
            # Call the per iteration solver
            cost = iterate!(iteratedata, data, problem, options)::Float64
            # Call the user-defined callback
            cost, terminate = options.callback(cost, problem, data)::Tuple{Float64, Int}
            # Store the cost if necessary
            if options.storecosts
                push!(costs, cost)
            end
            # Check for cost increase (only some iterators will do this)
            dcost = data.bestcost - cost
            if dcost >= 0
                data.bestcost = cost
                fails = 0
            else
                dcost = cost
                fails += 1
                if fails == 1
                    # Store the current best variables
                    if length(problem.variables) == length(problem.varbest)
                        updatetobest!(problem, data)
                    else
                        problem.varbest = copy(problem.variables)
                    end
                end
            end
            # Update the variables
            updatefromnext!(problem, data)
            if options.storetrajectory
                # Store the variable trajectory (as update vectors)
                push!(trajectory, copy(data.step))
            end
            # Check for termination
            maxstep = maximum(abs, data.step)
            converged = 0
            converged |= isinf(cost)                                     << 0 # Cost is infinite
            converged |= isnan(cost)                                     << 1 # Cost is NaN
            converged |= (dcost < data.bestcost * options.reldcost)      << 2 # Relative decrease in cost is too small
            converged |= (dcost < options.absdcost)                      << 3 # Absolute decrease in cost is too small
            converged |= isinf(maxstep)                                  << 4 # Infinity detected in the step
            converged |= isnan(maxstep)                                  << 5 # NaN detected in the step
            converged |= (maxstep < options.dstep)                       << 6 # Max of the step size is too small
            converged |= (fails > options.maxfails)                      << 7 # Max number of consecutive failed iterations reach
            converged |= (data.iternum >= options.maxiters)              << 8 # Max number of iterations reached
            converged |= terminate                                       << 9 # Terminated by the user-defined callback
            if converged != 0
                break
            end
            # Construct the linear problem
            data.timegradient += @elapsed begin
                zero!(data.linsystem)
                costgradhess!(data.linsystem, problem.variables, problem.costs)
            end
            data.gradientcomputations += 1
        end
        if data.bestcost < cost
            # Update the problem variables to the best ones found
            updatefrombest!(problem, data)
        end
    end
    # Return the result
    return NLLSResult(startcost, data.bestcost, t, timeinit, data.timecost, data.timegradient, data.timesolver, converged, data.iternum, data.costcomputations, data.gradientcomputations, data.linearsolvers, costs, trajectory)
end

function updatefromnext!(problem::NLLSProblem, ::NLLSInternalMultiVar)
    problem.variables, problem.varnext = problem.varnext, problem.variables
end

function updatefrombest!(problem::NLLSProblem, ::NLLSInternalMultiVar)
    problem.variables, problem.varbest = problem.varbest, problem.variables
end
updatetobest!(problem::NLLSProblem, data::NLLSInternalMultiVar) = updatefrombest!(problem::NLLSProblem, data::NLLSInternalMultiVar)

function updatefromnext!(problem::NLLSProblem, data::NLLSInternalSingleVar)
    problem.variables[data.linsystem.varindex] = problem.varnext[data.linsystem.varindex]
end

function updatefrombest!(problem::NLLSProblem, data::NLLSInternalSingleVar)
    problem.variables[data.linsystem.varindex] = problem.varbest[data.linsystem.varindex]
end

function updatetobest!(problem::NLLSProblem, data::NLLSInternalSingleVar)
    problem.varbest[data.linsystem.varindex] = problem.variables[data.linsystem.varindex]
end
