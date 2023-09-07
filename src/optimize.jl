function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions=NLLSOptions(), unfixed=0)::NLLSResult where VarTypes
    t = Base.time_ns()
    @assert length(problem.variables) > 0
    computehessian = !in(options.iterator, [gaussnewton, levenbergmarquardttall])
    @assert computehessian || all(type_ -> type_ <: AbstractResidual, keys(problem.costs.data))
    costgradient! = computehessian ? costgradhess! : costresjac!
    # Copy the variables
    if length(problem.variables) != length(problem.varnext)
        problem.varnext = copy(problem.variables)
    end
    # Compute the number of free variables (nblocks)
    nblocks, unfixed = getnblocks(unfixed, problem.variables)
    # Pre-allocate the temporary data
    if nblocks == 1
        # One unfixed variable
        varlen = UInt(nvars(problem.variables[unfixed]))
        return optimizeinternal!(problem, options, NLLSInternalSingleVar(unfixed, varlen, computehessian ? varlen : UInt(countcosts(resnum, problem.costs))), costgradient!, t)
    end
    # Multiple variables. Use a block sparse matrix
    mvls = computehessian ? makesymmvls(problem, unfixed, nblocks) : makemvls(problem, unfixed, nblocks)
    return optimizeinternal!(problem, options, NLLSInternalMultiVar(mvls), costgradient!, t)
end

function optimizeinternal!(problem::NLLSProblem{VarTypes}, options::NLLSOptions, data, costgradient!, starttimens)::NLLSResult where VarTypes
    # Call the optimizer with the required iterator struct
    if options.iterator == gaussnewton
        # Gauss-Newton (optimize jacobian form)
        gaussnewtondata = GaussNewtonData()
        return optimizeinternal!(problem, options, data, gaussnewtondata, costgradient!, (Base.time_ns() - starttimens) * 1.e-9)
    end
    if options.iterator == newton
        # Gauss-Newton (optimize jacobian form)
        newtondata = NewtonData()
        return optimizeinternal!(problem, options, data, newtondata, costgradient!, (Base.time_ns() - starttimens) * 1.e-9)
    end
    if options.iterator == levenbergmarquardt
        # Levenberg-Marquardt
        levmardata = LevMarData()
        return optimizeinternal!(problem, options, data, levmardata, costgradient!, (Base.time_ns() - starttimens) * 1.e-9)
    end
    if options.iterator == dogleg
        # Dogleg
        doglegdata = DoglegData()
        return optimizeinternal!(problem, options, data, doglegdata, costgradient!, (Base.time_ns() - starttimens) * 1.e-9)
    end
    if options.iterator == gradientdescent
        # Gradient descent
        gddata = GradientDescentData(1.0)
        return optimizeinternal!(problem, options, data, gddata, costgradient!, (Base.time_ns() - starttimens) * 1.e-9)
    end
    if options.iterator == levenbergmarquardttall
        # Levenberg-Marquardt
        levmardata = LevMarJacData()
        return optimizeinternal!(problem, options, data, levmardata, costgradient!, (Base.time_ns() - starttimens) * 1.e-9)
    end
    error("Iterator not recognized")
end

function optimizeinternal!(problem::NLLSProblem{VarTypes}, options::NLLSOptions, data, iteratedata, costgradient!, timeinit)::NLLSResult where VarTypes
    t = @elapsed begin
        # Initialize the linear problem
        data.timegradient += @elapsed data.bestcost = costgradient!(data.linsystem, problem.variables, problem.costs)
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
                costgradient!(data.linsystem, problem.variables, problem.costs)
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

function getnblocks(unfixed, variables)
    # Compute the number of free variables (nblocks)
    if isa(unfixed, DataType)
        unfixed = typeof.(variables) .== unfixed
    end
    unfixed_ = UInt(0)
    if isa(unfixed, Number)
        unfixed_ = UInt(unfixed)
        if unfixed_ > 0
            nblocks = UInt(1)
        else
            nblocks = UInt(length(variables))
            if nblocks == 1
                unfixed_ = UInt(1)
            else
                unfixed = trues(nblocks)
            end
        end
    else
        @assert length(unfixed) == length(variables)
        nblocks = UInt(sum(unfixed))
        @assert nblocks > 0
    end
    if nblocks == 1
        if unfixed_ == 0
            unfixed_ = UInt(findfirst(unfixed))
        end
        return nblocks, unfixed_
    end
    return nblocks, unfixed
end
