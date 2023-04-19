export optimize!

function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions=NLLSOptions())::NLLSResult where VarTypes
    t = Base.time_ns()
    # Pre-allocate the temporary data
    computehessian = in(options.iterator, [gaussnewton, levenbergmarquardt, dogleg])
    costgradient! = computehessian ? costgradhess! : costresjac!
    data = NLLSInternal{VarTypes}(problem, computehessian)
    # Copy the variables
    if length(problem.variables) != length(problem.varnext)
        problem.varnext = copy(problem.variables)
    end
    # Call the optimizer with the required iterator struct
    if options.iterator == gaussnewton
        # Gauss-Newton or Newton
        newtondata = NewtonData()
        return optimize!(problem, options, data, newtondata, costgradient!, (Base.time_ns() - t) * 1.e-9)
    end
    if options.iterator == levenbergmarquardt
        # Levenberg-Marquardt
        levmardata = LevMarData()
        return optimize!(problem, options, data, levmardata, costgradient!, (Base.time_ns() - t) * 1.e-9)
    end
    if options.iterator == dogleg
        # Dogleg
        doglegdata = DoglegData()
        return optimize!(problem, options, data, doglegdata, costgradient!, (Base.time_ns() - t) * 1.e-9)
    end
    error("Iterator not recognized")
end

function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions, data::NLLSInternal{VarTypes}, iteratedata, costgradient!, timeinit)::NLLSResult where VarTypes
    t = @elapsed begin
        # Initialize the linear problem
        data.timegradient += @elapsed data.bestcost = costgradient!(data.linsystem, problem.residuals, problem.variables)
        data.gradientcomputations += 1
        # Initialize the results
        startcost = data.bestcost
        costs = Vector{Float64}()
        trajectory = Vector{Vector{Float64}}()
        # Do the iterations
        fails = 0
        cost = data.bestcost
        while true
            data.iternum += 1
            # Call the per iteration solver
            cost = iterate!(iteratedata, data, problem, options)::Float64
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
                        problem.varbest, problem.variables = problem.variables, problem.varbest
                    else
                        problem.varbest = copy(problem.variables)
                    end
                end
            end
            # Update the variables
            problem.varnext, problem.variables = problem.variables, problem.varnext
            if options.storetrajectory
                # Store the variable trajectory (as update vectors)
                push!(trajectory, copy(data.step))
            end
            # Check for convergence
            if options.callback(problem, data, cost) || !(dcost >= data.bestcost * options.reldcost) || !(dcost >= options.absdcost) || (maximum(abs, data.step) < options.dstep) || (fails > options.maxfails) || data.iternum >= options.maxiters
                break
            end
            # Construct the linear problem
            data.timegradient += @elapsed begin
                zero!(data.linsystem)
                costgradient!(data.linsystem, problem.residuals, problem.variables)
            end
            data.gradientcomputations += 1
        end
        if data.bestcost < cost
            # Update the problem variables to the best ones found
            problem.varbest, problem.variables = problem.variables, problem.varbest
        end
    end
    # Return the result
    return NLLSResult(startcost, data.bestcost, t, timeinit, data.timecost, data.timegradient, data.timesolver, data.iternum, data.costcomputations, data.gradientcomputations, data.linearsolvers, costs, trajectory)
end
