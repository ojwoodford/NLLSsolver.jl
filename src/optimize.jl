export optimize!

function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions=NLLSOptions())::NLLSResult where VarTypes
    # Pre-allocate the temporary data
    t = Base.time_ns()
    computehessian = in(options.iterator, [gaussnewton, levenbergmarquardt, dogleg])
    data = NLLSInternal{VarTypes}(problem, computehessian)
    costgradient! = computehessian ? costgradhess! : costresjac!
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
        while true
            data.iternum += 1
            # Call the per iteration solver
            cost = iterate!(iteratedata, data, problem, options)
            if options.storecosts
                push!(costs, cost)
            end
            # Store the best result
            dcost = data.bestcost - cost
            if dcost > 0
                data.bestcost = cost
                copy!(data.bestvariables, data.variables, problem.unfixed)
                fails = 0
            else
                dcost = cost
                fails += 1
            end
            if options.storetrajectory
                # Store the variable trajectory (as update vectors)
                push!(trajectory, copy(data.step))
            end
            # Check for convergence
            if options.callback(problem, data, cost) || !(dcost >= data.bestcost * options.reldcost) || !(dcost >= options.absdcost) || (maximum(abs, data.step) < options.dstep) || (fails > options.maxfails) || data.iternum >= options.maxiters
                break
            end
            # Update the variables
            copy!(problem.variables, data.variables, problem.unfixed)
            # Construct the linear problem
            data.timegradient += @elapsed begin
                zero!(data.linsystem)
                costgradient!(data.linsystem, problem.residuals, problem.variables)
            end
            data.gradientcomputations += 1
        end
        # Update the problem variables
        copy!(problem.variables, data.bestvariables, problem.unfixed)
    end
    # Return the result
    return NLLSResult(startcost, data.bestcost, t, timeinit, data.timecost, data.timegradient, data.timesolver, data.iternum, data.costcomputations, data.gradientcomputations, data.linearsolvers, costs, trajectory)
end

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

function copy!(to::Vector, from::Vector, unfixed::UInt)
    to[unfixed] = from[unfixed]
end

function copy!(to::Vector, from::Vector, unfixed::BitVector)
    for (index, unfixed_) in enumerate(unfixed)
        if unfixed_
            to[index] = from[index]
        end
    end
end

