export optimize!

function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions=NLLSOptions())::NLLSResult where VarTypes
    # Call the optimizer with the required iterator struct
    if options.iterator == gaussnewton
        # Gauss-Newton or Newton
        return optimize!(problem, options, true, NewtonData())
    end
    if options.iterator == levenbergmarquardt
        # Levenberg-Marquardt
        return optimize!(problem, options, true, LevMarData())
    end
    if options.iterator == dogleg
        # Dogleg
        return optimize!(problem, options, true, DoglegData())
    end
end

function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions, computehessian::Bool, iteratedata)::NLLSResult where VarTypes
    t = @elapsed begin
        # Set up all the internal data structures
        costgradient! = computehessian ? costgradhess! : costresjac!
        data = NLLSInternal{VarTypes}(problem, computehessian)
        # Initialize the linear problem
        data.timegradient += @elapsed data.bestcost = costgradient!(data.linsystem, problem.residuals, problem.variables)
        data.gradientcomputations += 1
        # Initialize the results
        startcost = data.bestcost
        costs = Vector{Float64}()
        trajectory = Vector{Vector{Float64}}()
        # Do the iterations
        iter = 0
        fails = 0
        while true
            iter += 1
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
            if options.callback(problem, data, cost) || !(dcost >= data.bestcost * options.dcost) || (maximum(abs, data.step) < options.dstep) || (fails > options.maxfails) || iter >= options.maxiters
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
    return NLLSResult(startcost, data.bestcost, t, data.timecost, data.timegradient, data.timesolver, iter, data.costcomputations, data.gradientcomputations, data.linearsolvers, costs, trajectory)
end

function update!(to::Vector, from::Vector, linsystem::MultiVariateLS, step)
    # Update each variable
    @inbounds for (i, j) in enumerate(linsystem.blockindices)
        if j != 0
            a = update(from[i], step, linsystem.gradoffsets[j])
            to[i] = a
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

function linearsolve(A::AbstractMatrix, b::AbstractVector, options)
    return solve(LinearProblem(A, b), options).u
end

