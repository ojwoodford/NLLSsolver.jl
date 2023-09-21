
# Uni-variate optimization (single unfixed variable)
optimize!(problem::NLLSProblem, options::NLLSOptions, unfixed::Integer, callback=nullcallback, starttimens=Base.time_ns())::NLLSResult = getresult(setupiterator(optimizeinternal!, problem, options, NLLSInternal(UInt(unfixed), nvars(problem.variables[unfixed])), checkcallback(options, callback), starttimens))

# Multi-variate optimization
function optimize!(problem::NLLSProblem, options::NLLSOptions, unfixed::AbstractVector, callback)::NLLSResult
    starttime = Base.time_ns()
    @assert length(problem.variables) > 0
    # Compute the number of free variables (nblocks)
    nblocks = sum(unfixed)
    if nblocks == 1
        # One unfixed variable
        unfixed = findfirst(unfixed)
        return optimize!(problem, options, unfixed, callback, starttime)
    end
    # Multiple variables
    return getresult(setupiterator(optimizeinternal!, problem, options, NLLSInternal(makesymmvls(problem, unfixed, nblocks)), checkcallback(options, callback), starttime))
end

# Conversions for different types of "unfixed"
convertunfixed(::Nothing, problem) = trues(length(problem.variables))
convertunfixed(unfixed::Type, problem) = typeof.(problem.variables) .== unfixed
convertunfixed(unfixed, problem) = unfixed

# Default options
optimize!(problem::NLLSProblem, options::NLLSOptions=NLLSOptions(), unfixed=nothing, callback=nullcallback) = optimize!(problem, options, convertunfixed(unfixed, problem), callback)

# Optimize one variable at a time
optimizesingles!(problem::NLLSProblem, options::NLLSOptions, type::DataType, starttime=Base.time_ns())::NLLSResult = getresult(setupiterator(optimizesinglesinternal!, problem, options, NLLSInternal(UInt(1), nvars(type())), type, starttime))

checkcallback(::NLLSOptions{Nothing}, callback) = callback
checkcallback(options::NLLSOptions, ::Any) = options.callback

function setupiterator(func, problem::NLLSProblem, options::NLLSOptions, data::NLLSInternal, trailingargs...)
    # Call the optimizer with the required iterator struct
    if options.iterator == newton || options.iterator == gaussnewton
        # Newton's method, using Gauss' approximation to the Hessian (optimizing Hessian form)
        newtondata = NewtonData(problem, data)
        return func(problem, options, data, newtondata, trailingargs...)
    end
    if options.iterator == levenbergmarquardt
        # Levenberg-Marquardt
        levmardata = LevMarData(problem, data)
        return func(problem, options, data, levmardata, trailingargs...)
    end
    if options.iterator == dogleg
        # Dogleg
        doglegdata = DoglegData(problem, data)
        return func(problem, options, data, doglegdata, trailingargs...)
    end
    if options.iterator == gradientdescent
        # Gradient descent
        gddata = GradientDescentData(problem, data)
        return func(problem, options, data, gddata, trailingargs...)
    end
    error("Iterator not recognized")
end

# The meat of an optimization
function optimizeinternal!(problem::NLLSProblem, options::NLLSOptions, data, iteratedata, callback, starttime::UInt64)
    # Copy the variables
    if length(problem.variables) != length(problem.varnext)
        problem.varnext = copy(problem.variables)
    end
    data.timeinit += Base.time_ns() - starttime
    # Initialize the linear problem
    data.timegradient += @elapsed_ns data.bestcost = costgradhess!(data.linsystem, problem.variables, problem.costs)
    data.gradientcomputations += 1
    data.startcost = data.bestcost
    # Do the iterations
    fails = 0
    cost = data.bestcost
    converged = 0
    data.iternum = 0
    while true
        data.iternum += 1
        # Call the per iteration solver
        cost = iterate!(iteratedata, data, problem, options)::Float64
        # Call the user-defined callback
        cost, terminate = callback(cost, problem, data, iteratedata)::Tuple{Float64, Int}
        # Store the cost if necessary
        if options.storecosts
            push!(data.costs, cost)
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
                    problem.varbest = deepcopy(problem.variables)
                end
            end
        end
        # Update the variables
        updatefromnext!(problem, data)
        if options.storetrajectory
            # Store the variable trajectory (as update vectors)
            push!(data.trajectory, copy(data.linsystem.x))
        end
        # Check for termination
        maxstep = maximum(abs, data.linsystem.x)
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
        data.converged = converged
        if converged != 0
            break
        end
        # Construct the linear problem
        data.timegradient += @elapsed_ns begin
            zero!(data.linsystem)
            costgradhess!(data.linsystem, problem.variables, problem.costs)
        end
        data.gradientcomputations += 1
    end
    if data.bestcost < cost
        # Update the problem variables to the best ones found
        updatefrombest!(problem, data)
    end
    # Return the data to produce the final result
    data.timetotal += Base.time_ns() - starttime
    return data
end

# Optimizing variables one at a time (e.g. in alternation)
function optimizesinglesinternal!(problem::NLLSProblem{VT, CT}, options::NLLSOptions, data::NLLSInternal{LST}, iteratedata, ::Type{type}, starttime::UInt) where {VT, CT, LST<:UniVariateLS, type}
    # Initialize stats
    iternum = 0
    data.costcomputations = 2 # Count first and final cost computation here
    subprob = NLLSProblem{VT, CT}(problem.variables, CostStruct{CT}())
    costindices = sparse(getvarcostmap(problem)')
    data.timeinit = Base.time_ns() - starttime
    # Compute initial cost
    data.timecost = @elapsed_ns startcost = cost(problem)
    # Optimize each variable of the given type, in sequence
    indices = findall(v->isa(v, type), problem.variables)
    for ind in indices
        starttime_ = Base.time_ns()
        # Construct the subset of residuals that depend on this variable
        subproblem!(subprob, problem, @inbounds(view(costindices.rowval, costindices.colptr[ind]:costindices.colptr[ind+1]-1)))
        # Update the linear system
        data.linsystem = LST(data.linsystem, UInt(ind), nvars(problem.variables[ind]::type))
        # Reset the iterator data
        reset!(iteratedata, problem, data)
        stoptime = Base.time_ns()
        data.timeinit += stoptime - starttime_
        # Optimize the subproblem
        optimizeinternal!(subprob, options, data, iteratedata, nullcallback, stoptime)
        # Accumulate stats
        iternum += data.iternum
    end
    # Compute final cost
    data.timecost += @elapsed_ns data.bestcost = cost(problem)
    # Correct some stats
    data.startcost = startcost
    data.iternum = iternum
    data.converged = 0
    data.timetotal = Base.time_ns() - starttime
    return data
end


function updatefromnext!(problem::NLLSProblem, ::NLLSInternalMultiVar)
    problem.variables, problem.varnext = problem.varnext, problem.variables
end

function updatefrombest!(problem::NLLSProblem, ::NLLSInternalMultiVar)
    problem.variables, problem.varbest = problem.varbest, problem.variables
end
updatetobest!(problem::NLLSProblem, data::NLLSInternalMultiVar) = updatefrombest!(problem, data)

function updatefromnext!(problem::NLLSProblem, data::NLLSInternalSingleVar)
    @inbounds problem.variables[data.linsystem.varindex] = problem.varnext[data.linsystem.varindex]
end

function updatefrombest!(problem::NLLSProblem, data::NLLSInternalSingleVar)
    @inbounds problem.variables[data.linsystem.varindex] = problem.varbest[data.linsystem.varindex]
end

function updatetobest!(problem::NLLSProblem, data::NLLSInternalSingleVar)
    @inbounds problem.varbest[data.linsystem.varindex] = problem.variables[data.linsystem.varindex]
end
