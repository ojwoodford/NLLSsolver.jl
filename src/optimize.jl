# Uni-variate optimization (single unfixed variable)
optimize!(problem::NLLSProblem, options::NLLSOptions, unfixed::Integer, callback=nullcallback, starttimens=Base.time_ns()) = setupiterator(optimizeinternal!, checkvars!(problem), options, NLLSInternal(UInt(unfixed), nvars(problem.variables[unfixed]), starttimens), callback)::NLLSResult

# Multi-variate optimization
function optimize!(problem::NLLSProblem, options::NLLSOptions, unfixed::AbstractVector, callback)::NLLSResult
    starttimens = Base.time_ns()
    @assert length(problem.variables) > 0
    # Compute the number of free variables (nblocks)
    nblocks = sum(unfixed)
    if nblocks == 1
        # One unfixed variable
        return optimize!(problem, options, findfirst(unfixed), callback, starttimens)
    end
    # Multiple variables
    problem = checkvars!(problem)
    return setupiterator(optimizeinternal!, problem, options, NLLSInternal(makesymmvls(problem, unfixed, nblocks), starttimens), callback)::NLLSResult
end

# Conversions for different types of "unfixed"
convertunfixed(::Nothing, problem) = trues(length(problem.variables))
convertunfixed(unfixed::Type, problem) = typeof.(problem.variables) .== unfixed
convertunfixed(unfixed, problem) = unfixed

# Helper to ensure the NLLSProblem struct has the right variable storage (e.g. varnext) for optimization
function checkvars!(problem::NLLSProblem)::NLLSProblem
    # Copy the variables, if need be
    if length(problem.variables) != length(problem.varnext)
        problem.varnext = deepcopy(problem.variables)
    end
    return problem
end

# Default options
"""
    NLLSsolver.optimize!(problem::NLLSProblem, options=NLLSOptions(), 
                         unfixed=nothing, callback=nullcallback)

Optimize the cost defined by `problem`, updating variables in-place, and return 
`result::NLLSResult`.

Options such as which optimizer to use and termination criteria can be defined in 
options::[`NLLSOptions`](@ref). The default options specify Levenberg-Marquardt optimization.

If not all variables should be optimzed, this can be specified using `unfixed`, which 
defines which variables are unfixed, and therefore should be optimized. It can be an integer 
index of a single variable, a variable type, or a boolean vector. The default (`nothing`)
indicates that all variables should be optimized.

A callback function can be supplied via the `callback` argument. This function is called 
after each iteration of the optimization, and should take the following form:
```julia
    cost, terminate = callback(cost, problem, data::NLLSInternal, iteratedata)
````
where `cost` is the potentially updated problem cost (if the callback updates the problem), 
and `terminate` is an integer, where non-zero values indicate the optimizer should terminate.
The value of `terminate` is reported in the result of `optimizer!`. The default,  
[`nullcallback`](@ref), does nothing. Existing callbacks [`printoutcallback`](@ref) and 
[`storecostscallback`](@ref) print out and store per iteration information respectively. 
Callbacks can also be user defined.

Variables are optimized in place, with the `variables` element of `problem::`[`NLLSProblem`](@ref)
set to the optimal values found. Other pertinent information, such as the start and end
costs, iteration count, high level timings and reasons for termination, is returned in an 
[`NLLSResult`](@ref) object.
"""
optimize!(problem::NLLSProblem, options::NLLSOptions=NLLSOptions(), unfixed=nothing, callback=nullcallback) = optimize!(problem, options, convertunfixed(unfixed, problem), callback)

# Optimize one variable at a time
function optimizesingles!(problem::NLLSProblem{VT, CT}, options::NLLSOptions, type::DataType)::NLLSResult where {VT, CT}
    indices = findall(v->isa(v, type), problem.variables)
    return optimizesingles!(problem, options, indices, !isempty(indices) && dynamic(is_static(nvars(problem.variables[indices[1]]))))
end
function optimizesingles!(problem::NLLSProblem{VT, CT}, options::NLLSOptions, indices, sortedbysize=false)::NLLSResult where {VT, CT}
    problem = checkvars!(problem)
    # Get the indices per cost
    costindices = sparse(getvarcostmap(problem)')
    # Put the costs aside
    allcosts = problem.costs
    problem.costs = CostStruct{CT}()
    # If the variables are not static sized, then we can optimize all variables of the same size together, so sort the indices by variable size
    if !sortedbysize
        indices = indices[sortperm([nvars(problem.variables[i]) for i in indices])]
    end
    # Loop over all the variables
    result = NLLSResult(0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    first = 1
    while first <= length(indices)
        # Optimize all variables of the same size at once
        first, result = loopvarsizes(problem, options, nvars(problem.variables[indices[first]]), allcosts, costindices, indices, first, result)
    end
    # Put the costs back
    problem.costs = allcosts
    return result
end
loopvarsizes(problem::NLLSProblem, options::NLLSOptions, varsz, allcosts, costindices, indices, first, result::NLLSResult) = setupiterator(optimizesinglesinternal!, problem, options, NLLSInternal(UInt(1), varsz, UInt64(0)), allcosts, costindices, indices, first, result)


function setupiterator(func, problem::NLLSProblem, options::NLLSOptions, data::NLLSInternal, trailingargs...)
    # Call the optimizer with the required iterator struct
    if options.iterator == newton
        # Newton's method
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
function optimizeinternal!(problem::NLLSProblem, options::NLLSOptions, data, iteratedata, callback)::NLLSResult
    # Do any preoptimization for the iterator
    data.startcost = preoptimization(iteratedata, problem, options, data)::Float64
    # Other initializations
    fails = 0
    converged = 0
    data.iternum = 0
    stoptime = data.starttime + options.maxtime
    data.timeinit += Base.time_ns() - data.starttime
    # Initialize the linear problem
    data.timegradient += @elapsed_ns begin
            zero!(data.linsystem)
            cost = costgradhess!(data.linsystem, problem.variables, problem.costs)
        end
    data.gradientcomputations += 1
    data.bestcost = cost
    data.startcost = max(cost, data.startcost)
    # Do the iterations
    while true
        data.iternum += 1
        # Call the per iteration solver
        cost = iterate!(iteratedata, data, problem, options)::Float64
        computegradient = isequal(cost, -Inf)
        if computegradient
            # Construct the linear problem now, in order to compute the correct cost
            data.timegradient += @elapsed_ns begin
                zero!(data.linsystem)
                cost = costgradhess!(data.linsystem, problem.varnext, problem.costs)
            end
            data.gradientcomputations += 1
        end
        # Call the user-defined callback
        cost, terminate = callback(cost, problem, data, iteratedata)::Tuple{Float64, Int}
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
        # Check for termination
        maxstep = maximum(abs, data.linsystem.x)
        converged |= isinf(cost)                                     << 0 # Cost is infinite
        converged |= isnan(cost)                                     << 1 # Cost is NaN
        converged |= (dcost < data.bestcost * options.reldcost)      << 2 # Relative decrease in cost is too small
        converged |= (dcost < options.absdcost)                      << 3 # Absolute decrease in cost is too small
        converged |= isinf(maxstep)                                  << 4 # Infinity detected in the step
        converged |= isnan(maxstep)                                  << 5 # NaN detected in the step
        converged |= (maxstep < options.dstep)                       << 6 # Max of the step size is too small
        converged |= (fails > options.maxfails)                      << 7 # Max number of consecutive failed iterations reach
        converged |= (data.iternum >= options.maxiters)              << 8 # Max number of iterations reached
        converged |= (Base.time_ns() > stoptime)                     << 9 # Max amount of time exceeded
        converged |= terminate                                       << 16 # Terminated by the user-defined callback (room left for new flags above)
        if converged != 0
            break
        end
        if !computegradient
            # Construct the linear problem
            data.timegradient += @elapsed_ns begin
                zero!(data.linsystem)
                cost_ = costgradhess!(data.linsystem, problem.variables, problem.costs)
            end
            data.gradientcomputations += 1
            converged |= !isapprox(cost_, cost)                           << 10 # Cost computations don't agree
        end
    end
    if !(data.bestcost >= cost)
        # Update the problem variables to the best ones found
        updatefrombest!(problem, data)
    end
    # Return the data to produce the final result
    data.timetotal += Base.time_ns() - data.starttime
    return NLLSResult(data.startcost, data.bestcost, data.timetotal, data.timeinit, data.timecost, data.timegradient, data.timesolver, data.iternum, data.costcomputations, data.gradientcomputations, data.linearsolvers, converged)
end

# Optimizing variables one at a time (e.g. in alternation)
function optimizesinglesinternal!(problem::NLLSProblem, options::NLLSOptions, data::NLLSInternal{LST}, iteratedata, allcosts::CostStruct, costindices, varindices, first, result) where {LST<:UniVariateLS}
    iternum = result.niterations
    startcost = result.startcost
    bestcost = result.bestcost
    while first <= length(varindices)
        # Bail out if the variable size changes
        ind = varindices[first]
        if nvars(problem.variables[ind]) != length(data.linsystem.b)
            break
        end
        data.starttime = Base.time_ns()
        data.linsystem.varindex = UInt(ind)
        # Construct the subset of residuals that depend on this variable
        selectcosts!(problem.costs, allcosts, @inbounds(view(costindices.rowval, costindices.colptr[ind]:costindices.colptr[ind+1]-1)))
        # Reset the iterator data
        reset!(iteratedata, problem, data)
        # Optimize the subproblem
        optimizeinternal!(problem, options, data, iteratedata, nullcallback)
        # Increment
        startcost += data.startcost
        bestcost += data.bestcost
        iternum += data.iternum
        first += 1
    end
    return (first, NLLSResult(startcost, bestcost, result.timetotal+data.timetotal, result.timeinit+data.timeinit, result.timecost+data.timecost, result.timegradient+data.timegradient, result.timesolver+data.timesolver, iternum, result.costcomputations+data.costcomputations, result.gradientcomputations+data.gradientcomputations, result.linearsolvers+data.linearsolvers, 0))
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
