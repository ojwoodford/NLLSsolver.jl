# Uni-variate optimization (single unfixed variable)
optimize!(problem::NLLSProblem, options::NLLSOptions, unfixed::Integer, callback=nullcallback) = setupsinglevarls(optimizeinternal!, problem, options, unfixed, Stats(), callback)

# Multi-variate optimization
function optimize!(problem::NLLSProblem, options::NLLSOptions, unfixed::AbstractVector, callback=nullcallback)::NLLSResult
    startstats = Stats()
    @assert length(problem.variables) > 0
    # Compute the number of free variables (nblocks)
    nblocks = sum(unfixed)
    if nblocks == 1
        # One unfixed variable
        return setupsinglevarls(optimizeinternal!, problem, options, findfirst(unfixed), startstats, callback)
    else
        # Multiple variables
        return setupsmultivarls(optimizeinternal!, problem, options, unfixed, startstats, nblocks, callback)
    end
end

# Conversions for different types of "unfixed"
convertunfixed(::Nothing, problem) = trues(length(problem.variables))
convertunfixed(unfixed::DataType, problem) = isa.(problem.variables, unfixed)
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
    startstats = Stats()
    # Get the indices per cost
    costindices = sparse(getvarcostmap(problem)')
    # Put the costs aside
    allcosts = problem.costs
    problem.costs = CostStruct{CT}()
    # Group variables of the same size together to minimize allocations, so sort the indices by variable size
    if !sortedbysize
        indices = sort!(indices, lt=(i, j)->nvars(problem.variables[i])<=nvars(problem.variables[j]))
    end
    # Loop over all the variables
    result = NLLSResult(0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    first = 1
    while first <= length(indices)
        # Compute the run length of the current variable size
        varsz = dynamic(nvars(problem.variables[indices[first]]))::Int
        last = first + 1
        while last <= length(indices) && dynamic(nvars(problem.variables[indices[last]]))::Int == varsz
            last += 1
        end
        # Optimize all variables of the same size at once
        result = setupsinglevarls(optimizesinglesinternal!, problem, options, indices[first], startstats, allcosts, costindices, @inbounds(view(indices, first:last-1)), result)
        first = last
        startstats = Stats()
    end
    # Put the costs back
    problem.costs = allcosts
    return result
end

function setupsinglevarls(func, problem::NLLSProblem, options::NLLSOptions, unfixed::Integer, startstats, trailingargs...)::NLLSResult
    problem = checkvars!(problem)
    varlen = nvars(problem.variables[unfixed])
    if dynamic(is_static(varlen)) && varlen <= 16
        return setupstaticvarls(func, problem, options, unfixed, startstats, varlen, trailingargs)::NLLSResult
    else
        dynamicdata = NLLSInternal(LinearSystem{UniVariateLSdynamic_x, UniVariateLSdynamic_ls}((unfixed, dynamic(varlen)), (dynamic(varlen),)), startstats)
        return func(problem, options, dynamicdata, construct(options.iterator, problem, dynamicdata), trailingargs...)::NLLSResult
    end
end

function setupstaticvarls(func, problem::NLLSProblem, options::NLLSOptions, unfixed::Integer, startstats, varlen::StaticInt, trailingargs)::NLLSResult
    staticdata = NLLSInternal(LinearSystem{UniVariateLSstatic_x{dynamic(varlen)}, UniVariateLSstatic_ls{dynamic(varlen), dynamic(varlen*varlen)}}((unfixed,), ()), startstats)
    return func(problem, options, staticdata, construct(options.iterator, problem, staticdata), trailingargs...)::NLLSResult
end

function setupsmultivarls(func, problem::NLLSProblem, options::NLLSOptions, unfixed, startstats, nblocks, trailingargs...)::NLLSResult
    problem = checkvars!(problem)
    # Multiple variables. Decide whether to have a sparse or a dense system
    blocksizes, blockindices = block_sizes_indices(problem.variables, unfixed, nblocks)
    formarginalization = false # To be used in the future
    len = formarginalization ? 40 : sum(blocksizes)
    if len >= 40
        # Compute the block sparsity
        sparsity = block_sparsity(problem, unfixed)

        # Check sparsity level
        if formarginalization || sparse_dense_decision(len, sparsity, blocksizes)
            # Construct the BSM
            bsm = BlockSparseMatrix{Float64}(sparsity, blocksizes, blocksizes)

            # Construct the sparse MultiVariateLS
            sparsedata = NLLSInternal(MultiVariateLSsparse(bsm, blockindices, !formarginalization), startstats)
            return func(problem, options, sparsedata, construct(options.iterator, problem, sparsedata), trailingargs...)::NLLSResult
        end
    end

    # Construct the dense MultiVariateLS
    densedata = NLLSInternal(MultiVariateLSdense(blocksizes, blockindices), startstats)
    return func(problem, options, densedata, construct(options.iterator, problem, densedata), trailingargs...)::NLLSResult
end

# The meat of an optimization
@inline function optimizeloop!(problem::NLLSProblem, options::NLLSOptions, data, iteratedata, callback)
    # Initializations
    stoptime = Base.time_ns() + options.maxtime
    fails = 0
    converged = 0
    data.iternum = 0
    # Initialize the linear problem
    data.gradients += @elapsed_ns begin
            zero!(data.linsystem)
            cost = costgradhess!(data.linsystem, problem.variables, problem.costs)
        end
    data.bestcost = cost
    data.startcost = cost
    # Do the iterations
    while true
        data.iternum += 1
        # Call the per iteration solver
        cost = iterate!(iteratedata, data, problem, options)
        computegradient = isequal(cost, -Inf)
        if computegradient
            # Construct the linear problem now, in order to compute the correct cost
            data.gradients += @elapsed_ns begin
                zero!(data.linsystem)
                cost = costgradhess!(data.linsystem, problem.varnext, problem.costs)
            end
        end
        # Call the user-defined callback
        cost, terminate = callback(cost, problem, data, iteratedata)
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
        maxstep = maximum(abs, getx(data.linsystem))::Float64
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
            data.gradients += @elapsed_ns begin
                zero!(data.linsystem)
                cost_ = costgradhess!(data.linsystem, problem.variables, problem.costs)
            end
            converged |= !isapprox(cost_, cost)                      << 10 # Cost computations don't agree
        end
    end
    if !(data.bestcost >= cost)
        # Update the problem variables to the best ones found
        updatefrombest!(problem, data)
    end
    return converged
end

function optimizeinternal!(problem::NLLSProblem, options::NLLSOptions, data, iteratedata, callback)
    data.init = Stats()
    converged = optimizeloop!(problem, options, data, iteratedata, callback)
    data.optimize = Stats()
    return NLLSResult(data, converged)
end

# Optimizing variables one at a time (e.g. in alternation)
function optimizesinglesinternal!(problem::NLLSProblem, options::NLLSOptions, data::NLLSInternal{LST}, iteratedata, allcosts::CostStruct, costindices, varindices, result) where  LST<:NLLSsolver.UniVariateLS
    iternum = 0
    termination = 0
    startcost = 0.0
    bestcost = 0.0
    data.init = Stats()
    for ind in varindices
        data.linsystem = updateunfixed(data.linsystem, ind)
        # Construct the subset of residuals that depend on this variable
        selectcosts!(problem.costs, allcosts, @inbounds(view(costindices.rowval, costindices.colptr[ind]:costindices.colptr[ind+1]-1)))
        # Reset the iterator data
        reset!(iteratedata, problem, data)
        # Optimize the subproblem
        termination |= optimizeloop!(problem, options, data, iteratedata, nullcallback)
        # Increment stats
        startcost += data.startcost
        bestcost += data.bestcost
        iternum += data.iternum
    end
    data.iternum = iternum
    data.startcost = startcost
    data.bestcost = bestcost
    data.optimize = Stats()
    result += NLLSResult(data, termination)
    return result
end

function updatefromnext!(problem::NLLSProblem, ::NLLSInternalMultiVar)
    problem.variables, problem.varnext = problem.varnext, problem.variables
end

function updatefrombest!(problem::NLLSProblem, ::NLLSInternalMultiVar)
    problem.variables, problem.varbest = problem.varbest, problem.variables
end
updatetobest!(problem::NLLSProblem, data::NLLSInternalMultiVar) = updatefrombest!(problem, data)

updatefromnext!(problem::NLLSProblem, data) = updatefromnext!(problem, data.linsystem.x.varindex)
function updatefromnext!(problem::NLLSProblem, ind::UInt)
    @inbounds problem.variables[ind] = problem.varnext[ind]
end

updatefrombest!(problem::NLLSProblem, data) = updatefrombest!(problem, data.linsystem.x.varindex)
function updatefrombest!(problem::NLLSProblem, ind::UInt)
    @inbounds problem.variables[ind] = problem.varbest[ind]
end

updatetobest!(problem::NLLSProblem, data) = updatetobest!(problem, data.linsystem.x.varindex)
function updatetobest!(problem::NLLSProblem, ind::UInt)
    @inbounds problem.varbest[ind] = problem.variables[ind]
end
