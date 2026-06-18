using SparseArrays, Static
import Printf.@printf

@enum NLLSIterator newton levenbergmarquardt dogleg gradientdescent

struct NLLSOptions{S, N}
    reldcost::Float64           # Minimum relative reduction in cost required to avoid termination
    absdcost::Float64           # Minimum absolute reduction in cost required to avoid termination
    dstep::Float64              # Minimum L-infinity norm of the update vector required to avoid termination
    maxfails::Int               # Maximum number of consecutive iterations that have a higher cost than the current best before termination
    maxiters::Int               # Maximum number of outer iterations
    maxtime::UInt64             # Maximum optimization time allowed, in nano-seconds (converted from seconds in the constructor)
    iterator::StaticSymbol{S}   # Static Symbol of the inner iterator (see above for options for iterators)
    numthreads::StaticInt{N}    # Number of threads to use for parallel computations

    function NLLSOptions(reldcost, abscost, dstep, maxfails, maxiters, maxtime, iterator::StaticSymbol{S}, numthreads::StaticInt{N}) where {S, N}
        return new{S, N}(reldcost, abscost, dstep, maxfails, maxiters, maxtime, iterator, numthreads)
    end
end
function NLLSOptions(; maxiters=100, reldcost=1.e-15, absdcost=1.e-15, dstep=1.e-15, maxfails=3, maxtime=30.0, iterator=:levenbergmarquardt, callback=nothing, iteratordata=nothing, numthreads=StaticInt(Threads.nthreads()))
    @assert(isnothing(callback), "Callbacks should be passed directly to optimize!, not to the options struct.")
    @assert(isnothing(iteratordata), "Iteratordata should not be passed to the options struct.")
    NLLSOptions(reldcost, absdcost, dstep, maxfails, maxiters, UInt64(round(maxtime * 1e9)), static(Symbol(iterator)), static(numthreads))
end

struct NLLSResult
    # Costs
    startcost::Float64                    # The function cost prior to minimization
    bestcost::Float64                     # The lowest function cost achieved
    # Times (nano-seconds)
    timetotal::UInt64                      # The total time taken to run the optimization
    timeinit::UInt64                       # The time to initialize the internal data structures
    timecost::UInt64                       # Time spent computing the cost
    timegradient::UInt64                   # Time spent computing the residual gradients and constructing the linear problems
    timesolver::UInt64                     # Time spent solving the linear problems
    # Counts
    niterations::Int                       # Number of outer optimization iterations performed
    costcomputations::Int                  # Number of cost computations performed
    gradientcomputations::Int              # Number of residual gradient computations performed
    linearsolvers::Int                     # Number of linear solves performed
    # Termination reason
    termination::Int                       # Set of flags indicating which termination criteria were met - the value should not be relied upon
end
NLLSResult(data, termination) = NLLSResult(data.startcost, 
                                           data.bestcost, 
                                           data.optimize.time_ns-data.start.time_ns, 
                                           data.init.time_ns-data.start.time_ns, 
                                           data.costs.time_ns, 
                                           data.gradients.time_ns, 
                                           data.solves.time_ns, 
                                           data.iternum, 
                                           data.costs.count, 
                                           data.gradients.count, 
                                           data.solves.count, 
                                           termination)
Base.:+(a::NLLSResult, b::NLLSResult) = NLLSResult(a.startcost+b.startcost, 
                                                   a.bestcost+b.bestcost, 
                                                   a.timetotal+b.timetotal, 
                                                   a.timecost+b.timecost, 
                                                   a.timeinit+b.timeinit, 
                                                   a.timegradient+b.timegradient, 
                                                   a.timesolver+b.timesolver, 
                                                   a.niterations+b.niterations, 
                                                   a.costcomputations+b.costcomputations, 
                                                   a.gradientcomputations+b.gradientcomputations, 
                                                   a.linearsolvers+b.linearsolvers, 
                                                   a.termination|b.termination)

function Base.show(io::IO, x::NLLSResult)
    optimtime = x.timetotal - x.timeinit
    timedoptim = max(x.timecost + x.timegradient + x.timesolver, optimtime)
    @printf(io, "NLLSsolver optimization took %f seconds and %d iterations to reduce the cost from %e to %e (a %.2f%% reduction), using:
   %f seconds for initialization (%.2f%% of total time), and
   %f seconds for optimization (%.2f%% of total time), of which:
        %d cost computations accounted for %.2f%% of the time,
        %d gradient computations accounted for %.2f%%, and
        %d linear solver computations accounted for %.2f%%.\n", 
            x.timetotal*1e-9, x.niterations, x.startcost, x.bestcost, 100*(1-x.bestcost/x.startcost),
            x.timeinit*1e-9, 100.0*x.timeinit/x.timetotal,
            optimtime*1e-9, 100.0*optimtime/x.timetotal,
            x.costcomputations, 100.0*x.timecost/timedoptim,
            x.gradientcomputations, 100.0*x.timegradient/timedoptim,
            x.linearsolvers, 100.0*x.timesolver/timedoptim)
    if 0 != x.termination           ; println(io,  "Reason(s) for termination:"); end
    if 0 != x.termination & (1 << 0); println(io,  "   Cost is infinite."); end
    if 0 != x.termination & (1 << 1); println(io,  "   Cost is NaN."); end
    if 0 != x.termination & (1 << 2); println(io,  "   Relative decrease in cost below threshold."); end
    if 0 != x.termination & (1 << 3); println(io,  "   Absolute decrease in cost below threshold."); end
    if 0 != x.termination & (1 << 4); println(io,  "   Step contains an infinite value."); end
    if 0 != x.termination & (1 << 5); println(io,  "   Step contains a NaN."); end
    if 0 != x.termination & (1 << 6); println(io,  "   Step size below threshold."); end
    if 0 != x.termination & (1 << 7); println(io,  "   Too many consecutive iterations increasing the cost."); end
    if 0 != x.termination & (1 << 8); println(io,  "   Maximum number of outer iterations reached."); end
    if 0 != x.termination & (1 << 9); println(io,  "   Maximum allowed computation time exceeded."); end
    if 0 != x.termination & (1 << 10); println(io, "   Internal discrepancy in cost computations."); end
    userflags = x.termination >> 16
    if 0 != userflags; println(io, "   Terminated by user-defined callback, with flags: ", string(userflags, base=2)); end
end

mutable struct NLLSInternal{LSType}
    # Costs
    startcost::Float64
    bestcost::Float64
    # Times (nano-seconds)
    start::Stats
    init::Stats
    optimize::Stats
    # Stats for key computations
    costs::TimeCounter
    gradients::TimeCounter
    solves::TimeCounter
    # Counts
    iternum::Int
    # Linear system
    linsystem::LSType  

    function NLLSInternal(linsystem::LSType, startstats) where LSType
        return new{LSType}(0., 0., startstats, Stats(0), Stats(0), TimeCounter(), TimeCounter(), TimeCounter(), 0, linsystem)
    end
end

NLLSInternalMultiVar = Union{NLLSInternal{MultiVariateLSdense}, NLLSInternal{MultiVariateLSsparse}}
NLLSInternalSingleVar = NLLSInternal{UniVariateLS}
