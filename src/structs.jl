using SparseArrays, Static
import Printf.@printf

@enum NLLSIterator newton levenbergmarquardt dogleg gradientdescent
Base.String(iterator::NLLSIterator) = String(getiteratortype(iterator))

struct NLLSOptions
    reldcost::Float64           # Minimum relative reduction in cost required to avoid termination
    absdcost::Float64           # Minimum absolute reduction in cost required to avoid termination
    dstep::Float64              # Minimum L-infinity norm of the update vector required to avoid termination
    maxfails::Int               # Maximum number of consecutive iterations that have a higher cost than the current best before termination
    maxiters::Int               # Maximum number of outer iterations
    maxtime::UInt64             # Maximum optimization time allowed, in nano-seconds (converted from seconds in the constructor)
    iterator::Type              # Type of the inner iterator (see above for options)
    numthreads::StaticInt       # Number of threads to use for parallel computations - currently not used, but may be in the future
end
function NLLSOptions(; maxiters=100, reldcost=1.e-15, absdcost=1.e-15, dstep=1.e-15, maxfails=3, maxtime=30.0, iterator::NLLSIterator=levenbergmarquardt, callback=nothing, iteratordata=nothing, numthreads::StaticInt=StaticInt(Threads.nthreads()))
    @assert(isnothing(callback), "Callbacks should be passed directly to optimize!, not to the options struct.")
    @assert(isnothing(iteratordata), "Iteratordata should not be passed to the options struct.")
    NLLSOptions(reldcost, absdcost, dstep, maxfails, maxiters, UInt64(round(maxtime * 1e9)), getiteratortype(iterator), numthreads)
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
    niterations::Int                        # Number of outer optimization iterations performed
    costcomputations::Int                   # Number of cost computations performed
    gradientcomputations::Int               # Number of residual gradient computations performed
    linearsolvers::Int                      # Number of linear solves performed
    # Termination reason
    termination::Int                        # Set of flags indicating which termination criteria were met - the value should not be relied upon
end

function Base.show(io::IO, x::NLLSResult)
    otherstuff = x.timetotal - x.timecost - x.timegradient - x.timesolver - x.timeinit
    @printf(io, "NLLSsolver optimization took %f seconds and %d iterations to reduce the cost from %e to %e (a %.2f%% reduction), using:
   %d cost computations in %f seconds (%.2f%% of total time),
   %d gradient computations in %f seconds (%.2f%% of total time),
   %d linear solver computations in %f seconds (%.2f%% of total time),
   %f seconds for initialization (%.2f%% of total time), and
   %f seconds for other stuff (%.2f%% of total time).\n", 
            x.timetotal*1e-9, x.niterations, x.startcost, x.bestcost, 100*(1-x.bestcost/x.startcost), 
            x.costcomputations, x.timecost*1e-9, 100.0*x.timecost/x.timetotal,
            x.gradientcomputations, x.timegradient*1e-9, 100.0*x.timegradient/x.timetotal,
            x.linearsolvers, x.timesolver*1e-9, 100.0*x.timesolver/x.timetotal,
            x.timeinit*1e-9, 100.0*x.timeinit/x.timetotal,
            otherstuff*1e-9, 100.0*otherstuff/x.timetotal)
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
    starttime::UInt64
    timetotal::UInt64
    timeinit::UInt64
    # Stats for key computations
    costs::Stats
    gradients::Stats
    solves::Stats
    # Counts
    iternum::Int
    # Linear system
    linsystem::LSType  

    function NLLSInternal(linsystem::LSType, starttimens) where LSType
        return new{LSType}(0., 0., starttimens, 0, 0, Stats(), Stats(), Stats(), 0, linsystem)
    end
end
@inline NLLSInternal(unfixed::UInt, varlen, starttimens) = NLLSInternal(dynamic(is_static(varlen)) && varlen <= 16 ? LinearSystem{UniVariateLSstatic_x{dynamic(varlen)}, UniVariateLSstatic_ls{dynamic(varlen), dynamic(varlen*varlen)}}((unfixed,), ()) : 
                                                                                                                     LinearSystem{UniVariateLSdynamic_x, UniVariateLSdynamic_ls}((unfixed, dynamic(varlen)), (dynamic(varlen),)), starttimens)

NLLSInternalMultiVar = Union{NLLSInternal{MultiVariateLSdense}, NLLSInternal{MultiVariateLSsparse}}
NLLSInternalSingleVar = NLLSInternal{UniVariateLS}
