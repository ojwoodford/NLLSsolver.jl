using SparseArrays, Dates, Static
import IfElse.ifelse
import Printf.@printf

@enum NLLSIterator gaussnewton newton levenbergmarquardt dogleg gradientdescent
function Base.String(iterator::NLLSIterator) 
    if iterator == newton || iterator == gaussnewton
        return "Newton"
    end
    if iterator == levenbergmarquardt
        return "Levenberg-Marquardt"
    end
    if iterator == dogleg
        return "Dogleg"
    end
    if iterator == gradientdescent
        return "Gradient descent"
    end
    return "Unknown iterator"
end

struct NLLSOptions
    reldcost::Float64           # Minimum relative reduction in cost required to avoid termination
    absdcost::Float64           # Minimum absolute reduction in cost required to avoid termination
    dstep::Float64              # Minimum L-infinity norm of the update vector required to avoid termination
    maxfails::Int               # Maximum number of consecutive iterations that have a higher cost than the current best before termination
    maxiters::Int               # Maximum number of outer iterations
    iterator::NLLSIterator      # Inner iterator (see above for options)
    callback                    # Callback called every outer iteration - (cost, problem, data) -> (newcost, terminate::Bool) where terminate == true ends the optimization
    storecosts::Bool            # Indicates whether the cost per outer iteration should be stored
    storetrajectory::Bool       # Indicates whether the step per outer iteration should be stored
end
function NLLSOptions(; maxiters=100, reldcost=1.e-15, absdcost=1.e-15, dstep=1.e-15, maxfails=3, iterator=levenbergmarquardt, callback=(cost, args...)->(cost, 0), storecosts=false, storetrajectory=false)
    if iterator == gaussnewton
        Base.depwarn("gaussnewton is deprecated. Use newton instead", :NLLSOptions)
    end
    NLLSOptions(reldcost, absdcost, dstep, maxfails, maxiters, iterator, callback, storecosts, storetrajectory)
end

# Utility callback that prints out per-iteration results
function printoutcallback(cost, problem, data, trailingargs...)
    if data.iternum == 1
        # First iteration, so print out column headers and the zeroth iteration (i.e. start) values
        println("iter      cost      cost_change    |step|")
        @printf("% 4d % 8e  % 4.3e   % 3.2e\n", 0, data.bestcost, 0, 0)
    end
    @printf("% 4d % 8e  % 4.3e   % 3.2e\n", data.iternum, cost, data.bestcost-cost, norm(data.linsystem.x))
    return cost, 0
end
function printoutcallback(cost, data, trradius::Float64)
    if data.iternum == 1
        # First iteration, so print out column headers and the zeroth iteration (i.e. start) values
        println("iter      cost      cost_change    |step|    tr_radius")
        @printf("% 4d % 8e  % 4.3e   % 3.2e   % 2.1e\n", 0, data.bestcost, 0, 0, trradius)
    end
    @printf("% 4d % 8e  % 4.3e   % 3.2e   % 2.1e\n", data.iternum, cost, data.bestcost-cost, norm(data.linsystem.x), trradius)
    return cost, 0
end

struct NLLSResult
    startcost::Float64                      # The function cost prior to minimization
    bestcost::Float64                       # The lowest function cost achieved
    timetotal::Float64                      # The total time (in seconds) taken to run the optimization
    timeinit::Float64                       # The time (in seconds) to initialize the internal data structures
    timecost::Float64                       # Time (in seconds) spent computing the cost
    timegradient::Float64                   # Time (in seconds) spent computing the residual gradients and constructing the linear problems
    timesolver::Float64                     # Time (in seconds) spent solving the linear problems
    termination::Int                        # Set of flags indicating which termination criteria were met
    niterations::Int                        # Number of outer optimization iterations performed
    costcomputations::Int                   # Number of cost computations performed
    gradientcomputations::Int               # Number of residual gradient computations performed
    linearsolvers::Int                      # Number of linear solves performed
    costs::Vector{Float64}                  # Vector of costs at the end of each outer iteration
    trajectory::Vector{Vector{Float64}}     # Vector of update vectors at the end of each outer iteration
end

function Base.show(io::IO, x::NLLSResult)
    otherstuff = x.timetotal - x.timecost - x.timegradient - x.timesolver - x.timeinit
    @printf(io, "NLLSsolver optimization took %f seconds and %d iterations to reduce the cost from %f to %f (a %.2f%% reduction), using:
   %d cost computations in %f seconds (%.2f%% of total time),
   %d gradient computations in %f seconds (%.2f%% of total time),
   %d linear solver computations in %f seconds (%.2f%% of total time),
   %f seconds for initialization (%.2f%% of total time), and
   %f seconds for other stuff (%.2f%% of total time).\n", 
            x.timetotal, x.niterations, x.startcost, x.bestcost, 100*(1-x.bestcost/x.startcost), 
            x.costcomputations, x.timecost, 100*x.timecost/x.timetotal,
            x.gradientcomputations, x.timegradient, 100*x.timegradient/x.timetotal,
            x.linearsolvers, x.timesolver, 100*x.timesolver/x.timetotal,
            x.timeinit, 100*x.timeinit/x.timetotal,
            otherstuff, 100*otherstuff/x.timetotal)
    if 0 != x.termination           ; println(io, "Reason(s) for termination:"); end
    if 0 != x.termination & (1 << 0); println(io, "   Cost is infinite."); end
    if 0 != x.termination & (1 << 1); println(io, "   Cost is NaN."); end
    if 0 != x.termination & (1 << 2); println(io, "   Relative decrease in cost below threshold."); end
    if 0 != x.termination & (1 << 3); println(io, "   Absolute decrease in cost below threshold."); end
    if 0 != x.termination & (1 << 4); println(io, "   Step contains an infinite value."); end
    if 0 != x.termination & (1 << 5); println(io, "   Step contains a NaN."); end
    if 0 != x.termination & (1 << 6); println(io, "   Step size below threshold."); end
    if 0 != x.termination & (1 << 7); println(io, "   Too many consecutive iterations increasing the cost."); end
    if 0 != x.termination & (1 << 8); println(io, "   Maximum number of outer iterations reached."); end
    userflags = x.termination >> 9
    if 0 != userflags; println(io, "   Terminated by user-defined callback, with flags: ", string(userflags, base=2)); end
end

mutable struct NLLSInternal{LSType}
    bestcost::Float64
    timecost::UInt64
    timegradient::UInt64
    timesolver::UInt64
    iternum::Int
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int
    linsystem::LSType

    function NLLSInternal(linsystem::LSType) where LSType
        return new{LSType}(0., 0, 0, 0, 0, 0, 0, 0, linsystem)
    end
end
NLLSInternal(unfixed::UInt, varlen) = NLLSInternal(ifelse(is_static(varlen), UniVariateLSstatic{dynamic(varlen), dynamic(varlen*varlen)}(unfixed), UniVariateLSdynamic(unfixed, dynamic(varlen))))

NLLSInternalMultiVar = Union{NLLSInternal{MultiVariateLSdense}, NLLSInternal{MultiVariateLSsparse}}
NLLSInternalSingleVar = Union{NLLSInternal{UniVariateLSstatic{N, N2}}, NLLSInternal{UniVariateLSdynamic}} where {N, N2}
