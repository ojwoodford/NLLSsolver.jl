using SparseArrays, Dates
import Printf.@printf
export NLLSOptions, NLLSResult, NLLSIterator

@enum NLLSIterator gaussnewton levenbergmarquardt dogleg
function Base.String(iterator::NLLSIterator) 
    if iterator == gaussnewton
        return "Gauss-Newton"
    end
    if iterator == levenbergmarquardt
        return "Levenberg-Marquardt"
    end
    if iterator == dogleg
        return "Dogleg"
    end
    return "Unknown iterator"
end

struct NLLSOptions
    reldcost::Float64           # Minimum relative reduction in cost required to avoid termination
    absdcost::Float64           # Minimum relative reduction in cost required to avoid termination
    dstep::Float64              # Minimum L-infinity norm of the update vector required to avoid termination
    maxfails::Int               # Maximum number of consecutive iterations that have a higher cost than the current best before termination
    maxiters::Int               # Maximum number of outer iterations
    iterator::NLLSIterator      # Inner iterator (see above for options)
    callback                    # Callback called every outer iteration - returns true to terminate
    storecosts::Bool            # Indicates whether the cost per outer iteration should be stored
    storetrajectory::Bool       # Indicates whether the step per outer iteration should be stored
end
function NLLSOptions(; maxiters=100, reldcost=1.e-6, absdcost=1.e-15, dstep=1.e-6, maxfails=3, iterator=levenbergmarquardt, callback=(args...)->false, storecosts=false, storetrajectory=false)
    NLLSOptions(reldcost, absdcost, dstep, maxfails, maxiters, iterator, callback, storecosts, storetrajectory)
end

struct NLLSResult
    startcost::Float64                      # The function cost prior to minimization
    bestcost::Float64                       # The lowest function cost achieved
    timetotal::Float64                      # The time (in seconds) to run the optimization (excluding initialization)
    timeinit::Float64                       # The time (in seconds) to initialize the internal data structures
    timecost::Float64                       # Time (in seconds) spent computing the cost
    timegradient::Float64                   # Time (in seconds) spent computing the residual gradients and constructing the linear problems
    timesolver::Float64                     # Time (in seconds) spent solving the linear problems
    niterations::Int                        # Number of outer optimization iterations performed
    costcomputations::Int                   # Number of cost computations performed
    gradientcomputations::Int               # Number of residual gradient computations performed
    linearsolvers::Int                      # Number of linear solves performed
    costs::Vector{Float64}                  # Vector of costs at the end of each outer iteration
    trajectory::Vector{Vector{Float64}}     # Vector of update vectors at the end of each outer iteration
end

function Base.show(io::IO, x::NLLSResult)
    timetotal = x.timetotal + x.timeinit
    otherstuff = x.timetotal - x.timecost - x.timegradient - x.timesolver
    @printf(io, "NLLSsolver optimization took %f seconds and %d iterations to reduce the cost from %f to %f (a %.2f%% reduction), using:
   %d cost computations in %f seconds (%.2f%% of total time),
   %d gradient computations in %f seconds (%.2f%% of total time),
   %d linear solver computations in %f seconds (%.2f%% of total time),
   %f seconds for initialization (%.2f%% of total time), and
   %f seconds for other stuff (%.2f%% of total time).\n", 
            timetotal, x.niterations, x.startcost, x.bestcost, 100*(1-x.bestcost/x.startcost), 
            x.costcomputations, x.timecost, 100*x.timecost/timetotal,
            x.gradientcomputations, x.timegradient, 100*x.timegradient/timetotal,
            x.linearsolvers, x.timesolver, 100*x.timesolver/timetotal,
            x.timeinit, 100*x.timeinit/timetotal,
            otherstuff, 100*otherstuff/timetotal)
end

mutable struct NLLSInternal{VarTypes}
    variables::Vector{VarTypes}
    bestvariables::Vector{VarTypes}
    linsystem::Union{UniVariateLS, MultiVariateLS}
    step::Vector{Float64}
    bestcost::Float64
    timecost::Float64
    timegradient::Float64
    timesolver::Float64
    iternum::Int
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int
    starttimens::UInt64

    function NLLSInternal{VarTypes}(problem::NLLSProblem, computehessian::Bool) where VarTypes
        starttimens = Base.time_ns()
        @assert length(problem.variables) > 0
        # Compute the block offsets
        unfixed = UInt(0)
        if typeof(problem.unfixed) == UInt
            # Single variable block
            nblocks = UInt(1)
            blockindices[problem.unfixed] = 1
            unfixed = UInt(problem.unfixed)
        else
            nblocks = UInt(sum(problem.unfixed))
            @assert nblocks > 0
        end
        # Construct the Hessian
        if nblocks == 1
            # One unfixed variable
            if unfixed == 0
                unfixed = UInt(findfirst(problem.unfixed))
            end
            varlen = UInt(nvars(problem.variables[unfixed]))
            linsystem = UniVariateLS(unfixed, varlen, computehessian ? varlen : UInt(lengthresiduals(problem.residuals)))
            return new(copy(problem.variables), copy(problem.variables), linsystem, Vector{Float64}(undef, varlen), 0., 0., 0., 0., 0, 0, 0, 0, starttimens)
        end

        # Multiple variables. Use a block sparse matrix
        mvls = computehessian ? makesymmvls(problem.variables, problem.residuals, problem.unfixed, nblocks) : makemvls(problem.variables, problem.residuals, problem.unfixed, nblocks)
        return new(copy(problem.variables), copy(problem.variables), mvls, Vector{Float64}(undef, size(mvls.A, 2)), 0., 0., 0., 0., 0, 0, 0, 0, starttimens)
    end
end
function NLLSInternal(problem::NLLSProblem{VarTypes}, computehessian) where VarTypes
    return NLLSInternal{VarTypes}(problem, computehessian)
end