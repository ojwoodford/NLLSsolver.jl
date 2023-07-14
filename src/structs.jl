using SparseArrays, Dates, Static
import Printf.@printf

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
    absdcost::Float64           # Minimum absolute reduction in cost required to avoid termination
    dstep::Float64              # Minimum L-infinity norm of the update vector required to avoid termination
    maxfails::Int               # Maximum number of consecutive iterations that have a higher cost than the current best before termination
    maxiters::Int               # Maximum number of outer iterations
    iterator::NLLSIterator      # Inner iterator (see above for options)
    callback                    # Callback called every outer iteration - returns true to terminate
    storecosts::Bool            # Indicates whether the cost per outer iteration should be stored
    storetrajectory::Bool       # Indicates whether the step per outer iteration should be stored
end
function NLLSOptions(; maxiters=100, reldcost=1.e-15, absdcost=1.e-15, dstep=1.e-15, maxfails=3, iterator=levenbergmarquardt, callback=(args...)->false, storecosts=false, storetrajectory=false)
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
    termination::Int                        # Set of flags indicating which termination criteria were met
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
    println(io, "Reason(s) for termination:")
    if 0 != x.termination & (1 << 0); println(io, "   Terminated by user-defined callback."); end
    if 0 != x.termination & (1 << 1); println(io, "   Cost is infinite."); end
    if 0 != x.termination & (1 << 2); println(io, "   Cost is NaN."); end
    if 0 != x.termination & (1 << 3); println(io, "   Relative decrease in cost below threshold."); end
    if 0 != x.termination & (1 << 4); println(io, "   Absolute decrease in cost below threshold."); end
    if 0 != x.termination & (1 << 5); println(io, "   Step contains an infinite value."); end
    if 0 != x.termination & (1 << 6); println(io, "   Step contains a NaN."); end
    if 0 != x.termination & (1 << 7); println(io, "   Step size below threshold."); end
    if 0 != x.termination & (1 << 8); println(io, "   Too many consecutive iterations increasing the cost."); end
    if 0 != x.termination & (1 << 9); println(io, "   Maximum number of outer iterations reached."); end
end

mutable struct NLLSInternalSingleVar
    bestcost::Float64
    timecost::Float64
    timegradient::Float64
    timesolver::Float64
    iternum::Int
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int
    step::Vector{Float64}
    linsystem::UniVariateLS

    function NLLSInternalSingleVar(unfixed::UInt, varlen::Integer, n::Integer)
        return new(0., 0., 0., 0., 0, 0, 0, 0, Vector(undef, varlen), UniVariateLS(unfixed, varlen, n))
    end
end

mutable struct NLLSInternalMultiVar
    bestcost::Float64
    timecost::Float64
    timegradient::Float64
    timesolver::Float64
    iternum::Int
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int
    step::Vector{Float64}
    linsystem::MultiVariateLS

    # Maps of dependencies
    varresmap::SparseMatrixCSC{Bool, Int}
    resvarmap::SparseMatrixCSC{Bool, Int}
    varvarmap::SparseMatrixCSC{Bool, Int}
    mapsvalid::Bool

    function NLLSInternalMultiVar(mvls)
        return new(0., 0., 0., 0., 0, 0, 0, 0, Vector{Float64}(undef, size(mvls.A, 2)), mvls, spzeros(Bool, 0, 0), spzeros(Bool, 0, 0), spzeros(Bool, 0, 0), false)
    end
end

function updatevarresmap!(varresmap::SparseMatrixCSC{Bool, Int}, residuals::Vector, colind::Int, rowind::Int)
    numres = length(residuals)
    if numres > 0
        ndeps_ = known(ndeps(residuals[1]))
        srange = SR(0, ndeps_-1)
        @inbounds for res in residuals
            varresmap.rowval[srange.+rowind] .= varindices(res)
            rowind += ndeps_
            colind += 1
            varresmap.colptr[colind] = rowind
        end
    end
    return colind, rowind
end

function updatevarresmap!(varresmap::SparseMatrixCSC{Bool, Int}, residuals::ResidualStruct)
    # Pre-allocate all the necessary memory
    resize!(varresmap.rowval, countresiduals(resdeps, residuals))
    resize!(varresmap.colptr, countresiduals(reslen, residuals)+1)
    prevlen = length(varresmap.nzval)
    resize!(varresmap.nzval, length(varresmap.rowval))

    # Fill in the arrays
    varresmap.nzval[prevlen+1:length(varresmap.rowval)] .= true
    varresmap.colptr[1] = 1
    colind = 1
    rowind = 1
    @inbounds for res in values(residuals)
        colind, rowind = updatevarresmap!(varresmap, res, colind, rowind)
    end
end
