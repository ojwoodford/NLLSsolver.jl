using SparseArrays, LinearSolve, Dates
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
    dcost::Float64
    dstep::Float64
    maxfails::Int
    maxiters::Int
    iterator::NLLSIterator
    linearsolver
    callback
    storecosts::Bool
    storetrajectory::Bool
end
function NLLSOptions(; maxiters=100, dcost=1.e-6, dstep=1.e-6, maxfails=3, iterator=gaussnewton, callback=(args...)->false, storecosts=false, storetrajectory=false, linearsolver=nothing)
    NLLSOptions(dcost, dstep, maxfails, maxiters, iterator, linearsolver, callback, storecosts, storetrajectory)
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
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int
    wallclock::DateTime

    function NLLSInternal{VarTypes}(problem::NLLSProblem, computehessian) where VarTypes
        @assert length(problem.variables) > 0
        # Compute the block offsets
        if typeof(problem.unfixed) == UInt
            # Single variable block
            nblocks = UInt(1)
            blockindices[problem.unfixed] = 1
            unfixed = UInt(problem.unfixed)
        else
            nblocks = UInt(sum(problem.unfixed))
            @assert nblocks > 0
            unfixed = UInt(findfirst(problem.unfixed))
        end
        # Construct the Hessian
        if nblocks == 1
            # One unfixed variable
            varlen = UInt(nvars(problem.variables[unfixed]))
            return new(copy(problem.variables), copy(problem.variables), UniVariateLS(unfixed, varlen), Vector{Float64}(undef, varlen), 0., 0., 0., 0., 0, 0, 0, now())
        end

        # Multiple variables. Use a block sparse matrix
        mvls = makemvls(problem.variables, problem.residuals, problem.unfixed, nblocks)
        return new(copy(problem.variables), copy(problem.variables), mvls, Vector{Float64}(undef, length(mvls.gradient)), 0., 0., 0., 0., 0, 0, 0, now())
    end
end
function NLLSInternal(problem::NLLSProblem{VarTypes}, computehessian) where VarTypes
    return NLLSInternal{VarTypes}(problem, computehessian)
end

struct NLLSResult
    startcost::Float64
    bestcost::Float64
    timetotal::Float64
    timeinit::Float64
    timecost::Float64
    timegradient::Float64
    timesolver::Float64
    niterations::Int
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int
    costs::Vector{Float64}
    trajectory::Vector{Vector{Float64}}
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