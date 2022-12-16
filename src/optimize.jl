using SparseArrays, LinearSolve
import Printf.@printf
export optimize!, NLLSOptions, NLLSResult, NLLSIterator

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
    gradient::Vector{Float64}
    hessian::Union{Matrix{Float64}, BlockSparseMatrix{Float64}}
    step::Vector{Float64}
    blockoffsets::Vector{UInt}
    blockindices::Vector{UInt}
    bestcost::Float64
    lambda::Float64
    timecost::Float64
    timegradient::Float64
    timesolver::Float64
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int

    function NLLSInternal{VarTypes}(problem::NLLSProblem) where VarTypes
        # Compute the block offsets
        blockindices = zeros(UInt, length(problem.variables))
        blockoffsets = zeros(UInt, length(problem.variables))
        blocksizes = zeros(UInt, length(problem.variables))
        nblocks = UInt(0)
        if typeof(problem.unfixed) == UInt
            # Single variable block
            nblocks += 1
            blockindices[problem.unfixed] = 1
            blockoffsets[problem.unfixed] = 1
            @inbounds blocksizes[1] = nvars(problem.variables[problem.unfixed])
        else
            start = UInt(1)
            for (index, unfixed) in enumerate(problem.unfixed)
                if unfixed
                    nblocks += 1
                    @inbounds blockindices[index] = nblocks
                    @inbounds blockoffsets[index] = start
                    N = UInt(nvars(problem.variables[index]))
                    @inbounds blocksizes[nblocks] = N
                    start += N
                end
            end
        end
        # Construct the Hessian
        if nblocks == 1
            # One unfixed variable. Use a dense matrix hessian
            mat = zeros(Float64, blocksizes[1], blocksizes[1])
        else
            # Compute the block pairs

            # Use a block sparse matrix
            resize!(blocksizes, nblocks)
            mat = BlockSparseMatrix{Float64}(pairs, blocksizes, blocksizes)
        end
        # Initialize everything
        return new(deepcopy(problem.variables), zeros(Float64, start), mat, Vector{Float64}(undef, start), blockindices, blockindices, 0., 0., 0., 0., 0., 0, 0, 0)
    end
end
struct NLLSResult
    startcost::Float64
    bestcost::Float64
    timetotal::Float64
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
    otherstuff = x.timetotal - x.timecost - x.timegradient - x.timesolver
    @printf(io, "NLLSsolver optimization took %f seconds and %d iterations to reduce the cost from %f to %f (a %.2f%% reduction), using:
   %d cost computations in %f seconds (%.2f%% of total time),
   %d gradient computations in %f seconds (%.2f%% of total time),
   %d linear solver computations in %f seconds (%.2f%% of total time),
   %f seconds for other stuff (%.2f%% of total time).\n", 
            x.timetotal, x.niterations, x.startcost, x.bestcost, 100*(1-x.bestcost/x.startcost), 
            x.costcomputations, x.timecost, 100*x.timecost/x.timetotal,
            x.gradientcomputations, x.timegradient, 100*x.timegradient/x.timetotal,
            x.linearsolvers, x.timesolver, 100*x.timesolver/x.timetotal,
            otherstuff, 100*otherstuff/x.timetotal)
end

function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions=NLLSOptions())::NLLSResult where VarTypes
    t = @elapsed begin
        # Set up all the internal data structures
        data = NLLSInternal{VarTypes}(problem)
        bestvariables = Vector{VarTypes}(undef, length(data.variables))
        copy!(bestvariables, data.variables, problem.unfixed)
        # Initialize the linear problem
        data.timegradient += @elapsed data.bestcost = costgradhess!(data.gradient, data.hessian, problem.residuals, problem.variables, data.blockindices)
        data.gradientcomputations += 1
        # Initialize the results
        startcost = data.bestcost
        costs = Vector{Float64}()
        trajectory = Vector{Vector{Float64}}()
        fails = 0
        # Initialize the iterator
        if options.iterator == gaussnewton
            # Gauss-Newton or Newton
            iterator = newton_iteration!
        elseif options.iterator == levenbergmarquardt
            # Levenberg-Marquardt
            data.lambda = 1.
            iterator = levenberg_iteration!
        elseif options.iterator == dogleg
            # Dogleg
            data.lambda = 0.
            iterator = dogleg_iteration!
        end
        # Do the iterations
        iter = 0
        while true
            iter += 1
            # Call the per iteration solver
            cost = iterator(data, problem, options)
            if options.storecosts
                push!(costs, cost)
            end
            # Store the best result
            dcost = data.bestcost - cost
            if dcost > 0
                data.bestcost = cost
                copy!(bestvariables, data.variables, problem.unfixed)
                fails = 0
            else
                dcost = cost
                fails += 1
            end
            if options.storetrajectory
                # Store the variable trajectory (as update vectors)
                push!(trajectory, data.step)
            end
            # Check for convergence
            if options.callback(problem, data, cost) || !(dcost >= data.bestcost * options.dcost) || (maximum(abs, data.step) < options.dstep) || (fails > options.maxfails) || iter >= options.maxiters
                break
            end
            # Update the variables
            copy!(problem.variables, data.variables, problem.unfixed)
            # Construct the linear problem
            data.timegradient += @elapsed begin
                fill!(data.gradient, 0)
                zero!(data.hessian)
                costgradhess!(data.gradient, data.hessian, problem.residuals, problem.variables, data.blockindices)
            end
            data.gradientcomputations += 1
        end
        # Update the problem variables
        copy!(problem.variables, bestvariables, problem.unfixed)
    end
    # Return the result
    return NLLSResult(startcost, data.bestcost, t, data.timecost, data.timegradient, data.timesolver, iter, data.costcomputations, data.gradientcomputations, data.linearsolvers, costs, trajectory)
end

function update!(to::Vector, from::Vector, offsets::Vector, step)
    # Update each variable
    for (index, offset) in enumerate(offsets)
        if offset != 0
            to[index] = update(from[index], step, offset)
        end
    end
end

function update!(to::Vector, from::Vector, unfixed::UInt, step)
    # Update one variable
    to[unfixed] = update(from[unfixed], step)
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

# Iterators assume that the linear problem has been constructed
function newton_iteration!(data::NLLSInternal, problem::NLLSProblem, options::NLLSOptions)::Float64
    # Compute the step
    data.timesolver += @elapsed data.step = -solve(LinearProblem(data.hessian, data.gradient), options.linearsolver).u
    data.linearsolvers += 1
    # Update the new variables
    update!(data.variables, problem.variables, data.blockoffsets, data.step)
    # Return the cost
    data.timecost += @elapsed cost_ = cost(problem.residuals, data.variables)
    data.costcomputations += 1
    return cost_
end

function dogleg_iteration!(data::NLLSInternal, problem::NLLSProblem, options::NLLSOptions)::Float64
    data.timesolver += @elapsed begin
        # Compute the Cauchy step
        gnorm2 = data.gradient' * data.gradient
        a = gnorm2 / (data.gradient' * data.hessian * data.gradient + floatmin(eltype(data.gradient)))
        cauchy = -a * data.gradient
        alpha2 = a * a * gnorm2
        alpha = sqrt(alpha2)
        if data.lambda == 0
            # Make first step the Cauchy point
            data.lambda = alpha
        end
        if alpha < data.lambda
            # Compute the Newton step
            newton = -solve(LinearProblem(data.hessian, data.gradient), options.linearsolver).u
            beta = norm(newton)
            data.linearsolvers += 1
        end
    end
    cost_ = data.bestcost
    while true
        # Determine the step
        if !(alpha < data.lambda)
            # Along first leg
            data.step = (data.lambda / alpha) * cauchy
            linear_approx = data.lambda * (2 * alpha - data.lambda) / (2 * a)
        else
            # Along second leg
            if beta <= data.lambda
                # Do the full Newton step
                data.step = newton
                linear_approx = cost_
            else
                # Find the point along the Cauchy -> Newton line on the trust
                # region circumference
                leg = newton - cauchy
                sq_leg = leg' * leg
                c = cauchy' * leg
                step = sqrt(c * c + sq_leg * (data.lambda * data.lambda - alpha2))
                if c <= 0
                    step = (-c + step) / sq_leg
                else
                    step = (data.lambda * data.lambda - alpha2) / (c + step)
                end
                data.step = cauchy + step * leg
                linear_approx = 0.5 * (a * (1 - step) ^ 2 * gnorm2) + step * (2 - step) * cost_
            end
        end
        # Update the new variables
        update!(data.variables, problem.variables, data.blockoffsets, data.step)
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, data.variables)
        data.costcomputations += 1
        # Update lambda
        mu = (data.bestcost - cost_) / linear_approx
        if mu > 0.75
            data.lambda = max(data.lambda, 3 * norm(data.step))
        elseif mu < 0.25
            data.lambda *= 0.5
        end
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Return the cost
            return cost_
        end
    end
end

function levenberg_iteration!(data::NLLSInternal, problem::NLLSProblem, options::NLLSOptions)::Float64
    @assert data.lambda >= 0.
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        data.hessian += (data.lambda - lastlambda) * I
        lastlambda = data.lambda
        # Solve the linear system
        data.timesolver += @elapsed data.step = -solve(LinearProblem(data.hessian, data.gradient), options.linearsolver).u
        data.linearsolvers += 1
        # Update the new variables
        update!(data.variables, problem.variables, data.blockoffsets, data.step)
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, data.variables)
        data.costcomputations += 1
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Success (or convergence) - update lambda
            step_quality = (cost_ - data.bestcost) / (data.step' * (data.gradient + data.hessian * data.step * 0.5))
            data.lambda *= max(0.333, 1 - (step_quality - 1) ^ 3)
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        data.lambda *= mu;
        mu *= 2.;
    end
end
