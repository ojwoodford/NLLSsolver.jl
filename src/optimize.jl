using SparseArrays, LinearSolve, Dates
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
    linsystem::Union{UniVariateLS, MultiVariateLS}
    step::Vector{Float64}
    bestcost::Float64
    lambda::Float64
    timecost::Float64
    timegradient::Float64
    timesolver::Float64
    costcomputations::Int
    gradientcomputations::Int
    linearsolvers::Int
    wallclock::DateTime

    function NLLSInternal{VarTypes}(problem::NLLSProblem) where VarTypes
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
            return new(copy(problem.variables), UniVariateLS(unfixed, varlen), Vector{Float64}(undef, varlen), 0., 0., 0., 0., 0., 0, 0, 0, now())
        end

        # Multiple variables. Use a block sparse matrix
        mvls = MultiVariateLS(problem.variables, problem.residuals, problem.unfixed, nblocks)
        return new(copy(problem.variables), mvls, Vector{Float64}(undef, length(mvls.gradient)), 0., 0., 0., 0., 0., 0, 0, 0, now())
    end
end
function NLLSInternal(problem::NLLSProblem{VarTypes}) where VarTypes
    return NLLSInternal{VarTypes}(problem)
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
        data.timegradient += @elapsed data.bestcost = costgradhess!(data.linsystem, problem.residuals, problem.variables)
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
                push!(trajectory, copy(data.step))
            end
            # Check for convergence
            if options.callback(problem, data, cost) || !(dcost >= data.bestcost * options.dcost) || (maximum(abs, data.step) < options.dstep) || (fails > options.maxfails) || iter >= options.maxiters
                break
            end
            # Update the variables
            copy!(problem.variables, data.variables, problem.unfixed)
            # Construct the linear problem
            data.timegradient += @elapsed begin
                zero!(data.linsystem)
                costgradhess!(data.linsystem, problem.residuals, problem.variables)
            end
            data.gradientcomputations += 1
        end
        # Update the problem variables
        copy!(problem.variables, bestvariables, problem.unfixed)
    end
    # Return the result
    return NLLSResult(startcost, data.bestcost, t, data.timecost, data.timegradient, data.timesolver, iter, data.costcomputations, data.gradientcomputations, data.linearsolvers, costs, trajectory)
end

function update!(to::Vector, from::Vector, linsystem::MultiVariateLS, step)
    # Update each variable
    @inbounds for (i, j) in enumerate(linsystem.blockindices)
        if j != 0
            a = update(from[i], step, linsystem.gradoffsets[j])
            to[i] = a
        end
    end
end

function update!(to::Vector, from::Vector, linsystem::UniVariateLS, step)
    # Update one variable
    to[linsystem.varindex] = update(from[linsystem.varindex], step)
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

function linearsolve(A::AbstractMatrix, b::AbstractVector, options)
    return solve(LinearProblem(A, b), options).u
end

# Iterators assume that the linear problem has been constructed
function newton_iteration!(data::NLLSInternal, problem::NLLSProblem, options::NLLSOptions)::Float64
    # Compute the step
    data.timesolver += @elapsed begin
        hessian, gradient = gethessgrad(data.linsystem)
        data.step .= -linearsolve(hessian, gradient, options.linearsolver)
    end
    data.linearsolvers += 1
    # Update the new variables
    update!(data.variables, problem.variables, data.linsystem, data.step)
    # Return the cost
    data.timecost += @elapsed cost_ = cost(problem.residuals, data.variables)
    data.costcomputations += 1
    return cost_
end

function dogleg_iteration!(data::NLLSInternal, problem::NLLSProblem, options::NLLSOptions)::Float64
    hessian, gradient = gethessgrad(data.linsystem)
    data.timesolver += @elapsed begin
        # Compute the Cauchy step
        gnorm2 = gradient' * gradient
        a = gnorm2 / ((gradient' * hessian) * gradient + floatmin(eltype(gradient)))
        cauchy = -a * gradient
        alpha2 = a * a * gnorm2
        alpha = sqrt(alpha2)
        if data.lambda == 0
            # Make first step the Cauchy point
            data.lambda = alpha
        end
        if alpha < data.lambda
            # Compute the Newton step
            data.step .= -linearsolve(hessian, gradient, options.linearsolver)
            beta = norm(data.step)
            data.linearsolvers += 1
        end
    end
    cost_ = data.bestcost
    while true
        # Determine the step
        if !(alpha < data.lambda)
            # Along first leg
            data.step .= (data.lambda / alpha) * cauchy
            linear_approx = data.lambda * (2 * alpha - data.lambda) / (2 * a)
        else
            # Along second leg
            if beta <= data.lambda
                # Do the full Newton step
                linear_approx = cost_
            else
                # Find the point along the Cauchy -> Newton line on the trust
                # region circumference
                data.step .-= cauchy
                sq_leg = data.step' * data.step
                c = cauchy' * data.step
                step = sqrt(c * c + sq_leg * (data.lambda * data.lambda - alpha2))
                if c <= 0
                    step = (-c + step) / sq_leg
                else
                    step = (data.lambda * data.lambda - alpha2) / (c + step)
                end
                data.step .*= step
                data.step .+= cauchy
                linear_approx = 0.5 * (a * (1 - step) ^ 2 * gnorm2) + step * (2 - step) * cost_
            end
        end
        # Update the new variables
        update!(data.variables, problem.variables, data.linsystem, data.step)
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
    hessian, gradient = gethessgrad(data.linsystem)
    lastlambda = 0.
    mu = 2.
    while true
        # Dampen the Hessian
        uniformscaling!(hessian, data.lambda - lastlambda)
        lastlambda = data.lambda
        # Solve the linear system
        data.timesolver += @elapsed data.step .= -linearsolve(hessian, gradient, options.linearsolver)
        data.linearsolvers += 1
        # Update the new variables
        update!(data.variables, problem.variables, data.linsystem, data.step)
        # Compute the cost
        data.timecost += @elapsed cost_ = cost(problem.residuals, data.variables)
        data.costcomputations += 1
        # Check for exit
        if !(cost_ > data.bestcost) || (maximum(abs, data.step) < options.dstep)
            # Success (or convergence) - update lambda
            step_quality = (data.bestcost - cost_) / (((data.step' * hessian) * 0.5 - gradient') * data.step)
            data.lambda *= max(0.333, 1 - (step_quality - 1) ^ 3)
            # Return the cost
            return cost_
        end
        # Failure - increase lambda
        data.lambda *= mu;
        mu *= 2.;
    end
end
