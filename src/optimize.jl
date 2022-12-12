using SparseArrays, LinearSolve
export optimize!, NLLSOptions, NLLSResult
export gaussnewton_iteration!

struct NLLSOptions
    dcost::Float64
    dstep::Float64
    maxfails::Int
    maxiters::Int
    iterator
    callback
    storetrajectory::Bool
end
function NLLSOptions(; maxiters=100, dcost=0.001, dstep=1.e-6, maxfails=3, iterator=gaussnewton_iteration!, callback=(args...)->false, storetrajectory=false)
    NLLSOptions(dcost, dstep, maxfails, maxiters, iterator, callback, storetrajectory)
end

mutable struct NLLSInternal{VarTypes}
    variables::Vector{VarTypes}
    gradient::Vector{Float64}
    hessian::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int}}
    step::Vector{Float64}
    blockoffsets::Vector{UInt}

    function NLLSInternal{VarTypes}(problem::NLLSProblem) where VarTypes
        # Compute the block offsets
        offsets = zeros(UInt, length(problem.variables))
        start = UInt(1)
        for index in eachindex(offsets)
            if problem.unfixed[index]
                @inbounds offsets[index] = start
                start += convert(UInt, nvars(problem.variables[index]))
            end
        end
        start -= 1
        # Initialize everything
        return new(Vector{VarTypes}(undef, length(offsets)), zeros(Float64, start), zeros(Float64, start, start), Vector{Float64}(undef, start), offsets)
    end
end

struct NLLSResult
    costs::Vector{Float64}
    trajectory::Vector{Vector{Float64}}
end

function optimize!(problem::NLLSProblem{VarTypes}, options::NLLSOptions=NLLSOptions()) where VarTypes
    # Set up all the internal data structures
    data = NLLSInternal{VarTypes}(problem)
    bestvariables = problem.variables
    # Initialize the linear problem
    bestcost = costgradhess!(data.gradient, data.hessian, problem.residuals, problem.variables, data.blockoffsets)
    # Initialize the results
    result = NLLSResult([bestcost], Vector{Vector{Float64}}())
    fails = 0
    # Do the iterations
    iter = 0
    while true
        iter += 1
        # Call the per iteration solver
        cost = options.iterator(data, problem)
        push!(result.costs, cost)
        # Store the best result
        dcost = bestcost - cost
        if dcost > 0
            bestcost = cost
            bestvariables = data.variables
            fails = 0
        else
            dcost = cost
            fails += 1
        end
        if options.storetrajectory
            # Store the variable trajectory (as update vectors)
            push!(result.trajectory, data.step)
        end
        # Check for convergence
        if options.callback(problem, data, cost) || (dcost < bestcost * options.dcost) || (maximum(abs, data.step) < options.dstep) || (fails > options.maxfails) || iter >= options.maxiters
            break
        end
        # Update the variables
        problem.variables .= data.variables
        # Construct the linear problem
        fill!(data.gradient, 0)
        fill!(data.hessian, 0)
        costgradhess!(data.gradient, data.hessian, problem.residuals, problem.variables, data.blockoffsets)
    end
    # Update the problem variables
    problem.variables .= bestvariables
    return result
end

function update!(variables, offsets, step)
    # Update each variable
    for index in eachindex(offsets)
        if offsets[index] != 0
            variables[index] = update(variables[index], step, offsets[index])
        end
    end
end

# Iterators assume that the linear problem has been constructed
function gaussnewton_iteration!(data::NLLSInternal, problem::NLLSProblem)
    # Compute the step
    data.step = -solve(LinearProblem(data.hessian, data.gradient)).u
    # Update the new variables
    data.variables = copy(problem.variables)
    update!(data.variables, data.blockoffsets, data.step)
    # Return the cost
    return cost(problem.residuals, data.variables)
end
