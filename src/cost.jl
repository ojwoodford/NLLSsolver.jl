using FLoops: @floop, @reduce
import ForwardDiff, NLLSsolver.valuedispatch, NLLSsolver.NLLSProblem, NLLSsolver.AbstractResidual
export cost, costgradhess!, computeresjac

cost(problem::NLLSProblem) = cost(problem.residuals, problem.variables)

function cost(residuals::IdDict, vars::Vector)::Float64
    # Compute the cost of all residuals in the problem
    c = 0.
    for (key, res) in residuals
        c += cost(res, vars)
    end
    return c 
    # return sum(x -> cost(x.second::Vector{x.first}, vars), residuals; init=0.)
end

function cost(residuals::Vector, vars::Vector)::Float64
    # Compute the total cost of all residuals in a container
    c = 0.
    # @floop for res in residuals
    #     c_ = cost(res, vars)
    #     @reduce c += c_
    # end
    for res in residuals
        c += cost(res, vars)
    end
    return c
end

function cost(residual::Residual, vars::Vector)::Float64 where Residual <: AbstractResidual
    # Get the variables required to compute the residual
    v = getvars(residual, vars)

    # Compute the residual
    r = computeresidual(residual, v...)
    
    # Compute the robustified cost
    return robustify(robustkernel(residual), r' * r)[1]
end

# Count the numbers of variables in the inputs
@inline function countvars(vars::Tuple, ::Val{varflags}) where varflags
    if isempty(vars)
        return 0
    end
    return sum(ntuple(i -> ((varflags >> (i - 1)) & 1) != 0 ? nvars(vars[i]) : 0, length(vars)))
end

# Generate the updated variables
@inline function updatevars(vars, varflags, advar)
    return ntuple(i -> ((varflags >> (i - 1)) & 1) != 0 ? update(vars[i], advar, countvars(vars[1:i-1], Val(varflags))+1) : vars[i], length(vars))
end

@inline function blockoffsets(vars, varflags, blockind)
    return ntuple(i -> SR(1, nvars(vars[i]) * ((varflags >> (i - 1)) & 1)) .+ (blockind[i] - 1), length(vars))
end

@inline function localoffsets(vars, varflags)
    return ntuple(i -> SR(1, nvars(vars[i]) * ((varflags >> (i - 1)) & 1)) .+ countvars(vars[1:i-1], Val(varflags)), length(vars))
end

# Automatic Jacobian computation
function computeresjac(::Val{varflags}, residual::Residual, vars...) where {varflags, Residual <: AbstractResidual}
    # Compute the residual
    res = computeresidual(residual, vars...)

    # Compute the Jacobian
    nres = length(res)
    type = eltype(res)
    nvars = countvars(vars, Val(varflags))
    Z = zeros(SVector{nvars, type})
    jac = ForwardDiff.jacobian(z -> computeresidual(residual, updatevars(vars, varflags, z)...), Z)::SMatrix{nres, nvars, type, nres*nvars}

    # Return both
    return res, jac
end

@inline function updatelinearsystem!(grad, hess::Matrix, g, H, vars, varflags, blockind, residual)
    # Update the blocks in the problem
    grad .+= g
    hess .+= H
end

@inline function updatelinearsystem!(grad, hess::BlockSparseMatrix, g, H, vars, varflags, blockind, residual)
    # Update the blocks in the problem
    goffsets = blockoffsets(vars, varflags, blockind)
    loffsets = localoffsets(vars, varflags)
    blocks = varindices(residual)
    for i in eachindex(vars)
        if ((varflags >> (i - 1)) & 1) == 1
            @inbounds view(grad, goffsets[i]) .+= g[loffsets[i]]
            @inbounds block(hess, blocks[i], blocks[i], nvars(vars[i]), nvars(vars[i])) .+= H[loffsets[i],loffsets[i]]
            for j in i+1:lastindex(vars)
                if ((varflags >> (j - 1)) & 1) == 1
                    @inbounds block(hess, blocks[i], blocks[j], nvars(vars[i]), nvars(vars[j])) .+= H[loffsets[i],loffsets[j]]
                end
            end
        end
    end
end

function gradhesshelper!(grad, hess, residual::Residual, vars::Vector, blockind, ::Val{varflags})::Float64 where {varflags, Residual <: AbstractResidual}
    # Get the variables
    v = getvars(residual, vars)

    # Compute the residual
    res, jac = computeresjac(Val(varflags), residual, v...)

    # Compute the robustified cost and the IRLS weight
    c, w1, w2 = robustify(robustkernel(residual), res' * res)

    # If this residual has a weight...
    if w1 != 0    
        # Compute the unrobust gradient and Hessian
        g = jac' * res
        H = jac' * jac
        # Check for robust case
        if w1 != 1
            # IRLS reweighting of Hessian
            H *= w1
            if w2 != 0
                # Second order correction
                H += (2 * w2) * g * g'
            end
            # IRLS reweighting of gradient
            g *= w1
        end
        # Update the blocks in the problem
        updatelinearsystem!(grad, hess, g, H, v, varflags, blockind, residual)
    end

    # Return the cost
    return c
end

function getoffsets(residual, blockoffsets::Vector{UInt})
    return blockoffsets[varindices(residual)]
end

function getoffsets(residual, blockoffsets::UInt)
    return convert.(UInt, SVector(varindices(residual)) .== blockoffsets)
end

function costgradhess!(grad, hess, residual::Residual, vars::Vector, blockoffsets) where Residual <: AbstractResidual
    # Get the bitset for the input variables, as an integer
    blockoff = getoffsets(residual, blockoffsets)
    varflags = foldl((x, y) -> (x << 1) + (y != 0), reverse(blockoff), init=UInt(0))

    # If there are no variables, just return the cost
    if varflags == 0
        return cost(residual, vars)
    end

    # Dispatch gradient computation based on the varflags, and return the cost
    # return gradhesshelper!(grad, hess, residual, vars, blockind, Val(varflags))
    return valuedispatch(Val(1), Val((2^nvars(residual))-1), v -> gradhesshelper!(grad, hess, residual, vars, blockoff, v), varflags)
end

function costgradhess!(grad, hess, residuals::Vector, vars::Vector, blockoffsets)::Float64
    # Go over all resdiduals, updating the gradient & hessian, and aggregating the cost 
    c = 0.
    # @floop 
    for res in residuals
        c_ = costgradhess!(grad, hess, res, vars, blockoffsets)
        # @reduce
        c += c_
    end
    return c
end

function costgradhess!(grad, hess, residuals::IdDict, vars::Vector, blockoffsets)::Float64
    # Go over all resdiduals in the problem
    # return sum(x -> costgradhess!(grad, hess, x.second::Vector{x.first}, vars, blockoffsets), residuals; init=0.)
    c = 0.
    for (key, res) in residuals
        c += costgradhess!(grad, hess, res, vars, blockoffsets)
    end
    return c 
end
