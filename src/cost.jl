import ForwardDiff, NLLSsolver.valuedispatch
export cost, costgradhess!

function cost(residuals, vars::Vector)
    # Compute the total cost of all residuals
    c = 0.
    for res in residuals
        c += cost(res, vars)
    end
    return c
end

function cost(residual::Residual, vars::Vector) where Residual <: AbstractResidual
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
    return ntuple(i -> ((varflags >> (i - 1)) & 1) != 0 ? update(vars[i], advar, countvars(vars[1:i-1], Val(varflags))) : vars[i], length(vars))
end

# Compute the offsets of the variables
@inline function computeoffsets(vars, varflags, blockind)
    return vcat(ntuple(i -> SR(1, nvars(vars[i]) * ((varflags >> (i - 1)) & 1)) .+ (blockind[i] - 1), length(vars))...)
end

function updateblocks!(grad, hess, res, jac, w, blockoffsets)
    # IRLS weighting of the residual and Jacobian 
    if w != 1
        res = res * w
        jac = jac * w
    end

    # Update the blocks in the problem
    grad[blockoffsets] += jac' * res
    hess[blockoffsets, blockoffsets] += jac' * jac
    return nothing
end

# Automatic Jacobian computation
function computejacobian(residual::Residual, vars, ::Val{varflags}) where {varflags, Residual <: AbstractResidual}
    N = countvars(vars, Val(varflags))
    Z = zeros(SVector{N, eltype(residual)})
    return ForwardDiff.jacobian(z -> computeresidual(residual, updatevars(vars, varflags, z)...), Z)::SMatrix{reslen(residual), N, eltype(residual), reslen(residual)*N}
end

function gradhesshelper!(grad, hess, residual::Residual, vars::Vector, blockind, ::Val{varflags}) where {varflags, Residual <: AbstractResidual}
    # Get the variables
    v = getvars(residual, vars)

    # Compute the residual
    res = computeresidual(residual, v...)

    # Compute the robustified cost and the IRLS weight
    c, w = robustify(robustkernel(residual), res' * res)

    # If this residual has a weight...
    if w > 0
        # Compute the Jacobian
        jac = computejacobian(residual, v, Val(varflags))
    
        # Update the blocks in the problem
        updateblocks!(grad, hess, res, jac, w, computeoffsets(v, varflags, blockind))
    end

    # Return the cost
    return c
end

function costgradhess!(grad, hess, residual::Residual, vars::Vector, blockindex::Vector{Int}) where Residual <: AbstractResidual
    # Get the bitset for the input variables, as an integer
    blockind = blockindex[residual.varind]
    varflags = foldl((x, y) -> (x << 1) + (y != 0), reverse(blockind), init=0)

    # If there are no variables, just return the cost
    if varflags == 0
        return cost(residual, vars)
    end

    # Dispatch gradient computation based on the varflags, and return the cost
    #return gradhesshelper!(grad, hess, residual, vars, blockind, Val(varflags))
    return valuedispatch(Val(1), Val((2^nvars(residual))-1), v -> gradhesshelper!(grad, hess, residual, vars, blockind, v), varflags)
end

function costgradhess!(grad, hess, residuals, vars::Vector, blockindex::Vector{Int})
    # Go over all resdiduals, updating the gradient & hessian, and aggregating the cost 
    c = 0.
    for res in residuals
        c += costgradhess!(grad, hess, res, vars, blockindex)
    end
    return c
end
