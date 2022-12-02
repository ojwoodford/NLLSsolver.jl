using FLoops
import ForwardDiff, NLLSsolver.valuedispatch
export cost, costgradhess!, computeresjac

function cost(problem::NLLSProblem)::Float64
    # Compute the cost of all residuals in the problem
    return sum(x -> cost(x.second::Vector{x.first}, problem.variables), problem.residuals; init=0.)
end

function cost(residuals, vars::Vector{Any})::Float64
    # Compute the total cost of all residuals in a container
    c = 0.
    @floop for res in residuals
        @reduce c += cost(res, vars)
    end
    return c
end

function cost(residual::Residual, vars::Vector{Any})::Float64 where Residual <: AbstractResidual
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

function updateblocks!(grad, hess, res, jac, w1, w2, blockoffsets)
    # Triggs' second order correction
    if w2
        
    end
    # IRLS weighting of the residual and Jacobian
    if w1 != 1
        res = res * w1
        jac = jac * w1
    end

    # Update the blocks in the problem
    grad[blockoffsets] += jac' * res
    hess[blockoffsets, blockoffsets] += jac' * jac
    return nothing
end

# Automatic Jacobian computation
function computeresjac(::Val{varflags}, residual::Residual, vars...) where {varflags, Residual <: AbstractResidual}
    # Compute the residual
    res = computeresidual(residual, vars...)

    # Compute the Jacobian
    nres = length(res)
    type = nres > 1 ? eltype(res) : typeof(res)
    nvars = countvars(vars, Val(varflags))
    Z = zeros(SVector{nvars, type})
    jac = ForwardDiff.jacobian(z -> computeresidual(residual, updatevars(vars, varflags, z)...), Z)::SMatrix{nres, nvars, type, nres*nvars}

    # Return both
    return res, jac
end

function gradhesshelper!(grad, hess, residual::Residual, vars::Vector, blockind, ::Val{varflags}) where {varflags, Residual <: AbstractResidual}
    # Get the variables
    v = getvars(residual, vars)

    # Compute the residual
    res, jac = computeresjac(Val(varflags), residual, v...)

    # Compute the robustified cost and the IRLS weight
    c, w1, w2 = robustify(robustkernel(residual), res' * res)

    # If this residual has a weight...
    if w1 > 0    
        # Update the blocks in the problem
        updateblocks!(grad, hess, res, jac, w1, w2, computeoffsets(v, varflags, blockind))
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
