import ForwardDiff
export computeresjac 

# Count the numbers of variables in the inputs
@inline function countvars(vars, varflags)
    if isempty(vars)
        return 0
    end
    return sum(ntuple(i -> ((1 << (i - 1)) & varflags) != 0 ? nvars(vars[i]) : 0, length(vars)))
end

# Generate the updated variables
@inline function updatevars(vars, varflags, advar)
    return ntuple(i -> ((1 << (i - 1)) & varflags) != 0 ? update(vars[i], advar, countvars(vars[1:i-1], varflags)+1) : vars[i], length(vars))
end

# Automatic Jacobian computation
function computeresjac(::Val{varflags}, residual::Residual, vars...) where {varflags, Residual <: AbstractResidual}
    # Compute the residual
    res = computeresidual(residual, vars...)

    # Compute the Jacobian
    Z = zeros(SVector{countvars(vars, varflags), eltype(res)})
    jac = ForwardDiff.jacobian(z -> computeresidual(residual, updatevars(vars, varflags, z)...), Z)

    # Return both
    return res, jac
end