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

# Construct a dual with zero values and one partial derivative set to 1
@generated function dualzeros(::Type{T}, ::Val{N}) where {T, N}
    dx = Expr(:tuple, [:(ForwardDiff.Dual{T, T, N}(zero(T), single_seed(T, Val(N), Val($i)))) for i in 1:N]...)
    return :(SVector{N, ForwardDiff.Dual{T, T, N}}($(dx)))
end
@generated function single_seed(::Type{T}, ::Val{N}, ::Val{i}) where {T, N, i}
    ex = Expr(:tuple, [:(T(i == $j)) for j in 1:N]...)
    return :(ForwardDiff.Partials{N, T}($ex))
end

# Extract the residual and jacobian to static arrays
@generated function extract_resjac(dual::SVector{M, ForwardDiff.Dual{T, T, N}}) where {T, M, N}
    res = Expr(:tuple, [:(ForwardDiff.value(dual[$i])) for i in 1:M]...)
    jac = Expr(:tuple, [:(ForwardDiff.partials(T, dual[$i], $j)) for i in 1:M, j in 1:N]...)
    return :(SVector{M, T}($res), SMatrix{M, N, T, M*N}($jac))
end
@generated function extract_resjac(dual::ForwardDiff.Dual{T, T, N}) where {T, N}
    jac = Expr(:tuple, [:(ForwardDiff.partials(T, dual, $i)) for i in 1:N]...)
    return :(ForwardDiff.value(dual), SVector{N, T}($jac)')
end

# Automatic Jacobian computation
function computeresjac(::Val{varflags}, residual::Residual, vars...) where {varflags, Residual <: AbstractResidual}
    # Compute the residual with ForwardDiff Duals
    ydual = computeresidual(residual, updatevars(vars, varflags, dualzeros(eltype(residual), Val(countvars(vars, varflags))))...)

    # Return the residual and jacobian
    return extract_resjac(ydual)
end