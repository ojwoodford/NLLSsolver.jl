import ForwardDiff
export computeresjac 

# Construct a dual with zero values and one partial derivative set to 1
@generated function single_seed(::Type{T}, ::Val{N}, ::Val{i}) where {T, N, i}
    ex = Expr(:tuple, [:($T(i == $j)) for j in 1:N]...)
    return :(ForwardDiff.Partials{$N, $T}($ex))
end
@generated function dualzeros(::Type{T}, ::Val{N}, ::Val{First}, ::Val{Last}) where {T, N, First, Last}
    dx = Expr(:tuple, [:(ForwardDiff.Dual{$T, $T, $N}(zero($T), single_seed($T, Val($N), Val($i)))) for i in First:Last]...)
    return :(SVector{$Last-$First+1, ForwardDiff.Dual{$T, $T, $N}}($dx))
end

# Generate the updated variables
@generated function dualvars(vars::NTuple{N, Any}, ::Val{varflags}, ::Type{T}) where {N, varflags, T}
    counts = Expr(:tuple, 0, [@bitiset(varflags, i) ? :(nvars(fieldtypes($vars)[$i])) : 0 for i in 1:N]...)
    cumul = :(cumsum($counts))
    return Expr(:tuple, [@bitiset(varflags, i) ? :(update(vars[$i], dualzeros($T, Val($cumul[$N+1]), Val($cumul[$i]+1), Val($cumul[$i+1])))) : :(vars[$i]) for i in 1:N]...)
end

# Extract the residual and jacobian to static arrays
@generated function extract_resjac(dual::SVector{M, ForwardDiff.Dual{T, T, N}}) where {T, M, N}
    res = Expr(:tuple, [:(ForwardDiff.value(dual[$i])) for i in 1:M]...)
    jac = Expr(:tuple, [:(ForwardDiff.partials($T, dual[$i], $j)) for i in 1:M, j in 1:N]...)
    return :(SVector{$M, $T}($res), SMatrix{$M, $N, $T, $M*$N}($jac))
end
function extract_resjac(dual::ForwardDiff.Dual{T, T, N}) where {T, N}
    return ForwardDiff.value(dual), SVector{N, T}(dual.partials.values...)'
end

# Automatic Jacobian computation
function computeresjac(::Val{varflags}, residual::Residual, vars...) where {varflags, Residual <: AbstractResidual}
    @assert eltype(residual)!=Any "Define Base.eltype() for your residual type"
    
    # Construct ForwardDiff Dual arguments for the unfixed variables
    xdual = dualvars(vars, Val(varflags), eltype(residual))
    
    # Compute the residual with ForwardDiff Duals
    ydual = computeresidual(residual, xdual...)

    # Return the residual and jacobian
    return extract_resjac(ydual)
end