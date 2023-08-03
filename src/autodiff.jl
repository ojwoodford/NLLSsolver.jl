import ForwardDiff, DiffResults

# Decision on whether to use static-sized autodiff - compile-time decision where possible
function usestatic(varflags::StaticInt, vars)
    N = length(vars)
    totalnumvars = static(0)
    @unroll for i in 1:MAX_ARGS
        if i <= N && bitiset(varflags, i)
            nvi = nvars(vars[i])
            if dynamic(!is_static(nvi))
                return false
            end
            totalnumvars += nvi
        end
    end
    if totalnumvars > static(MAX_STATIC_VAR)
        return false
    end
    return true
end
usestatic(varflags, vars) = false

# Construct a vector of duals with zero values and one partial derivative set to 1
@generated function singleseed(::Type{T}, ::Val{N}, ::Val{i}) where {T, N, i}
    return ForwardDiff.Partials{N, T}(tuple([T(i == j) for j in 1:N]...))
end
@generated function dualzeros(::Type{T}, ::Val{N}, ::Val{First}, ::Val{Last}) where {T, N, First, Last}
    return SVector{Last-First+1, ForwardDiff.Dual{T, T, N}}(tuple([ForwardDiff.Dual{T, T, N}(zero(T), singleseed(T, Val(N), Val(i))) for i in First:Last]...))
end
dualzeros(T, N) = dualzeros(T, N, Val(1), N)

# Generate the updated (dualized) variables
@generated function dualvars(vars::NTuple{N, Any}, ::StaticInt{varflags}, ::Type{T}) where {N, varflags, T}
    counts = Expr(:tuple, [@bitiset(varflags, i) ? :(nvars(vars[$i])) : 0 for i = 1:N]...)
    cumul = :(cumsum((0, $counts...)))
    return Expr(:tuple, [@bitiset(varflags, i) ? :(update(vars[$i], dualzeros($T, Val($cumul[$N+1]), Val($cumul[$i]+1), Val($cumul[$i+1])))) : :(vars[$i]) for i = 1:N]...)
end

@generated function updatevars(vars::NTuple{N, Any}, ::StaticInt{varflags}, dualvars) where {N, varflags}
    counts = Expr(:tuple, [@bitiset(varflags, i) ? :(nvars(vars[$i])) : 0 for i = 1:N]...)
    cumul = :(cumsum((1, $counts...)))
    return Expr(:tuple, [@bitiset(varflags, i) ? :(update(vars[$i], dualvars, $cumul[$i])) : :(vars[$i]) for i = 1:N]...)
end

# Extract the residual and jacobian from duals to static arrays
@generated function extractresjac(dual::SVector{M, ForwardDiff.Dual{T, T, N}}) where {T, M, N}
    res = Expr(:tuple, [:(ForwardDiff.value(dual[$i])) for i in 1:M]...)
    jac = Expr(:tuple, [:(ForwardDiff.partials($T, dual[$i], $j)) for i in 1:M, j in 1:N]...)
    return :(SVector{$M, $T}($res), SMatrix{$M, $N, $T, $M*$N}($jac))
end
extractresjac(dual::ForwardDiff.Dual{T, T, N}) where {T, N} = ForwardDiff.value(dual), SVector{N, T}(dual.partials.values...)'

# Automatic Jacobian computation
computeresjac(varflags, residual, vars...) = usestatic(varflags, vars) ? computeresjacstatic(varflags, residual, vars) : computeresjacdynamic(varflags, residual, vars)

# Automatic statically-sized Jacobian computation
function computeresjacstatic(varflags::StaticInt, residual::Residual, vars) where Residual <: AbstractResidual
    # Perform static (i.e. fixed size vector) auto-differentiation
    @assert eltype(residual)!=Any "Define Base.eltype() for your residual type"

    # Construct ForwardDiff Dual arguments for the unfixed variables
    xdual = dualvars(vars, varflags, eltype(residual))
    
    # Compute the residual with ForwardDiff Duals
    ydual = computeresidual(residual, xdual...)

    # Return the residual and jacobian
    return extractresjac(ydual)
end

# Automatic dynamically-sized Jacobian computation
function computeresjacdynamic(varflags, residual::Residual, vars) where Residual <: AbstractResidual
    # Perform dynamic (i.e. variable size vector) auto-differentiation
    @assert eltype(residual)!=Any "Define Base.eltype() for your residual type"

    # Compute the number of free variables
    N = ndeps(residual)
    totalnumvars = 0
    @unroll for i in 1:MAX_ARGS
        if i <= N
            totalnumvars += ifelse(bitiset(varflags, i), dynamic(nvars(vars[i])), 0)
        end
    end

    # Use the ForwardDiff API to compute the jacobian
    M = nres(residual)
    out = DiffResults.DiffResult(dynamic(M), (totalnumvars,))
    if M > 1
        ForwardDiff.jacobian!(out, fixallbutlast(computeresjachelper, varflags, residual, vars), zeros(totalnumvars))
    else
        ForwardDiff.gradient!(out, fixallbutlast(computeresjachelper, varflags, residual, vars), zeros(totalnumvars))
    end

    # Return the residual and jacobian
    return value(out), jacobian(out)
end

computeresjachelper(varflags, residual, vars, x) = computeresidual(residual, updatevars(vars, varflags, x)...) 
