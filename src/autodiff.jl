import ForwardDiff, DiffResults
using Static, StaticArrays

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
    if totalnumvars > MAX_STATIC_VAR
        return false
    end
    return true
end
usestatic(varflags, vars) = false

# Count the number of variables across all unfixed variable blocks
function computetotalnumvars(varflags::StaticInt, vars)
    N = length(vars)
    totalnumvars = static(0)
    @unroll for i in 1:MAX_ARGS
        if i <= N && bitiset(varflags, i)
            totalnumvars += nvars(vars[i])
        end
    end
    return totalnumvars
end
function computetotalnumvars(varflags, vars)
    N = length(vars)
    totalnumvars = 0
    @unroll for i in 1:MAX_ARGS
        if i <= N
            totalnumvars += ifelse(bitiset(varflags, i), dynamic(nvars(vars[i])), 0)
        end
    end
    return totalnumvars
end

# Construct a vector of duals with zero values and one partial derivative set to 1
@generated function singleseed(::Type{T}, ::Val{N}, ::Val{i}) where {T, N, i}
    return ForwardDiff.Partials{N, T}(tuple([T(i == j) for j in 1:N]...))
end
@generated function dualzeros(::Type{T}, ::Val{N}, ::Val{First}, ::Val{Last}) where {T, N, First, Last}
    TG = ForwardDiff.Tag(computeresidual, T)
    return SVector{Last-First+1, ForwardDiff.Dual{TG, T, N}}(tuple([ForwardDiff.Dual{TG, T, N}(zero(T), singleseed(T, Val(N), Val(i))) for i in First:Last]...))
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
@generated function extractvaldual(dual::SVector{M, ForwardDiff.Dual{TG, T, N}}) where {TG, T, M, N}
    res = Expr(:tuple, [:(ForwardDiff.value(dual[$i])) for i in 1:M]...)
    jac = Expr(:tuple, [:(ForwardDiff.partials(dual[$i], $j)) for i in 1:M, j in 1:N]...)
    return :(SVector{$M, $T}($res), SMatrix{$M, $N, $T, $M*$N}($jac))
end
extractvaldual(dual::ForwardDiff.Dual{TG, T, N}) where {TG, T, N} = ForwardDiff.value(dual), SVector{N, T}(dual.partials.values...)'

# Automatic Jacobian computation
@inline computeresjac(varflags, residual, vars...) = usestatic(varflags, vars) ? computeresjacstatic(varflags, residual, vars) : computeresjacdynamic(varflags, residual, vars)

# Automatic statically-sized Jacobian computation
function computeresjacstatic(varflags::StaticInt, residual::AbstractResidual, vars)
    # Perform static (i.e. fixed size vector) auto-differentiation
    @assert eltype(residual)!=Any "Define Base.eltype() for your residual type"

    # Construct ForwardDiff Dual arguments for the unfixed variables
    xdual = dualvars(vars, varflags, eltype(residual))
    
    # Compute the residual with ForwardDiff Duals
    ydual = computeresidual(residual, xdual...)

    # Return the residual and jacobian
    return extractvaldual(ydual)
end

# Automatic dynamically-sized Jacobian computation
function computeresjacdynamic(varflags, residual::AbstractResidual, vars)
    # Perform dynamic (i.e. variable size vector) auto-differentiation
    @assert eltype(residual)!=Any "Define Base.eltype() for your residual type"

    # Compute the number of free variables
    totalnumvars = computetotalnumvars(varflags, vars)

    # Use the ForwardDiff API to compute the jacobian
    M = nres(residual)
    if M == 1
        # Scalar residual
        x = zeros(eltype(residual), totalnumvars)
        result = DiffResults.GradientResult(x)
        result = ForwardDiff.gradient!(result, fixallbutlast(computeresjachelper, varflags, residual, vars), x)
        # Return the residual and jacobian
        return DiffResults.value(result), DiffResults.gradient(result)'
    end

    # Vector residual
    resultlength = (totalnumvars + 1) * M
    resultstorage = zeros(eltype(residual), resultlength)
    result = DiffResults.DiffResult(view(resultstorage, 1:dynamic(M)), (reshape(view(resultstorage, M+1:resultlength), dynamic(M), totalnumvars),))
    result = ForwardDiff.jacobian!(result, fixallbutlast(computeresjachelper, varflags, residual, vars), view(resultstorage, 1:totalnumvars))
    # Return the residual and jacobian
    return DiffResults.value(result), DiffResults.jacobian(result)
end

computeresjachelper(varflags, residual, vars, x) = computeresidual(residual, updatevars(vars, varflags, x)...) 

# Hessian computation wrapper
function computehessian(f, x::AbstractArray)
    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, f, x)
    return DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
end

computehessian(f, x::T) where T <: Number = map(first, computehessian(f ∘ first, SVector{1, T}(x)))

# Gradient computation wrapper
function computegradient(f, x::AbstractArray)
    result = DiffResults.GradientResult(x)
    result = ForwardDiff.gradient!(result, f, x)
    return DiffResults.value(result), DiffResults.gradient(result)
end

computegradient(f, x::T) where T <: Number = map(first, computegradient(f ∘ first, SVector{1, T}(x)))

# Automatic computation of the cost, gradient and Hessian
function computecostgradhess(varflags, cost::AbstractCost, vars...)
    # Perform dynamic (i.e. variable size vector) auto-differentiation
    @assert eltype(cost)!=Any "Define Base.eltype() for your residual type"

    # Compute the number of free variables
    totalnumvars = computetotalnumvars(varflags, vars)

    # Construct the cost function that updates the variables
    costfunc = fixallbutlast(computegradhesshelper, varflags, cost, vars)

    # Compute hessian with either static or dynamic array
    if dynamic(is_static(totalnumvars)) && totalnumvars <= MAX_STATIC_VAR
        return computehessian(costfunc, zeros(SVector{dynamic(totalnumvars), eltype(cost)}))
    end
    return computehessian(costfunc, zeros(eltype(cost), dynamic(totalnumvars)))
end

computegradhesshelper(varflags, residual, vars, x) = computecost(residual, updatevars(vars, varflags, x)...)

@inline autorobustifydcost(kernel::AbstractRobustifier, cost) = computehessian(Base.Fix1(robustify, kernel), cost)
@inline autorobustifydkernel(kernel::AbstractAdaptiveRobustifier, cost::T) where T = computehessian(fixallbutlast(computerobustgradhesshelper, kernel, cost), zeros(SVector{nvars(kernel)+1, T}))
computerobustgradhesshelper(kernel, cost, x) = robustify(update(kernel, x), cost+x[end])

# Overloads of some specific functions
struct RotMatLie{T}
    x::T
    y::T
    z::T
end

function rodrigues(x::T, y::T, z::T) where T<:ForwardDiff.Dual
    @assert x == 0 && y == 0 && z == 0
    return RotMatLie{T}(x, y, z)
    # return SMatrix{3, 3, T, 9}(T(1), z, -y, -z, T(1), x, y, -x, T(1))
end

du(x) = ForwardDiff.partials(x)
Base.:*(r::AbstractMatrix, u::RotMatLie{T}) where T = SMatrix{3, 3, T, 9}(T(r[1,1], du(u.z)*r[1,2]-du(u.y)*r[1,3]), T(r[2,1], du(u.z)*r[2,2]-du(u.y)*r[2,3]), T(r[3,1], du(u.z)*r[3,2]-du(u.y)*r[3,3]),
                                                                          T(r[1,2], du(u.x)*r[1,3]-du(u.z)*r[1,1]), T(r[2,2], du(u.x)*r[2,3]-du(u.z)*r[2,1]), T(r[3,2], du(u.x)*r[3,3]-du(u.z)*r[3,1]),
                                                                          T(r[1,3], du(u.y)*r[1,1]-du(u.x)*r[1,2]), T(r[2,3], du(u.y)*r[2,1]-du(u.x)*r[2,2]), T(r[3,3], du(u.y)*r[3,1]-du(u.x)*r[3,2]))
Base.:*(u::RotMatLie{T}, r::AbstractMatrix) where T = SMatrix{3, 3, T, 9}(T(r[1,1], du(u.y)*r[3,1]-du(u.z)*r[2,1]), T(r[2,1], du(u.z)*r[1,1]-du(u.x)*r[3,1]), T(r[3,1], du(u.x)*r[2,1]-du(u.y)*r[1,1]),
                                                                          T(r[1,2], du(u.y)*r[3,2]-du(u.z)*r[2,2]), T(r[2,2], du(u.z)*r[1,2]-du(u.x)*r[3,2]), T(r[3,2], du(u.x)*r[2,2]-du(u.y)*r[1,2]),
                                                                          T(r[1,3], du(u.y)*r[3,3]-du(u.z)*r[2,3]), T(r[2,3], du(u.z)*r[1,3]-du(u.x)*r[3,3]), T(r[3,3], du(u.x)*r[2,3]-du(u.y)*r[1,3]))
