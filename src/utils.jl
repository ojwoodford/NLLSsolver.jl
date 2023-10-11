using StaticArrays, Static, SparseArrays, LoopVectorization

@inline function valuedispatch(lower::StaticInt, upper::StaticInt, val::Int, fun)
    if lower >= upper
        return fun(upper)
    end
    midpoint = lower + div(upper - lower, static(2))
    if val <= midpoint
        return valuedispatch(lower, midpoint, val, fun)
    end
    return valuedispatch(midpoint + static(1), upper, val, fun)
end

expandfunc(args, v) = args[1](args[2:end]..., v)
fixallbutlast(func, args...) = Base.Fix1(expandfunc, (func, args...))

SR(first, last) = StaticArrays.SUnitRange(dynamic(first), dynamic(last))

macro bitiset(flags, bit)
    esc(:(((1 << ($bit - 1)) & $flags) != 0))
end

bitiset(flags::StaticInt, bit) = (static(1 << (bit - 1)) & flags) != static(0)
bitiset(flags, bit) = (1 << (bit - 1)) & flags != 0

@inline uniontotuple(T::Union) = (uniontotuple(T.a)..., uniontotuple(T.b)...)
@inline uniontotuple(T::DataType) = (T,)

sqnorm(x::Number) = @fastmath x * x
function sqnorm(vec::AbstractVector)
    total = zero(eltype(vec))
    @turbo for i in eachindex(vec)
        total += vec[i] * vec[i]
    end
    return total
end

function runlengthencodesortedints(sortedints)
    runindices = Vector{Int}(undef, sortedints[end]+2)
    ind = 1
    currval = 1
    runindices[currval] = ind
    for val in sortedints
        while val >= currval
            currval += 1
            runindices[currval] = ind
        end
        ind += 1
    end
    runindices[currval+1] = ind
    return runindices
end

macro elapsed_ns(ex)
    quote
        local t0 = Base.time_ns()
        $(esc(ex))
        Base.time_ns() - t0
    end
end

function Base.cumsum!(A::AbstractVector)
    total = zero(eltype(A))
    @inbounds for ind in eachindex(A)
        @fastmath total += A[ind]
        A[ind] = total
    end
    return A
end

function fast_bAb(A::Matrix, b::Vector)
    total = zero(eltype(b))
    @tturbo for i in eachindex(b)
        subtotal = zero(eltype(b))
        for j in eachindex(b)
            subtotal += A[j,i] * b[j]
        end
        total += b[i] * subtotal
    end
    return total
end

function fast_bAb(A::StaticArray, b::StaticArray)
    total = zero(eltype(b))
    @turbo for i in eachindex(b)
        subtotal = zero(eltype(b))
        for j in eachindex(b)
            subtotal += A[j,i] * b[j]
        end
        total += b[i] * subtotal
    end
    return total
end

function fast_bAb(A::SparseMatrixCSC, b::Vector)
    total = zero(eltype(b))
    for i in eachindex(b)
        coltotal = zero(eltype(b))
        for j in nzrange(A, i)
            @inbounds @fastmath coltotal += A.nzval[j] * b[A.rowval[j]]
        end
        @inbounds @fastmath coltotal *= b[i]
        total += coltotal
    end
    return total
end

sparse_dense_decision(d, nnz) = nnz < (d - 40) * d

function block_sparse_nnz(sparsity, blocksizes)
    # Compute the number of non-zeros in a block sparse matrix
    nnz = 0
    for col in 1:size(sparsity, 2)
        @inbounds bc = blocksizes[col]
        for r in nzrange(sparsity, col)
            nnz += bc * @inbounds blocksizes[rowvals(sparsity)[r]]
        end
    end
    return nnz
end