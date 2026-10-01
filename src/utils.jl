using StaticArrays, Static, SparseArrays, LoopVectorization, Printf

@inline function valuedispatch(lower::StaticInt, upper::StaticInt, val::Int, fun)
    if lower >= upper
        return fun(upper)
    end
    midpoint = lower + div(upper - lower, static(2))
    if val <= midpoint
        return valuedispatch(lower, midpoint, val, fun)
    else
        return valuedispatch(midpoint + static(1), upper, val, fun)
    end
end

struct Bind{F, A}
    func::F
    args::A
end
function (bound::Bind)(args...)
    bound.func(bound.args..., args...)
end
bindleadingargs(func, args...) = Bind(func, args)

SR(first, last) = StaticArrays.SUnitRange(dynamic(first), dynamic(last))

macro bitiset(flags, bit)
    esc(:(((1 << ($bit - 1)) & $flags) != 0))
end

bitiset(flags::StaticInt, bit) = (static(1 << (bit - 1)) & flags) != static(0)
bitiset(flags, bit) = (1 << (bit - 1)) & flags != 0

uniontotuple(T::Union) = (uniontotuple(T.a)..., uniontotuple(T.b)...)
uniontotuple(T::DataType) = (T,)

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

struct TimeCounter
    time_ns::UInt64
    count::Int64
end
TimeCounter() = TimeCounter(0, 0)
Base.:+(x::TimeCounter, u::UInt64) = TimeCounter(x.time_ns + u, x.count + 1)

macro elapsed_ns(ex)
    quote
        local t0 = Base.time_ns()
        $(esc(ex))
        Base.time_ns() - t0
    end
end

struct StatsCounter
    num_allocs::Int64
    bytes_allocd::Int64
    time_ns::UInt64
    count::Int64
end
StatsCounter() = StatsCounter(0, 0, 0, 0)
struct Stats
    num_allocs::Int64
    bytes_allocd::Int64
    time_ns::UInt64
end
Stats(init) = Stats(init, init, init)
function Stats()
    gc_data = Base.gc_num()
    return Stats(gc_data.malloc + gc_data.realloc + gc_data.poolalloc + gc_data.bigalloc, gc_data.allocd, Base.time_ns())
end
Base.:+(x::StatsCounter, u::Stats) = StatsCounter(x.num_allocs + u.num_allocs, x.bytes_allocd + u.bytes_allocd, x.time_ns + u.time_ns, x.count + 1)

macro stats(ex)
    quote
        local t0 = Base.time_ns()
        local allocs = Base.gc_num()
        $(esc(ex))
        local diff = Base.GC_Diff(Base.gc_num(), allocs)
        Stats(Base.gc_alloc_count(diff), diff.allocd, Base.time_ns() - t0)
    end
end

function bytesstring(nb)
    bytessuffix = (" bytes", "KB", "MB", "GB", "TB")
    suffix = 1
    while nb >= 1024 && suffix <= length(bytessuffix)
        nb /= 1024
        suffix += 1
    end
    return @sprintf("%d%s", nb, bytessuffix[suffix])
end

function timestring(ns)
    timesuffix = ("ns", "ms", "s", " minutes", " hours")
    timeperiod = (1000.0, 1000.0, 1000.0, 60.0, 60.0)
    ns = Float64(ns)
    suffix = 1
    while ns >= timeperiod[suffix] && suffix <= length(timesuffix)
        ns /= timeperiod[suffix]
        suffix += 1
    end
    return @sprintf("%g%s", ns, timesuffix[suffix])
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

sparse_dense_decision(ndims, sparsity, blocksizes) = sparse_dense_decision(ndims, block_sparse_nnz(sparsity, blocksizes))
sparse_dense_decision(ndims, nnz) = (nnz * 64) < (25 * ndims * (ndims - 40)) # Threshold nnz (for lower triangle) = 25/64 * (d^2 - 40d)

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

function block_sparsity(problem, unfixed)
    # Compute the block sparsity
    sparsity = getvarcostmap(problem)
    sparsity = sparsity[unfixed,:]
    sparsity = triu(sparse(sparsity * sparsity' .> 0))
    return sparsity
end

function block_sizes_indices(variables, unfixed, nblocks)
    blockindices = zeros(UInt, length(variables))
    blocksizes = zeros(UInt, nblocks)
    nblocks = 0
    for (index, unfixed_) in enumerate(unfixed)
        if unfixed_
            nblocks += 1
            blockindices[index] = nblocks
            blocksizes[nblocks] = nvars(variables[index])
        end
    end
    return blocksizes, blockindices
end
