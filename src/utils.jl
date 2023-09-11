using StaticArrays, Static

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

@inline sqnorm(x) = (x' * x)

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
        Base.Experimental.@force_compile
        local t0 = Base.time_ns()
        $(esc(ex))
        Base.time_ns() - t0
    end
end
