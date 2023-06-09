using StaticArrays

function valuedispatch_expr(::Val{lower}, ::Val{upper}, val, fun) where {lower, upper}
    if lower >= upper
        return :( return $fun(Val($upper)) ) 
    end
    midpoint = lower + div(upper - lower, 2)
    expr_a = valuedispatch_expr(Val(lower), Val(midpoint), val, fun)
    expr_b = valuedispatch_expr(Val(midpoint+1), Val(upper), val, fun)
    return quote
        if $val <= $midpoint
            $expr_a
        else
            $expr_b
        end
    end
end

macro valuedispatch_macro(lower::Int, upper::Int, val, fun)
    return valuedispatch_expr(Val(lower), Val(upper), esc(val), esc(fun))
end

@generated function valuedispatch(::Val{lower}, ::Val{upper}, val, fun) where {lower, upper}
    return :( @valuedispatch_macro($lower, $upper, val, fun) )
end

expandfunc(args, v) = args[1](args[2:end]..., v)
fixallbutlast(func, args...) = Base.Fix1(expandfunc, (func, args...))

const SR = StaticArrays.SUnitRange

macro bitiset(flags, bit)
    esc(:(((1 << ($bit - 1)) & $flags) != 0))
end
