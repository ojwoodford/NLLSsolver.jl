export SR, @objtype, @bitiset
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
    ex = valuedispatch_expr(Val(lower), Val(upper), esc(val), esc(fun))
    return quote
        @nospecialize
        $ex
    end
end

@eval valuedispatch_1_3(val, fun) = @valuedispatch_macro(1, 3, val, fun)
@eval valuedispatch_1_7(val, fun) = @valuedispatch_macro(1, 7, val, fun)
@eval valuedispatch_1_15(val, fun) = @valuedispatch_macro(1, 15, val, fun)
@eval valuedispatch_1_32(val, fun) = @valuedispatch_macro(1, 32, val, fun)
@eval valuedispatch_1_63(val, fun) = @valuedispatch_macro(1, 63, val, fun)
@eval valuedispatch_1_127(val, fun) = @valuedispatch_macro(1, 127, val, fun)

expandfunc(args, v) = args[1](args[2:end]..., v)
fixallbutlast(func, args...) = Base.Fix1(expandfunc, (func, args...))

const SR = StaticArrays.SUnitRange

macro objtype(type)
    esc(:(::Union{$type, Type{$type}}))
end

macro bitiset(flags, bit)
    esc(:(((1 << ($bit - 1)) & $flags) != 0))
end
