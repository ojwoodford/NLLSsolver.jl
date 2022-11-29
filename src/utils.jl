export valuedispatch, SR
using StaticArrays

function valuedispatch_expr(::Val{lower}, ::Val{upper}, fun, val) where {lower, upper}
    if lower >= upper
        return :( $fun(Val($upper)) )
    end
    midpoint = lower + div(upper - lower, 2)
    expr_a = valuedispatch_expr(Val(lower), Val(midpoint), fun, val)
    expr_b = valuedispatch_expr(Val(midpoint+1), Val(upper), fun, val)
    quote
        if $val <= $midpoint
            $expr_a
        else
            $expr_b
        end
    end
end

macro valuedispatch_macro(lower::Int, upper::Int, fun, val)
    valuedispatch_expr(Val(lower), Val(upper), esc(fun), esc(val))
end

@generated function valuedispatch(::Val{lower}, ::Val{upper}, fun, val) where {lower, upper}
    :( @valuedispatch_macro($lower, $upper, fun, val) )
end

const SR = StaticArrays.SUnitRange
