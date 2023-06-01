export valuedispatch, SR, @objtype
using StaticArrays

function valuedispatch_expr(::Val{lower}, ::Val{upper}, val, fun, args) where {lower, upper}
    if lower >= upper
        return :( $fun(($args)..., Val($upper)) )
    end
    midpoint = lower + div(upper - lower, 2)
    expr_a = valuedispatch_expr(Val(lower), Val(midpoint), val, fun, args)
    expr_b = valuedispatch_expr(Val(midpoint+1), Val(upper), val, fun, args)
    quote
        if $val <= $midpoint
            $expr_a
        else
            $expr_b
        end
    end
end

macro valuedispatch_macro(lower::Int, upper::Int, val, fun, args)
    valuedispatch_expr(Val(lower), Val(upper), esc(val), esc(fun), esc(args))
end

@generated function valuedispatch(::Val{lower}, ::Val{upper}, val, fun, args) where {lower, upper}
    :( @valuedispatch_macro($lower, $upper, val, fun, args) )
end

const SR = StaticArrays.SUnitRange

macro objtype(type)
    esc(:(::Union{$type, Type{$type}}))
end
