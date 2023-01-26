# Unroll code copied from https://github.com/StephenVavasis/Unroll.jl

# Copyright (c) 2015 Stephen Vavasis

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software 
# is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

export @unroll

copy_and_substitute_tree(e, varname, newtext, mod) = e

copy_and_substitute_tree(e::Symbol, varname, newtext, mod) =
    e == varname ? newtext : e

function copy_and_substitute_tree(e::Expr, varname, newtext, mod)
    e2 = Expr(e.head)
    for subexp in e.args
        push!(e2.args, copy_and_substitute_tree(subexp, varname, newtext, mod))
    end
    if e.head == :if
        newe = e2
        try
            u = Core.eval(mod, e2.args[1])
            if u == true
                newe = e2.args[2]
            elseif u == false
                if length(e2.args) == 3
                    newe = e2.args[3]
                else
                    newe = :nothing
                end
            end
        catch
        end
        e2 = newe
    end
    e2 
end

macro unroll(expr)
    if expr.head != :for || length(expr.args) != 2 ||
        expr.args[1].head != :(=) || 
        typeof(expr.args[1].args[1]) != Symbol ||
        expr.args[2].head != :block
        error("Expression following unroll macro must be a for-loop as described in the documentation")
    end
    varname = expr.args[1].args[1]
    ret = Expr(:block)
    for k in Core.eval(__module__, expr.args[1].args[2])
        e2 = copy_and_substitute_tree(expr.args[2], varname, k, __module__)
        push!(ret.args, e2)
    end
    esc(ret)
end
