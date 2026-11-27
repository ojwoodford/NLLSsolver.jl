# Gradient descent as a user-defined iterator.
#
# In NLLSsolver 5, `:gradientdescent` was a built-in iterator. SecondOrderOptimization is a second-order
# optimizer and gradient descent is a first-order method, so it is no longer built in; it is instead the
# worked example of writing your own iterator. This file registers a backtracking gradient descent as
# `:gradientdescent` through the iterator interface, so that `NLLSOptions(iterator = :gradientdescent)`
# accepts it once this file (or your own equivalent) is loaded. It is taken from SecondOrderOptimization's
# tutorial:
#   https://ojwoodford.github.io/SecondOrderOptimization.jl/stable/tutorials/custom_iterator/
# and the interface it implements is documented at
#   https://ojwoodford.github.io/SecondOrderOptimization.jl/stable/iterator_interface/
#
# The iterator uses only the gradient and never solves the linear system, so it makes a short example. It is
# slow on curved valleys, as a first-order method is; `:levenbergmarquardt` reaches the Rosenbrock minimum in
# a fraction of the iterations.

using NLLSsolver.IteratorInterface, Static, LinearAlgebra

# The iterator's state between iterations: the step size, a multiple of the negative gradient.
mutable struct GradientDescentData
    stepsize::Float64
end

# Registers the name: NLLSOptions(iterator = :gradientdescent) now accepts it, and show(result) prints it.
IteratorInterface.iteratorname(::StaticSymbol{:gradientdescent}) = "gradient descent"

# One method, whatever the callback may change: this iterator holds no trial linearization.
IteratorInterface.construct(::StaticSymbol{:gradientdescent}, ls, ::StaticBool) = GradientDescentData(1.0)

# A reused workspace repeats a first run.
IteratorInterface.reset!(gdd::GradientDescentData) = (gdd.stepsize = 1.0; nothing)

# `f` and `fdf!` are only passed on, so type parameters make Julia specialise on them (otherwise each call allocates).
function IteratorInterface.iterate!(gdd::GradientDescentData, data, f::F, fdf!::FDF, options)::Float64 where {F, FDF}
    gradient = getgradient(data)   # finite: the loop stops on a non-finite gradient before iterating
    step = getstep(data)
    step .= .-gradient .* gdd.stepsize
    settrial!(data)
    cost = trialcost!(data, f)
    inner_iters = 0
    while !(cost <= getbestcost(data))
        inner_iters += 1
        if inner_iters > 100 || !(gdd.stepsize >= floatmin(Float64)) || !isfinite(gdd.stepsize)
            return cost
        end
        if isfinite(cost)
            # Minimiser of the quadratic through the cost and slope at 0 and the cost at the trial
            # step, as a fraction of the step, safeguarded to [0.1, 0.5]. The slope term is not
            # positive and the trial cost is above the best, so `costdiff` is negative, never zero.
            slope = dot(step, gradient)
            costdiff = getbestcost(data) + slope - cost
            gdd.stepsize *= clamp(0.5 * slope / costdiff, 0.1, 0.5)
        else
            gdd.stepsize *= 0.5
        end
        step .= .-gradient .* gdd.stepsize
        settrial!(data)
        cost = trialcost!(data, f)
    end
    gdd.stepsize *= 2.0
    return cost
end
