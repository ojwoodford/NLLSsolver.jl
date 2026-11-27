# optimizesingles!: optimize one variable at a time (alternation), as in NLLSsolver 5. Unexported, as in
# 5.3.4; VisualGeometryOptimization calls it as `NLLSsolver.optimizesingles!`. Rewritten on the public API
# of NLLSModel + SecondOrderOptimization (a subproblem per variable, solved through the front door), so it
# is expected to be slower than 5.x's hand-written single-variable loop (measured in the fixes file).

# The identity for summing per-variable results (no variables of the requested type). Builds a zero
# OptimizationResult, as NLLSsolver 5 did; the field order matches SecondOrderOptimization's own `+`.
emptysinglesresult(options::OptimizationOptions) = OptimizationResult(
    0.0, 0.0, UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
    0, 0, 0, 0, 0, Symbol(options.iterator), 0, 0, 0, 0)

"""
    NLLSsolver.optimizesingles!(model::Model, options, type::DataType; kwargs...) -> NLLSResult
    NLLSsolver.optimizesingles!(model::Model, options, indices; kwargs...) -> NLLSResult

Optimize each named variable on its own (alternation), the rest held fixed, and return the summed result.
**Deprecated** (and unexported, as in NLLSsolver 5). `type` selects every variable of that type; `indices`
names them directly. Keywords (`precision`, `reduction`, …) are forwarded to the optimizer; `numthreads`
defaults to 1, as NLLSsolver 5 ran singles on one thread.

Written on the public API: a subproblem is built for each variable and solved through `optimize!`. For an
equivalent on the new packages, loop over the variables with `subproblem` and
`optimize!(sub, options; unfixed = i)`.
"""
function optimizesingles!(problem::Model{VT, CT}, options::OptimizationOptions, indices::AbstractVector; numthreads=1, kwargs...) where {VT, CT}
    # Cost indices per requested variable, read in one pass over values(problem.costs) (so each list is in
    # that concatenated order), from each cost's varindices.
    requested = Set{Int}(indices)
    costindices = Dict{Int, Vector{Int}}(i => Int[] for i in requested)
    k = 0
    for group in values(problem.costs)
        for c in group
            k += 1
            for vi in varindices(c)
                haskey(costindices, vi) && push!(costindices[vi], k)
            end
        end
    end
    # Group the variables by size (so a size change is the only time the shared workspace is rebuilt).
    order = sort!(collect(requested); by = i -> Int(nvars(problem.variables[i])))
    sub = Model(problem.variables, CT)   # shares the variable vector with problem
    workspace = OptimizationWorkspace()
    result = nothing
    for i in order
        subproblem!(sub, problem, costindices[i])
        r = SecondOrderOptimization.optimize!(sub, options; unfixed=i, numthreads, workspace, kwargs...)
        result = result === nothing ? r : result + r
    end
    return result === nothing ? emptysinglesresult(options) : result
end

function optimizesingles!(problem::Model, options::OptimizationOptions, type::DataType; kwargs...)
    indices = findall(v -> isa(v, type), problem.variables)
    return optimizesingles!(problem, options, indices; kwargs...)
end
