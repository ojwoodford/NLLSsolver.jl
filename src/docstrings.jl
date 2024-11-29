

"""
    NLLSsolver.ndeps(mycost::AbstractCost)

Return the number of variables that `mycost` depends on. The return value must
be a static integer.

# Example

For a problem `MyCost <: AbstractCost` that has two variables, the user should
define the following function:

```julia
NLLSsolver.ndeps(::MyCost) = static(2)
```

!!! note
    Required user-specialized API function.
"""
function ndeps end


"""
    NLLSsolver.varindices(mycost::AbstractCost)

Return a static vector representing the indices of the variables that `mycost` depends on.

# Example

For a problem `MyCost <: AbstractCost` that depends on the first and third variables,
the user should define the following function:

```julia
NLLSsolver.varindices(::MyCost) = SVector(1, 3)
```

!!! note
    Required user-specialized API function.
"""
function varindices end


"""
    NLLSsolver.nres(mycost::AbstractCost)

Return the number of residuals that `mycost` generates. The return value must be
an integer, preferably static, and must not change during an optimization.

# Example

For a problem `MyCost <: AbstractCost` that generates a residual vector of length three,

```julia
NLLSsolver.nres(::MyCost) = static(3)
```

!!! tip
    If the residual length is known at compile time, return a static integer.

!!! note
    Required user-specialized API function.
"""
function nres end


"""
    NLLSsolver.getvars(mycost::AbstractCost, vars)

Return the variables that `mycost` depends on. The return value must be a tuple
of variables. It is important that the output variables are typed (i.e. not `Any`)
for good performance.

# Example

For a problem `MyCost <: AbstractCost` that depends on the first and third variables,

```julia
NLLSsolver.getvars(::MyCost, vars) = (vars[1], vars[3])
```

!!! tip
    If `vars` is heterogeneous in type, add type-annotations to the specific variables, e.g.
    `(vars[1]::Matrix{T}, vars[3]::Vector{T})` for a `MyCost{T}`.

!!! note
    Required user-specialized API function.
"""
function getvars end


"""
    NLLSsolver.computeresidual(mycost::AbstractCost, vars...)

Compute the residuals generated by `mycost` given the variables `vars`. The return
value must be a vector.

# Example

For the Rosenbrock problem with
```julia
struct Rosenbrock <: NLLSsolver.AbstractResidual
    a::Float64
    b::Float64
end
```

the user should define the following function:

```julia
function NLLSsolver.computeresidual(res::Rosenbrock, x)
    return SVector(res.a * (1 - x[1]), res.b * (x[1] ^ 2 - x[2]))
end
"""
function computeresidual end
