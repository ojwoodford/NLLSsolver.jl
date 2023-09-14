using OrderedCollections

struct VectorRepo{T}
    data::OrderedDict{DataType, Vector}
    function VectorRepo{T}() where T
        return new{T}(OrderedDict{DataType, Vector}())
    end
end
VectorRepo() = VectorRepo{Any}()

@inline function Base.get(vr::VectorRepo{T}, type::DataType)::Vector{type} where T
    @assert type<:T "Invalid type"
    return haskey(vr.data, type) ? vr.data[type]::Vector{type} : Vector{type}()
end

@inline function Base.get!(vr::VectorRepo{T}, type::DataType)::Vector{type} where T
    @assert type<:T "Invalid type"
    return get!(vr.data, type, Vector{type}())::Vector{type}
end

@inline Base.push!(vr::VectorRepo, v::T) where T = push!(get!(vr, T), v)
@inline Base.append!(vr::VectorRepo, v::Vector{T}) where T = append!(get!(vr, T), v)

# Get the keys
@inline Base.keys(vr::VectorRepo{Any}) = keys(vr.data)
# Get a typed Tuple of keys
@inline Base.keys(::VectorRepo{T}) where T = uniontotuple(T)

# Get a vector of the vectors (allocates)
@inline Base.values(vr::VectorRepo{Any}) = values(vr.data)
# Get a typed Tuple of vectors
@inline Base.values(vr::VectorRepo{T}) where T = (valuetuple(vr, T)...,)
@inline valuetuple(vr::VectorRepo, T::Union) = (valuetuple(vr, T.a)..., valuetuple(vr, T.b)...)
@inline valuetuple(vr::VectorRepo, T::DataType) = (get(vr, T),)

# # Iterators
# # Any case
# @inline function Base.iterate(vr::VectorRepo{Any})
#     it = iterate(vr.data)
#     if !isnothing(it)
#         return it[1].second, it[2]
#     end
#     return nothing
# end
# @inline function Base.iterate(vr::VectorRepo{Any}, state)
#     it = iterate(vr.data, state)
#     if !isnothing(it)
#         return it[1].second, it[2]
#     end
#     return nothing
# end
# # Union case
# @inline Base.iterate(vr::VectorRepo{T}) where T = iterate(vr, T)
# @inline Base.iterate(vr::VectorRepo, T::Union) = (get(vr, T.a), T.b)
# @inline Base.iterate(vr::VectorRepo, T::DataType) = (get(vr, T), nothing)
# @inline Base.iterate(::VectorRepo, ::Nothing) = nothing

# Sum reduction
@inline Base.sum(fun, vr::VectorRepo{Any}; init=0.0) = sum(fixallbutlast(vrsum, fun, init), values(vr.data); init=init)
@inline vrsum(fun, init, v::Vector) = sum(fun, v; init=init)
# Static dispatch if types are known
@inline Base.sum(fun, vr::VectorRepo{T}; init=0.0) where T = vrsum(fun, vr, init, T)
@inline vrsum(fun, vr::VectorRepo, init, T::Union) = vrsum(fun, vr, init, T.a) + vrsum(fun, vr, init, T.b)
@inline vrsum(fun, vr::VectorRepo, init, T::DataType) = sum(fun, get(vr, T); init=init)

# Sum reduction over a subset of the elements
@inline sumsubset(fun, subsetfun, vr::VectorRepo{Any}) = sum(fixallbutlast(sumsubset, fun, subsetfun), values(vr.data); init=0.0)
@inline sumsubset(fun, subsetfun, vr::VectorRepo{T}) where T = sumsubset(fun, subsetfun, vr, T)
@inline sumsubset(fun, subsetfun, vr::VectorRepo, T::Union) = sumsubset(fun, subsetfun, vr, T.a) + sumsubset(fun, subsetfun, vr, T.b)
@inline sumsubset(fun, subsetfun, vr::VectorRepo, T::DataType) = sumsubset(fun, subsetfun, get(vr, T))
@inline sumsubset(fun, subsetfun::Function, vector::Vector) = sumsubset(fun, subsetfun(vector), vector)
function sumsubset(fun, subset::Union{BitVector, Vector{Bool}}, vector::Vector)::Float64
    total = 0.0
    for (ind, val) in enumerate(subset)
        if val
            total += fun(vector[ind])
        end
    end
    return total
end
function sumsubset(fun, subset, vector::Vector)::Float64
    total = 0.0
    for ind in subset
        total += fun(vector[ind])
    end
    return total
end

# Map
Base.map(fun, vr::VectorRepo) = OrderedDict([eltype(val)=>fun(val) for val in values(vr)]...)
