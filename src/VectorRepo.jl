
struct VectorRepo{T}
    data::Dict{DataType, Vector}
    function VectorRepo{T}(args...) where T
        return new{T}(Dict{DataType, Vector}(args...))
    end
end
VectorRepo(args...) = VectorRepo{Any}(args...)

function Base.get(vr::VectorRepo{T}, ::Type{type})::Vector{type} where {T, type}
    @assert type<:T "Invalid type"
    return haskey(vr.data, type) ? vr.data[type]::Vector{type} : Vector{type}()
end

function Base.get!(vr::VectorRepo{T}, ::Type{type})::Vector{type} where {T, type}
    @assert type<:T "Invalid type"
    if haskey(vr.data, type)
        vec = vr.data[type]::Vector{type}
    else
        vec = Vector{type}()
        vr.data[type] = vec
    end
    return vec
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
@inline valuetuple(vr::VectorRepo, ::Type{T}) where T = (get(vr, T)::Vector{T},)

# Sum reduction
Base.sum(fun, vr::VectorRepo{Any}; init=0.0) = sum(bindleadingargs(vrsum, fun, init), values(vr.data); init=init)
vrsum(fun, init, v::Vector) = multithreadedsum(fun, v, init)
# Static dispatch if types are known
Base.sum(fun, vr::VectorRepo{T}; init=0.0) where T = vrsum(fun, vr, init, T)
vrsum(fun, vr::VectorRepo, init, T::Union) = vrsum(fun, vr, init, T.a) + vrsum(fun, vr, init, T.b)
vrsum(fun, vr::VectorRepo, init, ::Type{T}) where T = vrsum(fun, init, get(vr, T)::Vector{T})

# Sum reduction over a subset of the elements
# sumsubset(fun, vr::VectorRepo{Any}, subsetfun; init=0.0) = sum(bindleadingargs(sumsubsetvec, fun, subsetfun, init), values(vr.data); init=init)
# sumsubset(fun, vr::VectorRepo{T}, subsetfun; init=0.0) where T = vrsumsubset(fun, subsetfun, vr, init,T)
# vrsumsubset(fun, subsetfun, vr, init, T::Union) = vrsumsubset(fun, subsetfun, vr, init, T.a) + vrsumsubset(fun, subsetfun, vr, init, T.b)
# vrsumsubset(fun, subsetfun, vr, init, ::Type{T}) where T = sumsubsetvec(args, subsetfun, init, get(vr, T)::Vector{T})
# sumsubsetvec(fun, subsetfun, init, vector) = multithreadedsum(fun, view(vector, subsetfun(vector)), init)
