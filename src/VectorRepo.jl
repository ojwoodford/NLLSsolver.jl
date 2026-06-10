
struct VectorRepo{T}
    data::Dict{DataType, Vector}
    function VectorRepo{Any}()
        return new{Any}(Dict{DataType, Vector}())
    end
    function VectorRepo{T}() where T
        vr = new{T}(Dict{DataType, Vector}())
        foreach(t -> vr.data[t] = Vector{t}(), uniontotuple(T))
        return vr
    end
end
VectorRepo() = VectorRepo{Any}()

function Base.get(vr::VectorRepo{Any}, ::Type{type})::Vector{type} where type
    return haskey(vr.data, type) ? @inbounds(vr.data[type]::Vector{type}) : Vector{type}()
end

function Base.get(vr::VectorRepo{T}, ::Type{type})::Vector{type} where {T, type}
    @assert type<:T "Invalid type"
    return @inbounds vr.data[type]::Vector{type}
end

function Base.get!(vr::VectorRepo{Any}, ::Type{type})::Vector{type} where type
    if haskey(vr.data, type)
        vec = @inbounds vr.data[type]::Vector{type}
    else
        vec = Vector{type}()
        vr.data[type] = vec
    end
    return vec
end

Base.get!(vr::VectorRepo{T}, ::Type{type}) where {T, type} = get(vr, type)
Base.push!(vr::VectorRepo, v::T) where T = push!(get!(vr, T), v)
Base.append!(vr::VectorRepo, v::Vector{T}) where T = append!(get!(vr, T), v)

# Get the keys
Base.keys(vr::VectorRepo{Any}) = keys(vr.data)
# Get a typed Tuple of keys
Base.keys(::VectorRepo{T}) where T = uniontotuple(T)

# Get a vector of the vectors (allocates)
Base.values(vr::VectorRepo{Any}) = values(vr.data)
# Get a typed Tuple of vectors
Base.values(vr::VectorRepo{T}) where T = (valuetuple(vr, T)...,)
valuetuple(vr::VectorRepo, T::Union) = (valuetuple(vr, T.a)..., valuetuple(vr, T.b)...)
valuetuple(vr::VectorRepo, T::DataType) = (get(vr, T),)

# Sum reduction
Base.sum(fun, vr::VectorRepo{Any}; kw...) = sum(bindleadingargs(vrsum, fun, kw), values(vr.data); kw...)
vrsum(fun, kw, v::Vector) = sum(fun, v; kw...)
# Static dispatch if types are known
Base.sum(fun, vr::VectorRepo{T}; kw...) where T = vrsum(fun, vr, kw, T)
vrsum(fun, vr::VectorRepo, kw, T::Union) = vrsum(fun, vr, kw, T.a) + vrsum(fun, vr, kw, T.b)
vrsum(fun, vr::VectorRepo, kw, T::DataType) = vrsum(fun, kw, get(vr, T))

# Sum reduction over a subset of the elements
# sumsubset(fun, vr::VectorRepo{Any}, subsetfun; init=0.0) = sum(bindleadingargs(sumsubsetvec, fun, subsetfun, init), values(vr.data); init=init)
# sumsubset(fun, vr::VectorRepo{T}, subsetfun; init=0.0) where T = vrsumsubset(fun, subsetfun, vr, init,T)
# vrsumsubset(fun, subsetfun, vr, init, T::Union) = vrsumsubset(fun, subsetfun, vr, init, T.a) + vrsumsubset(fun, subsetfun, vr, init, T.b)
# vrsumsubset(fun, subsetfun, vr, init, T::DataType)= sumsubsetvec(args, subsetfun, init, get(vr, T))
# sumsubsetvec(fun, subsetfun, init, vector) = sum(fun, view(vector, subsetfun(vector)); init=init)
