struct VectorRepo{T}
    data::Dict{DataType, Vector}
    function VectorRepo{T}() where T
        return new{T}(Dict{DataType, Vector}())
    end
end
VectorRepo() = VectorRepo{Any}()

@inline Base.values(vr::VectorRepo) = values(vr.data)

function Base.get(vr::VectorRepo{T}, type::DataType) where T
    @assert type<:T "Invalid type"
    return haskey(vr.data, type) ? vr.data[type]::Vector{type} : Vector{type}()
end

function Base.get!(vr::VectorRepo{T}, type::DataType) where T
    @assert type<:T "Invalid type"
    return get!(vr.data, type, Vector{type}())::Vector{type}
end

@inline Base.push!(vr::VectorRepo, v::T) where T = push!(get!(vr, T), v)
@inline Base.append!(vr::VectorRepo, v::Vector{T}) where T = append!(get!(vr, T), v)

# Dynamic dispatch if types are not known 
@inline Base.sum(fun, vr::VectorRepo{Any}; init=0.0) = sum(fixallbutlast(vrsum, fun, init), values(vr.data); init=init)
@inline vrsum(fun, init, v::Vector) = sum(fun, v; init=init)
# Static dispatch if types are known
@inline Base.sum(fun, vr::VectorRepo{T}; init=0.0) where T = vrsum(fun, vr, init, T)
@inline vrsum(fun, vr::VectorRepo, init, T::Union) = vrsum(fun, vr, init, T.a) + vrsum(fun, vr, init, T.b)
@inline vrsum(fun, vr::VectorRepo, init, T::DataType) = sum(fun, get(vr, T); init=init)
