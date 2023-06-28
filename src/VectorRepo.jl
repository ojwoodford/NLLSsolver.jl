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
    return get(vr.data, type, Vector{type}())::Vector{type}
end

function Base.get!(vr::VectorRepo{T}, type::DataType) where T
    @assert type<:T "Invalid type"
    return get!(vr.data, type, Vector{type}())::Vector{type}
end

@inline Base.push!(vr::VectorRepo, v::T) where T = push!(get!(vr, T), v)
@inline Base.append!(vr::VectorRepo, v::Vector{T}) where T = append!(get!(vr, T), v)

# Dynamic dispatch if types are not known 
@inline Base.sum(fun, vr::VectorRepo{Any}) = sum(Base.Fix1(vrsum, fun), values(vr.data); init=0.0)
@inline vrsum(fun, v::Vector) = sum(fun, v; init=0.0)
# Static dispatch if types are known
@inline Base.sum(fun, vr::VectorRepo{T}) where T = vrsum(fun, vr, T)
@inline vrsum(fun, vr::VectorRepo, T::Union) = vrsum(fun, vr, T.a) + vrsum(fun, vr, T.b)
@inline vrsum(fun, vr::VectorRepo, T::DataType) = sum(fun, get(vr, T); init=0.0)