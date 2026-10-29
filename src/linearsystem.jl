using SparseArrays, Static, StaticArrays, LinearAlgebra, LDLFactorizations

# function getresblocksizes!(resblocksizes, residuals::Vector, ind::Int)::Int
#     @inbounds for res in residuals
#         resblocksizes[ind] = nres(res)
#         ind += 1
#     end
#     return ind
# end

struct LinearSystem{F, V}
    x::F
    ls::V
end
LinearSystem{F, V}(xargs::Tuple, lsargs::Tuple) where {F, V} = LinearSystem{F, V}(F(xargs...), V(lsargs...))
updateunfixed(old::LinearSystem{F, V}, unfixed) where {F, V} = LinearSystem{F, V}(F(unfixed, old.x), old.ls)

struct LinearSystemShared{F, V, N}
    x::F
    ls::SVector{N, V}
    costs::MVector{N, Float64}
end
function LinearSystemShared{F, V, N}(xargs::Tuple, lsargs::Tuple) where {F, V, N}
    return LinearSystemShared{F, V, N}(F(xargs...), SVector{N, V}(ntuple(i -> V(lsargs...), Val(N))), MVector{N, Float64}(undef))
end
updateunfixed(old::LinearSystemShared{F, V, N}, unfixed) where {F, V, N} = LinearSystemShared{F, V, N}(F(unfixed, old.x), old.ls, old.costs)

# Ephemeral task wrapper constructed on the stack
LinearSystem(ls::LinearSystemShared{F, V, N}, ind) where {F, V, N} = LinearSystem{F, V}(ls.x, ls.ls[ind])


getA(ls::LinearSystem) = ls.ls.A
getb(ls::LinearSystem) = ls.ls.b
getx(ls::LinearSystem) = ls.x.x

getA(ls::LinearSystemShared) = ls.ls[1].A
getb(ls::LinearSystemShared) = ls.ls[1].b
getx(ls::LinearSystemShared) = ls.x.x

getA(ls) = ls.A
getb(ls) = ls.b
getx(ls) = ls.x

# Uni-variate linear system
struct UniVariateLSdynamic_ls
    A::Matrix{Float64}
    b::Vector{Float64}

    function UniVariateLSdynamic_ls(varlen)
        return new(Matrix{Float64}(undef, varlen, varlen), Vector{Float64}(undef, varlen))
    end
end

struct UniVariateLSdynamic_x
    varindex::UInt
    x::Vector{Float64}

    function UniVariateLSdynamic_x(unfixed, varlen)
        return new(UInt(unfixed), Vector{Float64}(undef, varlen))
    end
    function UniVariateLSdynamic_x(unfixed, old::UniVariateLSdynamic_x)
        return new(UInt(unfixed), old.x)
    end
end

struct UniVariateLSstatic_ls{N, N2}
    A::MMatrix{N, N, Float64, N2}
    b::MVector{N, Float64}

    function UniVariateLSstatic_ls{N, N2}() where {N, N2}
        return new(MMatrix{N, N, Float64, N2}(undef), MVector{N, Float64}(undef))
    end
end

struct UniVariateLSstatic_x{N}
    varindex::UInt
    x::MVector{N, Float64}

    function UniVariateLSstatic_x{N}(unfixed) where N
        return new(UInt(unfixed), MVector{N, Float64}(undef))
    end
    function UniVariateLSstatic_x{N}(unfixed, old::UniVariateLSstatic_x{N}) where N
        return new(UInt(unfixed), old.x)
    end
end

UniVariateLSsingle = Union{LinearSystem{UniVariateLSdynamic_x, UniVariateLSdynamic_ls}, LinearSystem{UniVariateLSstatic_x{N}, UniVariateLSstatic_ls{N, N2}} where {N, N2}}
UniVariateLSdynamic = Union{LinearSystem{UniVariateLSdynamic_x, UniVariateLSdynamic_ls}, LinearSystemShared{UniVariateLSdynamic_x, UniVariateLSdynamic_ls, M} where M}
UniVariateLSstatic = Union{LinearSystem{UniVariateLSstatic_x{N}, UniVariateLSstatic_ls{N, N2}} where {N, N2}, LinearSystemShared{UniVariateLSstatic_x{N}, UniVariateLSstatic_ls{N, N2}, M} where {N, N2, M}}
UniVariateLS = Union{UniVariateLSdynamic, UniVariateLSstatic}

function computestartindices(blocksizes)
    startind = Vector{Int}(undef, length(blocksizes))
    startind[1] = 1
    @inbounds startind[2:end] .= view(blocksizes, 1:length(blocksizes)-1)
    return cumsum!(startind)
end

# Multi-variate linear system
struct MultiVariateLSsparse_x
    x::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{UInt} # One per block in b
    hessian::SparseMatrixCSC{Float64, Int} # Storage for sparse hessian
    sparseindices::Vector{Int}
    ldlfac::LDLFactorizations.LDLFactorization{Float64, Int, Int, Int}
end

struct MultiVariateLSsparse_ls
    A::BlockSparseMatrix{Float64}
    b::Vector{Float64}
end

struct MultiVariateLSdense_x
    x::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{UInt} # One per block in b
end

struct MultiVariateLSdense_ls
    A::BlockDenseMatrix{Float64}
    b::Vector{Float64}
end

const MultiVariateLSsparse = Union{LinearSystem{MultiVariateLSsparse_x, MultiVariateLSsparse_ls}, LinearSystemShared{MultiVariateLSsparse_x, MultiVariateLSsparse_ls, M} where M}
const MultiVariateLSdense = Union{LinearSystem{MultiVariateLSdense_x, MultiVariateLSdense_ls}, LinearSystemShared{MultiVariateLSdense_x, MultiVariateLSdense_ls, M} where M}
const MultiVariateLSsingle = Union{LinearSystem{MultiVariateLSsparse_x, MultiVariateLSsparse_ls}, LinearSystem{MultiVariateLSdense_x, MultiVariateLSdense_ls}}
const MultiVariateLS = Union{MultiVariateLSsparse, MultiVariateLSdense}

function MultiVariateLSsparse(A::BlockSparseMatrix, blockindices, storehessian=true)
    @assert A.rowblocksizes == A.columnblocksizes
    boffsets = computestartindices(A.rowblocksizes)
    blen = boffsets[end] + A.rowblocksizes[end] - 1
    if storehessian
        x = Vector{Float64}(undef, blen)
        sparseindices = makesparseindices(A, true)
        hessian = SparseMatrixCSC{Float64, Int}(sparseindices.m, sparseindices.n, sparseindices.colptr, sparseindices.rowval, Vector{Float64}(undef, length(sparseindices.nzval)))
        sparseindices = sparseindices.nzval
    else
        x = Vector{Float64}(undef, 1)
        sparseindices = Vector{Int}()
        hessian = spzeros(1, 1)
    end
    ldlfac = ldl_analyze(hessian)
    return LinearSystem(
        MultiVariateLSsparse_x(x, blockindices, boffsets, hessian, sparseindices, ldlfac),
        MultiVariateLSsparse_ls(A, zeros(Float64, blen))
    )
end

function MultiVariateLSsparse(A::BlockSparseMatrix, b, x, blockindices, boffsets, hessian, sparseindices, ldlfac)
    return LinearSystem(
        MultiVariateLSsparse_x(x, blockindices, boffsets, hessian, sparseindices, ldlfac),
        MultiVariateLSsparse_ls(A, b)
    )
end

function MultiVariateLSdense(blocksizes, blockindices)
    boffsets = computestartindices(blocksizes)
    blen = boffsets[end] + blocksizes[end]
    push!(boffsets, blen)
    blen -= 1
    return LinearSystem(
        MultiVariateLSdense_x(Vector{Float64}(undef, blen), blockindices, boffsets),
        MultiVariateLSdense_ls(BlockDenseMatrix{Float64}(boffsets), zeros(Float64, blen))
    )
end

function MultiVariateLSdense(A::BlockDenseMatrix, b, x, blockindices, boffsets)
    return LinearSystem(
        MultiVariateLSdense_x(x, blockindices, boffsets),
        MultiVariateLSdense_ls(A, b)
    )
end

copy_structure(ls::UniVariateLSdynamic_ls) = UniVariateLSdynamic_ls(size(ls.A, 1))
copy_structure(::UniVariateLSstatic_ls{N, N2}) where {N, N2} = UniVariateLSstatic_ls{N, N2}()
copy_structure(ls::MultiVariateLSdense_ls) = MultiVariateLSdense_ls(copy_structure(ls.A), zeros(Float64, length(ls.b)))
copy_structure(ls::MultiVariateLSsparse_ls) = MultiVariateLSsparse_ls(copy_structure(ls.A), zeros(Float64, length(ls.b)))
copy_structure(ls::LinearSystem{F, V}) where {F, V} = LinearSystem{F, V}(ls.x, copy_structure(ls.ls))
copy_structure(ls::LinearSystemShared{F, V, N}) where {F, V, N} = LinearSystemShared{F, V, N}(ls.x, SVector{N, V}(ntuple(i -> copy_structure(ls.ls[i]), Val(N))), MVector{N, Float64}(undef))

# The blocks of one linear system with the unknowns (and solver state) of another
sharex(ls::LinearSystem{F, V}, from) where {F, V} = LinearSystem{F, V}(from.x, ls.ls)
sharex(ls::LinearSystemShared{F, V, N}, from) where {F, V, N} = LinearSystemShared{F, V, N}(from.x, ls.ls, ls.costs)

sharedlinearsystem(ls::LinearSystem, ::StaticInt{1}) = ls
sharedlinearsystem(ls::LinearSystemShared, ::StaticInt) = ls
function sharedlinearsystem(ls::LinearSystem{F, V}, ::StaticInt{N}) where {F, V, N}
    buffers = SVector{N, V}(ntuple(i -> i == 1 ? ls.ls : copy_structure(ls.ls), Val(N)))
    costs = MVector{N, Float64}(undef)
    return LinearSystemShared{F, V, N}(ls.x, buffers, costs)
end
sharedlinearsystem(ls, n::Integer) = sharedlinearsystem(ls, static(n))
sharedlinearsystem(ls, problem, options) = sharedlinearsystem(ls, problem, options.numthreads)
sharedlinearsystem(ls, problem, ::StaticInt{1}) = ls
sharedlinearsystem(ls, problem, numthreads::StaticInt) = sharedlinearsystem(ls, countcosts(costnum, problem.costs) >= 10000 ? numthreads : static(1))

add!(dest::UniVariateLSdynamic_ls, src::UniVariateLSdynamic_ls) = (dest.A .+= src.A; dest.b .+= src.b; nothing)
add!(dest::UniVariateLSstatic_ls, src::UniVariateLSstatic_ls) = (dest.A .+= src.A; dest.b .+= src.b; nothing)
add!(dest::MultiVariateLSdense_ls, src::MultiVariateLSdense_ls) = (dest.A.data .+= src.A.data; dest.b .+= src.b; nothing)
add!(dest::MultiVariateLSsparse_ls, src::MultiVariateLSsparse_ls) = (dest.A.data .+= src.A.data; dest.b .+= src.b; nothing)
add!(dest::LinearSystem, src::LinearSystem) = add!(dest.ls, src.ls)

function updatesymlinearsystem!(linsystem::UniVariateLS, g, H, unusedargs...)
    # Update the blocks in the problem
    getb(linsystem) .+= g
    getA(linsystem) .+= H
end

function updatesymA!(A, a, vars, varflags, blockindices)
    # Update the blocks in the problem
    loffseti = static(0)
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && bitiset(varflags, i)
            nvi = nvars(vars[i])
            rangei = SR(1, nvi) .+ loffseti
            loffseti += nvi
            @inbounds block(A, blockindices[i], blockindices[i], nvi, nvi) .+= view(a, rangei, rangei)

            loffsetj = static(0)
            @unroll for j in 1:i-1
                if bitiset(varflags, j)
                    @inbounds nvj = nvars(vars[j])
                    rangej = SR(1, nvj) .+ loffsetj
                    loffsetj += nvj
                    if @inbounds blockindices[i] >= blockindices[j] # Make sure the block matrix is lower triangular
                        @inbounds block(A, blockindices[i], blockindices[j], nvi, nvj) .+= view(a, rangei, rangej)
                    else
                        @inbounds block(A, blockindices[j], blockindices[i], nvj, nvi) .+= view(a, rangej, rangei)
                    end
                end
            end
        end
    end
end

function updateb!(B, b, vars, varflags, boffsets, blockindices) 
    # Update the blocks in the problem
    loffset = static(1)
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && bitiset(varflags, i)
            nv = nvars(vars[i])
            range = SR(0, nv-1)
            @inbounds view(B, range .+ boffsets[blockindices[i]]) .+= view(b, range .+ loffset)
            loffset += nv
        end
    end
end

function updatesymlinearsystem!(linsystem::MultiVariateLS, g::AbstractVector, H::AbstractArray, vars, varflags, blockindices)
    updateb!(getb(linsystem), g, vars, varflags, linsystem.x.boffsets, blockindices)
    updatesymA!(getA(linsystem), H, vars, varflags, blockindices)
end

getgrad(ls::LinearSystem) = ls.ls.b
getgrad(ls::LinearSystemShared) = ls.ls[1].b
getgrad(linsystem) = getb(linsystem)

gethessian(ls::LinearSystem) = gethessian(ls.x, ls.ls)
gethessian(ls::LinearSystemShared) = gethessian(ls.x, ls.ls[1])
gethessian(::UniVariateLSdynamic_x, ls::UniVariateLSdynamic_ls) = ls.A
gethessian(::UniVariateLSstatic_x, ls::UniVariateLSstatic_ls) = ls.A
gethessian(::MultiVariateLSdense_x, ls::MultiVariateLSdense_ls) = symmetrifyfull(ls.A)
function gethessian(x::MultiVariateLSsparse_x, ls::MultiVariateLSsparse_ls)
    # Fill sparse hessian
    @inbounds for (i, si) in enumerate(x.sparseindices)
        x.hessian.nzval[i] = ls.A.data[si]
    end
    return x.hessian
end
gethessian(linsystem) = getA(linsystem)
gethessgrad(linsystem) = gethessian(linsystem), getgrad(linsystem)

function zero!(linsystem)
    fill!(getb(linsystem), 0)
    zero!(getA(linsystem))
end

getoffsets(block, linsystem::MultiVariateLS) = @inbounds(linsystem.x.blockindices[varindices(block)])
function getoffsets(block, linsystem::UniVariateLS)
    varind = varindices(block)
    if isa(varind, Number)
        return SVector(UInt(varind == linsystem.x.varindex))
    end
    return convert.(UInt, varind .== linsystem.x.varindex)
end

function update!(to::Vector, from::Vector, linsystem::MultiVariateLS, step=getx(linsystem))
    # Update each variable
    for (i, j) in enumerate(linsystem.x.blockindices)
        if j != 0
            @inbounds to[i] = update(from[i], step, linsystem.x.boffsets[j])
        end
    end
end

@inline function update!(to::Vector, from::Vector, linsystem::UniVariateLS, step=getx(linsystem))
    # Update one variable
    @inbounds to[linsystem.x.varindex] = update(from[linsystem.x.varindex], step)
end
