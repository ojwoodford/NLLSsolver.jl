using SparseArrays, Static, StaticArrays, LinearAlgebra

function getresblocksizes!(resblocksizes, residuals::Vector, ind::Int)::Int
    @inbounds for res in residuals
        resblocksizes[ind] = nres(res)
        ind += 1
    end
    return ind
end

# Uni-variate linear system
struct UniVariateLS
    A::Matrix{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    varindex::UInt

    function UniVariateLS(unfixed, varlen)
        return new(zeros(Float64, varlen, varlen), zeros(Float64, varlen), Vector{Float64}(undef, varlen), UInt(unfixed))
    end
end

struct UniVariateLSstatic{N, N2}
    A::MMatrix{N, N, Float64, N2}
    b::MVector{N, Float64}
    x::MVector{N, Float64}
    varindex::UInt

    function UniVariateLSstatic{N, N2}(unfixed) where {N, N2}
        return new(zeros(MMatrix{N, N, Float64, N2}), zeros(MVector{N, Float64}), MVector{N, Float64}(undef), UInt(unfixed))
    end
end

function computestartindices(blocksizes)
    startind = circshift(blocksizes, 1)
    startind[1] = 1
    return cumsum(startind)
end

# Multi-variate linear system
struct MultiVariateLS
    A::BlockSparseMatrix{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{UInt} # One per block in b
    sparseindices::SparseMatrixCSC{Int, Int} # Indices for turning block sparse matrix into sparse matrix

    function MultiVariateLS(A::BlockSparseMatrix, blockindices, sparseindices=spzeros(0, 0))
        @assert A.rowblocksizes==A.columnblocksizes
        boffsets = computestartindices(A.rowblocksizes)
        blen = boffsets[end] + A.rowblocksizes[end] - 1
        return new(A, zeros(Float64, blen), Vector{Float64}(undef, blen), blockindices, boffsets, sparseindices)
    end
end

function makesymmvls(problem, unfixed, nblocks)
    # Multiple variables. Use a block sparse matrix
    blockindices = zeros(UInt, length(problem.variables))
    blocksizes = zeros(UInt, nblocks)
    nblocks = 0
    for (index, unfixed_) in enumerate(unfixed)
        if unfixed_
            nblocks += 1
            blockindices[index] = nblocks
            blocksizes[nblocks] = nvars(problem.variables[index])
        end
    end

    # Compute the block sparsity
    sparsity = getvarcostmap(problem)
    sparsity = sparsity[unfixed,:]
    sparsity = sparse(UpperTriangular(sparse(sparsity * sparsity' .> 0)))

    # Construct the BSM
    bsm = BlockSparseMatrix{Float64}(sparsity, blocksizes, blocksizes)

    # Construct the sparse indices
    sparseindices = size(bsm, 2) > 1000 && 3 * nnz(bsm) < length(bsm) ? makesparseindices(bsm, true) : spzeros(0, 0)

    # Construct the MultiVariateLS
    return MultiVariateLS(bsm, blockindices, sparseindices)
end

function updatesymlinearsystem!(linsystem::Union{UniVariateLS, UniVariateLSstatic}, g, H, unusedargs...)
    # Update the blocks in the problem
    linsystem.b .+= g
    linsystem.A .+= H
end

function updatesymA!(A, a, vars, varflags, blockindices)
    # Update the blocks in the problem
    loffseti = static(0)
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && bitiset(varflags, i)
            nvi = nvars(vars[i])
            rangei = SR(1, nvi) .+ loffseti
            loffseti += nvi
            block(A, blockindices[i], blockindices[i], nvi, nvi) .+= @inbounds view(a, rangei, rangei)

            loffsetj = static(0)
            @unroll for j in 1:i-1
                if bitiset(varflags, j)
                    nvj = nvars(vars[j])
                    rangej = SR(1, nvj) .+ loffsetj
                    loffsetj += nvj
                    if blockindices[i] >= blockindices[j] # Make sure the BSM is lower triangular
                        block(A, blockindices[i], blockindices[j], nvi, nvj) .+= @inbounds view(a, rangei, rangej)
                    else
                        block(A, blockindices[j], blockindices[i], nvj, nvi) .+= @inbounds view(a, rangej, rangei)
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
    updateb!(linsystem.b, g, vars, varflags, linsystem.boffsets, blockindices)
    updatesymA!(linsystem.A, H, vars, varflags, blockindices)
end

uniformscaling!(linsystem, k) = uniformscaling!(linsystem.A, k)

getgrad(linsystem) = linsystem.b
gethessgrad(linsystem::Union{UniVariateLS, UniVariateLSstatic}) = (linsystem.A, linsystem.b)
function gethessgrad(linsystem::MultiVariateLS)
    if !isempty(linsystem.sparseindices)
        return sparse(linsystem.A, linsystem.sparseindices), linsystem.b
    end
    return symmetrifyfull(linsystem.A), linsystem.b
end

function zero!(linsystem)
    fill!(linsystem.b, 0)
    zero!(linsystem.A)
end

getoffsets(block, linsystem::MultiVariateLS) = @inbounds(linsystem.blockindices[varindices(block)])
function getoffsets(block, linsystem::Union{UniVariateLS, UniVariateLSstatic})
    varind = varindices(block)
    if isa(varind, Number)
        return SVector(UInt(varind == linsystem.varindex))
    end
    return convert.(UInt, varind .== linsystem.varindex)
end

function update!(to::Vector, from::Vector, linsystem::MultiVariateLS, step=linsystem.x)
    # Update each variable
    @inbounds for (i, j) in enumerate(linsystem.blockindices)
        if j != 0
            to[i] = update(from[i], step, linsystem.boffsets[j])
        end
    end
end

@inline function update!(to::Vector, from::Vector, linsystem::Union{UniVariateLS, UniVariateLSstatic}, step=linsystem.x)
    # Update one variable
    to[linsystem.varindex] = update(from[linsystem.varindex], step)
end