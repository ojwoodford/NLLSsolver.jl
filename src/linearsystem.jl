using SparseArrays, Static, StaticArrays, LinearAlgebra, LDLFactorizations

function getresblocksizes!(resblocksizes, residuals::Vector, ind::Int)::Int
    @inbounds for res in residuals
        resblocksizes[ind] = nres(res)
        ind += 1
    end
    return ind
end

# Uni-variate linear system
mutable struct UniVariateLSdynamic
    A::Matrix{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    varindex::UInt

    function UniVariateLSdynamic(unfixed, varlen)
        return new(zeros(Float64, varlen, varlen), zeros(Float64, varlen), Vector{Float64}(undef, varlen), UInt(unfixed))
    end

    function UniVariateLSdynamic(prev::UniVariateLSdynamic, unfixed, varlen)
        A = prev.A
        if varlen == length(prev.b)
            zero!(prev)
        else
            A = zeros(Float64, varlen, varlen)
            resize!(prev.b, varlen)
            fill!(prev.b, 0)
            resize!(prev.x, varlen)
        end
        return new(A, prev.b, prev.x, UInt(unfixed))
    end
end

mutable struct UniVariateLSstatic{N, N2}
    A::MMatrix{N, N, Float64, N2}
    b::MVector{N, Float64}
    x::MVector{N, Float64}
    varindex::UInt

    function UniVariateLSstatic{N, N2}(unfixed) where {N, N2}
        return new(zeros(MMatrix{N, N, Float64, N2}), zeros(MVector{N, Float64}), MVector{N, Float64}(undef), UInt(unfixed))
    end

    function UniVariateLSstatic{N, N2}(prev::UniVariateLSstatic{N, N2}, unfixed, ::Any) where {N, N2}
        zero!(prev)
        return new(prev.A, prev.b, prev.x, UInt(unfixed))
    end
end

UniVariateLS = Union{UniVariateLSdynamic, UniVariateLSstatic{N, N2}} where {N, N2}

function computestartindices(blocksizes)
    startind = Vector{Int}(undef, length(blocksizes))
    startind[1] = 1
    @inbounds startind[2:end] .= view(blocksizes, 1:length(blocksizes)-1)
    return cumsum!(startind)
end

# Multi-variate linear system
struct MultiVariateLSsparse
    A::BlockSparseMatrix{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{UInt} # One per block in b
    hessian::SparseMatrixCSC{Float64, Int} # Storage for sparse hessian
    sparseindices::Vector{Int}
    ldlfac::LDLFactorizations.LDLFactorization{Float64, Int, Int, Int}

    function MultiVariateLSsparse(A::BlockSparseMatrix, blockindices, storehessian=true)
        @assert A.rowblocksizes==A.columnblocksizes
        boffsets = computestartindices(A.rowblocksizes)
        blen = boffsets[end] + A.rowblocksizes[end] - 1
        if storehessian
            x = Vector{Float64}(undef, blen)
            sparseindices = makesparseindices(A, true)
            hessian = SparseMatrixCSC{Float64, Int}(sparseindices.m, sparseindices.n, sparseindices.colptr, sparseindices.rowval, Vector{Float64}(undef, length(sparseindices.nzval)))
            sparseindices = sparseindices.nzval
        else
            x = Vector{Float64}()
            sparseindices = Vector{Int}()
            hessian = spzeros(0, 0)
        end
        ldlfac = ldl_analyze(hessian)
        return new(A, zeros(Float64, blen), x, blockindices, boffsets, hessian, sparseindices, ldlfac)
    end
end

struct MultiVariateLSdense
    A::BlockDenseMatrix{Float64}
    b::Vector{Float64}
    x::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{UInt} # One per block in b

    function MultiVariateLSdense(blocksizes, blockindices)
        boffsets = computestartindices(blocksizes)
        blen = boffsets[end] + blocksizes[end]
        push!(boffsets, blen)
        blen -= 1
        return new(BlockDenseMatrix{Float64}(boffsets), zeros(Float64, blen), Vector{Float64}(undef, blen), blockindices, boffsets)
    end

    function MultiVariateLSdense(from::MultiVariateLSdense, fromblock::Integer)
        # Crop an existing linear system
        boffsets = view(from.A.rowblockoffsets, 1:fromblock)
        blen = boffsets[end] - 1
        return new(BlockDenseMatrix{Float64}(boffsets), zeros(Float64, blen), Vector{Float64}(undef, blen), blockindices, boffsets)
    end
end

MultiVariateLS = Union{MultiVariateLSsparse, MultiVariateLSdense}

function makesymmvls(problem, unfixed, nblocks, formarginalization=false)
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

    # Decide whether to have a sparse or a dense system
    len = formarginalization ? 40 : sum(blocksizes)
    if len >= 40
        # Compute the block sparsity
        sparsity = getvarcostmap(problem)
        sparsity = sparsity[unfixed,:]
        sparsity = triu(sparse(sparsity * sparsity' .> 0))

        # Check sparsity level
        if formarginalization || sparse_dense_decision(len, block_sparse_nnz(sparsity, blocksizes))
            # Construct the BSM
            bsm = BlockSparseMatrix{Float64}(sparsity, blocksizes, blocksizes)

            # Construct the sparse MultiVariateLS
            return MultiVariateLSsparse(bsm, blockindices, !formarginalization)
        end
    end

    # Construct the dense MultiVariateLS
    return MultiVariateLSdense(blocksizes, blockindices)
end

function updatesymlinearsystem!(linsystem::UniVariateLS, g, H, unusedargs...)
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
    updateb!(linsystem.b, g, vars, varflags, linsystem.boffsets, blockindices)
    updatesymA!(linsystem.A, H, vars, varflags, blockindices)
end

uniformscaling!(linsystem, k) = uniformscaling!(linsystem.A, k)


getgrad(linsystem) = linsystem.b
gethessian(linsystem::UniVariateLS) = linsystem.A
function gethessian(linsystem::MultiVariateLSsparse)
    # Fill sparse hessian
    @inbounds for (i, si) in enumerate(linsystem.sparseindices)
        linsystem.hessian.nzval[i] = linsystem.A.data[si]
    end
    return linsystem.hessian
end
gethessian(linsystem::MultiVariateLSdense) = symmetrifyfull(linsystem.A)
gethessgrad(linsystem) = gethessian(linsystem), linsystem.b

function zero!(linsystem)
    fill!(linsystem.b, 0)
    zero!(linsystem.A)
end

getoffsets(block, linsystem::MultiVariateLS) = @inbounds(linsystem.blockindices[varindices(block)])
function getoffsets(block, linsystem::UniVariateLS)
    varind = varindices(block)
    if isa(varind, Number)
        return SVector(UInt(varind == linsystem.varindex))
    end
    return convert.(UInt, varind .== linsystem.varindex)
end

function update!(to::Vector, from::Vector, linsystem::MultiVariateLS, step=linsystem.x)
    # Update each variable
    for (i, j) in enumerate(linsystem.blockindices)
        if j != 0
            @inbounds to[i] = update(from[i], step, linsystem.boffsets[j])
        end
    end
end

@inline function update!(to::Vector, from::Vector, linsystem::UniVariateLS, step=linsystem.x)
    # Update one variable
    @inbounds to[linsystem.varindex] = update(from[linsystem.varindex], step)
end
