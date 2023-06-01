using SparseArrays
export UniVariateLS, MultiVariateLS, makemvls, makesymmvls, gethessgrad, getjacres, zero!, uniformscaling!

function addvarvarpairs!(pairs, residuals::Vector, blockindices)
    for res in residuals
        addvarvarpairs!(pairs, res, blockindices)
    end
end

function addvarvarpairs!(pairs, residual, blockindices)
    blocks = blockindices[varindices(residual)]
    blocks = sort(blocks[blocks .!= 0], rev=true)
    @inbounds for (i, b) in enumerate(blocks)
        @inbounds for b_ in @view blocks[i+1:end]
            push!(pairs, SVector(b, b_))
        end
    end
end

function addresvarpairs!(pairs, resblocksizes, residuals::Vector, blockindices, ind)
    for res in residuals
        addresvarpairs!(pairs, resblocksizes, res, blockindices, ind)
        ind += 1
    end
end

function addresvarpairs!(pairs, resblocksizes, residual, blockindices, ind)
    blocks = blockindices[varindices(residual)]
    blocks = blocks[blocks .!= 0]
    @inbounds for b in blocks
        push!(pairs, SVector(ind, b))
    end
    resblocksizes[ind] = nres(residual)
end

# Uni-variate linear system
struct UniVariateLS
    A::Matrix{Float64}
    b::Vector{Float64}
    varindex::UInt

    function UniVariateLS(unfixed, varlen, reslen=varlen)
        return new(zeros(Float64, reslen, varlen), zeros(Float64, reslen), unfixed)
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
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{UInt} # One per block in b
    soloffsets::Vector{UInt} # One for each unfixed variable

    function MultiVariateLS(A::BlockSparseMatrix, blockindices)
        boffsets = computestartindices(A.rowblocksizes)
        blen = boffsets[end] + A.rowblocksizes[end] - 1
        soloffsets = A.rowblocksizes == A.columnblocksizes ? boffsets : computestartindices(A.columnblocksizes)
        return new(A, zeros(Float64, blen), blockindices, boffsets, soloffsets)
    end
end

function makemvls(vars, residuals, unfixed, nblocks)
    # Multiple variables. Use a block sparse matrix
    blockindices = zeros(UInt, length(vars))
    varblocksizes = zeros(UInt, nblocks)
    resblocksizes = zeros(UInt, numresiduals(residuals))
    pairs = Vector{SVector{2, Int}}()
    nblocks = 0
    # Get the variable block sizes
    for (index, unfixed_) in enumerate(unfixed)
        if unfixed_
            nblocks += 1
            blockindices[index] = nblocks
            N = nvars(vars[index])
            varblocksizes[nblocks] = N
        end
    end

    # Compute the residual-variable pairs
    ind = 1
    @inbounds for res in values(residuals)
        addresvarpairs!(pairs, resblocksizes, res, blockindices, ind)
        ind += length(res)
    end

    # Construct the MultiVariateLS
    return MultiVariateLS(BlockSparseMatrix{Float64}(pairs, resblocksizes, varblocksizes), blockindices)
end

function makesymmvls(vars, residuals, unfixed, nblocks)
    # Multiple variables. Use a block sparse matrix
    blockindices = zeros(UInt, length(vars))
    blocksizes = zeros(UInt, nblocks)
    nblocks = 0
    pairs = Vector{SVector{2, Int}}()
    for (index, unfixed_) in enumerate(unfixed)
        if unfixed_
            nblocks += 1
            blockindices[index] = nblocks
            N = nvars(vars[index])
            blocksizes[nblocks] = N
            push!(pairs, SVector(nblocks, nblocks))
        end
    end

    # Compute the off-diagonal pairs
    @inbounds for res in values(residuals)
        addvarvarpairs!(pairs, res, blockindices)
    end

    # Construct the MultiVariateLS
    return MultiVariateLS(BlockSparseMatrix{Float64}(pairs, blocksizes, blocksizes), blockindices)
end

function updatesymlinearsystem!(linsystem::UniVariateLS, g, H, unusedargs...)
    # Update the blocks in the problem
    linsystem.b .+= g
    linsystem.A .+= H
end

function updatelinearsystem!(linsystem::UniVariateLS, res, jac, ind, unusedargs...)
    # Update the blocks in the problem
    view(linsystem.b, SR(1, Size(res)[1]).+(ind-1)) .= res
    view(linsystem.A, SR(1, Size(res)[1]).+(ind-1), :) .= jac
end

function updatesymA!(A, a, vars, ::Val{varflags}, blockindices, loffsets) where varflags
    # Update the blocks in the problem
    @unroll for i in 1:MAX_ARGS
        if ((varflags >> (i - 1)) & 1) == 1
            @unroll for j in i:MAX_ARGS
                if ((varflags >> (j - 1)) & 1) == 1
                    if blockindices[i] >= blockindices[j] # Make sure the BSM is lower triangular
                        block(A, blockindices[i], blockindices[j], Val(nvars(vars[i])), Val(nvars(vars[j]))) .+= @inbounds view(a, loffsets[i], loffsets[j])
                    else
                        block(A, blockindices[j], blockindices[i], Val(nvars(vars[j])), Val(nvars(vars[i]))) .+= @inbounds view(a, loffsets[j], loffsets[i])
                    end
                end
            end
        end
    end
end

@inline function blockoffsets(vars, varflags, boffsets, blockindices)
    return ntuple(i -> SR(1, nvars(vars[i]) * ((varflags >> (i - 1)) & 1)) .+ (boffsets[blockindices[i]] - 1), length(vars))
end

@inline function localoffsets(vars, varflags)
    return ntuple(i -> SR(1, nvars(vars[i]) * ((varflags >> (i - 1)) & 1)) .+ countvars(vars[1:i-1], varflags), length(vars))
end

function updateb!(B, b, vars, ::Val{varflags}, boffsets, blockindices, loffsets) where varflags
    # Update the blocks in the problem
    goffsets = blockoffsets(vars, varflags, boffsets, blockindices)
    @unroll for i in 1:MAX_ARGS
        if ((varflags >> (i - 1)) & 1) == 1
            @inbounds view(B, goffsets[i]) .+= view(b, loffsets[i])
        end
    end
end

function updatesymlinearsystem!(linsystem::MultiVariateLS, g, H, vars, ::Val{varflags}, blockindices) where varflags
    loffsets = localoffsets(vars, varflags)
    updateb!(linsystem.b, g, vars, Val(varflags), linsystem.boffsets, blockindices, loffsets)
    updatesymA!(linsystem.A, H, vars, Val(varflags), blockindices, loffsets)
end

function updateA!(A, a, ::Val{varflags}, blockindices, loffsets, ind) where varflags
    # Update the blocks in the problem
    rows = A.indicestransposed.colptr[ind]:A.indicestransposed.colptr[ind+1]-1
    @inbounds dataptr = view(A.indicestransposed.nzval, rows)
    @inbounds rows = view(A.indicestransposed.rowval, rows)
    @unroll for i in 1:MAX_ARGS
        if ((varflags >> (i - 1)) & 1) == 1
            @inbounds view(A.data, SR(0, Size(a)[1]*Size(loffsets[i])[1]-1) .+ dataptr[findfirst(isequal(blockindices[i]), rows)]) .= reshape(view(a, :, loffsets[i]), :)
        end
    end
end

function updatelinearsystem!(linsystem::MultiVariateLS, res, jac, ind, vars, ::Val{varflags}, blockindices) where varflags
    view(linsystem.b, SR(0, Size(res)[1]-1) .+ linsystem.boffsets[ind]) .= res
    updateA!(linsystem.A, jac, Val(varflags), blockindices, localoffsets(vars, varflags), ind)
end

function uniformscaling!(linsystem, k)
    uniformscaling!(linsystem.A, k)
end

function gethessgrad(linsystem::UniVariateLS)
    return linsystem.A, linsystem.b
end

function gethessgrad(linsystem::MultiVariateLS)
    if size(linsystem.A, 2) > 1000 && 3 * nnz(linsystem.A) < length(linsystem.A)
        return symmetrifysparse(linsystem.A), linsystem.b
    end
    return symmetrifyfull(linsystem.A), linsystem.b
end

function getjacres(linsystem::UniVariateLS)
    return linsystem.A, linsystem.b
end

function getjacres(linsystem::MultiVariateLS)
    if size(linsystem.A, 2) > 1000 && 3 * nnz(linsystem.A) < length(linsystem.A)
        return SparseArrays.sparse(linsystem.A), linsystem.b
    end
    return Matrix(linsystem.A), linsystem.b
end

function zero!(linsystem)
    fill!(linsystem.b, 0)
    zero!(linsystem.A)
end