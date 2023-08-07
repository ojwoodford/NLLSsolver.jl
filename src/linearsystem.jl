using SparseArrays, Static

function addvarvarpairs!(pairs, costs::Vector, blockindices)
    for cost in costs
        if ndeps(cost) > 1
            addvarvarpairs!(pairs, cost, blockindices)
        end
    end
end

function addvarvarpairs!(pairs, cost, blockindices)
    blocks = blockindices[varindices(cost)]
    blocks = blocks[blocks .!= 0]
    if length(blocks) <= 1
        return
    end
    blocks = sort!(blocks, rev=true)
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
    resblocksizes = zeros(UInt, countcosts(resnum, residuals))
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

function makesymmvls(vars, costs, unfixed, nblocks)
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
    @inbounds for cost in values(costs)
        addvarvarpairs!(pairs, cost, blockindices)
    end

    # Construct the MultiVariateLS
    return MultiVariateLS(BlockSparseMatrix{Float64}(pairs, blocksizes, blocksizes), blockindices)
end

function updatesymlinearsystem!(linsystem::UniVariateLS, g, H, unusedargs...)
    # Update the blocks in the problem
    linsystem.b .+= g
    if !isnothing(H)
        linsystem.A .+= H
    end
end

function updatelinearsystem!(linsystem::UniVariateLS, res, jac, ind, unusedargs...)
    # Update the blocks in the problem
    view(linsystem.b, SR(1, length(res)).+(ind-1)) .= res
    view(linsystem.A, SR(1, length(res)).+(ind-1), :) .= jac
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

function updatesymlinearsystem!(linsystem::MultiVariateLS, g, H, vars, varflags, blockindices)
    updateb!(linsystem.b, g, vars, varflags, linsystem.boffsets, blockindices)
    if !isnothing(H)
        updatesymA!(linsystem.A, H, vars, varflags, blockindices)
    end
end

function updatelinearsystem!(linsystem::MultiVariateLS, res, jac, ind, vars, varflags, blockindices)
    nres = length(res)
    view(linsystem.b, SR(0, nres-1) .+ linsystem.boffsets[ind]) .= res
    # Update the blocks in the problem A matrix
    rows = linsystem.A.indicestransposed.colptr[ind]:linsystem.A.indicestransposed.colptr[ind+1]-1
    dataptr = @inbounds view(linsystem.A.indicestransposed.nzval, rows)
    rows = @inbounds view(linsystem.A.indicestransposed.rowval, rows)
    loffset = static(0)
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && bitiset(varflags, i)
            nv = nvars(vars[i])
            bi = blockindices[i]
            view(linsystem.A.data, SR(0, nres*nv-1) .+ dataptr[findfirst(Base.Fix1(isequal, bi), rows)]) .= reshape(view(jac, :, SR(1, nv).+loffset), :)
            loffset += nv
        end
    end
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