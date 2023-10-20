using StaticArrays, HybridArrays, Static

function marginalize!(to::MultiVariateLS, from::MultiVariateLSsparse, blockind::Int, blocksz::Int)
    # Get the list of blocks to marginalize out
    ind = from.A.indicestransposed.colptr[blockind]:from.A.indicestransposed.colptr[blockind+1]-1
    blocks = view(from.A.indicestransposed.rowval, ind)
    @assert blocks[end] == blockind && from.A.rowblocksizes[blockind] == blocksz
    @assert !isa(to, MultiVariateLSsparse) || all(view(blocks, 1:lastindex(blocks)-1) .<= length(to.A.rowblocksizes))
    @assert !isa(to, MultiVariateLSdense) || all(view(blocks, 1:lastindex(blocks)-1) .<= (length(to.A.rowblockoffsets) - 1))
    N = length(blocks) - 1
    dataindices = view(from.A.indicestransposed.nzval, ind)
    # Get the diagonal block (to be marginalized)
    diagblock = reshape(view(from.A.data, (0:blocksz*blocksz-1) .+ dataindices[end]), blocksz, blocksz)
    @static if VERSION ≥ v"1.9"
        diagblock = bunchkaufman(diagblock)
    end
    # For each non-marginalized block
    blockgrad = view(from.b, (0:blocksz-1) .+ from.boffsets[blockind])
    for a in 1:N
        # Multiply inverse by first block
        blocka = blocks[a]
        lena = Int(from.A.rowblocksizes[blocka])
        S = (diagblock \ reshape(view(from.A.data, (0:blocksz*lena-1) .+ dataindices[a]), blocksz, lena))'
        # Update gradient
        view(to.b, (0:lena-1) .+ to.boffsets[blocka]) .-= S * blockgrad
        # Update Hessian blocks
        for b in 1:a
            blockb = blocks[b]
            lenb = Int(from.A.rowblocksizes[blockb])
            B = reshape(view(from.A.data, (0:blocksz*lenb-1) .+ dataindices[b]), blocksz, lenb)
            block(to.A, blocka, blockb, lena, lenb) .-= S * B
        end
    end
end

function marginalize!(to::MultiVariateLS, from::MultiVariateLSsparse, blockind::Int, ::StaticInt{blocksz}) where blocksz
    # Get the list of blocks to marginalize out
    ind = from.A.indicestransposed.colptr[blockind]:from.A.indicestransposed.colptr[blockind+1]-1
    blocks = view(from.A.indicestransposed.rowval, ind)
    @assert blocks[end] == blockind && from.A.rowblocksizes[blockind] == blocksz
    @assert !isa(to, MultiVariateLSsparse) || all(view(blocks, 1:lastindex(blocks)-1) .<= length(to.A.rowblocksizes))
    @assert !isa(to, MultiVariateLSdense) || all(view(blocks, 1:lastindex(blocks)-1) .<= (length(to.A.rowblockoffsets) - 1))
    N = length(blocks) - 1
    dataindices = view(from.A.indicestransposed.nzval, ind)
    # Get the diagonal block (to be marginalized)
    diagblock = SMatrix{blocksz, blocksz}(SizedMatrix{blocksz, blocksz}(view(from.A.data, SR(0, blocksz*blocksz-1).+dataindices[end])))
    @static if VERSION ≥ v"1.9"
        diagblock = bunchkaufman(diagblock)
    end
    # For each non-marginalized block
    blockgrad = SizedVector{blocksz}(view(from.b, SR(0, blocksz-1) .+ from.boffsets[blockind]))
    for a in 1:N
        # Multiply inverse by first block
        blocka = blocks[a]
        lena = Int(from.A.rowblocksizes[blocka])
        S = (diagblock \ HybridArray{Tuple{blocksz, StaticArrays.Dynamic()}}(reshape(view(from.A.data, (0:blocksz*lena-1) .+ dataindices[a]), blocksz, lena)))'
        # Update gradient
        view(to.b, SR(0, lena-1) .+ to.boffsets[blocka]) .-= S * blockgrad
        # Update Hessian blocks
        for b in 1:a
            blockb = blocks[b]
            lenb = Int(from.A.rowblocksizes[blockb])
            block(to.A, blocka, blockb, lena, lenb) .-= S * HybridArray{Tuple{blocksz, StaticArrays.Dynamic()}}(reshape(view(from.A.data, (0:blocksz*lenb-1) .+ dataindices[b]), blocksz, lenb))
        end
    end
end

function marginalize!(to::MultiVariateLS, from::MultiVariateLSsparse, blocks::AbstractRange, blocksz)
    for block in blocks
        marginalize!(to, from, block, blocksz)
    end
end

function marginalize!(to::MultiVariateLS, from::MultiVariateLSsparse, fromblock = isa(to, MultiVariateLSsparse) ? length(to.A.rowblocksizes)+1 : length(to.A.rowblockoffsets))
    last = fromblock
    finish = length(from.A.rowblocksizes)
    while last <= finish
        first = last
        blocksz = Int(from.A.rowblocksizes[last])
        while true
            last += 1
            if last > finish || blocksz != Int(from.A.rowblocksizes[last])
                break
            end
        end
        range = first:last-1
        if blocksz <= MAX_BLOCK_SZ
            # marginalize!(to, from, first:last, static(blocksz))
            valuedispatch(static(1), static(MAX_BLOCK_SZ), blocksz, fixallbutlast(marginalize!, to, from, range))
        else
            marginalize!(to, from, range, blocksz)
        end
    end
end

function initcrop!(to::MultiVariateLSsparse, from::MultiVariateLSsparse, fromblock=length(to.A.rowblocksizes)+1)
    # Reset the linear system to all zeros
    zero!(to)
    # Copy over all the cropped bits
    to.b .= view(from.b, 1:lastindex(to.b))
    endind = from.A.indicestransposed.nzval[from.A.indicestransposed.colptr[fromblock]] - 1
    view(to.A.data, 1:endind) .= view(from.A.data, 1:endind)
end

function initcrop!(to::MultiVariateLSdense, from::MultiVariateLSsparse, fromblock=length(to.A.rowblockoffsets))
    # Reset the linear system to all zeros
    zero!(to)
    # Copy over all the cropped bits
    @inbounds to.b .= view(from.b, 1:lastindex(to.b))
    @inbounds for row = 1:fromblock-1
        lenr = Int(from.A.rowblocksizes[row])
        for colind = from.A.indicestransposed.colptr[row]:from.A.indicestransposed.colptr[row+1]-1
            col = from.A.indicestransposed.rowval[colind]
            lenc = Int(from.A.columnblocksizes[col])
            block(to.A, row, col, lenr, lenc) .= reshape(view(from.A.data, (0:lenr*lenc-1) .+ from.A.indicestransposed.nzval[colind]), lenr, lenc)
        end
    end 
end

function constructcrop(from::MultiVariateLSsparse, fromblock, forcesparse=false)
    toblocksizes = view(from.A.rowblocksizes, 1:fromblock-1)

    # Decide whether to have a sparse or a dense system
    len = sum(toblocksizes)
    @inbounds if forcesparse || len >= 40
        # Compute the crop sparsity
        cacheindices(from.A)
        toblock = fromblock - 1
        cropsparsity = view(from.A.indices + from.A.indicestransposed, :, 1:toblock) # Do view before sum when this issue is fixed: https://github.com/JuliaSparse/SparseArrays.jl/issues/441
        cropsparsity = triu(cropsparsity' * cropsparsity)

        # Check sparsity level
        if forcesparse || sparse_dense_decision(len, block_sparse_nnz(cropsparsity, toblocksizes))
            # Add any missing blocks to the cropped region
            start = from.A.indicestransposed.nzval[from.A.indicestransposed.colptr[fromblock]]
            blocksizes = convert.(Int, @inbounds view(from.A.rowblocksizes, 1:toblock))
            for c = 1:toblock
                nc = blocksizes[c]
                for rind = cropsparsity.colptr[c]:cropsparsity.colptr[c+1]-1
                    r = cropsparsity.rowval[rind]
                    v = from.A.indicestransposed[r,c] # Improve this line
                    if v == 0
                        # We need to add a block
                        v = start
                        start += nc * blocksizes[r]
                    end
                    cropsparsity.nzval[rind] = v
                end
            end
            A = BlockSparseMatrix{Float64}(start-1, cropsparsity, blocksizes, blocksizes)

            # Construct the sparse linear system
            return MultiVariateLSsparse(A, from.blockindices[1:findfirst(isequal(fromblock), from.blockindices)])
        end
    end

    # Construct a dense linear system
    return MultiVariateLSdense(toblocksizes, from.blockindices)
end
