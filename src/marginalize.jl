using LinearSolve, StaticArrays, HybridArrays
export marginalize!, initcrop!, constructcrop

function marginalize!(to::MultiVariateLS, from::MultiVariateLS, block::Integer, ::Val{blocksz}) where blocksz
    # Get the list of blocks to marginalize out
    ind = from.hessian.indicestransposed.colptr[block]:from.hessian.indicestransposed.colptr[block+1]-1
    blocks = from.hessian.indicestransposed.rowval[ind]
    @assert blocks[end] == block && all(view(blocks, 1:length(blocks)-1) .<= length(to.hessian.rowblocksizes))
    N = length(blocks) - 1
    dataindices = from.hessian.indicestransposed.nzval[ind]
    blocksizes = from.hessian.rowblocksizes[blocks]
    @assert blocksizes[end] == blocksz
    # Compute inverse for the diagonal block (to be marginalized)
    inverseblock = inv(SizedMatrix{blocksz, blocksz}(view(from.hessian.data, SR(0, blocksz*blocksz-1).+dataindices[end])))
    # For each non-marginalized block
    blockgrad = SizedVector{blocksz}(view(from.gradient, SR(0, blocksz-1) .+ from.gradoffsets[block]))
    for a in 1:N
        # Multiply inverse by first block
        lena = blocksizes[a]
        blocka = blocks[a]
        S = HybridArray{Tuple{blocksz, StaticArrays.Dynamic()}}(reshape(view(from.hessian.data, (0:blocksz*lena-1) .+ dataindices[a]), blocksz, lena))' * inverseblock
        # Update gradient
        view(to.gradient, SR(0, lena-1) .+ to.gradoffsets[blocka]) .-= S * blockgrad
        # Update Hessian blocks
        for b in 1:a
            lenb = blocksizes[b]
            B = HybridArray{Tuple{blocksz, StaticArrays.Dynamic()}}(reshape(view(from.hessian.data, (0:blocksz*lenb-1) .+ dataindices[b]), blocksz, lenb))
            reshape(view(to.hessian.data, (0:lena*lenb-1) .+ to.hessian.indicestransposed[blocks[b],blocka]), lena, lenb) .-= S * B
        end
    end
end

function marginalize!(to::MultiVariateLS, from::MultiVariateLS, blocks::AbstractRange, ::Val{blocksz}) where blocksz
    for block in blocks
        marginalize!(to, from, block, Val(blocksz))
    end
end

function marginalize!(to::MultiVariateLS, from::MultiVariateLS, fromblock=length(to.hessian.rowblocksizes)+1)
    for block in fromblock:length(from.hessian.rowblocksizes)
        #marginalize!(to, from, block, Val(from.hessian.rowblocksizes[block]))
        valuedispatch(Val(1), Val(32), v -> marginalize!(to, from, block, v), from.hessian.rowblocksizes[block])
    end
end

function initcrop!(to::MultiVariateLS, from::MultiVariateLS, fromblock=length(to.hessian.rowblocksizes)+1)
    # Reset the linear system to all zeros
    zero!(to)
    # Copy over all the cropped bits
    to.gradient .= view(from.gradient, SR(1, length(to.gradient)))
    endind = from.hessian.indices.nzval[from.hessian.indices.colptr[fromblock]] - 1
    view(to.hessian.data, 1:endind) .= view(from.hessian.data, 1:endind)
end

function hasoverlap(A, B) # Assumed to be two sorted lists
    a = 1
    b = 1
    @inbounds while a <= length(A) && b <= length(B) && A[a] != B[b]
        tf = A[a] < B[b]
        a += tf
        b += !tf
    end
    return a <= length(A) && b <= length(B)
end

function constructcrop(from::MultiVariateLS, fromblock)
    # Create a dense map of the reduced area
    toblock = fromblock - 1
    indices = Matrix(view(from.hessian.indicestransposed, 1:toblock, 1:toblock))
    start = from.hessian.indices.nzval[from.hessian.indices.colptr[fromblock]]
    blocksizes = convert.(Int, from.hessian.rowblocksizes[1:toblock])
    for c = 1:toblock
        cindices = view(from.hessian.indices.rowval, from.hessian.indices.colptr[c]:from.hessian.indices.colptr[c+1]-1)
        nc = blocksizes[c]
        for r = 1:c
            if indices[r,c] != 0
                continue
            end
            if !hasoverlap(cindices, view(from.hessian.indices.rowval, from.hessian.indices.colptr[r]:from.hessian.indices.colptr[r+1]-1))
                continue
            end
            # We need to add a block
            indices[r,c] = start
            start += nc * blocksizes[r]
        end
    end
    hessian = BlockSparseMatrix{Float64}(start-1, indices, blocksizes, blocksizes)
    return MultiVariateLS(hessian, from.blockindices)
end

