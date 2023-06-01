using StaticArrays, HybridArrays
export marginalize!, initcrop!, constructcrop

function marginalize!(to::MultiVariateLS, from::MultiVariateLS, block::Integer, ::Val{blocksz}) where blocksz
    # Get the list of blocks to marginalize out
    ind = from.A.indicestransposed.colptr[block]:from.A.indicestransposed.colptr[block+1]-1
    blocks = from.A.indicestransposed.rowval[ind]
    @assert blocks[end] == block && all(view(blocks, 1:lastindex(blocks)-1) .<= length(to.A.rowblocksizes))
    N = length(blocks) - 1
    dataindices = from.A.indicestransposed.nzval[ind]
    blocksizes = from.A.rowblocksizes[blocks]
    @assert blocksizes[end] == blocksz
    # Compute inverse for the diagonal block (to be marginalized)
    inverseblock = inv(SizedMatrix{blocksz, blocksz}(view(from.A.data, SR(0, blocksz*blocksz-1).+dataindices[end])))
    # For each non-marginalized block
    blockgrad = SizedVector{blocksz}(view(from.b, SR(0, blocksz-1) .+ from.boffsets[block]))
    for a in 1:N
        # Multiply inverse by first block
        lena = blocksizes[a]
        blocka = blocks[a]
        S = HybridArray{Tuple{blocksz, StaticArrays.Dynamic()}}(reshape(view(from.A.data, (0:blocksz*lena-1) .+ dataindices[a]), blocksz, lena))' * inverseblock
        # Update gradient
        view(to.b, SR(0, lena-1) .+ to.boffsets[blocka]) .-= S * blockgrad
        # Update Hessian blocks
        for b in 1:a
            lenb = blocksizes[b]
            B = HybridArray{Tuple{blocksz, StaticArrays.Dynamic()}}(reshape(view(from.A.data, (0:blocksz*lenb-1) .+ dataindices[b]), blocksz, lenb))
            reshape(view(to.A.data, (0:lena*lenb-1) .+ to.A.indicestransposed[blocks[b],blocka]), lena, lenb) .-= S * B
        end
    end
end

function marginalize!(to::MultiVariateLS, from::MultiVariateLS, blocks::AbstractRange, ::Val{blocksz}) where blocksz
    for block in blocks
        marginalize!(to, from, block, Val(blocksz))
    end
end

function marginalize!(to::MultiVariateLS, from::MultiVariateLS, fromblock=length(to.A.rowblocksizes)+1)
    for block in fromblock:length(from.A.rowblocksizes)
        # marginalize!(to, from, block, Val(Int(from.A.rowblocksizes[block])))
        valuedispatch(Val(1), Val(32), Int(from.A.rowblocksizes[block]), marginalize!, (to, from, block, v))
    end
end

function initcrop!(to::MultiVariateLS, from::MultiVariateLS, fromblock=length(to.A.rowblocksizes)+1)
    # Reset the linear system to all zeros
    zero!(to)
    # Copy over all the cropped bits
    to.b .= view(from.b, 1:lastindex(to.b))
    cacheindices(from.A)
    endind = from.A.indices.nzval[from.A.indices.colptr[fromblock]] - 1
    view(to.A.data, 1:endind) .= view(from.A.data, 1:endind)
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
    cacheindices(from.A)
    indices = Matrix(view(from.A.indices, 1:toblock, 1:toblock))
    start = from.A.indices.nzval[from.A.indices.colptr[fromblock]]
    blocksizes = convert.(Int, from.A.rowblocksizes[1:toblock])
    for c = 1:toblock
        cindices = view(from.A.indices.rowval, from.A.indices.colptr[c]:from.A.indices.colptr[c+1]-1)
        nc = blocksizes[c]
        for r = c:toblock
            if indices[r,c] != 0
                continue
            end
            if !hasoverlap(cindices, view(from.A.indices.rowval, from.A.indices.colptr[r]:from.A.indices.colptr[r+1]-1))
                continue
            end
            # We need to add a block
            indices[r,c] = start
            start += nc * blocksizes[r]
        end
    end
    A = BlockSparseMatrix{Float64}(start-1, indices', blocksizes, blocksizes)
    return MultiVariateLS(A, from.blockindices)
end

