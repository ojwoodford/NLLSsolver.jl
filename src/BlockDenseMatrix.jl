struct BlockDenseMatrix{T}
    data::Matrix{T}
    rowblockoffsets::Vector{Int}
    columnblockoffsets::Vector{Int}

    function BlockDenseMatrix{T}(rowblockoffsets, columnblockoffsets=rowblockoffsets) where T
        return new{T}(zeros(T, rowblockoffsets[end]-1, columnblockoffsets[end]-1), rowblockoffsets, columnblockoffsets)
    end
end

uniformscaling!(bdm::BlockDenseMatrix, k) = uniformscaling!(bdm.data, k)

zero!(bdm::BlockDenseMatrix) = fill!(bdm.data, 0)
@inline block(bdm::BlockDenseMatrix, i, j, ::StaticInt{rows}, ::StaticInt{cols}) where {rows, cols} = @inbounds SizedMatrix{rows, cols}(view(bdm.data, SR(0, rows-1).+bdm.rowblockoffsets[i], SR(0, cols-1).+bdm.columnblockoffsets[j]))
@inline block(bdm::BlockDenseMatrix, i, j, rows::Integer, cols::Integer) = @inbounds view(bdm.data, SR(0, rows-1).+bdm.rowblockoffsets[i], SR(0, cols-1).+bdm.columnblockoffsets[j])
@inline block(bdm::BlockDenseMatrix, i, j, ::StaticInt{rows}, cols::Integer) where rows = HybridArray{Tuple{rows, StaticArrays.Dynamic()}}(block(bdm, i, j, rows, cols))
@inline block(bdm::BlockDenseMatrix, i, j) = @inbounds block(bdm, i, j, bdm.rowblockoffsets[i+1]-bdm.rowblockoffsets[i], bdm.columnblockoffsets[j+1]-bdm.columnblockoffsets[j])
@inline validblock(::BlockDenseMatrix, ::Integer, ::Integer) = true
Base.size(bdm::BlockDenseMatrix) = size(bdm.data)
Base.size(bdm::BlockDenseMatrix, dim) = size(bdm.data, dim)
Base.length(bdm::BlockDenseMatrix) = length(bdm.data)
Base.eltype(::BlockDenseMatrix{T}) where T = T

function symmetrifyfull(bdm::BlockDenseMatrix)
    @assert bdm.rowblockoffsets == bdm.columnblockoffsets
    # Symmetrify in-place: add the upper part
    for r in 2:size(bdm.data, 1)
        for c in 1:r-1
            @inbounds bdm.data[c,r] = bdm.data[r,c]
        end
    end
    # Return the matrix
    return bdm.data
end
