using StaticArrays
import SparseArrays
export BlockSparseMatrix, symmetrifysparse, symmetrifyfull, block, crop

struct BlockSparseMatrix{T}
    data::Vector{T}
    indices::SparseArrays.SparseMatrixCSC{UInt, UInt}
    rowblocksizes::Vector{UInt8}
    columnblocksizes::Vector{UInt8}
    m::UInt
    n::UInt

    function BlockSparseMatrix{T}(pairs, rowblocksizes, colblocksizes) where T
        # Check the block sizes
        rbs = convert.(UInt8, rowblocksizes)
        cbs = convert.(UInt8, colblocksizes)
        @assert all(i -> 0 < i, rbs)
        @assert all(i -> 0 < i, cbs)
        m = sum(rbs)
        n = sum(cbs)
        if size(pairs, 1) > 1 && size(pairs, 2) == 2
            # Sort the indices for faster construction, and ensure unique
            pairs_ = unique(sortslices(pairs, dims=1), dims=1)
            # Compute the block pointers and length of data storage
            start = UInt(1)
            indices = Vector{UInt}(undef, size(pairs_, 1))
            for ind in eachindex(indices)
                indices[ind] = start
                start += convert(UInt, rbs[pairs_[ind,1]]) * cbs[pairs_[ind,2]]
            end
            # Construct the block matrix
            nzvals = start - 1
            sp = SparseArrays.sparse(view(pairs_, :, 1), view(pairs_, :, 2), indices, length(rbs), length(cbs))
        elseif length(pairs) == 2
            # Single pair
            nzvals = convert(UInt, rbs[pairs[1]]) * cbs[pairs[2]]
            sp = SparseArrays.sparse([pairs[1]], [pairs[2]], [UInt(1)], length(rbs), length(cbs))
        else
            ArgumentError("Unexpected pairs")
        end
        return new(zeros(T, nzvals), sp, rbs, cbs, m, n)
    end
end

@inline zero!(bsm::BlockSparseMatrix) = fill!(bsm.data, 0)
@inline block(bsm::BlockSparseMatrix, i, j, rows, cols) = SizedMatrix{rows, cols}(view(bsm.data, StaticArrays.SUnitRange(0, rows*cols-1).+bsm.indices[i,j]))
function block(bsm::BlockSparseMatrix, i, j)
    rows = bsm.rowblocksizes[i]
    cols = bsm.columnblocksizes[j]
    index = bsm.indices[i,j]
    return reshape(view(bsm.data, index:index+rows*cols-1), (rows, cols))
end
Base.size(bsm::BlockSparseMatrix) = (bsm.m, bsm.n)
Base.size(bsm::BlockSparseMatrix, dim) = dim == 1 ? bsm.m : (dim == 2 ? bsm.n : 1)
SparseArrays.nnz(bsm::BlockSparseMatrix) = length(bsm.data)

@inline zero!(aa::AbstractArray) = fill!(aa, 0)
@inline block(aa::AbstractArray, i, j, rows, cols) = SizedMatrix{rows, cols}(aa)
@inline block(aa::AbstractArray, i, j) = aa
SparseArrays.nnz(aa::AbstractArray) = length(aa)

function symmetrifysparse(bsm::BlockSparseMatrix{T}) where T
    # Preallocate arrays
    @assert bsm.rowblocksizes == bsm.columnblocksizes
    nzvals = length(bsm.data) * 2 - sum((Vector(diag(bsm.indices)) .!= 0) .* (bsm.rowblocksizes .^ 2))
    rows = Vector{UInt}(undef, nzvals)
    cols = Vector{UInt}(undef, nzvals)
    values = Vector{T}(undef, nzvals)
    # Sparsify each block column
    start = cumsum(bsm.rowblocksizes) .+ 1
    pushfirst!(start, UInt(1))
    ind = 0
    for col in range(1, length(bsm.rowblocksizes))
        sprow = bsm.indices[col,:]
        block_rows, block_indices = SparseArrays.findnz(max(bsm.indices[:,col], sprow))
        transposed = findall(block_indices .== sprow[block_rows])
        colstride = ones(UInt, length(block_indices))
        rowstride = bsm.rowblocksizes[block_rows]
        colstride[transposed] .= rowstride[transposed]
        rowstride[transposed] .= 1
        col_ = start[col]
        for innercol in range(1, bsm.rowblocksizes[col])
            for (r, br) in enumerate(block_rows)
                s = start[br]
                c = bsm.rowblocksizes[br]
                v = block_indices[r]
                while c > 0
                    ind += 1
                    rows[ind] = s
                    s += 1
                    cols[ind] = col_
                    values[ind] = bsm.data[v]
                    v += rowstride[r]
                    c -= 1
                end
            end
            block_indices .+= colstride
            col_ += 1
        end
    end
    # Construct the sparse matrix
    m = start[end] - 1
    return SparseArrays.sparse(rows, cols, values, m, m)
end

function SparseArrays.sparse(bsm::BlockSparseMatrix{T}) where T
    # Preallocate arrays
    nzvals = length(bsm.data)
    rows = Vector{UInt}(undef, nzvals)
    cols = Vector{UInt}(undef, nzvals)
    values = Vector{T}(undef, nzvals)
    # Sparsify each block column
    start = cumsum(bsm.rowblocksizes) .+ 1
    pushfirst!(start, UInt(1))
    ind = 0
    for col in range(1, length(bsm.rowblocksizes))
        block_rows, block_indices = SparseArrays.findnz(bsm.indices[:,col])
        col_ = start[col]
        for innercol in range(1, bsm.rowblocksizes[col])
            for (r, br) in enumerate(block_rows)
                s = start[br]
                c = bsm.rowblocksizes[br]
                while c > 0
                    ind += 1
                    rows[ind] = s
                    s += 1
                    cols[ind] = col_
                    values[ind] = bsm.data[block_indices[r]]
                    block_indices[r] += 1
                    c -= 1
                end
            end
            col_ += 1
        end
    end
    # Construct the sparse matrix
    m = start[end] - 1
    return SparseArrays.sparse(rows, cols, values, m, m)
end

function symmetrifyfull(bsm::BlockSparseMatrix{T}) where T
    @assert bsm.rowblocksizes == bsm.columnblocksizes
    # Allocate the output
    starts = cumsum(bsm.rowblocksizes)
    pushfirst!(starts, UInt(0))
    output = zeros(T, starts[end], starts[end])
    # Copy blocks into the output
    for (r, c, index) in zip(SparseArrays.findnz(bsm.indices)...)
        rs = starts[r]
        rf = starts[r+1]
        cs = starts[c]
        cf = starts[c+1]
        r_ = rf - rs
        c_ = cf - cs
        V = reshape(view(bsm.data, index:index+r_*c_-1), (r_, c_))
        output[rs+1:rf,cs+1:cf] .= V
        if r != c
            output[cs+1:cf,rs+1:rf] .= V'
        end
    end
    # Return the matrix
    return output
end

function symmetrifyfull!(mat::Matrix{T}, bsm::BlockSparseMatrix{T}, blocks) where T
    @assert bsm.rowblocksizes == bsm.columnblocksizes
    lengths = zeros(UInt, length(bsm.rowblocksizes))
    lengths[blocks] .= bsm.rowblocksizes[blocks]
    m = sum(lengths)
    @assert size(mat) == (m, m)
    starts = zeros(UInt, length(bsm.rowblocksizes))
    starts[blocks] .= cumsum(vcat(1, bsm.rowblocksizes[blocks[1:end-1]]))
    # Copy blocks into the output
    fill!(mat, 0)
    for (r, c, index) in zip(SparseArrays.findnz(bsm.indices)...)
        cs = starts[c]
        if cs == 0
            continue
        end
        rs = starts[r]
        if rs == 0
            continue
        end
        r_ = lengths[r]
        c_ = lengths[c]
        V = reshape(view(bsm.data, index:index+(r_*c_)-1), (r_, c_))
        mat[rs:rs+r_-1,cs:cs+c_-1] .= V
        if r != c
            mat[cs:cs+c_-1,rs:rs+r_-1] .= V'
        end
    end
    return nothing
end

function Base.Matrix(bsm::BlockSparseMatrix{T}) where T
    # Allocate the output
    rowstarts = cumsum(bsm.rowblocksizes)
    pushfirst!(rowstarts, UInt(0))
    colstarts = cumsum(bsm.columnblocksizes)
    pushfirst!(colstarts, UInt(0))
    output = zeros(T, rowstarts[end], colstarts[end])
    # Copy blocks into the output
    for (r, c, index) in zip(SparseArrays.findnz(bsm.indices)...)
        rs = rowstarts[r]
        rf = rowstarts[r+1]
        cs = colstarts[c]
        cf = colstarts[c+1]
        r_ = rf - rs
        c_ = cf - cs
        output[rs+1:rf,cs+1:cf] .= reshape(view(bsm.data, index:index+r_*c_-1), (r_, c_))
    end
    # Return the matrix
    return output
end

function crop(bsm::BlockSparseMatrix{T}, rowblocks, colblocks) where T
    # Create the output
    (rows, cols, indices) = SparseArrays.findnz(view(bsm.indices, rowblocks, colblocks))
    output = BlockSparseMatrix{T}(hcat(rows, cols), bsm.rowblocksizes[rowblocks], colblocksizes[colblocks])
    # Copy all the blocks
    for (r, c, indin) in zip(rows, cols, indices)
        len = convert(UInt, output.rowblocksizes[r]) * output.colblocksizes[c] - 1
        indout = output.indices[r,c]
        view(output.data, indout:indout+len) .= view(bsm.data, indin:indin+len)
    end
    # Return the new matrix
    return output
end

b = BlockSparseMatrix{Float64}([2 1; 3 2; 1 3; 3 3], [2,3,2], [2,3,2])
block(b, 2, 1) .= randn(3, 2)
block(b, 3, 2) .= randn(2, 3)
block(b, 1, 3) .= randn(2, 2)
block(b, 3, 3) .= randn(2, 2)
# SparseArrays.sparse(b)
# Matrix(b)
# M = zeros(Float64, 7, 7)
# symmetrifyfull!(M, b, [1, 2, 3])
# M
symmetrifyfull(b)