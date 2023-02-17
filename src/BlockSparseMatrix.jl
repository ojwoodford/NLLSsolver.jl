using StaticArrays, HybridArrays, LinearAlgebra
import SparseArrays
export BlockSparseMatrix, makesparse, symmetrifysparse, symmetrifyfull, block, crop, uniformscaling!

struct BlockSparseMatrix{T}
    data::Vector{T}                                   # Storage for all the matrix data
    indices::SparseArrays.SparseMatrixCSC{Int, Int}   # Start indices for each block matrix
    indicestransposed::SparseArrays.SparseMatrixCSC{Int, Int}   # Start indices for each block matrix, stored transposed
    rowblocksizes::Vector{UInt8}                      # Number of rows for each block of rows
    columnblocksizes::Vector{UInt8}                   # Number of columns for each block of columns
    m::Int                                            # Number of rows of the represented matrix
    n::Int                                            # Number of columns of the represented matrix

    function BlockSparseMatrix{T}(nnzvals, indicestransposed, rowblocksizes, colblocksizes) where T
        # Check the block sizes
        rbs = convert.(UInt8, rowblocksizes)
        cbs = convert.(UInt8, colblocksizes)
        @assert all(i -> 0 < i, rbs)
        @assert all(i -> 0 < i, cbs)
        @assert size(indicestransposed) == (length(cbs), length(rbs))
        return new(zeros(T, nnzvals), SparseArrays.sparse(indicestransposed'), indicestransposed, rbs, cbs, sum(rbs), sum(cbs))
    end

    function BlockSparseMatrix{T}(pairs::Vector{SVector{2, U}}, rowblocksizes, colblocksizes) where {T, U}
        # Sort the indices for faster construction, and ensure unique
        pairs_ = sort(pairs)
        unique!(pairs_)
        # Compute the block pointers and length of data storage
        start = 1
        indices = Vector{Int}(undef, length(pairs_))
        for ind in eachindex(indices)
            indices[ind] = start
            # Store the blocks rowwise
            start += convert(Int, rowblocksizes[pairs_[ind][1]]) * convert(Int, colblocksizes[pairs_[ind][2]])
        end
        # Initialize the block pointer sparse matrix
        indicestransposed = SparseArrays.sparse([p[2] for p in pairs_], [p[1] for p in pairs_], indices, length(colblocksizes), length(rowblocksizes))
        # Construct the BlockSparseMatrix
        return BlockSparseMatrix{T}(start - 1, indicestransposed, rowblocksizes, colblocksizes)
    end
end

function uniformscaling!(M::AbstractMatrix, k)
    LinearAlgebra.checksquare(M)
    for i in axes(M, 1)
        @inbounds M[i,i] += k
    end
end

function uniformscaling!(M::BlockSparseMatrix, k)
    @assert M.rowblocksizes === M.columnblocksizes
    for (i, blocksz) in enumerate(M.rowblocksizes)
        @inbounds ind = M.indices[i,i]
        @assert ind > 0
        for j in ind:(blocksz+1):(ind+blocksz^2)
            @inbounds M.data[j] += k
        end
    end
end

zero!(bsm::BlockSparseMatrix) = fill!(bsm.data, 0)
@inline block(bsm::BlockSparseMatrix, i, j, ::Val{rows}, ::Val{cols}) where {rows, cols} = SizedMatrix{rows, cols}(view(bsm.data, SR(0, rows*cols-1).+bsm.indices[i,j]))
@inline block(bsm::BlockSparseMatrix, i, j, rows::Integer, cols::Integer) = reshape(view(bsm.data, SR(0, rows*cols-1).+bsm.indices[i,j]), (rows, cols))
@inline block(bsm::BlockSparseMatrix, i, j, ::Val{rows}, cols::Integer) where rows = HybridArray{Tuple{rows, StaticArrays.Dynamic()}}(block(bsm, i, j, rows, cols))
@inline block(bsm::BlockSparseMatrix, i, j) = block(bsm, i, j, bsm.rowblocksizes[i], bsm.columnblocksizes[j])
Base.size(bsm::BlockSparseMatrix) = (bsm.m, bsm.n)
Base.size(bsm::BlockSparseMatrix, dim) = dim == 1 ? bsm.m : (dim == 2 ? bsm.n : 1)
Base.length(bsm::BlockSparseMatrix) = bsm.m * bsm.n
@inline function Base.eltype(::BlockSparseMatrix{T}) where T
    return T
end
SparseArrays.nnz(bsm::BlockSparseMatrix) = length(bsm.data)

@inline zero!(aa::AbstractArray) = fill!(aa, 0)
SparseArrays.nnz(aa::AbstractArray) = length(aa)

function makesparse(bsm::BlockSparseMatrix{T}, indices, nzvals) where T
    # Preallocate arrays
    rows = Vector{Int}(undef, nzvals)
    cols = Vector{Int}(undef, nzvals)
    values = Vector{T}(undef, nzvals)
    # Sparsify each block column
    startrow = cumsum(bsm.rowblocksizes) .+ 1
    pushfirst!(startrow, UInt(1))
    startcol = cumsum(bsm.columnblocksizes) .+ 1
    pushfirst!(startcol, UInt(1))
    ind = 0
    for col in eachindex(bsm.columnblocksizes)
        sprows = indices.colptr[col]:indices.colptr[col+1]-1
        if isempty(sprows)
            continue
        end
        @inbounds block_indices = indices.nzval[sprows]
        @inbounds block_rows = indices.rowval[sprows]
        transposed = block_indices .< 0
        block_indices = abs.(block_indices)
        rowstride = ones(UInt, length(block_indices))
        @inbounds rowstride[transposed] .= bsm.columnblocksizes[col]
        @inbounds colstride = bsm.rowblocksizes[block_rows]
        @inbounds colstride[transposed] .= 1
        @inbounds col_ = startcol[col]
        for innercol in 1:bsm.columnblocksizes[col]
            for (r, br) in enumerate(block_rows)
                @inbounds s = startrow[br]
                @inbounds c = bsm.rowblocksizes[br]
                @inbounds v = block_indices[r]
                while c > 0
                    ind += 1
                    @inbounds rows[ind] = s
                    s += 1
                    @inbounds cols[ind] = col_
                    @inbounds values[ind] = bsm.data[v]
                    @inbounds v += rowstride[r]
                    c -= 1
                end
            end
            block_indices .+= colstride
            col_ += 1
        end
    end
    # Construct the sparse matrix
    return SparseArrays.sparse(rows, cols, values, startrow[end]-1, startcol[end]-1)
end

function symmetrifysparse(bsm::BlockSparseMatrix{T}) where T
    # Preallocate arrays
    @assert bsm.rowblocksizes == bsm.columnblocksizes
    indices = bsm.indices - bsm.indicestransposed
    @inbounds @view(indices[diagind(indices)]) .= diag(bsm.indices)
    nzvals = length(bsm.data) * 2 - sum((Vector(diag(bsm.indices)) .!= 0) .* (bsm.rowblocksizes .^ 2))
    return makesparse(bsm, indices, nzvals)
end

SparseArrays.sparse(bsm::BlockSparseMatrix) = makesparse(bsm, bsm.indices, length(bsm.data))

function symmetrifyfull(bsm::BlockSparseMatrix{T}) where T
    @assert bsm.rowblocksizes == bsm.columnblocksizes
    # Allocate the output
    len = sum(bsm.rowblocksizes)
    output = Matrix{T}(undef, len, len)
    # Fill in the matrix
    symmetrifyfull!(output, bsm)
    # Return the matrix
    return output
end

function symmetrifyfull!(mat::Matrix{T}, bsm::BlockSparseMatrix{T}, blocks=nothing) where T
    @assert bsm.rowblocksizes == bsm.columnblocksizes
    if blocks === nothing
        lengths = bsm.rowblocksizes
        starts = cumsum(vcat(1, bsm.rowblocksizes[1:end-1]))
    else
        lengths = zeros(UInt, length(bsm.rowblocksizes))
        lengths[blocks] .= bsm.rowblocksizes[blocks]
        starts = zeros(UInt, length(bsm.rowblocksizes))
        starts[blocks] .= cumsum(vcat(1, bsm.rowblocksizes[blocks[1:end-1]]))
    end
    m = sum(lengths)
    @assert size(mat) == (m, m)
    # Copy blocks into the output
    fill!(mat, 0)
    for (c, r, index) in zip(SparseArrays.findnz(bsm.indicestransposed)...)
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
        @inbounds mat[rs:rs+r_-1,cs:cs+c_-1] .= V
        if r != c
            @inbounds mat[cs:cs+c_-1,rs:rs+r_-1] .= V'
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
    for (c, r, index) in zip(SparseArrays.findnz(bsm.indicestransposed)...)
        @inbounds rs = rowstarts[r]
        @inbounds rf = rowstarts[r+1]
        @inbounds cs = colstarts[c]
        @inbounds cf = colstarts[c+1]
        r_ = rf - rs
        c_ = cf - cs
        @inbounds output[rs+1:rf,cs+1:cf] .= reshape(view(bsm.data, index:index+r_*c_-1), (r_, c_))
    end
    # Return the matrix
    return output
end

function crop(bsm::BlockSparseMatrix{T}, rowblocks, colblocks) where T
    # Create the output
    (cols, rows, indices) = SparseArrays.findnz(view(bsm.indices, colblocks, rowblocks))
    output = BlockSparseMatrix{T}(hcat(rows, cols), bsm.rowblocksizes[rowblocks], colblocksizes[colblocks])
    # Copy all the blocks
    for (c, r, indin) in zip(cols, rows, indices)
        len = convert(UInt, output.rowblocksizes[r]) * output.colblocksizes[c] - 1
        indout = output.indices[c,r]
        view(output.data, indout:indout+len) .= view(bsm.data, indin:indin+len)
    end
    # Return the new matrix
    return output
end
