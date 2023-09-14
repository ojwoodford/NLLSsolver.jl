using StaticArrays, HybridArrays, LinearAlgebra
import SparseArrays

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
        @assert all(i -> 0 < i, rbs)
        sumrbs = sum(rbs)
        if rowblocksizes == colblocksizes
            cbs = rbs
            sumcbs = sumrbs
        else
            cbs = convert.(UInt8, colblocksizes)
            @assert all(i -> 0 < i, cbs)
            sumcbs = sum(cbs)
        end
        @assert size(indicestransposed) == (length(cbs), length(rbs))
        return new(zeros(T, nnzvals), SparseArrays.spzeros(length(rbs), length(cbs)), SparseArrays.sparse(indicestransposed), rbs, cbs, sumrbs, sumcbs)
    end

    function BlockSparseMatrix{T}(sparsitytransposed::SparseArrays.SparseMatrixCSC{Bool, Int}, rowblocksizes, colblocksizes) where T
        @assert (size(sparsitytransposed) == (length(colblocksizes), length(rowblocksizes)))
        # Construct the block indices sparse matrix
        indicestransposed = SparseArrays.SparseMatrixCSC{Int, Int}(sparsitytransposed.m, sparsitytransposed.n, sparsitytransposed.colptr, sparsitytransposed.rowval, Vector{Int}(undef, length(sparsitytransposed.rowval)))
        # Compute the block pointers and length of data storage
        start = 1
        ind = 1
        for row = 1:length(rowblocksizes)
            rowwidth = convert(Int, @inbounds rowblocksizes[row])
            for col in @inbounds view(sparsitytransposed.rowval, sparsitytransposed.colptr[row]:sparsitytransposed.colptr[row+1]-1)
                @inbounds indicestransposed.nzval[ind] = start
                ind +=1
                start += rowwidth * convert(Int, @inbounds colblocksizes[col])
            end
        end
        # Construct the BlockSparseMatrix
        return BlockSparseMatrix{T}(start - 1, indicestransposed, rowblocksizes, colblocksizes)
    end

    function BlockSparseMatrix{T}(bsm::BlockSparseMatrix{T}, m::Int, n::Int=m) where T
        return new(bsm.data, bsm.indices, bsm.indicestransposed, bsm.rowblocksizes, bsm.columnblocksizes, m, n)
    end
end

function clearindicescache(bsm)
    resize!(bsm.indices.nzval, 0)
end

function cacheindices(bsm)
    if isempty(bsm.indices.nzval)
        # Cache the untransposed indices
        resize!(bsm.indices.nzval, length(bsm.indicestransposed.nzval))
        resize!(bsm.indices.rowval, length(bsm.indicestransposed.nzval))
        resize!(bsm.indices.colptr, size(bsm.indicestransposed, 2)+1)
        transpose!(bsm.indices, bsm.indicestransposed)
    end
end

function updatelastrowsym!(bsm::BlockSparseMatrix{T}, rowindices, blocksz) where T
    @assert(bsm.rowblocksizes == bsm.columnblocksizes)
    diff = blocksz - bsm.rowblocksizes[end]
    if diff != 0
        blocksz_ = convert(UInt8, blocksz)
        bsm.rowblocksizes[end] = blocksz_
        bsm.colblocksizes[end] = blocksz_
        bsm = BlockSparseMatrix{T}(bsm, bsm.m + diff)
    end
    diff = length(rowindices) - (bsm.indicestransposed.colptr[end] - bsm.indicestransposed.colptr[end-1])
    if diff != 0
    end
    return bsm
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
        @inbounds ind = M.indicestransposed[i,i]
        @assert ind > 0
        for j in ind:(blocksz+1):(ind+blocksz^2)
            @inbounds M.data[j] += k
        end
    end
end

zero!(bsm::BlockSparseMatrix) = fill!(bsm.data, 0)
block(bsm::BlockSparseMatrix, i, j, ::StaticInt{rows}, ::StaticInt{cols}) where {rows, cols} = SizedMatrix{rows, cols}(view(bsm.data, SR(0, rows*cols-1).+bsm.indicestransposed[j,i]))
block(bsm::BlockSparseMatrix, i, j, rows::Integer, cols::Integer) = reshape(view(bsm.data, SR(0, rows*cols-1).+bsm.indicestransposed[j,i]), (rows, cols))
block(bsm::BlockSparseMatrix, i, j, ::StaticInt{rows}, cols::Integer) where rows = HybridArray{Tuple{rows, StaticArrays.Dynamic()}}(block(bsm, i, j, rows, cols))
block(bsm::BlockSparseMatrix, i, j) = block(bsm, i, j, bsm.rowblocksizes[i], bsm.columnblocksizes[j])
validblock(bsm::BlockSparseMatrix, i, j) = bsm.indicestransposed[j,i] != 0
Base.size(bsm::BlockSparseMatrix) = (bsm.m, bsm.n)
Base.size(bsm::BlockSparseMatrix, dim) = dim == 1 ? bsm.m : (dim == 2 ? bsm.n : 1)
Base.length(bsm::BlockSparseMatrix) = bsm.m * bsm.n
Base.eltype(::BlockSparseMatrix{T}) where T = T
SparseArrays.nnz(bsm::BlockSparseMatrix) = length(bsm.data)

zero!(aa::AbstractArray) = fill!(aa, 0)
SparseArrays.nnz(aa::AbstractArray) = length(aa)

function cumsum1!(A, B)
    total = A[1]
    @inbounds for ind in eachindex(B)
        total += B[ind]
        A[ind+1] = total
    end
    return A
end

function diagonalblockspace(indices, blocksizes)
    total = Int(0)
    @inbounds for col = 1:length(indices.colptr)-1
        ind = indices.colptr[col]
        if indices.colptr[col+1] > ind
            row = indices.rowval[ind]
            @assert(row >= col, "BSM must be lower triangular for symmetrification")
            if row == col
                bs = Int(blocksizes[row])
                total += bs * bs
            end
        end
    end
    return total
end

function makesparseindices(bsm::BlockSparseMatrix, symmetrify::Bool=false)
    @assert !symmetrify || bsm.rowblocksizes == bsm.columnblocksizes
    # Preallocate arrays
    cacheindices(bsm)
    nzvals = symmetrify ? length(bsm.data) * 2 - diagonalblockspace(bsm.indices, bsm.rowblocksizes) : length(bsm.data)
    rows = Vector{Int}(undef, nzvals)
    indices = Vector{Int}(undef, nzvals)
    startrow = Vector{Int}(undef, length(bsm.rowblocksizes)+1)
    startrow[1] = 1
    cumsum1!(startrow, bsm.rowblocksizes)
    cols = Vector{Int}(undef, symmetrify ? startrow[end] : sum(bsm.columnblocksizes)+1)
    # Sparsify each block column
    ind = 1
    col = 1
    @inbounds cols[1] = 1
    @inbounds for (col_, colblocksize) in enumerate(bsm.columnblocksizes)
        lower_rows = bsm.indices.colptr[col_]:bsm.indices.colptr[col_+1]-1
        upper_rows = symmetrify ? (bsm.indicestransposed.colptr[col_]:bsm.indicestransposed.colptr[col_+1]-1-(!isempty(lower_rows) && bsm.indices.rowval[lower_rows[1]] == col_)) : 1:0
        for innercol in 0:colblocksize-1
            # Above diagonal blocks if symmetrifying (transposed)
            for r in upper_rows
                row = bsm.indicestransposed.rowval[r]
                s = startrow[row]
                c = bsm.rowblocksizes[row]
                v = bsm.indicestransposed.nzval[r] + innercol
                for i in 0:c-1
                    rows[ind] = s + i
                    indices[ind] = v
                    ind += 1
                    v += colblocksize
                end
            end
            # Diagonal and below diagonal blocks
            for r in lower_rows
                row = bsm.indices.rowval[r]
                s = startrow[row]
                c = bsm.rowblocksizes[row]
                v = bsm.indices.nzval[r] + innercol * c
                for i in 0:c-1
                    rows[ind] = s + i
                    indices[ind] = v + i
                    ind += 1
                end
            end
            col += 1
            cols[col] = ind
        end
    end
    # Construct the sparse matrix
    return SparseArrays.SparseMatrixCSC{Int, Int}(startrow[end]-1, col-1, cols, rows, indices)
end

@inline function SparseArrays.sparse(bsm::BlockSparseMatrix{T}, indices=makesparseindices(bsm)) where T
    return SparseArrays.SparseMatrixCSC{T, Int}(indices.m, indices.n, indices.colptr, indices.rowval, bsm.data[indices.nzval])
end

@inline symmetrifysparse(bsm::BlockSparseMatrix{T}) where T = SparseArrays.sparse(bsm, makesparseindices(bsm, true))

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
