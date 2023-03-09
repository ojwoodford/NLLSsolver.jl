using SparseArrays
export UniVariateLS, MultiVariateLS, makemvls, gethessgrad, zero!, uniformscaling!

function addpairs!(pairs, residuals::Vector, blockindices)
    for res in residuals
        addpairs!(pairs, res, blockindices)
    end
end

function addpairs!(pairs, residual, blockindices)
    blocks = blockindices[varindices(residual)]
    blocks = sort(blocks[blocks .!= 0])
    @inbounds for (i, b) in enumerate(blocks)
        @inbounds for b_ in @view blocks[i+1:end]
            push!(pairs, SVector(b, b_))
        end
    end
end

# Uni-variate linear system
struct UniVariateLS
    A::Matrix{Float64}
    b::Vector{Float64}
    varindex::UInt

    function UniVariateLS(unfixed, varlen)
        return new(zeros(Float64, varlen, varlen), zeros(Float64, varlen), unfixed)
    end
end

# Multi-variate linear system
struct MultiVariateLS
    A::BlockSparseMatrix{Float64}
    b::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    boffsets::Vector{UInt} # One per unfixed variable
    symmetricindices::SparseMatrixCSC{Int, Int}
    nzvals::Int

    function MultiVariateLS(A::BlockSparseMatrix, blockindices)
        symmetricindices = A.indices - A.indicestransposed
        @inbounds @view(symmetricindices[diagind(symmetricindices)]) .= diag(A.indices)
        nzvals = length(A.data) * 2 - sum((Vector(diag(A.indices)) .!= 0) .* (convert.(Int, A.rowblocksizes) .^ 2))
        boffsets = cumsum(A.rowblocksizes)
        varlen = boffsets[end]
        circshift!(boffsets, -1)
        boffsets[1] = 0
        boffsets .+= 1
        return new(A, zeros(Float64, varlen), blockindices, boffsets, symmetricindices, nzvals)
    end

    function MultiVariateLS(pairs, blocksizes, blockindices=1:length(blocksizes))
        return MultiVariateLS(BlockSparseMatrix{Float64}(pairs, blocksizes, blocksizes), blockindices)
    end
end

function makemvls(vars, residuals, unfixed, nblocks)
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
    @inbounds for (key, res) in residuals
        addpairs!(pairs, res, blockindices)
    end

    # Construct the MultiVariateLS
    return MultiVariateLS(pairs, blocksizes, blockindices)
end

function updatelinearsystem!(linsystem::UniVariateLS, g, H, unusedargs...)
    # Update the blocks in the problem
    linsystem.b .+= g
    linsystem.A .+= H
end

function updatesymA!(A, a, vars, ::Val{varflags}, blockindices, loffsets) where varflags
    # Update the blocks in the problem
    @unroll for i in 1:10
        if ((varflags >> (i - 1)) & 1) == 1
            @unroll for j in i:10
                if ((varflags >> (j - 1)) & 1) == 1
                    if blockindices[i] <= blockindices[j]
                        @inbounds block(A, blockindices[i], blockindices[j], Val(nvars(vars[i])), Val(nvars(vars[j]))) .+= a[loffsets[i],loffsets[j]]
                    else
                        @inbounds block(A, blockindices[j], blockindices[i], Val(nvars(vars[j])), Val(nvars(vars[i]))) .+= a[loffsets[j],loffsets[i]]
                    end
                end
            end
        end
    end
end

@inline function blockoffsets(vars, varflags, blockoff)
    return ntuple(i -> SR(1, nvars(vars[i]) * ((varflags >> (i - 1)) & 1)) .+ (blockoff[i] - 1), length(vars))
end

@inline function localoffsets(vars, varflags)
    return ntuple(i -> SR(1, nvars(vars[i]) * ((varflags >> (i - 1)) & 1)) .+ countvars(vars[1:i-1], Val(varflags)), length(vars))
end

function updateb!(B, b, vars, ::Val{varflags}, goffsets, loffsets) where varflags
    # Update the blocks in the problem
    goffsets = blockoffsets(vars, varflags, goffsets)
    @unroll for i in 1:10
        if ((varflags >> (i - 1)) & 1) == 1
            @inbounds view(B, goffsets[i]) .+= b[loffsets[i]]
        end
    end
end

function updatelinearsystem!(linsystem::MultiVariateLS, g, H, vars, ::Val{varflags}, blockindices) where varflags
    loffsets = localoffsets(vars, varflags)
    updateb!(linsystem.b, g, vars, Val(varflags), linsystem.boffsets[blockindices], loffsets)
    updatesymA!(linsystem.A, H, vars, Val(varflags), blockindices, loffsets)
end

function uniformscaling!(linsystem, k)
    uniformscaling!(linsystem.A, k)
end

function gethessgrad(linsystem::UniVariateLS)
    return linsystem.A, linsystem.b
end

function gethessgrad(linsystem::MultiVariateLS)
    if size(linsystem.A, 1) > 1000 && 3 * nnz(linsystem.A) < length(linsystem.A)
        return makesparse(linsystem.A, linsystem.symmetricindices, linsystem.nzvals), linsystem.b
    end
    return symmetrifyfull(linsystem.A), linsystem.b
end

function getresjac(linsystem::UniVariateLS)
    return linsystem.A, linsystem.b
end

function getresjac(linsystem::MultiVariateLS)
    if size(linsystem.A, 1) > 1000 && 3 * nnz(linsystem.A) < length(linsystem.A)
        return SparseArrays.sparse(linsystem.A), linsystem.b
    end
    return Matrix(linsystem.A), linsystem.b
end

function zero!(linsystem)
    fill!(linsystem.b, 0)
    zero!(linsystem.A)
end