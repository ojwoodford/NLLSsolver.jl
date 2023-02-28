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
    hessian::Matrix{Float64}
    gradient::Vector{Float64}
    varindex::UInt

    function UniVariateLS(unfixed, varlen)
        return new(zeros(Float64, varlen, varlen), zeros(Float64, varlen), unfixed)
    end
end

# Multi-variate linear system
struct MultiVariateLS
    hessian::BlockSparseMatrix{Float64}
    gradient::Vector{Float64}
    blockindices::Vector{UInt} # One for each variable
    gradoffsets::Vector{UInt} # One per unfixed variable
    symmetricindices::SparseMatrixCSC{Int, Int}
    nzvals::Int

    function MultiVariateLS(hessian::BlockSparseMatrix, blockindices)
        @assert hessian.rowblocksizes == hessian.columnblocksizes
        symmetricindices = hessian.indices - hessian.indicestransposed
        @inbounds @view(symmetricindices[diagind(symmetricindices)]) .= diag(hessian.indices)
        nzvals = length(hessian.data) * 2 - sum((Vector(diag(hessian.indices)) .!= 0) .* (convert.(Int, hessian.rowblocksizes) .^ 2))
        gradoffsets = cumsum(hessian.rowblocksizes)
        varlen = gradoffsets[end]
        circshift!(gradoffsets, -1)
        gradoffsets[1] = 0
        gradoffsets .+= 1
        return new(hessian, zeros(Float64, varlen), blockindices, gradoffsets, symmetricindices, nzvals)
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
    linsystem.gradient .+= g
    linsystem.hessian .+= H
end

function updatehessian!(hessian, H, vars, ::Val{varflags}, blockindices, loffsets) where varflags
    # Update the blocks in the problem
    @unroll for i in 1:10
        if ((varflags >> (i - 1)) & 1) == 1
            @unroll for j in i:10
                if ((varflags >> (j - 1)) & 1) == 1
                    if blockindices[i] <= blockindices[j]
                        @inbounds block(hessian, blockindices[i], blockindices[j], Val(nvars(vars[i])), Val(nvars(vars[j]))) .+= H[loffsets[i],loffsets[j]]
                    else
                        @inbounds block(hessian, blockindices[j], blockindices[i], Val(nvars(vars[j])), Val(nvars(vars[i]))) .+= H[loffsets[j],loffsets[i]]
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

function updategradient!(gradient, g, vars, ::Val{varflags}, goffsets, loffsets) where varflags
    # Update the blocks in the problem
    goffsets = blockoffsets(vars, varflags, goffsets)
    @unroll for i in 1:10
        if ((varflags >> (i - 1)) & 1) == 1
            @inbounds view(gradient, goffsets[i]) .+= g[loffsets[i]]
        end
    end
end

function updatelinearsystem!(linsystem::MultiVariateLS, g, H, vars, ::Val{varflags}, blockindices) where varflags
    loffsets = localoffsets(vars, varflags)
    updategradient!(linsystem.gradient, g, vars, Val(varflags), linsystem.gradoffsets[blockindices], loffsets)
    updatehessian!(linsystem.hessian, H, vars, Val(varflags), blockindices, loffsets)
end

function uniformscaling!(linsystem, k)
    uniformscaling!(linsystem.hessian, k)
end

function gethessgrad(linsystem::UniVariateLS)
    return linsystem.hessian, linsystem.gradient
end

function gethessgrad(linsystem::MultiVariateLS)
    if size(linsystem.hessian, 1) > 1000 && 3 * nnz(linsystem.hessian) < length(linsystem.hessian)
        return makesparse(linsystem.hessian, linsystem.symmetricindices, linsystem.nzvals), linsystem.gradient
    end
    return symmetrifyfull(linsystem.hessian), linsystem.gradient
end

function zero!(linsystem)
    fill!(linsystem.gradient, 0)
    zero!(linsystem.hessian)
end