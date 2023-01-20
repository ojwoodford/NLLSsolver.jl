export UniVariateLS, MultiVariateLS, gethessgrad, zero!, uniformscaling!

function addpairs!(pairs, residuals::Vector, blockindices)
    for res in residuals
        addpairs!(pairs, res, blockindices)
    end
end

function addpairs!(pairs, residual, blockindices)
    blocks = blockindices[varindices(residual)]
    blocks = blocks[blocks .!= 0]
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
    blockindices::Vector{UInt}
    gradoffsets::Vector{UInt}
    
    function MultiVariateLS(vars, residuals, unfixed, nblocks)
        # Multiple variables. Use a block sparse matrix
        blockindices = zeros(UInt, length(vars))
        gradoffsets = zeros(UInt, nblocks)
        blocksizes = zeros(UInt, nblocks)
        start = UInt(1)
        nblocks = UInt(0)
        pairs = Vector{SVector{2, UInt}}()
        for (index, unfixed_) in enumerate(unfixed)
            if unfixed_
                nblocks += 1
                blockindices[index] = nblocks
                gradoffsets[nblocks] = start
                N = UInt(nvars(vars[index]))
                blocksizes[nblocks] = N
                start += N
                push!(pairs, SVector(nblocks, nblocks))
            end
        end
        varlen = start - 1

        # Compute the off-diagonal pairs
        @inbounds for (key, res) in residuals
            addpairs!(pairs, res, blockindices)
        end

        # Construct block sparse matrix
        return new(BlockSparseMatrix{Float64}(pairs, blocksizes, blocksizes), zeros(Float64, varlen), blockindices, gradoffsets)
    end
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
                    @inbounds block(hessian, blockindices[i], blockindices[j], nvars(vars[i]), nvars(vars[j])) .+= H[loffsets[i],loffsets[j]]
                end
            end
        end
    end
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
        return symmetrifysparse(linsystem.hessian), linsystem.gradient
    end
    return symmetrifyfull(linsystem.hessian), linsystem.gradient
end

function zero!(linsystem)
    fill!(linsystem.gradient, 0)
    zero!(linsystem.hessian)
end