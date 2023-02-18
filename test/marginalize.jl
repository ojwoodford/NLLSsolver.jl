using NLLSsolver, SparseArrays, StaticArrays, Test

@testset "marginalize.jl" begin
    # Define the size of test problem
    blocksizes = [1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1]
    fromblock = 7
    pairs_ = [2, 1, 6, 1, 7, 1, 9, 1, 8, 2, 7, 3, 8, 3, 10, 3, 12, 4, 4, 3, 9, 5, 10, 5, 11, 5, 6, 5, 6, 4, 11, 6, 12, 6]
    
    # Intitialize a sparse linear system randomly
    pairs = [SVector(i, i) for i in 1:length(blocksizes)]
    for i in 1:2:length(pairs_)
        push!(pairs, SVector(pairs_[i], pairs_[i+1]))
    end
    from = MultiVariateLS(pairs, blocksizes)
    from.hessian.data .= randn(length(from.hessian.data))
    from.gradient .= randn(length(from.gradient))
    # Make the diagonal blocks symmetric
    for (ind, sz) in enumerate(blocksizes)
        diagblock = block(from.hessian, ind, ind, sz, sz)
        diagblock .= diagblock + diagblock'
    end

    # Construct the cropped system
    to = constructcrop(from, fromblock)
    initcrop!(to, from)

    # Check that the crop is correct
    hessian = symmetrifyfull(from.hessian)
    croplen = sum(blocksizes[1:fromblock-1])
    @test view(hessian, 1:croplen, 1:croplen) == symmetrifyfull(to.hessian)
    @test view(from.gradient, 1:croplen) == to.gradient
    @test all((to.gradoffsets .+ blocksizes[1:fromblock-1]) .<= (length(to.gradient) + 1))

    # Compute the marginalized system
    marginalize!(to, from)

    # Compute the ground truth reduced system
    S = hessian[1:croplen,croplen+1:end] / hessian[croplen+1:end,croplen+1:end]
    hessian = hessian[1:croplen,1:croplen] - S * hessian[croplen+1:end,1:croplen]
    gradient = from.gradient[1:croplen] - S * from.gradient[croplen+1:end]

    # Check that the result is correct
    @test hessian ≈ symmetrifyfull(to.hessian)
    @test gradient ≈ to.gradient
end