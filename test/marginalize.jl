using NLLSsolver, SparseArrays, StaticArrays, Test, Random

@testset "marginalize.jl" begin
    # Define the size of test problem
    blocksizes = [1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1]
    fromblock = 7
    rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 6, 7, 9, 8, 7, 8, 10, 12, 4, 9, 10, 11, 6, 6, 11, 12]
    cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 1, 1, 1, 2, 3, 3, 3,  4,  3, 5, 5,  5,  5, 4, 6,  6]
    
    # Intitialize a sparse linear system randomly
    from = NLLSsolver.MultiVariateLS(NLLSsolver.BlockSparseMatrix{Float64}(sparse(cols, rows, trues(length(rows))), blocksizes, blocksizes), 1:length(blocksizes))
    Random.seed!(1)
    from.A.data .= randn(length(from.A.data))
    from.b .= randn(length(from.b))
    # Make the diagonal blocks symmetric
    for (ind, sz) in enumerate(blocksizes)
        diagblock = NLLSsolver.block(from.A, ind, ind, sz, sz)
        diagblock .= diagblock + diagblock'
    end

    # Construct the cropped system
    to = NLLSsolver.constructcrop(from, fromblock)
    NLLSsolver.initcrop!(to, from)

    # Check that the crop is correct
    hessian = NLLSsolver.symmetrifyfull(from.A)
    croplen = sum(blocksizes[1:fromblock-1])
    @test view(hessian, 1:croplen, 1:croplen) == NLLSsolver.symmetrifyfull(to.A)
    @test view(from.b, 1:croplen) == to.b
    @test all((to.boffsets .+ blocksizes[1:fromblock-1]) .<= (length(to.b) + 1))

    # Compute the ground truth reduced system
    N = size(hessian, 2)
    S = view(hessian, 1:croplen, croplen+1:N) / view(hessian, croplen+1:N, croplen+1:N)
    hessian = view(hessian, 1:croplen, 1:croplen) - S * view(hessian, croplen+1:N, 1:croplen)
    gradient = view(from.b, 1:croplen) - S * view(from.b, croplen+1:N)

    # Compute the marginalized system using dynamic block sizes
    for block in fromblock:length(from.A.rowblocksizes)
        NLLSsolver.marginalize!(to, from, block, Int(from.A.rowblocksizes[block]))
    end

    # Check that the result is correct
    @test isapprox(hessian, NLLSsolver.symmetrifyfull(to.A); rtol=1.e-13)
    @test isapprox(gradient, to.b; rtol=1.e-13)

    # Reset the 'to' system
    NLLSsolver.initcrop!(to, from)

    # Compute the marginalized system using fixed block sizes
    NLLSsolver.marginalize!(to, from)

    # Check that the result is correct
    @test isapprox(hessian, NLLSsolver.symmetrifyfull(to.A); rtol=1.e-13)
    @test isapprox(gradient, to.b; rtol=1.e-13)
end
