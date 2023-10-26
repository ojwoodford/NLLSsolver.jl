using NLLSsolver, SparseArrays, StaticArrays, Test, Random

@testset "marginalize.jl" begin
    # Define the size of test problem
    blocksizes = [1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1]
    fromblock = 7
    rows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 2, 6, 7, 9, 8, 7, 8, 10, 12, 4, 9, 10, 11, 6, 6, 11, 12]
    cols = [1, 2, 3, 2, 5, 6, 7, 8, 9, 10, 11, 12, 1, 1, 1, 1, 2, 3, 3, 3,  4,  3, 5, 5,  5,  5, 4, 6,  6]
    
    # Intitialize a sparse linear system randomly
    from = NLLSsolver.MultiVariateLSsparse(NLLSsolver.BlockSparseMatrix{Float64}(sparse(cols, rows, trues(length(rows))), blocksizes, blocksizes), 1:length(blocksizes))
    Random.seed!(1)
    from.A.data .= randn(length(from.A.data))
    from.b .= randn(length(from.b))
    # Make the diagonal blocks symmetric
    for (ind, sz) in enumerate(blocksizes)
        if NLLSsolver.validblock(from.A, ind, ind)
            diagblock = NLLSsolver.block(from.A, ind, ind, sz, sz)
            diagblock .= diagblock * diagblock'
        end
    end

    # Construct the cropped system
    to_d = NLLSsolver.constructcrop(from, fromblock)
    @test isa(to_d, NLLSsolver.MultiVariateLSdense)
    to_s = NLLSsolver.constructcrop(from, fromblock, true)
    @test isa(to_s, NLLSsolver.MultiVariateLSsparse)
    NLLSsolver.initcrop!(to_d, from)
    NLLSsolver.initcrop!(to_s, from)

    # Check that the crops are correct
    hessian = NLLSsolver.symmetrifyfull(from.A)
    croplen = sum(blocksizes[1:fromblock-1])
    @test view(hessian, 1:croplen, 1:croplen) == NLLSsolver.symmetrifyfull(to_d.A)
    @test view(from.b, 1:croplen) == to_d.b
    @test all((to_d.boffsets[1:end-1] .+ blocksizes[1:fromblock-1]) .<= (length(to_d.b) + 1))
    @test view(hessian, 1:croplen, 1:croplen) == NLLSsolver.symmetrifyfull(to_s.A)
    @test view(from.b, 1:croplen) == to_s.b
    @test all((to_s.boffsets .+ blocksizes[1:fromblock-1]) .<= (length(to_s.b) + 1))

    # Compute the ground truth variable update
    gtupdate = hessian \ from.b
    gtupdate = gtupdate[1:croplen]

    # Compute the ground truth reduced system
    N = size(hessian, 2)
    S = view(hessian, 1:croplen, croplen+1:N) / view(hessian, croplen+1:N, croplen+1:N)
    hessian = view(hessian, 1:croplen, 1:croplen) - S * view(hessian, croplen+1:N, 1:croplen)
    gradient = view(from.b, 1:croplen) - S * view(from.b, croplen+1:N)

    # Compute the marginalized system using dynamic block sizes
    for block in fromblock:length(from.A.rowblocksizes)
        NLLSsolver.marginalize!(to_d, from, block, Int(from.A.rowblocksizes[block]))
        NLLSsolver.marginalize!(to_s, from, block, Int(from.A.rowblocksizes[block]))
    end

    # Check that the results are the same
    hess_d = NLLSsolver.symmetrifyfull(to_d.A)
    hess_s = NLLSsolver.symmetrifyfull(to_s.A)
    @test hess_d ≈ hess_s
    @test to_d.b == to_s.b

    # Check that the reduced systems give the correct variable update
    @test hessian \ gradient ≈ gtupdate
    @test hess_d \ to_d.b ≈ gtupdate
    @test hess_s \ to_s.b ≈ gtupdate

    # Reset the 'to' systems
    NLLSsolver.initcrop!(to_d, from)
    NLLSsolver.initcrop!(to_s, from)

    # Compute the marginalized system using static block sizes
    NLLSsolver.marginalize!(to_d, from)
    NLLSsolver.marginalize!(to_s, from)

    # Check that the results are the same
    hess_d = NLLSsolver.symmetrifyfull(to_d.A)
    hess_s = NLLSsolver.symmetrifyfull(to_s.A)
    @test hess_d ≈ hess_s
    @test to_d.b == to_s.b

    # Check that the reduced systems give the correct variable update
    @test hess_d \ to_d.b ≈ gtupdate
    @test hess_s \ to_s.b ≈ gtupdate
end
