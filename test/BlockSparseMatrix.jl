using NLLSsolver, SparseArrays, StaticArrays, Test

@testset "BlockSparseMatrix.jl" begin
    # Construct a test matrix
    out = [0. 0  0  0  0  10;
           0  0  0  0  0  11; 
           0  0  0  0  0  12;
           1  4  0  0  0  0 ;
           2  5  0  0  0  0 ;
           3  6  0  0  0  0 ;
           0  0  7  8  9  13]

    # Construct a matching BSM
    b = BlockSparseMatrix{Float64}([SVector(2, 1), SVector(3, 2), SVector(1, 3), SVector(3, 3)], [3,3,1], [2,3,1])
    block(b, 2, 1) .= reshape(1.:6., (3, 2))
    block(b, 3, 2, 1, 3) .= SMatrix{1, 3, Float64, 3}(7:9)
    block(b, 1, 3, 3, 1) .= [10.; 11; 12]
    block(b, 3, 3) .= 13.

    @test eltype(b) == Float64
    @test size(b) == (7, 6)
    @test length(b) == 42
    @test nnz(b) == 13

    m = Matrix(b)
    @test typeof(m) == Matrix{Float64}
    @test size(m) == (7, 6)
    @test m == out

    s = sparse(b)
    @test issparse(s)
    @test typeof(s) == SparseMatrixCSC{Float64, Int64}
    @test size(s) == (7, 6)
    @test nnz(s) == 13
    @test Matrix(s) == out

    # Construct a test matrix
    out2 = [0  0  1   3   5   0 ; 
            0  0  2   4   6   0 ;
            0  0  7   8   9   13;
            0  0  8   10  11  14;
            0  0  9   11  12  15;
            0  0  0   0   0   16]
    outsym = max.(out2, out2')

    # Construct a matching BSM
    bs = BlockSparseMatrix{Int64}([SVector(1, 2), SVector(2, 3), SVector(2, 2), SVector(3, 3)], [2,3,1], [2,3,1])
    block(bs, 1, 2) .= reshape(1:6, (2, 3))
    block(bs, 2, 3, 3, 1) .= SVector{3, Int64}(13:15)
    block(bs, 2, 2, 3, 3) .= [7 8 9; 8 10 11; 9 11 12]
    block(bs, 3, 3) .= 16

    @test eltype(bs) == Int64
    @test size(bs) == (6, 6)
    @test length(bs) == 36
    @test nnz(bs) == 19

    m2 = Matrix(bs)
    @test typeof(m2) == Matrix{Int64}
    @test size(m2) == (6, 6)
    @test m2 == out2

    s2 = sparse(bs)
    @test issparse(s2)
    @test typeof(s2) == SparseMatrixCSC{Int64, Int64}
    @test size(s2) == (6, 6)
    @test nnz(s2) == 19
    @test Matrix(s2) == out2

    m3 = symmetrifyfull(bs)
    @test typeof(m3) == Matrix{Int64}
    @test size(m3) == (6, 6)
    @test m3 == outsym

    s3 = symmetrifysparse(bs)
    @test issparse(s3)
    @test typeof(s3) == SparseMatrixCSC{Int64, Int64}
    @test size(s3) == (6, 6)
    @test nnz(s3) == 28
    @test Matrix(s3) == outsym
end