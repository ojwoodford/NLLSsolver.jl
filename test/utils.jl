using NLLSsolver, Test, SparseArrays, StaticArrays, LinearAlgebra, Static

test_fast_bAb(A, b) = @test NLLSsolver.fast_bAb(A, b) ≈ dot(b, A, b)

@testset "utils.jl" begin
    @test NLLSsolver.runlengthencodesortedints([0, 0, 1, 1, 3]) == [1, 3, 5, 5, 6]
    @test NLLSsolver.runlengthencodesortedints([3]) == [1, 1, 1, 1, 2]
    @test NLLSsolver.runlengthencodesortedints([0]) == [1, 2]

    x = [1, 2, 3]
    @test cumsum!(x) == [1, 3, 6]
    @test x == [1, 3, 6]

    test_fast_bAb(randn(20, 20), randn(20))
    test_fast_bAb(sprandn(100, 100, 0.02), randn(100))
    test_fast_bAb(randn(SMatrix{10, 10, Float64, 100}), randn(SVector{10, Float64}))

    @test NLLSsolver.timestring(500) == "500ns"
    @test NLLSsolver.timestring(1500) == "1.5μs"
    @test NLLSsolver.timestring(1500000) == "1.5ms"
    @test NLLSsolver.timestring(1500000000) == "1.5s"
    @test NLLSsolver.timestring(90000000000) == "1.5 minutes"
    @test NLLSsolver.timestring(7200000000000) == "2 hours"
    @test NLLSsolver.timestring(720000000000000) == "200 hours"

    @test NLLSsolver.bytesstring(500) == "500 bytes"
    @test NLLSsolver.bytesstring(2048) == "2KB"
    @test NLLSsolver.bytesstring(1048576 * 5) == "5MB"
    @test NLLSsolver.bytesstring(1073741824 * 3) == "3GB"
    @test NLLSsolver.bytesstring(1099511627776 * 4) == "4TB"
    @test NLLSsolver.bytesstring(1099511627776 * 4000) == "4000TB"

    # valuedispatch: converts a runtime Int into the matching StaticInt
    for v in 0:7
        @test NLLSsolver.valuedispatch(static(0), static(7), v, identity) === static(v)
    end
    @test NLLSsolver.valuedispatch(static(3), static(3), 3, x -> x + static(1)) === static(4)

    # bindleadingargs
    f = NLLSsolver.bindleadingargs(-, 10)
    @test f(3) == 7
    @test NLLSsolver.bindleadingargs(+, 1, 2)(3, 4) == 10
    @test NLLSsolver.bindleadingargs(() -> 5)() == 5

    # SR
    r = NLLSsolver.SR(2, 5)
    @test r isa StaticArrays.SUnitRange
    @test collect(r) == [2, 3, 4, 5]
    @test length(NLLSsolver.SR(static(0), static(2))) == 3

    # bitiset (function, static and dynamic) and @bitiset
    flags = 0b1010
    for bit in 1:4
        expected = isodd(flags >> (bit - 1))
        @test NLLSsolver.bitiset(flags, bit) == expected
        @test NLLSsolver.@bitiset(flags, bit) == expected
        @test dynamic(NLLSsolver.bitiset(static(flags), bit)) == expected
    end

    # uniontotuple
    @test NLLSsolver.uniontotuple(Float64) == (Float64,)
    @test Set(NLLSsolver.uniontotuple(Union{Int, Float64, String})) == Set([Int, Float64, String])

    # sqnorm, fast_norm, fast_dot
    a = randn(1001)
    b = randn(1001)
    @test NLLSsolver.sqnorm(-3.0) == 9.0
    @test NLLSsolver.sqnorm(a) ≈ dot(a, a)
    @test NLLSsolver.sqnorm(Float64[]) == 0.0
    @test NLLSsolver.sqnorm(SVector(3.0, 4.0)) == 25.0
    @test NLLSsolver.fast_norm(a) ≈ norm(a)
    @test NLLSsolver.fast_norm(SVector(3.0, 4.0)) == 5.0
    @test NLLSsolver.fast_dot(a, b) ≈ dot(a, b)
    @test NLLSsolver.fast_dot(Float64[], Float64[]) == 0.0
    @test NLLSsolver.fast_dot([1, 2, 3], [0.5, 0.5, 0.5]) == 3.0
    @test NLLSsolver.fast_dot([1, 2, 3], [0.5, 0.5, 0.5]) isa Float64
    @test NLLSsolver.fast_dot(view(a, 2:2:1000), view(b, 1:2:999)) ≈ dot(view(a, 2:2:1000), view(b, 1:2:999))
    @test NLLSsolver.fast_dot(SVector(1.0, 2.0), SVector(3.0, 4.0)) == 11.0

    # TimeCounter and @elapsed_ns
    tc = NLLSsolver.TimeCounter()
    @test tc.time_ns == 0 && tc.count == 0
    tc += UInt64(100)
    tc += UInt64(50)
    @test tc.time_ns == 150 && tc.count == 2
    y = 0
    t = NLLSsolver.@elapsed_ns y = sum(1:10)
    @test t isa UInt64
    @test y == 55

    # StatsCounter, Stats and @stats
    sc = NLLSsolver.StatsCounter()
    @test (sc.num_allocs, sc.bytes_allocd, sc.time_ns, sc.count) == (0, 0, 0, 0)
    sc += NLLSsolver.Stats(1)
    sc += NLLSsolver.Stats(2, 3, UInt64(4))
    @test (sc.num_allocs, sc.bytes_allocd, sc.time_ns, sc.count) == (3, 4, 5, 2)
    z = nothing
    st = NLLSsolver.@stats z = zeros(1000)
    @test length(z) == 1000
    @test st.num_allocs >= 1
    @test st.bytes_allocd >= 8000
    @test st.time_ns > 0

    # sparse_dense_decision: sparse iff nnz * 64 < 25 * ndims * (ndims - 40)
    @test NLLSsolver.sparse_dense_decision(100, 2343)
    @test !NLLSsolver.sparse_dense_decision(100, 2344)
    @test !NLLSsolver.sparse_dense_decision(40, 0)
    @test !NLLSsolver.sparse_dense_decision(10, 0)

    # block_sparse_nnz: upper triangular block sparsity, blocks of size 2 and 3
    sparsity = sparse([true true; false true])
    @test NLLSsolver.block_sparse_nnz(sparsity, [2, 3]) == 2 * 2 + 2 * 3 + 3 * 3
    @test NLLSsolver.block_sparse_nnz(sparse([true false; false true]), [2, 3]) == 2 * 2 + 3 * 3
    @test NLLSsolver.sparse_dense_decision(5, sparsity, [2, 3]) == NLLSsolver.sparse_dense_decision(5, 19)

    # block_sizes_indices: only unfixed variables get blocks
    vars = Any[1.0, EuclideanVector(1.0, 2.0, 3.0), 2.0, EuclideanVector(1.0, 2.0)]
    bsizes, bindices = NLLSsolver.block_sizes_indices(vars, [true, true, false, true], 3)
    @test bsizes == [1, 3, 2]
    @test bindices == [1, 2, 0, 3]

    # computevarflags: bit i-1 set iff variable i is unfixed (non-zero offset)
    @test NLLSsolver.computevarflags(SVector(5, 0, 3)) == 0b101
    @test NLLSsolver.computevarflags(SVector(0, 0)) == 0
    @test NLLSsolver.computevarflags(SVector(1, 2, 3, 4)) == 0b1111
    @test NLLSsolver.computevarflags((7,)) == 1

    s1 = NLLSsolver.Stats()
    x = zeros(1000)
    s2 = NLLSsolver.Stats()
    @test s2.bytes_allocd >= s1.bytes_allocd
    @test s2.time_ns >= s1.time_ns
    @test s2.num_allocs >= s1.num_allocs

    # Variable nvars
    @test NLLSsolver.nvars(NLLSsolver.ZeroToInfScalar(1.0)) == 1
    @test NLLSsolver.nvars(NLLSsolver.ZeroToOneScalar(0.5)) == 1

    # Iterator strings and resets
    @test String(NLLSsolver.newton) == "Newton"
    @test String(NLLSsolver.levenbergmarquardt) == "Levenberg-Marquardt"
    @test String(NLLSsolver.levenbergmarquardt_dtd) == "Levenberg-Marquardt (D'D scaling)"
    @test String(NLLSsolver.dogleg) == "Dogleg"
    @test String(NLLSsolver.gradientdescent) == "gradient descent"
    @test String(reinterpret(NLLSsolver.NLLSIterator, Int32(99))) == "unknown iterator"
    @test NLLSsolver.reset!(NLLSsolver.NewtonData()) === nothing

    # Linear solver utilities
    @test NLLSsolver.invsym([2.0 0.5; 0.5 2.0]) ≈ inv([2.0 0.5; 0.5 2.0])
    @test NLLSsolver.invsym(SMatrix{2, 2}(2.0, 0.5, 0.5, 2.0)) ≈ inv([2.0 0.5; 0.5 2.0])

    # BlockDenseMatrix utilities
    bdm = NLLSsolver.BlockDenseMatrix{Float64}([1, 3, 5])
    @test NLLSsolver.validblock(bdm, 1, 1) == true
    @test size(bdm) == (4, 4)
    @test size(bdm, 1) == 4
    @test length(bdm) == 16
    @test eltype(bdm) == Float64
    NLLSsolver.uniformscaling!(bdm, 1.0)
    @test bdm.data[1, 1] == 1.0
    @test NLLSsolver.block(bdm, 1, 1) isa AbstractMatrix
    @test NLLSsolver.block(bdm, 1, 1, static(2), 2) isa AbstractMatrix

    # Problem construction and counting
    p_vec = NLLSsolver.NLLSProblem([1.0, 2.0])
    @test p_vec isa NLLSsolver.NLLSProblem
    p_vec_cost = NLLSsolver.NLLSProblem([1.0, 2.0], NLLSsolver.VectorRepo())
    @test p_vec_cost isa NLLSsolver.NLLSProblem
    @test NLLSsolver.convertunfixed(Float64, p_vec) == [true, true]
    @test NLLSsolver.convertunfixed([true, false], p_vec) == [true, false]

    # Robust cost evaluation
    @test NLLSsolver.robustifydcost(NLLSsolver.HuberKernel(1.0), 0.5) isa Tuple
    @test NLLSsolver.robustifydcost(NLLSsolver.Scaled(NLLSsolver.NoRobust(), 2.0), 0.5) == (1.0, 2.0, 0.0)

    # Residuals: SimpleError3 and SimpleError4
    se3 = NLLSsolver.SimpleError3{Float64, Float64, Float64}(SVector(1.0), 1, 2, 3)
    @test NLLSsolver.ndeps(se3) == 3
    @test NLLSsolver.nres(se3) == 1
    @test NLLSsolver.varindices(se3) == SVector(1, 2, 3)
    @test NLLSsolver.getvars(se3, [1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)
    @test eltype(se3) == Float64
    se4 = NLLSsolver.SimpleError4{Float64, Float64, Float64, Float64}(SVector(1.0), 1, 2, 3, 4)
    @test NLLSsolver.ndeps(se4) == 4
    @test NLLSsolver.nres(se4) == 1
    @test NLLSsolver.varindices(se4) == SVector(1, 2, 3, 4)
    @test NLLSsolver.getvars(se4, [1.0, 2.0, 3.0, 4.0]) == (1.0, 2.0, 3.0, 4.0)
    @test eltype(se4) == Float64
    @test NLLSsolver.generatemeasurement() === nothing
end
