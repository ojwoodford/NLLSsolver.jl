using NLLSsolver, StaticArrays, Test

function testcamera(cam)
    x = @SVector randn(2)
    err = (@SVector randn(2)) * 1.e-6
    xe = x + err
    W = @SMatrix randn(2, 2)

    # Test image to ideal transformations
    y, Wy = image2ideal(cam, x, W)
    @test image2ideal(cam, x) == y
    @test isapprox(ideal2image(cam, y), x)

    # Test warping of the weight matrix
    ye = image2ideal(cam, xe)
    @test isapprox(Wy * (ye - y), W * err; rtol=1.e-3)

    # Test the update of all zeros returns the same camera
    @test cam == update(cam, zeros(nvars(cam)))
end

@testset "camera.jl" begin
    halfimsz = SA[640, 480]
    x = @SVector randn(2)
    x = x .* (0.3 * halfimsz)
    err = (@SVector randn(2)) * 1.e-6
    xe = x + err
    W = @SMatrix randn(2, 2)

    # Test pixel to image transformations
    y, Wy = pixel2image(halfimsz, x, W)
    @test pixel2image(halfimsz, x) == y
    @test isapprox(image2pixel(halfimsz, y), x)

    # Test warping of the weight matrix
    ye = pixel2image(halfimsz, xe)
    @test isapprox(Wy * (ye - y), W * err; rtol=1.e-5)

    # Test cameras
    testcamera(SimpleCamera(abs(randn())))
    f = abs.(@SVector randn(2))
    c = (@SVector randn(2)) * 0.05
    testcamera(NoDistortionCamera(f, c))
    testcamera(ExtendedUnifiedCamera(f, c, 0.8, 0.001))
    testcamera(ExtendedUnifiedCamera(f, c, 0.3, 0.01))
end
