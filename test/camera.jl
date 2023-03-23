using NLLSsolver, StaticArrays, ForwardDiff, Test

function testcamera(cam, x)
    # Test image to ideal transformations
    y = ideal2image(cam, x)
    x_ = image2ideal(cam, y)
    @test isapprox(x, x_)

    # Test warping of the weight matrix
    x__, W = image2ideal(cam, y, @SMatrix [1. 0.; 0. 1.])
    @test x__ == x_
    @test isapprox(W, ForwardDiff.jacobian(x -> ideal2image(cam, x), x))

    # Test the update of all zeros returns the same camera
    @test cam == update(cam, zeros(nvars(cam)))
end

@testset "camera.jl" begin
    halfimsz = SA[640, 480]
    x = SVector(7.0/11., 2.0/3.)
    xi = x .* (0.3 * halfimsz)

    # Test pixel to image transformations
    y = pixel2image(halfimsz, xi)
    @test isapprox(image2pixel(halfimsz, y), xi)

    # Test warping of the weight matrix
    y_, W = pixel2image(halfimsz, xi, @SMatrix [1. 0.; 0. 1.])
    @test y_ == y
    @test isapprox(W, ForwardDiff.jacobian(x -> image2pixel(halfimsz, x), y))

    # Test cameras
    f = SVector(11.0/13., 17.0/13.)
    c = SVector(3.0/5., 5.0/7.) * 0.05
    testcamera(SimpleCamera(f[1]), x)
    testcamera(NoDistortionCamera(f, c), x)
    testcamera(ExtendedUnifiedCamera(f, c, 0.1, 0.1), x)
    testcamera(ExtendedUnifiedCamera(f, c, 0.5, 0.2), x)

    # Test fitting Extended Unified lens distortion model from barrel distortion model
    k1 = -1.e-5
    k2 = -1.e-7
    halfimsz = 1.
    lens = barrel2eulens(k1, k2, halfimsz)
    display(lens)
    maxerror = 0.
    for x in range(0, halfimsz, 100)
        x2 = x ^ 2
        maxerror = max(maxerror, abs(ideal2distorted(lens, x) - x * (1 + x2 * (k1 + x2 * k2))))
    end
    @test maxerror < 1.e-6
end
