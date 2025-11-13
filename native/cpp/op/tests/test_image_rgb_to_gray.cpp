#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ptensor/op/image_rgb_to_gray.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include "ptensor/p10_error.hpp"

namespace p10::op {
using Catch::Approx;
using p10::testing::IsError;
using p10::testing::IsOk;

TEST_CASE("RGB to gray: Basic conversion", "[image_rgb_to_gray]") {
    auto rgb = Tensor::zeros(make_shape(10, 10, 3), Dtype::Uint8).unwrap();
    Tensor gray;

    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());
    REQUIRE(gray.shape(0).unwrap() == 10);
    REQUIRE(gray.shape(1).unwrap() == 10);
    REQUIRE(gray.dims() == 2);
    REQUIRE(gray.dtype() == Dtype::Uint8);
}

TEST_CASE("RGB to gray: Pure red converts correctly", "[image_rgb_to_gray]") {
    auto rgb = Tensor::zeros(make_shape(5, 5, 3), Dtype::Uint8).unwrap();
    auto rgb_span = rgb.as_span3d<uint8_t>().unwrap();

    for (size_t h = 0; h < rgb_span.height(); ++h) {
        for (size_t w = 0; w < rgb_span.width(); ++w) {
            auto pixel = rgb_span.channel(h, w);
            pixel[0] = 255;
            pixel[1] = 0;
            pixel[2] = 0;
        }
    }

    Tensor gray;
    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());

    auto gray_span = gray.as_span2d<uint8_t>().unwrap();
    for (size_t h = 0; h < gray_span.height(); ++h) {
        for (size_t w = 0; w < gray_span.width(); ++w) {
            REQUIRE(gray_span.row(h)[w] == Approx(53).margin(1));
        }
    }
}

TEST_CASE("RGB to gray: Pure green converts correctly", "[image_rgb_to_gray]") {
    auto rgb = Tensor::zeros(make_shape(5, 5, 3), Dtype::Uint8).unwrap();
    auto rgb_span = rgb.as_span3d<uint8_t>().unwrap();

    for (size_t h = 0; h < rgb_span.height(); ++h) {
        for (size_t w = 0; w < rgb_span.width(); ++w) {
            auto pixel = rgb_span.channel(h, w);
            pixel[0] = 0;
            pixel[1] = 255;
            pixel[2] = 0;
        }
    }

    Tensor gray;
    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());

    auto gray_span = gray.as_span2d<uint8_t>().unwrap();
    // Expected: 0.72 * 255 = 183.6 H 183
    for (size_t h = 0; h < gray_span.height(); ++h) {
        for (size_t w = 0; w < gray_span.width(); ++w) {
            REQUIRE(gray_span.row(h)[w] == Approx(183).margin(1));
        }
    }
}

TEST_CASE("RGB to gray: Pure blue converts correctly", "[image_rgb_to_gray]") {
    auto rgb = Tensor::zeros(make_shape(5, 5, 3), Dtype::Uint8).unwrap();
    auto rgb_s = rgb.as_span3d<uint8_t>().unwrap();

    for (size_t h = 0; h < rgb_s.height(); ++h) {
        for (size_t w = 0; w < rgb_s.width(); ++w) {
            auto pixel = rgb_s.channel(h, w);
            pixel[0] = 0;
            pixel[1] = 0;
            pixel[2] = 255;
        }
    }

    Tensor gray;
    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());

    auto gray_span = gray.as_span2d<uint8_t>().unwrap();
    for (size_t h = 0; h < gray_span.height(); ++h) {
        for (size_t w = 0; w < gray_span.width(); ++w) {
            REQUIRE(gray_span.row(h)[w] == Approx(17).margin(1));
        }
    }
}

TEST_CASE("RGB to gray: White converts to white", "[image_rgb_to_gray]") {
    auto rgb = Tensor::full(make_shape(8, 8, 3), 255, Dtype::Uint8).unwrap();
    Tensor gray;

    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());

    auto gray_span = gray.as_span2d<uint8_t>().unwrap();
    for (size_t h = 0; h < gray_span.height(); ++h) {
        for (size_t w = 0; w < gray_span.width(); ++w) {
            REQUIRE(gray_span.row(h)[w] == 255);
        }
    }
}

TEST_CASE("RGB to gray: Black converts to black", "[image_rgb_to_gray]") {
    auto rgb = Tensor::zeros(make_shape(8, 8, 3), Dtype::Uint8).unwrap();
    Tensor gray;

    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());

    auto gray_span = gray.as_span2d<uint8_t>().unwrap();
    for (size_t h = 0; h < gray_span.height(); ++h) {
        for (size_t w = 0; w < gray_span.width(); ++w) {
            REQUIRE(gray_span.row(h)[w] == 0);
        }
    }
}

TEST_CASE("RGB to gray: Mixed color values", "[image_rgb_to_gray]") {
    auto rgb = Tensor::zeros(make_shape(3, 3, 3), Dtype::Uint8).unwrap();
    auto rgb_span = rgb.as_span3d<uint8_t>().unwrap();

    // Set a specific pixel with known RGB values
    auto pixel = rgb_span.channel(1, 1);
    pixel[0] = 100;  // R
    pixel[1] = 150;  // G
    pixel[2] = 200;  // B

    Tensor gray;
    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());

    auto gray_span = gray.as_span2d<uint8_t>().unwrap();
    // Expected: 0.21 * 100 + 0.72 * 150 + 0.07 * 200 = 21 + 108 + 14 = 143
    REQUIRE(gray_span.row(1)[1] == Approx(143).margin(1));
}

TEST_CASE("RGB to gray: Different image sizes", "[image_rgb_to_gray]") {
    SECTION("Small image 1x1") {
        auto rgb = Tensor::zeros(make_shape(1, 1, 3), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());
        REQUIRE(gray.shape() == make_shape(1, 1));
    }

    SECTION("Wide image") {
        auto rgb = Tensor::zeros(make_shape(10, 100, 3), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());
        REQUIRE(gray.shape() == make_shape(10, 100));
    }

    SECTION("Tall image") {
        auto rgb = Tensor::zeros(make_shape(100, 10, 3), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());
        REQUIRE(gray.shape() == make_shape(100, 10));
    }

    SECTION("Large square image") {
        auto rgb = Tensor::zeros(make_shape(256, 256, 3), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());
        REQUIRE(gray.shape() == make_shape(256, 256));
    }
}

TEST_CASE("RGB to gray: Luminosity formula accuracy", "[image_rgb_to_gray]") {
    // Test that the conversion uses the correct luminosity formula
    auto rgb = Tensor::zeros(make_shape(1, 1, 3), Dtype::Uint8).unwrap();
    auto rgb_span = rgb.as_span3d<uint8_t>().unwrap();

    // Test multiple RGB combinations
    struct TestCase {
        uint8_t r, g, b;
        uint8_t expected_gray;
    };

    TestCase test_cases[] = {
        {128, 128, 128, 128},  // Gray
        {255, 255, 255, 255},  // White
        {0, 0, 0, 0},  // Black
        {50, 100, 150, 91
        },  // 0.21*50 + 0.72*100 + 0.07*150 = 10.5 + 72 + 10.5 = 93, but check actual
        {200, 50, 100, 85},  // 0.21*200 + 0.72*50 + 0.07*100 = 42 + 36 + 7 = 85
    };

    for (const auto& test : test_cases) {
        auto pixel = rgb_span.channel(0, 0);
        pixel[0] = test.r;
        pixel[1] = test.g;
        pixel[2] = test.b;

        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());

        auto gray_span = gray.as_span2d<uint8_t>().unwrap();
        float expected = 0.21f * test.r + 0.72f * test.g + 0.07f * test.b;
        REQUIRE(gray_span.row(0)[0] == Approx(expected).margin(1));
    }
}

TEST_CASE("RGB to gray: Error - Wrong number of dimensions", "[image_rgb_to_gray]") {
    SECTION("2D tensor instead of 3D") {
        auto rgb = Tensor::zeros(make_shape(10, 10), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }

    SECTION("4D tensor instead of 3D") {
        auto rgb = Tensor::zeros(make_shape(10, 10, 3, 1), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }

    SECTION("1D tensor") {
        auto rgb = Tensor::zeros(make_shape(30), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }
}

TEST_CASE("RGB to gray: Error - Wrong number of channels", "[image_rgb_to_gray]") {
    SECTION("1 channel") {
        auto rgb = Tensor::zeros(make_shape(10, 10, 1), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }

    SECTION("4 channels (RGBA)") {
        auto rgb = Tensor::zeros(make_shape(10, 10, 4), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }

    SECTION("2 channels") {
        auto rgb = Tensor::zeros(make_shape(10, 10, 2), Dtype::Uint8).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }
}

TEST_CASE("RGB to gray: Error - Wrong dtype", "[image_rgb_to_gray]") {
    SECTION("Float32 instead of Uint8") {
        auto rgb = Tensor::zeros(make_shape(10, 10, 3), Dtype::Float32).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }

    SECTION("Int32 instead of Uint8") {
        auto rgb = Tensor::zeros(make_shape(10, 10, 3), Dtype::Int32).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }

    SECTION("Float64 instead of Uint8") {
        auto rgb = Tensor::zeros(make_shape(10, 10, 3), Dtype::Float64).unwrap();
        Tensor gray;
        REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsError(P10Error::InvalidArgument));
    }
}

TEST_CASE("RGB to gray: Gradient pattern", "[image_rgb_to_gray]") {
    auto rgb = Tensor::zeros(make_shape(16, 16, 3), Dtype::Uint8).unwrap();
    auto rgb_span = rgb.as_span3d<uint8_t>().unwrap();

    for (size_t h = 0; h < rgb_span.height(); ++h) {
        for (size_t w = 0; w < rgb_span.width(); ++w) {
            auto pixel = rgb_span.channel(h, w);
            pixel[0] = static_cast<uint8_t>((h * 255) / 15);  // Vertical gradient
            pixel[1] = static_cast<uint8_t>((w * 255) / 15);  // Horizontal gradient
            pixel[2] = 128;  // Constant
        }
    }

    Tensor gray;
    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());
    REQUIRE(gray.shape() == make_shape(16, 16));

    // Verify some values
    auto gray_span = gray.as_span2d<uint8_t>().unwrap();

    float expected_tl = 0.21f * 0 + 0.72f * 0 + 0.07f * 128;
    REQUIRE(gray_span.row(0)[0] == Approx(expected_tl).margin(1));

    float expected_br = 0.21f * 255 + 0.72f * 255 + 0.07f * 128;
    REQUIRE(gray_span.row(15)[15] == Approx(expected_br).margin(1));
}

TEST_CASE("RGB to gray: Output tensor reuse", "[image_rgb_to_gray]") {
    // Test that the function properly handles reusing the output tensor
    auto rgb = Tensor::zeros(make_shape(10, 10, 3), Dtype::Uint8).unwrap();
    Tensor gray;

    REQUIRE_THAT(image_rgb_to_gray(rgb, gray), IsOk());
    REQUIRE(gray.shape() == make_shape(10, 10));

    auto rgb2 = Tensor::zeros(make_shape(20, 20, 3), Dtype::Uint8).unwrap();
    REQUIRE_THAT(image_rgb_to_gray(rgb2, gray), IsOk());
    REQUIRE(gray.shape() == make_shape(20, 20));
}

}  // namespace p10::op
