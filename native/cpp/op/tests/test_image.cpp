#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/image.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {

TEST_CASE("op::image::to tensor", "[imageop]") {
    Tensor image_tensor = Tensor::zeros(make_shape(256, 256, 3), Dtype::Uint8).unwrap();
    Tensor float_tensor;
    image_to_tensor(image_tensor, float_tensor);
    REQUIRE(float_tensor.shape() == make_shape(3, 256, 256));
    REQUIRE(float_tensor.dtype() == Dtype::Float32);
}

TEST_CASE("op::image::to tensor with invalid type", "[imageop]") {
    Tensor image_tensor = Tensor::zeros(make_shape(256, 256, 3), Dtype::Float32).unwrap();
    Tensor float_tensor;
    REQUIRE(image_to_tensor(image_tensor, float_tensor) == P10Error::InvalidArgument);
}

TEST_CASE("op::image::to tensor with invalid shape", "[imageop]") {
    Tensor image_tensor = Tensor::zeros(make_shape(256, 256), Dtype::Uint8).unwrap();
    Tensor float_tensor;
    REQUIRE(image_to_tensor(image_tensor, float_tensor) == P10Error::InvalidArgument);
}

TEST_CASE("op::image::from tensor", "[imageop]") {
    Tensor float_tensor = Tensor::zeros(make_shape(3, 256, 256), Dtype::Float32).unwrap();
    Tensor image_tensor;
    image_from_tensor(float_tensor, image_tensor);
    REQUIRE(image_tensor.shape() == make_shape(256, 256, 3));
    REQUIRE(image_tensor.dtype() == Dtype::Uint8);
}

TEST_CASE("op::image::from tensor with invalid type", "[imageop]") {
    Tensor float_tensor = Tensor::zeros(make_shape(3, 256, 256), Dtype::Uint8).unwrap();
    Tensor image_tensor;
    REQUIRE_THROWS_AS(image_from_tensor(float_tensor, image_tensor), P10Error);
}

TEST_CASE("op::image::from tensor with invalid shape", "[imageop]") {
    Tensor float_tensor = Tensor::zeros(make_shape(256, 256), Dtype::Float32).unwrap();
    Tensor image_tensor;
    REQUIRE_THROWS_AS(image_from_tensor(float_tensor, image_tensor), P10Error);
}

TEST_CASE("op::image::to tensor and back conversion", "[imageop]") {
    // Create a test image with known values
    Tensor original_image = Tensor::zeros(make_shape(64, 64, 3), Dtype::Uint8).unwrap();
    auto image_span = original_image.as_span3d<uint8_t>().unwrap();

    // Set some test values
    for (size_t row = 0; row < image_span.height(); row++) {
        for (size_t col = 0; col < image_span.width(); col++) {
            auto channel = image_span.channel(row, col);
            channel[0] = static_cast<uint8_t>(row % 256);
            channel[1] = static_cast<uint8_t>(col % 256);
            channel[2] = static_cast<uint8_t>((row + col) % 256);
        }
    }

    // Convert to tensor
    Tensor float_tensor;
    REQUIRE(image_to_tensor(original_image, float_tensor) == P10Error::Ok);

    // Convert back to image
    Tensor result_image;
    REQUIRE(image_from_tensor(float_tensor, result_image) == P10Error::Ok);

    // Verify dimensions
    REQUIRE(result_image.shape() == original_image.shape());
    REQUIRE(result_image.dtype() == original_image.dtype());

    // Verify values (allowing for small rounding errors)
    auto original_span = original_image.as_span3d<uint8_t>().unwrap();
    auto result_span = result_image.as_span3d<uint8_t>().unwrap();

    for (size_t row = 0; row < original_span.height(); row++) {
        for (size_t col = 0; col < original_span.width(); col++) {
            auto original_channel = original_span.channel(row, col);
            auto result_channel = result_span.channel(row, col);

            for (size_t c = 0; c < 3; c++) {
                // Allow for ±1 difference due to float conversion rounding
                REQUIRE(std::abs(int(original_channel[c]) - int(result_channel[c])) <= 1);
            }
        }
    }
}

TEST_CASE("op::image::from tensor value clamping", "[imageop]") {
    // Create a float tensor with values outside [0, 1] range
    Tensor float_tensor = Tensor::zeros(make_shape(3, 8, 8), Dtype::Float32).unwrap();
    auto tensor_span = float_tensor.as_planar_span3d<float>().unwrap();

    // Set values: channel 0 = -0.5 (below 0), channel 1 = 0.5 (normal), channel 2 = 1.5 (above 1)
    for (size_t row = 0; row < tensor_span.height(); row++) {
        for (size_t col = 0; col < tensor_span.width(); col++) {
            tensor_span[0][row][col] = -0.5f;
            tensor_span[1][row][col] = 0.5f;
            tensor_span[2][row][col] = 1.5f;
        }
    }

    // Convert to image
    Tensor image_tensor;
    REQUIRE(image_from_tensor(float_tensor, image_tensor) == P10Error::Ok);

    auto image_span = image_tensor.as_span3d<uint8_t>().unwrap();

    // Verify clamping
    for (size_t row = 0; row < image_span.height(); row++) {
        for (size_t col = 0; col < image_span.width(); col++) {
            auto channel = image_span.channel(row, col);
            REQUIRE(channel[0] == 0);  // -0.5 * 255 clamped to 0
            REQUIRE(channel[1] == 127);  // 0.5 * 255 = 127.5 ≈ 127
            REQUIRE(channel[2] == 255);  // 1.5 * 255 clamped to 255
        }
    }
}

}  // namespace p10::op
