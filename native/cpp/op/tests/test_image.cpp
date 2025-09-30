#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/tensor.hpp>

#include "imageop/image_to_tensor.hpp"
#include "imageop/resize_image.hpp"
#include "testing.hpp"

namespace p10::op {

TEST_CASE("Image to tensor", "[imageop]") {
    Tensor image_tensor = Tensor::zeros(DType::UINT8, {256, 256, 3});
    Tensor float_tensor;
    image_to_tensor(image_tensor, float_tensor);
    REQUIRE(float_tensor.shape() == std::vector<int64_t>({3, 256, 256}));
    REQUIRE(float_tensor.dtype() == DType::FLOAT32);
}

TEST_CASE("Image to tensor with invalid type", "[imageop]") {
    Tensor image_tensor = Tensor::zeros(DType::FLOAT32, {256, 256, 3});
    Tensor float_tensor;
    // REQUIRE(image_to_tensor(image_tensor, float_tensor) ==
    //         PtensorError::INVALID_ARGUMENT);
}

TEST_CASE("Image to tensor with invalid shape", "[imageop]") {
    Tensor image_tensor = Tensor::zeros(DType::UINT8, {256, 256, 2});
    Tensor float_tensor;
    // REQUIRE(image_to_tensor(image_tensor, float_tensor) ==
    //       PtensorError::INVALID_ARGUMENT);
}

TEST_CASE("Resize image with bilinear", "[imageop]") {
    auto sample_image = testing::samples::image01();
    auto mode = GENERATE(
        std ::make_pair(InterpolationMethod::NEAREST, "nearest"),
        std::make_pair(InterpolationMethod::BILINEAR, "bilinear")
    );
    DYNAMIC_SECTION("Testing resize with method " << mode.second) {
        ResizeImage resize_image(mode.first);

        SECTION("Downsample image") {
            Tensor resized_image;
            resize_image(sample_image, resized_image, 128, 128);
            REQUIRE(resized_image.shape() == std::vector<int64_t>({128, 128, 3}));
            REQUIRE(resized_image.dtype() == DType::UINT8);
            io::save_image(
                (testing::get_output_path()
                 / (std::string("000001_downsampled_") + mode.second + ".jpg"))
                    .string(),
                resized_image
            );
        }
        SECTION("Upsample image") {
            Tensor resized_image;
            resize_image(sample_image, resized_image, 1024, 1024);
            REQUIRE(resized_image.shape() == std::vector<int64_t>({1024, 1024, 3}));
            io::save_image(
                (testing::get_output_path()
                 / (std::string("000001_upsampled_") + mode.second + ".jpg"))
                    .string(),
                resized_image
            );
        }
    }
}

}  // namespace p10::op