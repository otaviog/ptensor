
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/image_layout.hpp>
#include <ptensor/op/resize.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

#include "testing.hpp"

namespace p10::op {

TEST_CASE("Tensorop: Resize", "[tensorop][resize][integration]") {
    auto [sample_image, image_file] = testing::samples::image01();
    Tensor sample_tensor;
    REQUIRE(image_to_tensor(sample_image, sample_tensor) == P10Error::Ok);
    const int64_t source_height = sample_tensor.shape(1).unwrap();
    const int64_t source_width = sample_tensor.shape(2).unwrap();
    const auto test_output_path = testing::get_output_path("op/resize");

    SECTION("Downsample image") {
        Tensor resized_tensor;
        REQUIRE(resize(sample_tensor, resized_tensor, 128, 128).is_ok());
        REQUIRE(resized_tensor.shape() == make_shape(3, 128, 128));
        REQUIRE(resized_tensor.dtype() == Dtype::Float32);

        Tensor resized_image;
        REQUIRE(image_from_tensor(resized_tensor, resized_image) == P10Error::Ok);
        REQUIRE(
            io::save_image(
                (test_output_path / testing::suffixed(image_file, "downsample-nearest")).string(),
                resized_image
            )
                .is_ok()
        );
    }

    SECTION("Upsample image") {
        Tensor resized_tensor;
        REQUIRE(resize(sample_tensor, resized_tensor, 1024, 1024).is_ok());
        REQUIRE(resized_tensor.shape() == make_shape(3, 1024, 1024));
        REQUIRE(resized_tensor.dtype() == Dtype::Float32);

        Tensor resized_image;
        REQUIRE(image_from_tensor(resized_tensor, resized_image) == P10Error::Ok);
        REQUIRE(
            io::save_image(
                (test_output_path / testing::suffixed(image_file, "upsample-nearest")).string(),
                resized_image
            )
                .is_ok()
        );
    }

    SECTION("Resizing to the same size is an exact copy") {
        // Nearest-neighbor at a 1:1 ratio maps every output pixel to the same
        // source pixel, so the result is bit-identical to the input.
        Tensor resized_tensor;
        REQUIRE(
            resize(sample_tensor, resized_tensor, source_width, source_height).is_ok()
        );
        REQUIRE_THAT(
            testing::compare_tensors(sample_tensor, resized_tensor),
            testing::is_ok()
        );
    }

    SECTION("Resizing a constant image keeps the constant") {
        // Nearest sampling only copies existing pixels, so a uniform image stays
        // uniform at any target size.
        const auto constant = Tensor::full(make_shape(3, 32, 32), 0.5).unwrap();
        Tensor resized_tensor;
        REQUIRE(resize(constant, resized_tensor, 64, 48).is_ok());
        REQUIRE(resized_tensor.shape() == make_shape(3, 48, 64));

        const auto expected = Tensor::full(make_shape(3, 48, 64), 0.5).unwrap();
        REQUIRE_THAT(
            testing::compare_tensors(expected, resized_tensor),
            testing::is_ok()
        );
    }
}

}  // namespace p10::op
