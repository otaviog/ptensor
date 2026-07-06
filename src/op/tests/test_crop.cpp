#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/crop.hpp>
#include <ptensor/op/image_layout.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

#include "testing.hpp"

namespace p10::op {
TEST_CASE("Op: Crop image", "[tensorop][crop][integration]") {
    auto [sample_image, image_file] = testing::samples::image01();
    Tensor sample_tensor;
    REQUIRE(op::image_to_tensor(sample_image, sample_tensor) == P10Error::Ok);
    const int64_t source_height = sample_tensor.shape(1).unwrap();
    const int64_t source_width = sample_tensor.shape(2).unwrap();
    const auto test_output_path = testing::get_output_path("op/crop");

    SECTION("Apply crop operator") {
        Tensor cropped_tensor;
        REQUIRE(op::crop(sample_tensor, 10, 10, 50, 50, cropped_tensor).is_ok());
        REQUIRE(cropped_tensor.shape() == make_shape(3, 50, 50));

        Tensor cropped_image;
        REQUIRE(op::image_from_tensor(cropped_tensor, cropped_image) == P10Error::Ok);
        REQUIRE(
            io::save_image(
                (test_output_path / testing::suffixed(image_file, "crop")).string(),
                cropped_image
            )
                .is_ok()
        );
    }

    SECTION("Cropping the whole image is an exact copy") {
        // A full-extent crop copies every pixel unchanged.
        Tensor cropped_tensor;
        REQUIRE(
            op::crop(sample_tensor, 0, 0, source_width, source_height, cropped_tensor).is_ok()
        );
        REQUIRE_THAT(
            testing::compare_tensors(sample_tensor, cropped_tensor),
            testing::is_ok()
        );
    }

    SECTION("Should fail with invalid crop parameters") {
        Tensor cropped_tensor;
        REQUIRE(op::crop(sample_tensor, -1, 0, 50, 50, cropped_tensor).is_error());
        REQUIRE(op::crop(sample_tensor, 0, -1, 50, 50, cropped_tensor).is_error());
        REQUIRE(op::crop(sample_tensor, 0, 0, 0, 5000, cropped_tensor).is_error());
        REQUIRE(op::crop(sample_tensor, 0, 0, 5000, 0, cropped_tensor).is_error());
    }
}
}  // namespace p10::op
