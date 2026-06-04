#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/crop.hpp>
#include <ptensor/op/image_layout.hpp>

#include "testing.hpp"

namespace p10::op {
TEST_CASE("Op: Crop image", "[tensorop]") {
    auto [sample_image, image_file] = testing::samples::image01();
    Tensor sample_tensor;

    op::image_to_tensor(sample_image, sample_tensor);

    SECTION("Apply crop operator") {
        Tensor cropped_tensor;
        op::crop(sample_tensor, 10, 10, 50, 50, cropped_tensor).expect("Failed to crop image");

        REQUIRE(cropped_tensor.shape() == make_shape(3, 50, 50));

        Tensor cropped_image;
        op::image_from_tensor(cropped_tensor, cropped_image);
        io::save_image(
            (testing::get_output_path() / testing::suffixed(image_file, "crop")).string(),
            cropped_image
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
