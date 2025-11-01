#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/blur.hpp>
#include <ptensor/op/image.hpp>

#include "testing.hpp"

namespace p10::op {
TEST_CASE("Op: Blur image", "[tensorop]") {
    auto [sample_image, image_file] = testing::samples::image01();
    Tensor sample_tensor;

    op::image_to_tensor(sample_image, sample_tensor);
    SECTION("Apply blur operator") {
        auto blur_op = GaussianBlur::create(25, 1.5f).unwrap();
        Tensor blurred_tensor;
        blur_op.transform(sample_tensor, blurred_tensor).expect("Failed to blur image");

        REQUIRE(blurred_tensor.shape() == sample_tensor.shape());

        Tensor blurred_image;
        op::image_from_tensor(blurred_tensor, blurred_image);
        io::save_image(
            (testing::get_output_path() / testing::suffixed(image_file, "blur")).string(),
            blurred_image
        );
    }

    SECTION("Should fail with invalid kernel size") {
        REQUIRE(GaussianBlur::create(GaussianBlur::MAX_KERNEL_SIZE + 1, 1.5f).is_error());
        REQUIRE(GaussianBlur::create(0, 1.5f).is_error());
    }
}
}  // namespace p10::op
