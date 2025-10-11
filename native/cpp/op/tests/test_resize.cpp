
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/image.hpp>
#include <ptensor/op/resize.hpp>

#include "testing.hpp"

namespace p10::op {
TEST_CASE("Tensorop: Resize", "[tensorop]") {
    auto [sample_image, image_file] = testing::samples::image01();
    Tensor sample_tensor;
    image_to_tensor(sample_image, sample_tensor);
    SECTION("Downsample image") {
        Tensor resized_tensor;
        REQUIRE(resize(sample_tensor, resized_tensor, 128, 128).is_ok());
        REQUIRE(resized_tensor.shape() == make_shape(3, 128, 128));
        REQUIRE(resized_tensor.dtype() == Dtype::Float32);
        Tensor resized_image;
        image_from_tensor(resized_tensor, resized_image);
        io::save_image(
            (testing::get_output_path() / testing::suffixed(image_file, "downsample-nearest")).string(),
            resized_image
        );
    }
    SECTION("Upsample image") {
        Tensor resized_tensor;
        REQUIRE(resize(sample_tensor, resized_tensor, 1024, 1024).is_ok());
        REQUIRE(resized_tensor.shape() == make_shape(3, 1024, 1024));
        REQUIRE(resized_tensor.dtype() == Dtype::Float32);
        Tensor resized_image;
        image_from_tensor(resized_tensor, resized_image);
        io::save_image(
            (testing::get_output_path() / testing::suffixed(image_file, "upsample-nearest.jpg")).string(),
            resized_image
        );
    }
}
}  // namespace p10::op
