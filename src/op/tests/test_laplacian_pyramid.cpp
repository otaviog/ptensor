#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/op/image_layout.hpp>
#include <ptensor/op/laplacian_pyramid.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

#include "testing.hpp"

namespace p10::op {

TEST_CASE("op: laplacian pyramid decompose and reconstruct", "[imageop][laplacian][integration]") {
    constexpr size_t PYRAMID_LEVELS = 4;
    const auto [sample_image, image_file] = testing::samples::image01();

    Tensor sample_tensor;
    REQUIRE(op::image_to_tensor(sample_image, sample_tensor) == P10Error::Ok);

    auto lp_process = LaplacianPyramid::create(5).expect("Can't create pyramid");

    std::vector<Tensor> pyramid(PYRAMID_LEVELS);
    REQUIRE(lp_process.transform(sample_tensor, pyramid).is_ok());

    const auto test_output_path = testing::get_output_path("op/laplacian_pyramid");
    // Save each level for eyeballing next to the other op outputs.
    for (size_t level = 0; level < PYRAMID_LEVELS; ++level) {
        Tensor level_image;
        REQUIRE(op::image_from_tensor(pyramid[level], level_image) == P10Error::Ok);
        REQUIRE(
            io::save_image(
                (test_output_path
                 / testing::suffixed(image_file, "laplacian-level-" + std::to_string(level)))
                    .string(),
                level_image
            )
                .is_ok()
        );
    }

    // Reconstruction collapses the pyramid back to the source image.
    Tensor reconstructed;
    REQUIRE(LaplacianPyramid::reconstruct(pyramid, reconstructed).is_ok());

    Tensor reconstructed_image;
    REQUIRE(op::image_from_tensor(reconstructed, reconstructed_image) == P10Error::Ok);
    REQUIRE(
        io::save_image(
            (test_output_path / testing::suffixed(image_file, "laplacian-rec")).string(),
            reconstructed_image
        )
            .is_ok()
    );

    // A laplacian pyramid reconstructs the original up to float rounding: each
    // level stores the residual the next level cannot represent, so summing
    // them back recovers the source. Compare in float space (before the uint8
    // quantization) with a small tolerance for accumulated blur/resample error.
    REQUIRE_THAT(
        testing::compare_tensors(
            sample_tensor,
            reconstructed,
            testing::CompareOptions().tolerance(1e-3)
        ),
        testing::is_ok()
    );
}

}  // namespace p10::op
