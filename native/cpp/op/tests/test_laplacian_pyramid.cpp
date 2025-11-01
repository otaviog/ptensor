#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/op/image.hpp>
#include <ptensor/op/laplacian_pyramid.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/tensor.hpp>

#include "testing.hpp"

namespace p10::op {

TEST_CASE("op: Create laplacian pyramid", "[health]") {
    constexpr size_t PYRAMID_LEVELS = 4;
    const auto [sample_image, image_file] = testing::samples::image01();

    Tensor sample_tensor;
    op::image_to_tensor(sample_image, sample_tensor);

    auto lp_process = LaplacianPyramid::create(5).expect("Can't create pyramid");

    std::vector<Tensor> pyramid(PYRAMID_LEVELS);
    REQUIRE(lp_process.transform(sample_tensor, pyramid).is_ok());

    Tensor reconstructed, reconstructed_image;
    lp_process.reconstruct(pyramid, reconstructed);
    REQUIRE(reconstructed.shape() == sample_tensor.shape());

    op::image_from_tensor(reconstructed, reconstructed_image);
    REQUIRE(
        io::save_image(
            (testing::get_output_path() / testing::suffixed(image_file, "laplacian-rec")).string(),
            reconstructed_image
        )
            .is_ok()
    );
}

}  // namespace p10::op
