#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <p10/io/image.hpp>
#include <p10/tensor.hpp>

#include "health/laplacian_pyramid.hpp"
#include "imageop/image_to_tensor.hpp"
#include "testing.hpp"

namespace p10::health {

TEST_CASE("health: Create laplacian pyramid", "[health]") {
    constexpr size_t PYRAMID_LEVELS = 4;

    Tensor sample_tensor;
    imageop::image_to_tensor(testing::samples::image01(), sample_tensor);

    auto lp_process =
        LaplacianPyramid::create(5).expect("Can't create pyramid");

    std::vector<Tensor> pyramid(PYRAMID_LEVELS);
    REQUIRE(lp_process.process(sample_tensor, pyramid).is_ok());

    Tensor reconstructed, reconstructed_image;
    lp_process.reconstruct(pyramid, reconstructed);
    REQUIRE(reconstructed.shape(0) == sample_tensor.shape(0));
    REQUIRE(reconstructed.shape(1) == sample_tensor.shape(1));
    REQUIRE(reconstructed.shape(2) == sample_tensor.shape(2));

    imageop::image_from_tensor(reconstructed, reconstructed_image);
    REQUIRE(io::save_image(
                (testing::get_output_path() / "000001_laplacian_recontruct.jpg")
                    .string(),
                reconstructed_image)
                .is_ok());
}

}  // namespace p10::health