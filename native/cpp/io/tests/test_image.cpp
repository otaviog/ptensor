#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/io/image.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "catch2/matchers/catch_matchers.hpp"

namespace p10::io {

TEST_CASE("save_image and load_image roundtrip", "[io][image]") {
    const std::string filename = "test_image.png";

    // Create a simple 8x8 RGB image (height, width, channels)
    auto tensor = Tensor::zeros(make_shape(8, 8, 3), TensorOptions().dtype(Dtype::Uint8)).unwrap();

    // Fill with some pattern
    auto data = tensor.as_span1d<uint8_t>().unwrap();
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = static_cast<uint8_t>(i % 256);
    }

    // Save image
    auto save_err = save_image(filename, tensor);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    // Load image
    auto loaded_result = load_image(filename);
    REQUIRE_THAT(loaded_result, testing::IsOk());

    auto loaded_tensor = loaded_result.unwrap();
    REQUIRE(loaded_tensor.shape() == tensor.shape());
    REQUIRE(loaded_tensor.dtype() == Dtype::Uint8);

    // Verify data (may have slight differences due to compression)
    auto loaded_data = loaded_tensor.as_span1d<uint8_t>().unwrap();
    REQUIRE(loaded_data.size() == data.size());

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("save_image with grayscale tensor", "[io][image]") {
    const std::string filename = "test_grayscale.png";

    // Create a grayscale image (height, width)
    auto tensor =
        Tensor::full(make_shape(16, 16), 128, TensorOptions().dtype(Dtype::Uint8)).unwrap();

    auto save_err = save_image(filename, tensor);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    auto loaded_result = load_image(filename);
    REQUIRE_THAT(loaded_result, p10::testing::IsOk());

    auto loaded_tensor = loaded_result.unwrap();
    REQUIRE(loaded_tensor.dims() >= 2);
    REQUIRE(loaded_tensor.shape()[0].unwrap() == 16);
    REQUIRE(loaded_tensor.shape()[1].unwrap() == 16);

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("save_image with RGBA tensor", "[io][image]") {
    const std::string filename = "test_rgba.png";

    // Create an RGBA image (height, width, 4 channels)
    auto tensor =
        Tensor::zeros(make_shape(10, 10, 4), TensorOptions().dtype(Dtype::Uint8)).unwrap();

    auto save_err = save_image(filename, tensor);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    REQUIRE(std::filesystem::exists(filename));

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("load_image with non-existent file returns error", "[io][image]") {
    auto result = load_image("non_existent_image.png");
    REQUIRE(!result.is_ok());
}

TEST_CASE("save_image creates valid file", "[io][image]") {
    const std::string filename = "test_save_creates_file.jpg";

    auto tensor =
        Tensor::full(make_shape(5, 5, 3), 1.0, TensorOptions().dtype(Dtype::Uint8)).unwrap();

    auto save_err = save_image(filename, tensor);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    REQUIRE(std::filesystem::exists(filename));

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("save_image with different formats", "[io][image]") {
    auto tensor =
        Tensor::full(make_shape(4, 4, 3), 200, TensorOptions().dtype(Dtype::Uint8)).unwrap();

    SECTION("PNG format") {
        const std::string filename = "test.png";
        auto err = save_image(filename, tensor);
        REQUIRE_THAT(err, p10::testing::IsOk());
        REQUIRE(std::filesystem::exists(filename));
        std::filesystem::remove(filename);
    }

    SECTION("JPEG format") {
        const std::string filename = "test.jpg";
        auto err = save_image(filename, tensor);
        REQUIRE_THAT(err, p10::testing::IsOk());
        REQUIRE(std::filesystem::exists(filename));
        std::filesystem::remove(filename);
    }

    SECTION("BMP format") {
        const std::string filename = "test.bmp";
        auto err = save_image(filename, tensor);
        REQUIRE_THAT(err, p10::testing::IsOk());
        REQUIRE(std::filesystem::exists(filename));
        std::filesystem::remove(filename);
    }
}

TEST_CASE("load_image returns correct dtype", "[io][image]") {
    const std::string filename = "test_dtype.png";

    auto tensor = Tensor::zeros(make_shape(6, 6, 3), TensorOptions().dtype(Dtype::Uint8)).unwrap();

    auto save_err = save_image(filename, tensor);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    auto loaded_result = load_image(filename);
    REQUIRE_THAT(loaded_result, testing::IsOk());

    auto loaded = loaded_result.unwrap();
    REQUIRE(loaded.dtype() == Dtype::Uint8);

    // Cleanup
    std::filesystem::remove(filename);
}

}  // namespace p10::io
