#include <filesystem>
#include <map>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ptensor/io/numpy.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "catch2/matchers/catch_matchers.hpp"

namespace p10::io {

TEST_CASE("io::numpy::save_npz and load_npz with single tensor", "[io][numpy]") {
    const std::string filename = "test_single.npz";

    // Create a test tensor
    auto tensor =
        Tensor::full(make_shape({3, 4}).unwrap(), 1.0, TensorOptions().dtype(Dtype::Float32))
            .unwrap();

    // Save to npz
    std::map<std::string, Tensor> tensors_to_save;
    tensors_to_save["data"] = tensor.clone().unwrap();
    auto save_err = save_npz(filename, tensors_to_save);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    // Load from npz
    auto loaded_result = load_npz(filename);
    REQUIRE_THAT(loaded_result, testing::IsOk());

    auto loaded_tensors = loaded_result.unwrap();
    REQUIRE(loaded_tensors.size() == 1);
    REQUIRE(loaded_tensors.count("data") == 1);

    auto& loaded_tensor = loaded_tensors["data"];
    REQUIRE(loaded_tensor.shape() == tensor.shape());
    REQUIRE(loaded_tensor.dtype() == tensor.dtype());

    // Verify data
    auto original_data = tensor.as_span1d<float>().unwrap();
    auto loaded_data = loaded_tensor.as_span1d<float>().unwrap();
    for (size_t i = 0; i < original_data.size(); i++) {
        REQUIRE(loaded_data[i] == original_data[i]);
    }

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("io::numpy::save_npz and load_npz with multiple tensors", "[io][numpy]") {
    const std::string filename = "test_multiple.npz";

    // Create test tensors
    auto tensor1 = Tensor::full(make_shape({2, 3}).unwrap(), 1.0).unwrap();
    auto tensor2 = Tensor::full(make_shape({4, 5}).unwrap(), 42.0f).unwrap();
    auto tensor3 =
        Tensor::full(make_shape({10}).unwrap(), 5, TensorOptions().dtype(Dtype::Uint8)).unwrap();

    // Save to npz
    std::map<std::string, Tensor> tensors_to_save;
    tensors_to_save["ones"] = tensor1.clone().unwrap();
    tensors_to_save["forties"] = tensor2.clone().unwrap();
    tensors_to_save["bytes"] = tensor3.clone().unwrap();
    auto save_err = save_npz(filename, tensors_to_save);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    // Load from npz
    auto loaded_result = load_npz(filename);
    REQUIRE_THAT(loaded_result, testing::IsOk());

    auto loaded_tensors = loaded_result.unwrap();
    REQUIRE(loaded_tensors.size() == 3);

    // Verify each tensor
    SECTION("verify ones tensor") {
        REQUIRE(loaded_tensors.count("ones") == 1);
        auto& loaded = loaded_tensors["ones"];
        REQUIRE(loaded.shape() == tensor1.shape());
        REQUIRE(loaded.dtype() == tensor1.dtype());

        auto data = loaded.as_span1d<float>().unwrap();
        for (size_t i = 0; i < data.size(); i++) {
            REQUIRE(data[i] == 1.0f);
        }
    }

    SECTION("verify forties tensor") {
        REQUIRE(loaded_tensors.count("forties") == 1);
        auto& loaded = loaded_tensors["forties"];
        REQUIRE(loaded.shape() == tensor2.shape());
        REQUIRE(loaded.dtype() == tensor2.dtype());

        auto data = loaded.as_span1d<float>().unwrap();
        for (size_t i = 0; i < data.size(); i++) {
            REQUIRE(data[i] == 42.0f);
        }
    }

    SECTION("verify bytes tensor") {
        REQUIRE(loaded_tensors.count("bytes") == 1);
        auto& loaded = loaded_tensors["bytes"];
        REQUIRE(loaded.shape() == tensor3.shape());
        REQUIRE(loaded.dtype() == tensor3.dtype());

        auto data = loaded.as_span1d<uint8_t>().unwrap();
        for (size_t i = 0; i < data.size(); i++) {
            REQUIRE(data[i] == static_cast<uint8_t>(5));
        }
    }

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("io::numpy::save_npz with different data types", "[io][numpy]") {
    const std::string filename = "test_dtypes.npz";

    auto float32_tensor =
        Tensor::full(make_shape({2, 2}).unwrap(), 3.14f, TensorOptions().dtype(Dtype::Float32))
            .unwrap();
    auto uint8_tensor =
        Tensor::full(make_shape({2, 3}).unwrap(), 3, TensorOptions().dtype(Dtype::Uint8)).unwrap();

    std::map<std::string, Tensor> tensors_to_save;
    tensors_to_save["float32"] = std::move(float32_tensor);
    tensors_to_save["uint8"] = std::move(uint8_tensor);

    auto save_err = save_npz(filename, tensors_to_save);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    auto loaded_result = load_npz(filename);
    REQUIRE_THAT(loaded_result, p10::testing::IsOk());

    auto loaded_tensors = loaded_result.unwrap();
    REQUIRE(loaded_tensors.size() == 2);

    // Verify float32
    auto& loaded_float32 = loaded_tensors["float32"];
    REQUIRE(loaded_float32.dtype() == Dtype::Float32);
    auto float32_data = loaded_float32.as_span1d<float>().unwrap();
    for (size_t i = 0; i < float32_data.size(); i++) {
        REQUIRE(float32_data[i] == Catch::Approx(3.14f));
    }

    // Verify uint8
    auto& loaded_uint8 = loaded_tensors["uint8"];
    REQUIRE(loaded_uint8.dtype() == Dtype::Uint8);
    auto uint8_data = loaded_uint8.as_span1d<uint8_t>().unwrap();
    for (size_t i = 0; i < uint8_data.size(); i++) {
        REQUIRE(uint8_data[i] == static_cast<uint8_t>(3));
    }

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("io::numpy::load_npz with non-existent file returns error", "[io][numpy]") {
    auto result = load_npz("non_existent_file.npz");
    REQUIRE(!result.is_ok());
}

TEST_CASE("io::numpy::save_npz creates valid file", "[io][numpy]") {
    const std::string filename = "test_file_exists.npz";

    auto tensor = Tensor::zeros(make_shape({2, 2}).unwrap()).unwrap();
    std::map<std::string, Tensor> tensors_to_save;
    tensors_to_save["test"] = std::move(tensor);

    auto save_err = save_npz(filename, tensors_to_save);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    // Check file exists
    REQUIRE(std::filesystem::exists(filename));

    // Cleanup
    std::filesystem::remove(filename);
}

}  // namespace p10::io
