#include <filesystem>
#include <map>

#include <catch2/catch_test_macros.hpp>
#include <ptensor/io/numpy.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include "catch2/matchers/catch_matchers.hpp"

namespace p10::io {

TEST_CASE("save_npz and load_npz with single tensor", "[io][numpy]") {
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

TEST_CASE("save_npz and load_npz with multiple tensors", "[io][numpy]") {
    const std::string filename = "test_multiple.npz";

    // Create test tensors
    auto tensor1 = Tensor::full(make_shape({2, 3}).unwrap(), 1.0).unwrap();
    auto tensor2 = Tensor::full(make_shape({4, 5}).unwrap(), 42.0f).unwrap();
    auto tensor3 =
        Tensor::full(make_shape({10}).unwrap(), 5, TensorOptions().dtype(Dtype::Int32)).unwrap();

    // Save to npz
    std::map<std::string, Tensor> tensors_to_save;
    tensors_to_save["ones"] = tensor1.clone().unwrap();
    tensors_to_save["forties"] = tensor2.clone().unwrap();
    tensors_to_save["range"] = tensor3.clone().unwrap();
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

    SECTION("verify range tensor") {
        SKIP("Skipping: needs to change cnpy to support more dtypes");
        REQUIRE(loaded_tensors.count("range") == 1);
        auto& loaded = loaded_tensors["range"];
        REQUIRE(loaded.shape() == tensor3.shape());
        REQUIRE(loaded.dtype() == tensor3.dtype());

        auto data = loaded.as_span1d<int32_t>().unwrap();
        for (size_t i = 0; i < data.size(); i++) {
            REQUIRE(data[i] == static_cast<int32_t>(i));
        }
    }

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("save_npz with different data types", "[io][numpy]") {
    SKIP("Skipping: needs to change cnpy to support more dtypes");
    const std::string filename = "test_dtypes.npz";

    auto float64_tensor =
        Tensor::full(make_shape({2, 2}).unwrap(), 3.14, TensorOptions().dtype(Dtype::Float64))
            .unwrap();
    auto int32_tensor =
        Tensor::full(make_shape({2, 3}).unwrap(), 3, TensorOptions().dtype(Dtype::Int32)).unwrap();

    std::map<std::string, Tensor> tensors_to_save;
    tensors_to_save["float64"] = std::move(float64_tensor);
    tensors_to_save["int32"] = std::move(int32_tensor);

    auto save_err = save_npz(filename, tensors_to_save);
    REQUIRE_THAT(save_err, p10::testing::IsOk());

    auto loaded_result = load_npz(filename);
    REQUIRE_THAT(loaded_result, p10::testing::IsOk());

    auto loaded_tensors = loaded_result.unwrap();
    REQUIRE(loaded_tensors.size() == 2);

    // Verify float64
    auto& loaded_float64 = loaded_tensors["float64"];
    REQUIRE(loaded_float64.dtype() == Dtype::Float64);
    auto float64_data = loaded_float64.as_span1d<double>().unwrap();
    for (size_t i = 0; i < float64_data.size(); i++) {
        REQUIRE(float64_data[i] == 3.14);
    }

    // Verify int32
    auto& loaded_int32 = loaded_tensors["int32"];
    REQUIRE(loaded_int32.dtype() == Dtype::Int32);
    auto int32_data = loaded_int32.as_span1d<int32_t>().unwrap();
    for (size_t i = 0; i < int32_data.size(); i++) {
        REQUIRE(int32_data[i] == static_cast<int32_t>(i));
    }

    // Cleanup
    std::filesystem::remove(filename);
}

TEST_CASE("load_npz with non-existent file returns error", "[io][numpy]") {
    auto result = load_npz("non_existent_file.npz");
    REQUIRE(!result.is_ok());
}

TEST_CASE("save_npz creates valid file", "[io][numpy]") {
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
