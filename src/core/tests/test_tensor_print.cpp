#include <cmath>
#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/tensor_print.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

namespace p10 {

// ============================================================================
// Tensor Creation Tests
// ============================================================================

TEST_CASE("Tensor::prints different shapes", "[tensor][print]") {
    SECTION("Prints empty tensor") {
        auto tensor = Tensor();
        std::string tensor_str = to_string(tensor);

        REQUIRE(tensor_str == "Tensor(shape=[], dtype=float32, values=[])");
    }

    SECTION("Tensor::prints [2x3]", "[tensor][print]") {
        auto tensor = Tensor::full(make_shape(2, 3), 3.0).expect("Could not create tensor");
        std::string tensor_str = to_string(tensor);

        REQUIRE(
            tensor_str
            == "Tensor(shape=[2, 3], dtype=float32, values=[[3.0000, 3.0000, 3.0000],\n[3.0000, 3.0000, 3.0000]])"
        );
    }

    SECTION("Tensor::prints [2x2x3]", "[tensor][print]") {
        auto tensor = Tensor::from_range(make_shape(2, 2, 3)).expect("Could not create tensor");
        std::string tensor_str = to_string(tensor);

        REQUIRE(
            tensor_str
            == "Tensor(shape=[2, 2, 3], dtype=float32, values=[[[0.0000, 1.0000, 2.0000],\n[3.0000, 4.0000, 5.0000]],\n[[6.0000, 7.0000, 8.0000],\n[9.0000, 10.0000, 11.0000]]])"
        );
    }
}

TEST_CASE("Tensor::prints handles truncation", "[tensor][print]") {
    auto tensor = Tensor::from_range(make_shape(20)).expect("Could not create tensor");
    SECTION("Truncate large tensor") {
        std::string tensor_str =
            to_string(tensor, TensorStringOptions().max_elements(10).float_precision(2));

        REQUIRE(
            tensor_str
            == "Tensor(shape=[20], dtype=float32, values=[0.00, 1.00, 2.00, 3.00, 4.00, ..., 15.00, 16.00, 17.00, 18.00, 19.00])"
        );
    }

    SECTION("When max_element is 0") {
        std::string tensor_str =
            to_string(tensor, TensorStringOptions().max_elements(0).float_precision(2));

        REQUIRE(tensor_str == "Tensor(shape=[20], dtype=float32, values=[...])");
    }
}

TEST_CASE("Tensor::to_json emits parseable JSON", "[tensor][print][json]") {
    auto tensor = Tensor::from_range(make_shape(2, 3)).expect("Could not create tensor");
    const std::string json = to_json(tensor);

    // Spot-check the structure rather than the full base64 payload.
    REQUIRE(json.starts_with(R"({"dtype":"float32","shape":[2, 3],"stride":[3, 1],"blob":")"));
    REQUIRE(json.ends_with(R"("})"));

    const char* debug_ptr = to_json_debug(tensor);
    REQUIRE(debug_ptr != nullptr);
    REQUIRE(std::string(debug_ptr) == json);
}

TEST_CASE("Tensor stats debug helpers", "[tensor][debug][stats]") {
    auto tensor = Tensor::from_range(make_shape(2, 3)).expect("Could not create tensor");
    // Values are 0..5, so min=0, max=5, mean=2.5.
    REQUIRE(tensor_min_debug(tensor) == Catch::Approx(0.0));
    REQUIRE(tensor_max_debug(tensor) == Catch::Approx(5.0));
    REQUIRE(tensor_mean_debug(tensor) == Catch::Approx(2.5));

    Tensor const empty;
    REQUIRE(std::isnan(tensor_min_debug(empty)));
    REQUIRE(std::isnan(tensor_max_debug(empty)));
    REQUIRE(std::isnan(tensor_mean_debug(empty)));
}

TEST_CASE("Tensor::prints handles float precision", "[tensor][print]") {
    auto tensor = Tensor::from_range(make_shape(3)).expect("Could not create tensor");
    tensor.visit([](auto span) {
        using SpanType = decltype(span)::value_type;
        span[0] = SpanType(1.123456);
        span[1] = SpanType(2.123456);
        span[2] = SpanType(3.123456);
    });

    SECTION("Default precision") {
        std::string tensor_str = to_string(tensor);

        REQUIRE(tensor_str == "Tensor(shape=[3], dtype=float32, values=[1.1235, 2.1235, 3.1235])");
    }

    SECTION("Precision 2") {
        std::string tensor_str = to_string(tensor, TensorStringOptions().float_precision(2));

        REQUIRE(tensor_str == "Tensor(shape=[3], dtype=float32, values=[1.12, 2.12, 3.12])");
    }

    SECTION("Precision 4") {
        std::string tensor_str = to_string(tensor, TensorStringOptions().float_precision(4));

        REQUIRE(tensor_str == "Tensor(shape=[3], dtype=float32, values=[1.1235, 2.1235, 3.1235])");
    }
}

}  // namespace p10