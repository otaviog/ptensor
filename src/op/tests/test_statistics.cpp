#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/op/statistics.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {
using Catch::Approx;

TEST_CASE("statistics::mean over from_range", "[op][statistics]") {
    auto dtype = GENERATE(Dtype::Float32, Dtype::Float64, Dtype::Int32, Dtype::Uint8);
    DYNAMIC_SECTION("dtype " << to_string(dtype)) {
        auto tensor = Tensor::from_range(make_shape(2, 3), dtype).unwrap();
        // values 0..5, mean = 2.5
        REQUIRE(mean(tensor) == Approx(2.5));
    }
}

TEST_CASE("statistics::min and max return value+index", "[op][statistics]") {
    auto tensor = Tensor::from_range(make_shape(4), Dtype::Float32).unwrap();
    tensor.visit([](auto span) {
        using scalar_t = typename decltype(span)::value_type;
        span[0] = scalar_t(3);
        span[1] = scalar_t(-1);
        span[2] = scalar_t(7);
        span[3] = scalar_t(2);
    });

    const auto [min_value, min_index] = min(tensor);
    REQUIRE(min_value == Approx(-1.0));
    REQUIRE(min_index == 1);

    const auto [max_value, max_index] = max(tensor);
    REQUIRE(max_value == Approx(7.0));
    REQUIRE(max_index == 2);
}

TEST_CASE("statistics on empty tensor returns NaN", "[op][statistics]") {
    Tensor empty;
    REQUIRE(std::isnan(mean(empty)));
    REQUIRE(std::isnan(min(empty).first));
    REQUIRE(std::isnan(max(empty).first));
}

}  // namespace p10::op
