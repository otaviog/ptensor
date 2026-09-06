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

TEST_CASE("statistics::mean over an axis", "[op][statistics]") {
    // values 0..5 laid out as [[0, 1, 2], [3, 4, 5]]
    auto tensor = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();

    SECTION("reduce axis 0") {
        Tensor reduced;
        REQUIRE(mean(tensor, 0, reduced).is_ok());
        REQUIRE(reduced.shape() == make_shape(3));
        REQUIRE(reduced.dtype() == Dtype::Float64);
        auto span = reduced.as_span1d<double>().unwrap();
        REQUIRE(span[0] == Approx(1.5));  // (0 + 3) / 2
        REQUIRE(span[1] == Approx(2.5));  // (1 + 4) / 2
        REQUIRE(span[2] == Approx(3.5));  // (2 + 5) / 2
    }

    SECTION("reduce axis 1") {
        Tensor reduced;
        REQUIRE(mean(tensor, 1, reduced).is_ok());
        REQUIRE(reduced.shape() == make_shape(2));
        auto span = reduced.as_span1d<double>().unwrap();
        REQUIRE(span[0] == Approx(1.0));  // (0 + 1 + 2) / 3
        REQUIRE(span[1] == Approx(4.0));  // (3 + 4 + 5) / 3
    }

    SECTION("negative axis indexes from the end") {
        Tensor reduced;
        REQUIRE(mean(tensor, -1, reduced).is_ok());
        REQUIRE(reduced.shape() == make_shape(2));
    }

    SECTION("out of range axis is an error") {
        Tensor reduced;
        REQUIRE(mean(tensor, 2, reduced).is_error());
    }
}

TEST_CASE("statistics::mean over an axis of a 3D tensor", "[op][statistics]") {
    // values 0..23 laid out as shape [2, 3, 4]
    auto tensor = Tensor::from_range(make_shape(2, 3, 4), Dtype::Float32).unwrap();

    SECTION("reduce middle axis 1") {
        Tensor reduced;
        REQUIRE(mean(tensor, 1, reduced).is_ok());
        REQUIRE(reduced.shape() == make_shape(2, 4));
        auto span = reduced.as_span1d<double>().unwrap();
        // out[0,0] = mean(0, 4, 8) = 4 ; out[1,3] = mean(15, 19, 23) = 19
        REQUIRE(span[0] == Approx(4.0));
        REQUIRE(span[7] == Approx(19.0));
    }
}

TEST_CASE("statistics::min and max return value+index", "[op][statistics]") {
    auto tensor = Tensor::from_range(make_shape(4), Dtype::Float32).unwrap();
    tensor.visit([](auto span) {
        using scalar_t = decltype(span)::value_type;
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
    Tensor const empty;
    REQUIRE(std::isnan(mean(empty)));
    REQUIRE(std::isnan(min(empty).first));
    REQUIRE(std::isnan(max(empty).first));
}

}  // namespace p10::op
