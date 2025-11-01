#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/op/elemwise.hpp>
#include <ptensor/op/image.hpp>
#include <ptensor/op/resize.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {
using Catch::Approx;

TEST_CASE("Tensorop: Add", "[tensorop]") {
    auto type = GENERATE(Dtype::Float32, Dtype::Int64, Dtype::Uint8);
    DYNAMIC_SECTION("Testing addition with type " << to_string(type)) {
        Tensor out;
        auto a = Tensor::from_range(make_shape(2, 3), type).unwrap();
        auto b = Tensor::from_range(make_shape(2, 3), type).unwrap();

        REQUIRE(add_elemwise(a, b, out).is_ok());
        REQUIRE(out.shape(0).unwrap() == 2);
        REQUIRE(out.shape(1).unwrap() == 3);
        REQUIRE(out.size() == 6);
        REQUIRE(out.dtype() == type);

        out.visit([](auto span) {
            using SpanType = decltype(span)::value_type;
            for (int i = 0; i < 6; ++i) {
                REQUIRE(span[i] == Approx(static_cast<SpanType>(i * 2)));
            }
        });
    }
}

TEST_CASE("Tensorop: Subtract", "[tensor]") {
    auto type = GENERATE(Dtype::Float32, Dtype::Int64, Dtype::Uint8);
    DYNAMIC_SECTION("Testing subtraction with type " << to_string(type)) {
        Tensor out;
        auto a = Tensor::from_range(make_shape(2, 3), type).unwrap();
        auto b = Tensor::from_range(make_shape(2, 3), type).unwrap();

        REQUIRE(subtract_elemwise(a, b, out).is_ok());
        REQUIRE(out.shape(0).unwrap() == 2);
        REQUIRE(out.shape(1).unwrap() == 3);
        REQUIRE(out.size() == 6);
        REQUIRE(out.dtype() == type);

        out.visit([](auto span) {
            for (int i = 0; i < 6; ++i) {
                REQUIRE(span[i] == Approx(0.0));
            }
        });
    }
}

}  // namespace p10::op
