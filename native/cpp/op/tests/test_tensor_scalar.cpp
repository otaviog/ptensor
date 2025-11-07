#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/op/tensor_scalar.hpp>
#include <ptensor/tensor.hpp>

namespace p10::op {
using Catch::Approx;

TEST_CASE("Tensorop: Multiply Scalar", "[tensorop]") {
    auto type = GENERATE(Dtype::Float32, Dtype::Int64, Dtype::Uint8);
    DYNAMIC_SECTION("Testing multiplication with type " << to_string(type)) {
        auto a = Tensor::from_range(make_shape(2, 3), type).unwrap();

        multiply_scalar(a, 2.0);

        a.visit([](auto span) {
            using SpanType = decltype(span)::value_type;
            for (int i = 0; i < 6; ++i) {
                REQUIRE(span[i] == Approx(static_cast<SpanType>(i * 2)));
            }
        });
    }
}
}  // namespace p10::op
