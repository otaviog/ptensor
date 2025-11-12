#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <ptensor/op/stack.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

namespace p10::op {
using Catch::Approx;
using p10::testing::IsErr;
using p10::testing::IsOk;

TEST_CASE("Stack: Basic stacking along axis 0", "[stack]") {
    auto type = GENERATE(Dtype::Float32, Dtype::Int32, Dtype::Uint8);
    DYNAMIC_SECTION("Testing stack with type " << to_string(type)) {
        auto t1 = Tensor::from_range(make_shape(2, 3), type).unwrap();
        auto t2 = Tensor::from_range(make_shape(2, 3), type).unwrap();

        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;

        REQUIRE_THAT(stack(inputs, 0, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 2);
        REQUIRE(output.shape(2).unwrap() == 3);
        REQUIRE(output.dtype() == type);

        // Verify the stacked data
        output.visit([&](auto span) {
            using SpanType = decltype(span)::value_type;
            auto t1_span = t1.as_span1d<SpanType>().unwrap();
            auto t2_span = t2.as_span1d<SpanType>().unwrap();

            // First tensor along axis 0
            for (size_t i = 0; i < 6; ++i) {
                REQUIRE(span[i] == t1_span[i]);
            }
            // Second tensor along axis 0
            for (size_t i = 0; i < 6; ++i) {
                REQUIRE(span[6 + i] == t2_span[i]);
            }
        });
    }
}

TEST_CASE("Stack: Stacking along axis 1", "[stack]") {
    auto type = GENERATE(Dtype::Float32, Dtype::Int32);
    DYNAMIC_SECTION("Testing stack along axis 1 with type " << to_string(type)) {
        auto t1 = Tensor::from_range(make_shape(2, 3), type).unwrap();
        auto t2 = Tensor::from_range(make_shape(2, 3), type).unwrap();

        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;

        REQUIRE_THAT(stack(inputs, 1, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 2);
        REQUIRE(output.shape(2).unwrap() == 3);
        REQUIRE(output.dtype() == type);
    }
}

TEST_CASE("Stack: Stacking along axis 2", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
    auto t2 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();

    Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
    Tensor output;

    REQUIRE_THAT(stack(inputs, 2, output), IsOk());
    REQUIRE(output.shape(0).unwrap() == 2);
    REQUIRE(output.shape(1).unwrap() == 3);
    REQUIRE(output.shape(2).unwrap() == 2);
    REQUIRE(output.dtype() == Dtype::Float32);
}

TEST_CASE("Stack: Negative axis indexing", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
    auto t2 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();

    SECTION("Axis -1 (last axis)") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, -1, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 3);
        REQUIRE(output.shape(2).unwrap() == 2);
    }

    SECTION("Axis -2 (second to last)") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, -2, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 2);
        REQUIRE(output.shape(2).unwrap() == 3);
    }

    SECTION("Axis -3 (first axis)") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, -3, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 2);
        REQUIRE(output.shape(2).unwrap() == 3);
    }
}

TEST_CASE("Stack: Multiple tensors", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
    auto t2 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
    auto t3 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
    auto t4 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();

    Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap(), t3.clone().unwrap(), t4.clone().unwrap()};
    Tensor output;

    REQUIRE_THAT(stack(inputs, 0, output), IsOk());
    REQUIRE(output.shape(0).unwrap() == 4);
    REQUIRE(output.shape(1).unwrap() == 2);
    REQUIRE(output.shape(2).unwrap() == 3);
}

TEST_CASE("Stack: 1D tensors", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(5), Dtype::Float32).unwrap();
    auto t2 = Tensor::from_range(make_shape(5), Dtype::Float32).unwrap();

    SECTION("Axis 0") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, 0, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 5);
    }

    SECTION("Axis 1") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, 1, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 5);
        REQUIRE(output.shape(1).unwrap() == 2);
    }
}

TEST_CASE("Stack: 3D tensors", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(2, 3, 4), Dtype::Float32).unwrap();
    auto t2 = Tensor::from_range(make_shape(2, 3, 4), Dtype::Float32).unwrap();

    SECTION("Axis 0") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, 0, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 2);
        REQUIRE(output.shape(2).unwrap() == 3);
        REQUIRE(output.shape(3).unwrap() == 4);
    }

    SECTION("Axis 1") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, 1, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 2);
        REQUIRE(output.shape(2).unwrap() == 3);
        REQUIRE(output.shape(3).unwrap() == 4);
    }

    SECTION("Axis 2") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, 2, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 3);
        REQUIRE(output.shape(2).unwrap() == 2);
        REQUIRE(output.shape(3).unwrap() == 4);
    }

    SECTION("Axis 3") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, 3, output), IsOk());
        REQUIRE(output.shape(0).unwrap() == 2);
        REQUIRE(output.shape(1).unwrap() == 3);
        REQUIRE(output.shape(2).unwrap() == 4);
        REQUIRE(output.shape(3).unwrap() == 2);
    }
}

TEST_CASE("Stack: Error - Empty input", "[stack]") {
    std::vector<Tensor> inputs;
    Tensor output;

    REQUIRE_THAT(stack(inputs, 0, output), IsErr());
}

TEST_CASE("Stack: Error - Mismatched shapes", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
    auto t2 = Tensor::from_range(make_shape(2, 4), Dtype::Float32).unwrap();

    Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
    Tensor output;

    REQUIRE_THAT(stack(inputs, 0, output), IsErr());
}

TEST_CASE("Stack: Error - Mismatched dtypes", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
    auto t2 = Tensor::from_range(make_shape(2, 3), Dtype::Int32).unwrap();

    Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
    Tensor output;

    REQUIRE_THAT(stack(inputs, 0, output), IsErr());
}

TEST_CASE("Stack: Error - Axis out of bounds", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();
    auto t2 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();

    SECTION("Positive axis too large") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, 3, output), IsErr());
    }

    SECTION("Negative axis too large") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, -4, output), IsErr());
    }
}

TEST_CASE("Stack: Data integrity check", "[stack]") {
    auto t1 = Tensor::zeros(make_shape(2, 2), Dtype::Float32).unwrap();
    auto t2 = Tensor::full(make_shape(2, 2), 1.0, Dtype::Float32).unwrap();

    Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
    Tensor output;

    REQUIRE_THAT(stack(inputs, 0, output), IsOk());

    auto data = output.as_span1d<float>().unwrap();

    // First tensor (all zeros)
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(data[i] == Approx(0.0f));
    }

    // Second tensor (all ones)
    for (size_t i = 4; i < 8; ++i) {
        REQUIRE(data[i] == Approx(1.0f));
    }
}

TEST_CASE("Stack: Data integrity with different values", "[stack]") {
    auto t1 = Tensor::empty(make_shape(2, 2), Dtype::Float32).unwrap();
    auto t2 = Tensor::empty(make_shape(2, 2), Dtype::Float32).unwrap();

    auto data1 = t1.as_span1d<float>().unwrap();
    auto data2 = t2.as_span1d<float>().unwrap();

    for (size_t i = 0; i < 4; ++i) {
        data1[i] = static_cast<float>(i);
        data2[i] = static_cast<float>(i + 10);
    }

    SECTION("Stack along axis 0") {
        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;
        REQUIRE_THAT(stack(inputs, 0, output), IsOk());
        auto out_data = output.as_span1d<float>().unwrap();

        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(out_data[i] == Approx(static_cast<float>(i)));
        }
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(out_data[4 + i] == Approx(static_cast<float>(i + 10)));
        }
    }
}

TEST_CASE("Stack: Single tensor", "[stack]") {
    auto t1 = Tensor::from_range(make_shape(2, 3), Dtype::Float32).unwrap();

    Tensor inputs[] = {t1.clone().unwrap()};
    Tensor output;

    REQUIRE_THAT(stack(inputs, 0, output), IsOk());
    REQUIRE(output.shape(0).unwrap() == 1);
    REQUIRE(output.shape(1).unwrap() == 2);
    REQUIRE(output.shape(2).unwrap() == 3);
}

TEST_CASE("Stack: Different dtypes coverage", "[stack]") {
    auto type = GENERATE(Dtype::Int64, Dtype::Uint8, Dtype::Float64);
    DYNAMIC_SECTION("Testing stack with type " << to_string(type)) {
        auto t1 = Tensor::from_range(make_shape(2, 3), type).unwrap();
        auto t2 = Tensor::from_range(make_shape(2, 3), type).unwrap();

        Tensor inputs[] = {t1.clone().unwrap(), t2.clone().unwrap()};
        Tensor output;

        REQUIRE_THAT(stack(inputs, 0, output), IsOk());
        REQUIRE(output.dtype() == type);
    }
}

}  // namespace p10::op
