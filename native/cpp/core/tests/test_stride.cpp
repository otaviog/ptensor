#include <catch2/catch_test_macros.hpp>
#include <ptensor/shape.hpp>
#include <ptensor/stride.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

#include "ptensor/p10_error.hpp"

namespace p10 {

// ============================================================================
// Stride Creation Tests
// ============================================================================

TEST_CASE("Stride creation from shape", "[stride][creation]") {
    SECTION("Empty shape (0D)") {
        Shape shape;
        Stride stride = Stride::from_contiguous_shape(shape);
        REQUIRE(stride.dims() == 0);
    }

    SECTION("1D shape") {
        auto shape = make_shape(5);
        Stride stride = Stride::from_contiguous_shape(shape);
        REQUIRE(stride.dims() == 1);
        REQUIRE(stride[0].unwrap() == 1);
    }

    SECTION("2D shape - row major") {
        auto shape = make_shape(2, 3);
        Stride stride = Stride::from_contiguous_shape(shape);
        REQUIRE(stride.dims() == 2);
        REQUIRE(stride[0].unwrap() == 3);  // stride along first dim
        REQUIRE(stride[1].unwrap() == 1);  // stride along second dim
    }

    SECTION("3D shape - row major") {
        auto shape = make_shape(2, 3, 4);
        Stride stride = Stride::from_contiguous_shape(shape);
        REQUIRE(stride.dims() == 3);
        REQUIRE(stride[0].unwrap() == 12);  // 3 * 4
        REQUIRE(stride[1].unwrap() == 4);
        REQUIRE(stride[2].unwrap() == 1);
    }

    SECTION("4D shape - row major") {
        auto shape = make_shape(2, 3, 4, 5);
        Stride stride = Stride::from_contiguous_shape(shape);
        REQUIRE(stride.dims() == 4);
        REQUIRE(stride[0].unwrap() == 60);  // 3 * 4 * 5
        REQUIRE(stride[1].unwrap() == 20);  // 4 * 5
        REQUIRE(stride[2].unwrap() == 5);
        REQUIRE(stride[3].unwrap() == 1);
    }
}

TEST_CASE("Stride creation from initializer list", "[stride][creation]") {
    auto stride = make_stride(12, 4, 1);
    REQUIRE(stride.dims() == 3);
    REQUIRE(stride[0].unwrap() == 12);
    REQUIRE(stride[1].unwrap() == 4);
    REQUIRE(stride[2].unwrap() == 1);
}

TEST_CASE("Stride creation from span", "[stride][creation]") {
    const std::vector<int64_t> strides_vec = {6, 2, 1};
    auto stride = make_stride(std::span<const int64_t>(strides_vec)).unwrap();

    REQUIRE(stride.dims() == 3);
    REQUIRE(stride[0].unwrap() == 6);
    REQUIRE(stride[1].unwrap() == 2);
    REQUIRE(stride[2].unwrap() == 1);
}

// ============================================================================
// Stride Validation Tests
// ============================================================================

TEST_CASE("Stride exceeding maximum dimensions", "[stride][validation]") {
    REQUIRE_THAT(make_stride({1, 2, 3, 4, 5, 6, 7, 8}), testing::IsOk());
    REQUIRE_THAT(make_stride({1, 2, 3, 4, 5, 6, 7, 8, 9}), testing::IsError(P10Error::OutOfRange));
}

// ============================================================================
// String Conversion Tests
// ============================================================================

TEST_CASE("Stride to_string conversion", "[stride][formatting]") {
    REQUIRE(to_string(Stride()) == "[]");
    REQUIRE(to_string(make_stride(1)) == "[1]");
    REQUIRE(to_string(make_stride(3, 1)) == "[3, 1]");
    REQUIRE(to_string(make_stride(12, 4, 1)) == "[12, 4, 1]");
}

}  // namespace p10
