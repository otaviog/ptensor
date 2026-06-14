#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <p10_internal/simd/tile.hpp>
#include <ptensor/accessor2D.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

namespace p10::simd {

namespace {
    constexpr size_t SIMD_BLOCK = 4;

    // Fixed 4x4 transpose. Constant bounds let the compiler unroll and vectorize
    // it (gather of 4 rows, scattered store) without hand-written intrinsics.
    void simd_transpose(Accessor2D<const int32_t> src, Accessor2D<int32_t> dst) {
        for (size_t i = 0; i < SIMD_BLOCK; i++) {
            for (size_t j = 0; j < SIMD_BLOCK; j++) {
                dst[j][i] = src[i][j];
            }
        }
    }

    // Generic transpose for the leftover border regions of any shape.
    void scalar_transpose(Accessor2D<const int32_t> src, Accessor2D<int32_t> dst) {
        for (auto i = 0; i < src.rows(); i++) {
            for (auto j = 0; j < src.cols(); j++) {
                dst[j][i] = src[i][j];
            }
        }
    }
}  // namespace

TEST_CASE("Simd::tile2d_autoblock", "[simd][tile]") {
    constexpr size_t SHAPE_WIDTH = 1220;
    constexpr size_t SHAPE_HEIGHT = 1360;
    const auto int32 = TensorOptions().dtype(Dtype::Int32);

    Tensor input_image = Tensor::from_range(make_shape(SHAPE_HEIGHT, SHAPE_WIDTH), int32).unwrap();

    Tensor output_image;
    output_image.create(make_shape(SHAPE_WIDTH, SHAPE_HEIGHT), int32);

    Tensor expected_image;
    expected_image.create(make_shape(SHAPE_WIDTH, SHAPE_HEIGHT), int32);

    const auto src = input_image.as_span2d<const int32_t>().unwrap();
    const auto dst = output_image.as_span2d<int32_t>().unwrap();
    const auto expected = expected_image.as_span2d<int32_t>().unwrap();

    // Reference: transpose the whole image in one shot with the scalar kernel.
    scalar_transpose(
        src({.row = 0, .col = 0, .height = SHAPE_HEIGHT, .width = SHAPE_WIDTH}),
        expected({.row = 0, .col = 0, .height = SHAPE_WIDTH, .width = SHAPE_HEIGHT})
    );

    // The transposed element of a src region lands at the transposed dst region.
    tile2d_autoblock<SIMD_BLOCK, int32_t>(
        SHAPE_HEIGHT,
        SHAPE_WIDTH,
        [&](auto region) { simd_transpose(src(region), dst(region.transposed())); },
        [&](auto region) { scalar_transpose(src(region), dst(region.transposed())); }
    );

    REQUIRE_THAT(testing::compare_tensors(output_image, expected_image), testing::is_ok());
}

TEST_CASE("Simd::tile2d", "[simd][tile]") {
    constexpr size_t SHAPE_WIDTH = 1220;
    constexpr size_t SHAPE_HEIGHT = 1360;
    const auto int32 = TensorOptions().dtype(Dtype::Int32);

    Tensor input_image = Tensor::from_range(make_shape(SHAPE_HEIGHT, SHAPE_WIDTH), int32).unwrap();

    Tensor output_image;
    output_image.create(make_shape(SHAPE_WIDTH, SHAPE_HEIGHT), int32);

    Tensor expected_image;
    expected_image.create(make_shape(SHAPE_WIDTH, SHAPE_HEIGHT), int32);

    const auto src = input_image.as_span2d<const int32_t>().unwrap();
    const auto dst = output_image.as_span2d<int32_t>().unwrap();
    const auto expected = expected_image.as_span2d<int32_t>().unwrap();

    // Reference: transpose the whole image in one shot with the scalar kernel.
    scalar_transpose(
        src({.row = 0, .col = 0, .height = SHAPE_HEIGHT, .width = SHAPE_WIDTH}),
        expected({.row = 0, .col = 0, .height = SHAPE_WIDTH, .width = SHAPE_HEIGHT})
    );

    // The transposed element of a src region lands at the transposed dst region.
    tile2d<int32_t>(
        SHAPE_HEIGHT,
        SHAPE_WIDTH,
        [&](auto region) { scalar_transpose(src(region), dst(region.transposed())); },
        Portable<SIMD_BLOCK>([&](auto region) { simd_transpose(src(region), dst(region.transposed())); })
    );

    REQUIRE_THAT(testing::compare_tensors(output_image, expected_image), testing::is_ok());
}

}  // namespace p10::simd
