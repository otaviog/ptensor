#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <p10_internal/simd/tile2d.hpp>
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
    auto dst = output_image.as_span2d<int32_t>().unwrap();
    auto expected = expected_image.as_span2d<int32_t>().unwrap();

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
    auto dst = output_image.as_span2d<int32_t>().unwrap();
    auto expected = expected_image.as_span2d<int32_t>().unwrap();

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

TEST_CASE("Simd::tile2d_blocked border", "[simd][tile]") {
    // Paint every cell each kernel visits; the split must cover the domain
    // exactly once, with the SIMD kernel confined to the halo-inset interior.
    constexpr int64_t ROWS = 200;
    constexpr int64_t COLS = 160;
    constexpr TileBorder BORDER {.horizontal = 2, .vertical = 3};
    constexpr size_t CACHE = 32;
    constexpr size_t SIMD = 8;

    std::vector<int> simd_paint(static_cast<size_t>(ROWS * COLS), 0);
    std::vector<int> total_paint(static_cast<size_t>(ROWS * COLS), 0);

    const auto paint = [&](std::vector<int>& grid, const TileRegion2D& region) {
        for (int64_t r = region.row; r < region.row + region.height; r++) {
            for (int64_t c = region.col; c < region.col + region.width; c++) {
                grid[static_cast<size_t>((r * COLS) + c)]++;
            }
        }
    };

    tile2d_blocked<CACHE, SIMD, TileExecution::SEQUENTIAL, BORDER>(
        ROWS,
        COLS,
        [&](const TileRegion2D& region) {
            paint(simd_paint, region);
            paint(total_paint, region);
        },
        [&](const TileRegion2D& region) { paint(total_paint, region); }
    );

    for (int64_t r = 0; r < ROWS; r++) {
        for (int64_t c = 0; c < COLS; c++) {
            const size_t idx = static_cast<size_t>((r * COLS) + c);
            CAPTURE(r, c);
            // Every cell written exactly once (no gaps, no overlap).
            REQUIRE(total_paint[idx] == 1);
            // SIMD cells only inside the inset interior; the frame is scalar-only.
            const bool inset = r >= BORDER.vertical && r < ROWS - BORDER.vertical
                && c >= BORDER.horizontal && c < COLS - BORDER.horizontal;
            if (simd_paint[idx] == 1) {
                REQUIRE(inset);
            }
        }
    }
}

}  // namespace p10::simd
