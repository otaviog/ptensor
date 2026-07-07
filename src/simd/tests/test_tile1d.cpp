#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <p10_internal/simd/tile1d.hpp>
#include <ptensor/tensor.hpp>
#include <ptensor/testing/catch2_assertions.hpp>
#include <ptensor/testing/compare_tensors.hpp>

namespace p10::simd {

TEST_CASE("Simd::tile1d", "[simd][tile]") {
    // Size deliberately not a multiple of the SIMD chunk so the scalar tail runs.
    constexpr size_t SIZE = 10003;
    constexpr size_t SIMD_CHUNK = 8;
    const auto int32 = TensorOptions().dtype(Dtype::Int32);

    Tensor input = Tensor::from_range(make_shape(SIZE), int32).unwrap();

    Tensor output;
    output.create(make_shape(SIZE), int32);

    const auto src = input.as_span1d<const int32_t>().unwrap();
    const auto dst = output.as_span1d<int32_t>().unwrap();

    // x -> 2x via a fixed SIMD chunk for the body and a scalar tail.
    const auto simd_double = [&](const TileRegion1D& region) {
        for (int64_t k = 0; k < region.size; k++) {
            dst[region.offset + k] = src[region.offset + k] * 2;
        }
    };
    const auto scalar_double = [&](const TileRegion1D& region) {
        for (int64_t k = 0; k < region.size; k++) {
            dst[region.offset + k] = src[region.offset + k] * 2;
        }
    };

    dynamic_tile1d<SIMD_CHUNK, int32_t>(SIZE, simd_double, scalar_double);

    for (size_t i = 0; i < SIZE; i++) {
        REQUIRE(dst[i] == src[i] * 2);
    }
}

}  // namespace p10::simd
