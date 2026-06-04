#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <p10_internal/simd/tile.hpp>

namespace p10::simd {

void generic_impl(Accessor2D<const float> src, Accessor2D<float> dst) {
    for (auto row = 0; row<src.height(); row++) {
        for (auto col = 0; col<src.width(); col++) {
            dst[col][row] = src[row][col];
        }
    }
}

TEST_CASE("Simd::tile", "[bitwise math]") {
    tile<4, 64, float>
        (image.as_span2d<const float>(),
         [](auto region) {
             generic_impl(source.acessor(region), dst.accessor(transpose_region(region)));
         },
         
            
}

}  // namespace p10::simd
