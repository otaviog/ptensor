#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <ptensor/simd/bitwise_math.hpp>

namespace p10::simd {
TEST_CASE("Simd::bitwise_modulo", "[bitwise math]") {
    REQUIRE(bitwise_modulo<8>(0) == 0);
    REQUIRE(bitwise_modulo<8>(1) == 1);
    REQUIRE(bitwise_modulo<8>(7) == 7);
    REQUIRE(bitwise_modulo<8>(8) == 0);
    REQUIRE(bitwise_modulo<8>(9) == 1);
    REQUIRE(bitwise_modulo<8>(15) == 7);
}

}  // namespace p10::simd
