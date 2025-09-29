#include <catch2/catch_test_macros.hpp>
#include <ptensor/shape.hpp>

namespace p10 {
TEST_CASE("Shape::Many dims detection", "[tensor]") {
    REQUIRE_FALSE(make_shape({2, 3, 4, 5, 6, 7, 8, 9, 10}).is_ok());
}
}  // namespace p10