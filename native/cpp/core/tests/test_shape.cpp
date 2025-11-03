#include <catch2/catch_test_macros.hpp>
#include <ptensor/shape.hpp>
#include <ptensor/testing/catch2_assertions.hpp>

namespace p10 {

// ============================================================================
// Shape Creation Tests
// ============================================================================

TEST_CASE("Shape creation from initializer list", "[shape][creation]") {
    SECTION("1D shape") {
        auto shape = make_shape({5}).unwrap();
        REQUIRE(shape.dims() == 1);
        REQUIRE(shape[0].unwrap() == 5);
        REQUIRE(shape.count() == 5);
    }

    SECTION("2D shape") {
        auto shape = make_shape({2, 3}).unwrap();
        REQUIRE(shape.dims() == 2);
        REQUIRE(shape[0].unwrap() == 2);
        REQUIRE(shape[1].unwrap() == 3);
        REQUIRE(shape.count() == 6);
    }

    SECTION("3D shape") {
        auto shape = make_shape({2, 3, 4}).unwrap();
        REQUIRE(shape.dims() == 3);
        REQUIRE(shape[0].unwrap() == 2);
        REQUIRE(shape[1].unwrap() == 3);
        REQUIRE(shape[2].unwrap() == 4);
        REQUIRE(shape.count() == 24);
    }

    SECTION("4D shape") {
        auto shape = make_shape({2, 3, 4, 5}).unwrap();
        REQUIRE(shape.dims() == 4);
        REQUIRE(shape.count() == 120);
    }
}

TEST_CASE("Shape creation from span", "[shape][creation]") {
    const std::vector<int64_t> dims = {3, 4, 5};
    auto shape = make_shape(std::span<const int64_t>(dims)).unwrap();

    REQUIRE(shape.dims() == 3);
    REQUIRE(shape[0].unwrap() == 3);
    REQUIRE(shape[1].unwrap() == 4);
    REQUIRE(shape[2].unwrap() == 5);
    REQUIRE(shape.count() == 60);
}

TEST_CASE("Shape creation from individual parameters", "[shape][creation]") {
    SECTION("1D shape from single parameter") {
        auto shape = make_shape(7);
        REQUIRE(shape.dims() == 1);
        REQUIRE(shape[0].unwrap() == 7);
        REQUIRE(shape.count() == 7);
    }

    SECTION("2D shape from two parameters") {
        auto shape = make_shape(3, 5);
        REQUIRE(shape.dims() == 2);
        REQUIRE(shape[0].unwrap() == 3);
        REQUIRE(shape[1].unwrap() == 5);
        REQUIRE(shape.count() == 15);
    }

    SECTION("3D shape from three parameters") {
        auto shape = make_shape(2, 4, 6);
        REQUIRE(shape.dims() == 3);
        REQUIRE(shape[0].unwrap() == 2);
        REQUIRE(shape[1].unwrap() == 4);
        REQUIRE(shape[2].unwrap() == 6);
        REQUIRE(shape.count() == 48);
    }

    SECTION("4D shape from four parameters") {
        auto shape = make_shape(2, 3, 4, 5);
        REQUIRE(shape.dims() == 4);
        REQUIRE(shape[0].unwrap() == 2);
        REQUIRE(shape[1].unwrap() == 3);
        REQUIRE(shape[2].unwrap() == 4);
        REQUIRE(shape[3].unwrap() == 5);
        REQUIRE(shape.count() == 120);
    }
}

TEST_CASE("Shape empty initialization", "[shape][creation]") {
    auto shape = make_shape({}).unwrap();
    REQUIRE(shape.dims() == 0);
    REQUIRE(shape.empty());
    REQUIRE(shape.count() == 0);
}

// ============================================================================
// Shape Validation Tests
// ============================================================================

TEST_CASE("Shape exceeding maximum dimensions", "[shape][validation]") {
    REQUIRE_THAT(make_shape({1, 2, 3, 4, 5, 6, 7, 8}), testing::IsOk());
    REQUIRE_THAT(make_shape({1, 2, 3, 4, 5, 6, 7, 8, 9}), testing::IsErr());
}

TEST_CASE("Shape with zero dimensions", "[shape][validation]") {
    SECTION("single zero dimension") {
        auto shape = make_shape({0, 3}).unwrap();
        REQUIRE(shape.dims() == 2);
        REQUIRE(shape[0].unwrap() == 0);
        REQUIRE(shape[1].unwrap() == 3);
        REQUIRE(shape.count() == 0);
    }

    SECTION("multiple zero dimensions") {
        auto shape = make_shape({0, 0, 3}).unwrap();
        REQUIRE(shape.count() == 0);
    }
}

TEST_CASE("Shape with large dimensions", "[shape][validation]") {
    auto shape = make_shape({1000, 2000}).unwrap();
    REQUIRE(shape.count() == 2000000);
}

// ============================================================================
// Shape Properties Tests
// ============================================================================

TEST_CASE("Shape::count calculates total elements", "[shape][properties]") {
    REQUIRE(make_shape({}).unwrap().count() == 0);
    REQUIRE(make_shape({5}).unwrap().count() == 5);
    REQUIRE(make_shape({2, 3}).unwrap().count() == 6);
    REQUIRE(make_shape({2, 3, 4}).unwrap().count() == 24);
    REQUIRE(make_shape({2, 3, 4, 5}).unwrap().count() == 120);
    REQUIRE(make_shape({10, 10, 10}).unwrap().count() == 1000);
}

TEST_CASE("Shape::dims returns number of dimensions", "[shape][properties]") {
    REQUIRE(make_shape({}).unwrap().dims() == 0);
    REQUIRE(make_shape({5}).unwrap().dims() == 1);
    REQUIRE(make_shape({2, 3}).unwrap().dims() == 2);
    REQUIRE(make_shape({2, 3, 4}).unwrap().dims() == 3);
    REQUIRE(make_shape({2, 3, 4, 5, 6, 7, 8}).unwrap().dims() == 7);
}

TEST_CASE("Shape::empty detects empty shape", "[shape][properties]") {
    REQUIRE(make_shape({}).unwrap().empty());
    REQUIRE_FALSE(make_shape({1}).unwrap().empty());
    REQUIRE_FALSE(make_shape({2, 3}).unwrap().empty());
}

TEST_CASE("Shape::operator[] accesses dimensions", "[shape][properties]") {
    auto shape = make_shape({2, 3, 4}).unwrap();

    SECTION("valid indices") {
        REQUIRE_THAT(shape[0], testing::IsOk());
        REQUIRE_THAT(shape[1], testing::IsOk());
        REQUIRE_THAT(shape[2], testing::IsOk());
        REQUIRE(shape[0].unwrap() == 2);
        REQUIRE(shape[1].unwrap() == 3);
        REQUIRE(shape[2].unwrap() == 4);
    }

    SECTION("out of bounds index") {
        REQUIRE_THAT(shape[3], testing::IsErr());
        REQUIRE_THAT(shape[10], testing::IsErr());
    }
}

// ============================================================================
// Shape Comparison Tests
// ============================================================================

TEST_CASE("Shape equality comparison", "[shape][comparison]") {
    SECTION("equal shapes") {
        auto shape1 = make_shape({2, 3, 4}).unwrap();
        auto shape2 = make_shape({2, 3, 4}).unwrap();
        REQUIRE(shape1 == shape2);
        REQUIRE_FALSE(shape1 != shape2);
    }

    SECTION("different dimensions") {
        auto shape1 = make_shape({2, 3}).unwrap();
        auto shape2 = make_shape({2, 4}).unwrap();
        REQUIRE(shape1 != shape2);
        REQUIRE_FALSE(shape1 == shape2);
    }

    SECTION("different number of dims") {
        auto shape1 = make_shape({2, 3}).unwrap();
        auto shape2 = make_shape({2, 3, 4}).unwrap();
        REQUIRE(shape1 != shape2);
    }

    SECTION("empty shapes") {
        auto shape1 = make_shape({}).unwrap();
        auto shape2 = make_shape({}).unwrap();
        REQUIRE(shape1 == shape2);
    }
}

// ============================================================================
// Shape Iteration Tests
// ============================================================================

TEST_CASE("Shape iteration with begin/end", "[shape][iteration]") {
    auto shape = make_shape({2, 3, 4}).unwrap();

    std::vector<int64_t> collected;
    for (auto it = shape.begin(); it != shape.end(); ++it) {
        collected.push_back(*it);
    }

    REQUIRE(collected.size() == 3);
    REQUIRE(collected[0] == 2);
    REQUIRE(collected[1] == 3);
    REQUIRE(collected[2] == 4);
}

TEST_CASE("Shape as_span conversion", "[shape][iteration]") {
    auto shape = make_shape({2, 3, 4}).unwrap();
    auto span = shape.as_span();

    REQUIRE(span.size() == 3);
    REQUIRE(span[0] == 2);
    REQUIRE(span[1] == 3);
    REQUIRE(span[2] == 4);
}

// ============================================================================
// String Conversion Tests
// ============================================================================

TEST_CASE("Shape to_string conversion", "[shape][formatting]") {
    REQUIRE(to_string(make_shape({}).unwrap()) == "[]");
    REQUIRE(to_string(make_shape({5}).unwrap()) == "[5]");
    REQUIRE(to_string(make_shape({2, 3}).unwrap()) == "[2, 3]");
    REQUIRE(to_string(make_shape({2, 3, 4}).unwrap()) == "[2, 3, 4]");
    REQUIRE(to_string(make_shape({1, 2, 3, 4, 5}).unwrap()) == "[1, 2, 3, 4, 5]");
}

}  // namespace p10
